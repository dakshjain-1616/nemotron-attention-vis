#!/usr/bin/env python3
"""
Nemotron Attention Visualizer
------------------------------
Loads a HuggingFace causal-LM (default: nvidia/Nemotron-Cascade-2-30B-A3B),
runs a prompt through it with attention outputs enabled, extracts per-layer
per-head attention matrices, and renders an interactive HTML heatmap.

Usage:
    python visualize_attention.py --prompt "Explain quantum computing"
    python visualize_attention.py --model Qwen/Qwen3-0.6B --prompt "Hello world"
    python visualize_attention.py --mock   # no download needed
    python visualize_attention.py --mock --export-csv --rollout-view
    python visualize_attention.py --compare "Prompt A" "Prompt B" --mock
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__version__ = "1.0.0"

# ─── Rich setup ───────────────────────────────────────────────────────────────

try:
    from rich.console import Console as _RichConsole
    _console = _RichConsole()
    _RICH = True
except ImportError:
    _console = None
    _RICH = False


def _rprint(msg: str, style: str = "") -> None:
    """Print using Rich styling when available, else plain print."""
    if _RICH:
        _console.print(msg, style=style)
    else:
        print(msg)

# ─── Environment helpers ──────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


def _env_list(key: str) -> Optional[List[int]]:
    """Parse comma-separated ints from an env var; return None if empty."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None


# ─── New Feature 1: Attention Rollout ─────────────────────────────────────────

def compute_attention_rollout(attentions: List[np.ndarray]) -> np.ndarray:
    """
    Compute attention rollout (Abnar & Zuidema, 2020).

    Each layer's attention is averaged across heads, mixed with the identity
    matrix (to account for residual connections), normalized, and multiplied
    through all layers.  The result shows the effective attention path from
    input tokens to the final representation.

    Parameters
    ----------
    attentions : list of np.ndarray, shape (num_heads, seq, seq) per layer

    Returns
    -------
    np.ndarray of shape (seq, seq) — rollout attention matrix, rows sum to 1
    """
    if not attentions:
        raise ValueError("attentions list is empty")

    seq_len = attentions[0].shape[-1]
    rollout = np.eye(seq_len, dtype=np.float32)

    for layer_attn in attentions:
        # Average across all heads → (seq, seq)
        avg_attn = layer_attn.mean(axis=0).astype(np.float32)
        # Mix with identity to model skip/residual connections
        mixed = 0.5 * avg_attn + 0.5 * np.eye(seq_len, dtype=np.float32)
        # Row-normalise so each token's distribution sums to 1
        row_sums = mixed.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        mixed = mixed / row_sums
        rollout = mixed @ rollout

    return rollout


# ─── New Feature 2: Token Importance Scoring ──────────────────────────────────

def compute_token_importance(attentions: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-token importance as mean received attention across all layers and heads.

    A token is "important" if many other tokens attend strongly to it.
    Formally: column sums of the layer/head-averaged attention matrix, normalised.

    Parameters
    ----------
    attentions : list of np.ndarray, shape (num_heads, seq, seq) per layer

    Returns
    -------
    np.ndarray of shape (seq,) — importance score per token, sums to 1
    """
    if not attentions:
        raise ValueError("attentions list is empty")

    all_layers = np.stack(attentions, axis=0)          # (L, H, seq, seq)
    avg = all_layers.mean(axis=(0, 1))                  # (seq, seq)
    col_sums = avg.sum(axis=0)                          # (seq,)
    total = col_sums.sum()
    return col_sums / total if total > 0 else col_sums


# ─── New Feature 3: CSV Export ────────────────────────────────────────────────

def export_attention_csv(attention_data: Dict, csv_path: "Path") -> "Path":
    """
    Export per-layer, per-head attention statistics as a CSV file.

    Columns: layer_idx, head_idx, entropy_mean, max_weight, gini_coefficient

    Parameters
    ----------
    attention_data : dict from extractor.extract()
    csv_path       : destination path (parent dirs created automatically)

    Returns
    -------
    Path to the created CSV file
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for real_layer_idx, arr in zip(attention_data["layers"], attention_data["attentions"]):
        for head_idx in range(arr.shape[0]):
            head = arr[head_idx]  # (seq, seq)
            entropy_per_row = -np.sum(head * np.log(head + 1e-10), axis=-1)
            flat = head.flatten()
            flat_sorted = np.sort(flat)
            n = len(flat_sorted)
            denom = n * flat_sorted.sum() + 1e-10
            gini = float(
                (2 * np.dot(flat_sorted, np.arange(1, n + 1)) - (n + 1) * flat_sorted.sum())
                / denom
            )
            rows.append({
                "layer_idx": real_layer_idx,
                "head_idx": head_idx,
                "entropy_mean": round(float(entropy_per_row.mean()), 6),
                "max_weight": round(float(head.max()), 6),
                "gini_coefficient": round(max(0.0, gini), 6),
            })

    fieldnames = ["layer_idx", "head_idx", "entropy_mean", "max_weight", "gini_coefficient"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[renderer] Saved attention CSV  → {csv_path}")
    return csv_path


# ─── New Feature 4: Multi-prompt Comparison ───────────────────────────────────

def compare_prompts(
    prompts: List[str],
    model_name: Optional[str] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
    comparison_filename: str = "comparison.html",
    num_layers: int = 8,
    num_heads: int = 16,
    mock_seed: Optional[int] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> "Path":
    """
    Run multiple prompts through the same model and produce a comparison HTML.

    The output page shows side-by-side token importance charts, per-prompt
    attention statistics, and rollout final-row comparisons.

    Parameters
    ----------
    prompts             : list of input strings to compare
    model_name          : HuggingFace model ID
    mock                : use synthetic data
    output_dir          : output directory
    comparison_filename : filename for the generated HTML
    num_layers          : (mock) simulated layers
    num_heads           : (mock) simulated heads per layer
    mock_seed           : base RNG seed (incremented per prompt)
    device              : torch device string
    hf_token            : HuggingFace token

    Returns
    -------
    Path to the generated comparison HTML file
    """
    if not prompts:
        raise ValueError("provide at least one prompt")

    resolved_model = model_name or _env("MODEL_NAME", "nvidia/Nemotron-Cascade-2-30B-A3B")
    resolved_mock = mock or _env_bool("MOCK_MODE")
    out_dir = Path(output_dir or _env("OUTPUT_DIR", "outputs"))
    base_seed = mock_seed if mock_seed is not None else _env_int("MOCK_SEED", 42)

    if resolved_mock:
        extractor: "MockAttentionExtractor | AttentionExtractor" = MockAttentionExtractor(
            model_name=resolved_model,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=base_seed,
        )
        extractor.load()
    else:
        extractor = AttentionExtractor(
            model_name=resolved_model,
            device=device,
            hf_token=hf_token,
        )
        extractor.load()

    comparison_results = []
    for i, prompt in enumerate(prompts):
        if resolved_mock:
            extractor.seed = base_seed + i  # type: ignore[union-attr]
        data = extractor.extract(prompt)
        importance = compute_token_importance(data["attentions"])
        rollout = compute_attention_rollout(data["attentions"])
        stats = HeatmapRenderer._compute_stats(data["attentions"])
        comparison_results.append({
            "prompt": prompt,
            "tokens": data["tokens"],
            "seq_len": data["seq_len"],
            "importance": importance.tolist(),
            "rollout_last_row": rollout[-1].tolist(),
            "stats": stats,
        })

    html_content = _render_comparison_html(comparison_results, resolved_model, resolved_mock)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / comparison_filename
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[compare]  Saved comparison     → {out_path}")
    return out_path


def _render_comparison_html(
    results: List[Dict],
    model_name: str,
    mock: bool,
    color_scale: Optional[str] = None,
) -> str:
    import html as _html
    import json as _json

    resolved_color_scale = color_scale or _env("COLOR_SCALE", "Viridis")
    plotly_version = _env("PLOTLY_CDN_VERSION", "2.27.0")
    mock_badge = ' <span style="background:#ff8c00;color:#000;font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:20px;vertical-align:middle;margin-left:8px;">MOCK DATA</span>' if mock else ""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_json = _json.dumps({"results": results, "model": model_name, "generated_at": ts, "color_scale": resolved_color_scale})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nemotron Attention Comparison — {_html.escape(model_name)}</title>
<script src="https://cdn.plot.ly/plotly-{plotly_version}.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d0d14; color: #e0e0f0; padding: 20px; }}
  header {{ text-align: center; padding: 24px 0 16px; border-bottom: 1px solid #2a2a40; margin-bottom: 24px; }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header .sub {{ font-size: 0.82rem; color: #888; margin-top: 6px; }}
  header .sub code {{ background: #1e1e2e; padding: 2px 6px; border-radius: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin-bottom: 24px; }}
  .card {{ background: #141420; border: 1px solid #2a2a40; border-radius: 10px; padding: 16px; }}
  .card h3 {{ font-size: 0.95rem; color: #c0c0e0; margin-bottom: 10px; }}
  .card .prompt-label {{ font-size: 0.8rem; color: #7b61ff; margin-bottom: 8px; font-style: italic; }}
  .stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 12px; }}
  .stats-table th, .stats-table td {{ padding: 7px 10px; border-bottom: 1px solid #2a2a40; text-align: right; }}
  .stats-table th {{ color: #888; text-align: left; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; }}
  .stats-table td:first-child {{ text-align: left; color: #aaa; }}
  footer {{ text-align: center; font-size: 0.72rem; color: #555; padding-top: 16px; border-top: 1px solid #1e1e2e; margin-top: 20px; }}
</style>
</head>
<body>
<header>
  <h1>&#x1F9E0; Attention Comparison{mock_badge}</h1>
  <div class="sub">Model: <code>{_html.escape(model_name)}</code> &nbsp;|&nbsp; {len(results)} prompts &nbsp;|&nbsp; Generated: {ts}</div>
</header>
<div id="charts-grid" class="grid"></div>
<div class="card">
  <h3>Statistics Summary</h3>
  <table class="stats-table" id="summary-table">
    <thead><tr><th>Prompt</th><th>Mean Entropy</th><th>Max Weight</th><th>Min Non-zero</th><th>Tokens</th></tr></thead>
    <tbody id="summary-body"></tbody>
  </table>
</div>
<footer>Nemotron Attention Visualizer &mdash; Comparison Report &mdash; {_html.escape(model_name)}</footer>
<script>
const DATA = {data_json};

const grid = document.getElementById('charts-grid');
const tbody = document.getElementById('summary-body');

DATA.results.forEach((r, idx) => {{
  // Importance chart card
  const card = document.createElement('div');
  card.className = 'card';
  const chartId = 'imp_' + idx;
  card.innerHTML = '<div class="prompt-label">&ldquo;' + r.prompt + '&rdquo;</div>' +
                   '<div id="' + chartId + '"></div>';
  grid.appendChild(card);

  // Sort tokens by importance
  const pairs = r.tokens.map((t, i) => [t, r.importance[i]]).sort((a, b) => b[1] - a[1]);
  const topN = pairs.slice(0, Math.min(15, pairs.length));

  Plotly.newPlot(chartId, [{{
    x: topN.map(p => p[1]),
    y: topN.map(p => p[0]),
    type: 'bar',
    orientation: 'h',
    marker: {{ color: topN.map(p => p[1]), colorscale: DATA.color_scale || 'Viridis', showscale: false }},
    hovertemplate: '<b>%{{y}}</b>: %{{x:.4f}}<extra></extra>',
  }}], {{
    paper_bgcolor: '#141420',
    plot_bgcolor: '#141420',
    font: {{ color: '#e0e0f0', size: 10 }},
    margin: {{ t: 30, r: 10, b: 40, l: 90 }},
    title: {{ text: 'Token Importance (top ' + topN.length + ')', font: {{ size: 12, color: '#c0c0e0' }} }},
    xaxis: {{ gridcolor: '#1e1e2e', tickfont: {{ color: '#aaa' }}, title: {{ text: 'Score', font: {{ color: '#888', size: 10 }} }} }},
    yaxis: {{ tickfont: {{ color: '#aaa', size: 9 }}, gridcolor: '#1e1e2e' }},
    height: 280,
  }}, {{ responsive: true, displayModeBar: false }});

  // Summary table row
  const s = r.stats;
  const row = document.createElement('tr');
  const shortPrompt = r.prompt.length > 40 ? r.prompt.slice(0, 40) + '…' : r.prompt;
  row.innerHTML = '<td title="' + r.prompt + '">' + shortPrompt + '</td>' +
                  '<td>' + s.entropy_mean.toFixed(4) + '</td>' +
                  '<td>' + s.max.toFixed(4) + '</td>' +
                  '<td>' + s.min_nonzero.toFixed(4) + '</td>' +
                  '<td>' + r.seq_len + '</td>';
  tbody.appendChild(row);
}});
</script>
</body>
</html>"""


# ─── Attention extractor ──────────────────────────────────────────────────────

class AttentionExtractor:
    """Loads a transformer model and extracts multi-layer attention weights."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_tokens: int = 128,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.hf_token = hf_token or _env("HF_TOKEN") or None
        self.device = device or _env("DEVICE") or self._detect_device()
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load(self, max_retries: Optional[int] = None) -> "AttentionExtractor":
        """Download and load model + tokenizer with automatic retry on transient errors."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        retries = max_retries if max_retries is not None else _env_int("LOAD_RETRIES", 2)
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                print(f"[loader] Loading tokenizer: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    trust_remote_code=True,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                print(f"[loader] Loading model on device={self.device} ...")
                load_kwargs: Dict = dict(
                    token=self.hf_token,
                    trust_remote_code=True,
                    output_attentions=True,
                )
                if self.device == "cuda":
                    load_kwargs["torch_dtype"] = torch.bfloat16
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs
                )
                if self.device not in ("cuda",):
                    self.model = self.model.to(self.device)
                self.model.eval()
                print("[loader] Model ready.")
                return self

            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait_s = 2 ** attempt
                    print(
                        f"[loader] Attempt {attempt + 1}/{retries + 1} failed: {e}\n"
                        f"[loader] Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)
                else:
                    raise RuntimeError(
                        f"Failed to load model '{self.model_name}' after {retries + 1} attempt(s).\n"
                        f"Last error: {last_error}\n"
                        f"Tips:\n"
                        f"  • Set MOCK_MODE=true to use synthetic data (no download needed)\n"
                        f"  • Set HF_TOKEN=<your_token> for gated models\n"
                        f"  • Set FALLBACK_MODEL=Qwen/Qwen3-0.6B for a small open model\n"
                        f"  • Set DEVICE=cpu to avoid GPU memory issues"
                    ) from last_error

        return self  # unreachable; satisfies type checker

    def extract(self, prompt: str) -> Dict:
        """
        Run a forward pass and return attention data.

        Returns
        -------
        dict with keys:
            tokens   : list[str]
            layers   : list[int]   — layer indices that produced attention
            attentions: list[np.ndarray]  shape (num_heads, seq, seq) per layer
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load() before .extract().")

        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = [t.replace("▁", " ").replace("Ġ", " ").strip() or t for t in tokens]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        raw_attentions = outputs.attentions
        if raw_attentions is None:
            raise RuntimeError(
                "Model did not return attentions. "
                "Some hybrid SSM/Mamba layers do not produce attention maps; "
                "only pure-attention layers are captured."
            )

        layers = []
        attentions = []
        for layer_idx, attn in enumerate(raw_attentions):
            if attn is None:
                continue
            arr = attn[0].float().cpu().numpy()
            layers.append(layer_idx)
            attentions.append(arr)

        return {
            "model": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "layers": layers,
            "attentions": attentions,
            "num_attention_layers": len(layers),
            "num_heads": attentions[0].shape[0] if attentions else 0,
            "seq_len": len(tokens),
        }


# ─── Mock extractor (no model download) ──────────────────────────────────────

class MockAttentionExtractor:
    """
    Generates realistic synthetic attention patterns without loading a model.
    Useful for testing, CI, and demos on machines without GPU/large RAM.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-Cascade-2-30B-A3B",
        num_layers: int = 8,
        num_heads: int = 16,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seed = seed if seed is not None else _env_int("MOCK_SEED", 42)

    def load(self) -> "MockAttentionExtractor":
        print(f"[mock] Mock mode active — simulating {self.model_name}")
        return self

    def extract(self, prompt: str) -> Dict:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        rng = np.random.default_rng(self.seed)

        raw_tokens = prompt.split()
        tokens = ["<s>"] + raw_tokens + ["</s>"]
        seq_len = len(tokens)

        attentions = []
        for layer_idx in range(self.num_layers):
            heads = []
            for head_idx in range(self.num_heads):
                mat = self._generate_head(rng, seq_len, layer_idx, head_idx)
                heads.append(mat)
            attentions.append(np.stack(heads, axis=0))

        layers = list(range(self.num_layers))
        return {
            "model": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "layers": layers,
            "attentions": attentions,
            "num_attention_layers": self.num_layers,
            "num_heads": self.num_heads,
            "seq_len": seq_len,
            "mock": True,
        }

    def _generate_head(
        self,
        rng: np.random.Generator,
        seq_len: int,
        layer_idx: int,
        head_idx: int,
    ) -> np.ndarray:
        """Generate a plausible causal attention pattern."""
        mat = np.zeros((seq_len, seq_len), dtype=np.float32)
        pattern = (layer_idx * self.num_heads + head_idx) % 4

        for i in range(seq_len):
            raw = np.zeros(seq_len, dtype=np.float32)

            if pattern == 0:
                window = max(1, seq_len // 4)
                for j in range(max(0, i - window), i + 1):
                    raw[j] = rng.exponential(1.0 / (abs(i - j) + 1))

            elif pattern == 1:
                raw[0] = rng.uniform(0.3, 0.7)
                raw[max(0, i)] = rng.uniform(0.2, 0.5)
                raw = raw + rng.uniform(0, 0.05, seq_len)

            elif pattern == 2:
                raw[i] = rng.uniform(0.5, 0.9)
                for j in range(max(0, i - 2), i):
                    raw[j] = rng.uniform(0.05, 0.3)
                raw = raw + rng.uniform(0, 0.02, seq_len)

            else:
                raw = rng.dirichlet(np.ones(seq_len) * 0.5 * (layer_idx + 1))

            raw[i + 1 :] = 0.0
            raw_sum = raw.sum()
            if raw_sum > 0:
                mat[i] = raw / raw_sum
            else:
                mat[i, i] = 1.0

        return mat


# ─── HTML renderer ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nemotron Attention Map — {model_name}</title>
<script src="https://cdn.plot.ly/plotly-{plotly_cdn_version}.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d0d14;
    color: #e0e0f0;
    min-height: 100vh;
    padding: 20px;
  }}
  header {{
    text-align: center;
    padding: 24px 0 16px;
    border-bottom: 1px solid #2a2a40;
    margin-bottom: 24px;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: 0.02em; }}
  header .subtitle {{ font-size: 0.85rem; color: #888; margin-top: 6px; }}
  header .subtitle code {{ background: #1e1e2e; padding: 2px 6px; border-radius: 4px; }}
  .view-toggle {{
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    justify-content: center;
    flex-wrap: wrap;
  }}
  .view-toggle button {{
    background: #1e1e2e;
    color: #e0e0f0;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }}
  .view-toggle button.active {{
    background: #7b61ff;
    border-color: #7b61ff;
    color: #fff;
    font-weight: 600;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
    justify-content: center;
    margin-bottom: 20px;
    padding: 16px;
    background: #141420;
    border-radius: 10px;
    border: 1px solid #2a2a40;
  }}
  .control-group {{ display: flex; flex-direction: column; gap: 6px; }}
  .control-group label {{ font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.08em; }}
  select, input[type=range] {{
    background: #1e1e2e;
    color: #e0e0f0;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.9rem;
    cursor: pointer;
    min-width: 180px;
  }}
  select:focus {{ outline: 2px solid #7b61ff; }}
  .range-row {{ display: flex; align-items: center; gap: 8px; }}
  .range-val {{ font-size: 0.85rem; color: #7b61ff; min-width: 28px; text-align: right; }}
  .panel {{
    background: #141420;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 20px;
  }}
  .panel h3 {{
    font-size: 1rem;
    color: #c0c0e0;
    margin-bottom: 12px;
    font-weight: 600;
  }}
  #heatmap {{ width: 100%; }}
  .stats-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin-bottom: 20px;
  }}
  .stat-chip {{
    background: #1e1e2e;
    border: 1px solid #3a3a5a;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.8rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }}
  .stat-chip .val {{ font-size: 1.1rem; font-weight: 700; color: #7b61ff; }}
  .layer-stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }}
  .layer-stats-table th, .layer-stats-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid #2a2a40;
    text-align: right;
  }}
  .layer-stats-table th {{
    color: #888;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
  }}
  .layer-stats-table td:first-child {{ text-align: left; color: #7b61ff; font-weight: 600; }}
  .token-list {{
    background: #141420;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 16px;
    font-size: 0.85rem;
    word-break: break-all;
    line-height: 2;
    margin-bottom: 20px;
  }}
  .token-list .tok {{
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #3a3a5a;
    border-radius: 4px;
    padding: 1px 6px;
    margin: 2px;
    cursor: pointer;
    transition: background 0.15s;
  }}
  .token-list .tok:hover {{ background: #2e2e4e; }}
  .mock-badge {{
    display: inline-block;
    background: #ff8c00;
    color: #000;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    vertical-align: middle;
    margin-left: 8px;
  }}
  .export-btn {{
    background: #1e1e2e;
    color: #7b61ff;
    border: 1px solid #7b61ff;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: background 0.15s;
    margin-left: 12px;
  }}
  .export-btn:hover {{ background: #7b61ff; color: #fff; }}
  footer {{
    text-align: center;
    font-size: 0.75rem;
    color: #555;
    padding-top: 20px;
    border-top: 1px solid #1e1e2e;
    margin-top: 24px;
  }}
  footer a {{ color: #7b61ff; text-decoration: none; }}
</style>
</head>
<body>
<header>
  <h1>&#x1F9E0; Nemotron Attention Visualizer{mock_badge}</h1>
  <div class="subtitle">
    Model: <code>{model_name}</code> &nbsp;|&nbsp;
    Prompt: <code>{prompt_escaped}</code> &nbsp;|&nbsp;
    <span style="color:#555">{generated_at}</span>
  </div>
</header>

<div class="stats-bar">
  <div class="stat-chip"><span class="val">{num_tokens}</span> Tokens</div>
  <div class="stat-chip"><span class="val">{num_layers}</span> Attn Layers</div>
  <div class="stat-chip"><span class="val">{num_heads}</span> Heads / Layer</div>
  <div class="stat-chip"><span class="val">{entropy_mean}</span> Mean Entropy</div>
  <div class="stat-chip"><span class="val">{max_weight}</span> Peak Weight</div>
</div>

<div class="view-toggle">
  <button id="btnHeatmap" class="active" onclick="switchView('heatmap')">&#x1F525; Attention Heatmap</button>
  <button id="btnRollout" onclick="switchView('rollout')">&#x1F300; Rollout View</button>
  <button id="btnImportance" onclick="switchView('importance')">&#x1F4CA; Token Importance</button>
  <button id="btnStats" onclick="switchView('stats')">&#x1F4CB; Layer Stats</button>
</div>

<!-- ── View: Attention Heatmap ── -->
<div id="view-heatmap">
  <div class="controls">
    <div class="control-group">
      <label>Attention Layer</label>
      <select id="layerSelect" onchange="updatePlot()"></select>
    </div>
    <div class="control-group">
      <label>Attention Head</label>
      <select id="headSelect" onchange="updatePlot()"></select>
    </div>
    <div class="control-group">
      <label>Color Scale</label>
      <select id="colorSelect" onchange="updatePlot()">
        <option value="Viridis" selected>Viridis</option>
        <option value="Plasma">Plasma</option>
        <option value="Hot">Hot</option>
        <option value="Blues">Blues</option>
        <option value="YlOrRd">YlOrRd</option>
        <option value="RdPu">RdPu</option>
      </select>
    </div>
    <div class="control-group">
      <label>Opacity <span id="opacityVal">0.9</span></label>
      <div class="range-row">
        <input type="range" id="opacityRange" min="0.3" max="1.0" step="0.05" value="0.9"
               oninput="document.getElementById('opacityVal').textContent=this.value; updatePlot()">
      </div>
    </div>
    <div class="control-group">
      <label>Aggregate</label>
      <select id="aggregateSelect" onchange="updatePlot()">
        <option value="none">Single Head</option>
        <option value="mean">Mean (all heads)</option>
        <option value="max">Max (all heads)</option>
      </select>
    </div>
  </div>
  <div class="panel">
    <div id="heatmap"></div>
  </div>
</div>

<!-- ── View: Rollout ── -->
<div id="view-rollout" style="display:none">
  <div class="panel">
    <h3>Attention Rollout</h3>
    <p style="font-size:0.82rem;color:#888;margin-bottom:12px">
      Effective attention after propagating through all {num_layers} layers
      (Abnar &amp; Zuidema 2020). Rows represent query tokens; columns represent
      keys. Higher values mean stronger end-to-end attention signal.
    </p>
    <div id="rollout-chart"></div>
  </div>
</div>

<!-- ── View: Token Importance ── -->
<div id="view-importance" style="display:none">
  <div class="panel">
    <h3>Token Importance</h3>
    <p style="font-size:0.82rem;color:#888;margin-bottom:12px">
      Mean received attention per token, aggregated across all layers and heads.
      High-importance tokens are those most frequently attended to by other tokens.
    </p>
    <div id="importance-chart"></div>
  </div>
</div>

<!-- ── View: Layer Stats ── -->
<div id="view-stats" style="display:none">
  <div class="panel">
    <h3>Per-Layer Attention Statistics
      <button class="export-btn" onclick="downloadCSV()">&#x2B07; Download CSV</button>
    </h3>
    <table class="layer-stats-table">
      <thead>
        <tr>
          <th>Layer</th>
          <th>Mean Entropy</th>
          <th>Max Weight</th>
          <th>Gini Coeff</th>
          <th>Heads</th>
        </tr>
      </thead>
      <tbody id="layer-stats-body"></tbody>
    </table>
  </div>
</div>

<div class="token-list">
  <strong>Tokens ({num_tokens}):</strong><br>
  {token_chips}
</div>

<footer>
  Generated by <a href="https://heyneo.so">NEO</a> &mdash;
  Nemotron Attention Visualizer &mdash;
  <a href="https://huggingface.co/{model_name_url}">Model on HuggingFace</a>
</footer>

<script>
const DATA = {data_json};

const layerSelect     = document.getElementById('layerSelect');
const headSelect      = document.getElementById('headSelect');
const colorSelect     = document.getElementById('colorSelect');
const opacityRange    = document.getElementById('opacityRange');
const aggregateSelect = document.getElementById('aggregateSelect');

// Populate layer dropdown
DATA.layers.forEach((lyr, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = 'Layer ' + lyr;
  layerSelect.appendChild(opt);
}});

function populateHeads(numHeads) {{
  headSelect.innerHTML = '';
  for (let h = 0; h < numHeads; h++) {{
    const opt = document.createElement('option');
    opt.value = h;
    opt.textContent = 'Head ' + h;
    headSelect.appendChild(opt);
  }}
}}
populateHeads(DATA.num_heads);

// ── View switching ──
let rolloutRendered = false;
let importanceRendered = false;

function switchView(view) {{
  ['heatmap','rollout','importance','stats'].forEach(v => {{
    document.getElementById('view-' + v).style.display = v === view ? '' : 'none';
    const btn = document.getElementById('btn' + v.charAt(0).toUpperCase() + v.slice(1));
    if (btn) btn.classList.toggle('active', v === view);
  }});
  if (view === 'rollout' && !rolloutRendered) {{ renderRollout(); rolloutRendered = true; }}
  if (view === 'importance' && !importanceRendered) {{ renderImportance(); importanceRendered = true; }}
  if (view === 'stats') {{ buildLayerStats(); }}
}}

// ── Heatmap ──
function getMatrix() {{
  const layerIdx  = parseInt(layerSelect.value);
  const headIdx   = parseInt(headSelect.value);
  const agg       = aggregateSelect.value;
  const layerData = DATA.attentions[layerIdx];

  if (agg === 'mean') {{
    const seq = layerData[0].length;
    const result = Array.from({{length: seq}}, () => new Array(seq).fill(0));
    for (let h = 0; h < layerData.length; h++)
      for (let i = 0; i < seq; i++)
        for (let j = 0; j < seq; j++)
          result[i][j] += layerData[h][i][j] / layerData.length;
    return result;
  }} else if (agg === 'max') {{
    const seq = layerData[0].length;
    const result = Array.from({{length: seq}}, () => new Array(seq).fill(0));
    for (let h = 0; h < layerData.length; h++)
      for (let i = 0; i < seq; i++)
        for (let j = 0; j < seq; j++)
          result[i][j] = Math.max(result[i][j], layerData[h][i][j]);
    return result;
  }}
  return layerData[headIdx];
}}

function updatePlot() {{
  const matrix     = getMatrix();
  const colorscale = colorSelect.value;
  const opacity    = parseFloat(opacityRange.value);
  const tokens     = DATA.tokens;

  Plotly.react('heatmap', [{{
    z: matrix, x: tokens, y: tokens,
    type: 'heatmap', colorscale: colorscale, opacity: opacity,
    hoverongaps: false,
    hovertemplate: '<b>From:</b> %{{y}}<br><b>To:</b> %{{x}}<br><b>Attn:</b> %{{z:.4f}}<extra></extra>',
    colorbar: {{ title: 'Weight', titlefont: {{ color: '#aaa' }}, tickfont: {{ color: '#aaa' }}, outlinecolor: '#3a3a5a', outlinewidth: 1 }},
  }}], {{
    paper_bgcolor: '#141420', plot_bgcolor: '#141420',
    font: {{ color: '#e0e0f0', size: 11 }},
    margin: {{ t: 40, r: 20, b: 100, l: 100 }},
    xaxis: {{ tickangle: -45, tickfont: {{ size: 10, color: '#aaa' }}, gridcolor: '#1e1e2e', linecolor: '#3a3a5a', title: 'Key (token attended to)', titlefont: {{ size: 12, color: '#aaa' }} }},
    yaxis: {{ tickfont: {{ size: 10, color: '#aaa' }}, gridcolor: '#1e1e2e', linecolor: '#3a3a5a', autorange: 'reversed', title: 'Query (current token)', titlefont: {{ size: 12, color: '#aaa' }} }},
    title: {{ text: buildTitle(), font: {{ size: 13, color: '#c0c0e0' }} }},
  }}, {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['select2d','lasso2d'], displaylogo: false }});
}}

function buildTitle() {{
  const agg = aggregateSelect.value;
  const layer = DATA.layers[parseInt(layerSelect.value)];
  if (agg !== 'none') return 'Layer ' + layer + ' — ' + agg.charAt(0).toUpperCase() + agg.slice(1) + ' across all heads';
  return 'Layer ' + layer + ' · Head ' + headSelect.value;
}}

updatePlot();

// ── Rollout ──
function renderRollout() {{
  if (!DATA.rollout) return;
  Plotly.react('rollout-chart', [{{
    z: DATA.rollout, x: DATA.tokens, y: DATA.tokens,
    type: 'heatmap', colorscale: colorSelect.value,
    hovertemplate: '<b>From:</b> %{{y}}<br><b>To:</b> %{{x}}<br><b>Rollout:</b> %{{z:.4f}}<extra></extra>',
    colorbar: {{ title: 'Rollout', titlefont: {{ color: '#aaa' }}, tickfont: {{ color: '#aaa' }} }},
  }}], {{
    paper_bgcolor: '#141420', plot_bgcolor: '#141420',
    font: {{ color: '#e0e0f0', size: 11 }},
    margin: {{ t: 40, r: 20, b: 100, l: 100 }},
    xaxis: {{ tickangle: -45, tickfont: {{ size: 10, color: '#aaa' }}, gridcolor: '#1e1e2e', title: 'Key', titlefont: {{ size: 12, color: '#aaa' }} }},
    yaxis: {{ tickfont: {{ size: 10, color: '#aaa' }}, gridcolor: '#1e1e2e', autorange: 'reversed', title: 'Query', titlefont: {{ size: 12, color: '#aaa' }} }},
    title: {{ text: 'End-to-end Effective Attention (Rollout across ' + DATA.layers.length + ' layers)', font: {{ size: 13, color: '#c0c0e0' }} }},
  }}, {{ responsive: true, displayModeBar: true, displaylogo: false }});
}}

// ── Token Importance ──
function renderImportance() {{
  if (!DATA.token_importance) return;
  const pairs = DATA.tokens.map((t, i) => [t, DATA.token_importance[i]]).sort((a, b) => b[1] - a[1]);
  Plotly.react('importance-chart', [{{
    x: pairs.map(p => p[1]),
    y: pairs.map(p => p[0]),
    type: 'bar', orientation: 'h',
    marker: {{ color: pairs.map(p => p[1]), colorscale: colorSelect.value, showscale: true,
               colorbar: {{ title: 'Score', titlefont: {{ color: '#aaa' }}, tickfont: {{ color: '#aaa' }} }} }},
    hovertemplate: '<b>%{{y}}</b>: %{{x:.4f}}<extra></extra>',
  }}], {{
    paper_bgcolor: '#141420', plot_bgcolor: '#141420',
    font: {{ color: '#e0e0f0', size: 11 }},
    margin: {{ t: 50, r: 60, b: 60, l: 120 }},
    title: {{ text: 'Token Importance — Mean Received Attention (sorted)', font: {{ size: 13, color: '#c0c0e0' }} }},
    xaxis: {{ title: 'Importance Score', titlefont: {{ size: 12, color: '#aaa' }}, gridcolor: '#1e1e2e' }},
    yaxis: {{ tickfont: {{ size: 10, color: '#aaa' }}, gridcolor: '#1e1e2e' }},
    height: Math.max(300, DATA.tokens.length * 24 + 100),
  }}, {{ responsive: true, displayModeBar: false }});
}}

// ── Layer stats table ──
function buildLayerStats() {{
  const tbody = document.getElementById('layer-stats-body');
  if (!DATA.per_layer_stats || tbody.children.length > 0) return;
  DATA.per_layer_stats.forEach(s => {{
    const row = document.createElement('tr');
    row.innerHTML =
      '<td>Layer ' + s.layer_idx + '</td>' +
      '<td>' + s.entropy_mean.toFixed(4) + '</td>' +
      '<td>' + s.max_weight.toFixed(4) + '</td>' +
      '<td>' + s.gini.toFixed(4) + '</td>' +
      '<td>' + s.num_heads + '</td>';
    tbody.appendChild(row);
  }});
}}

// ── CSV download (client-side) ──
function downloadCSV() {{
  if (!DATA.per_layer_stats) return;
  let csv = 'layer_idx,head_idx,entropy_mean,max_weight,gini_coefficient\\n';
  DATA.per_layer_stats.forEach(s => {{
    for (let h = 0; h < s.num_heads; h++) {{
      csv += s.layer_idx + ',' + h + ',' + s.entropy_mean.toFixed(6) + ',' + s.max_weight.toFixed(6) + ',' + s.gini.toFixed(6) + '\\n';
    }}
  }});
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'attention_stats.csv';
  a.click();
}}

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight') {{
    const s = layerSelect;
    if (s.selectedIndex < s.options.length - 1) {{ s.selectedIndex++; updatePlot(); }}
  }} else if (e.key === 'ArrowLeft') {{
    const s = layerSelect;
    if (s.selectedIndex > 0) {{ s.selectedIndex--; updatePlot(); }}
  }} else if (e.key === 'ArrowUp') {{
    const s = headSelect;
    if (s.selectedIndex > 0) {{ s.selectedIndex--; updatePlot(); }}
  }} else if (e.key === 'ArrowDown') {{
    const s = headSelect;
    if (s.selectedIndex < s.options.length - 1) {{ s.selectedIndex++; updatePlot(); }}
  }} else if (e.key === '1') {{ switchView('heatmap'); }}
    else if (e.key === '2') {{ switchView('rollout'); }}
    else if (e.key === '3') {{ switchView('importance'); }}
    else if (e.key === '4') {{ switchView('stats'); }}
}});
</script>
</body>
</html>
"""


class HeatmapRenderer:
    """Renders attention data as a self-contained interactive HTML file."""

    def __init__(
        self,
        color_scale: str = "Viridis",
        output_dir: Optional[str] = None,
    ):
        self.color_scale = color_scale or _env("COLOR_SCALE", "Viridis")
        self.output_dir = Path(output_dir or _env("OUTPUT_DIR", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        attention_data: Dict,
        html_filename: Optional[str] = None,
        json_filename: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Render the attention heatmap to HTML and save raw data to JSON.

        Returns
        -------
        (html_path, json_path)
        """
        html_filename = html_filename or _env("OUTPUT_HTML", "attention_map.html")
        json_filename = json_filename or _env("OUTPUT_JSON", "attention_data.json")

        html_path = self.output_dir / html_filename
        json_path = self.output_dir / json_filename

        serialised = self._serialise(attention_data)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serialised, f, indent=2)
        print(f"[renderer] Saved attention data → {json_path}")

        html_content = self._render_html(attention_data, serialised)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[renderer] Saved heatmap       → {html_path}")

        return html_path, json_path

    def _serialise(self, data: Dict) -> Dict:
        """Convert numpy arrays to nested Python lists for JSON, including rollout & importance."""
        attentions_list = [arr.tolist() for arr in data["attentions"]]
        stats = self._compute_stats(data["attentions"])
        per_layer_stats = self._compute_per_layer_stats(data["layers"], data["attentions"])

        rollout_matrix = compute_attention_rollout(data["attentions"]).tolist()
        token_importance = compute_token_importance(data["attentions"]).tolist()

        return {
            "model": data["model"],
            "prompt": data["prompt"],
            "tokens": data["tokens"],
            "layers": data["layers"],
            "num_heads": data["num_heads"],
            "seq_len": data["seq_len"],
            "num_attention_layers": data["num_attention_layers"],
            "mock": data.get("mock", False),
            "attentions": attentions_list,
            "rollout": rollout_matrix,
            "token_importance": token_importance,
            "stats": stats,
            "per_layer_stats": per_layer_stats,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def _compute_stats(attentions: List[np.ndarray]) -> Dict:
        """Compute summary statistics across all attention layers/heads."""
        all_vals = np.concatenate([a.flatten() for a in attentions])
        nonzero = all_vals[all_vals > 1e-6]
        entropy_mean = float(
            np.mean([
                -np.sum(row * np.log(row + 1e-10))
                for a in attentions
                for head in a
                for row in head
            ])
        )
        return {
            "mean_nonzero": float(np.mean(nonzero)) if len(nonzero) else 0.0,
            "max": float(np.max(all_vals)),
            "min_nonzero": float(np.min(nonzero)) if len(nonzero) else 0.0,
            "entropy_mean": round(entropy_mean, 6),
        }

    @staticmethod
    def _compute_per_layer_stats(layers: List[int], attentions: List[np.ndarray]) -> List[Dict]:
        """Compute per-layer mean entropy, peak weight, and Gini coefficient."""
        results = []
        for layer_idx, arr in zip(layers, attentions):
            flat = arr.flatten()
            flat_sorted = np.sort(flat)
            n = len(flat_sorted)
            s = flat_sorted.sum()
            gini = float(
                (2 * np.dot(flat_sorted, np.arange(1, n + 1)) - (n + 1) * s) / (n * s + 1e-10)
            )
            entropy_per_row = -np.sum(arr * np.log(arr + 1e-10), axis=-1)
            results.append({
                "layer_idx": layer_idx,
                "entropy_mean": round(float(entropy_per_row.mean()), 6),
                "max_weight": round(float(arr.max()), 6),
                "gini": round(max(0.0, gini), 6),
                "num_heads": arr.shape[0],
            })
        return results

    def _render_html(self, data: Dict, serialised: Dict) -> str:
        import html as _html
        import json as _json

        tokens = data["tokens"]
        token_chips = " ".join(
            f'<span class="tok" title="index {i}">{_html.escape(t)}</span>'
            for i, t in enumerate(tokens)
        )
        mock_badge = (
            ' <span class="mock-badge">MOCK DATA</span>'
            if data.get("mock")
            else ""
        )
        model_name = data["model"]
        stats = serialised["stats"]

        return _HTML_TEMPLATE.format(
            model_name=model_name,
            model_name_url=model_name,
            prompt_escaped=_html.escape(data["prompt"]),
            num_tokens=data["seq_len"],
            num_layers=data["num_attention_layers"],
            num_heads=data["num_heads"],
            generated_at=serialised["generated_at"],
            entropy_mean=round(stats["entropy_mean"], 3),
            max_weight=round(stats["max"], 3),
            token_chips=token_chips,
            mock_badge=mock_badge,
            data_json=_json.dumps(serialised),
            plotly_cdn_version=_env("PLOTLY_CDN_VERSION", "2.27.0"),
        )


# ─── Convenience function ─────────────────────────────────────────────────────

def visualize(
    prompt: str,
    model_name: Optional[str] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
    html_filename: Optional[str] = None,
    json_filename: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 128,
    hf_token: Optional[str] = None,
    num_layers: int = 8,
    num_heads: int = 16,
    mock_seed: Optional[int] = None,
    # ── New parameters ──
    export_csv: bool = False,
    fallback_model: Optional[str] = None,
    load_retries: int = 2,
    layer_filter: Optional[List[int]] = None,
) -> Tuple[Path, Path]:
    """
    High-level entry point: extract attention and render heatmap.

    Parameters
    ----------
    prompt         : Input text to analyze
    model_name     : HuggingFace model ID (overrides MODEL_NAME env var)
    mock           : Use synthetic data instead of loading a real model
    output_dir     : Where to save outputs (overrides OUTPUT_DIR env var)
    html_filename  : Output HTML filename
    json_filename  : Output JSON filename
    device         : Torch device string ("cuda", "cpu", "mps")
    max_tokens     : Max input tokens
    hf_token       : HuggingFace API token
    num_layers     : (mock only) number of simulated layers
    num_heads      : (mock only) number of simulated heads per layer
    mock_seed      : (mock only) RNG seed for reproducibility
    export_csv     : Also export per-head stats as CSV alongside JSON
    fallback_model : Small model to try if the primary fails (e.g. Qwen/Qwen3-0.6B)
    load_retries   : How many times to retry on transient load errors
    layer_filter   : Restrict visualization to these layer indices only

    Returns
    -------
    (html_path, json_path)
    """
    resolved_model = model_name or _env("MODEL_NAME", "nvidia/Nemotron-Cascade-2-30B-A3B")
    resolved_mock = mock or _env_bool("MOCK_MODE")
    resolved_fallback = fallback_model or _env("FALLBACK_MODEL", "")
    resolved_export_csv = export_csv or _env_bool("EXPORT_CSV")
    resolved_layer_filter = layer_filter or _env_list("LAYER_FILTER")

    if resolved_mock:
        extractor: "MockAttentionExtractor | AttentionExtractor" = MockAttentionExtractor(
            model_name=resolved_model,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=mock_seed,
        )
        extractor.load()
    else:
        extractor = AttentionExtractor(
            model_name=resolved_model,
            device=device,
            max_tokens=max_tokens,
            hf_token=hf_token,
        )
        try:
            extractor.load(max_retries=load_retries)
        except RuntimeError as primary_err:
            if resolved_fallback and resolved_fallback != resolved_model:
                print(f"[warn] Primary model failed: {primary_err}")
                print(f"[warn] Attempting fallback model: {resolved_fallback}")
                extractor = AttentionExtractor(
                    model_name=resolved_fallback,
                    device=device,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                )
                extractor.load(max_retries=load_retries)
            else:
                raise

    attention_data = extractor.extract(prompt)

    # Apply layer filter if specified
    if resolved_layer_filter is not None:
        filter_set = set(resolved_layer_filter)
        filtered_layers = []
        filtered_attentions = []
        for lyr, attn in zip(attention_data["layers"], attention_data["attentions"]):
            if lyr in filter_set:
                filtered_layers.append(lyr)
                filtered_attentions.append(attn)
        if not filtered_layers:
            print(
                f"[warn] LAYER_FILTER={resolved_layer_filter} matched no layers "
                f"(available: {attention_data['layers']}). Showing all layers."
            )
        else:
            attention_data["layers"] = filtered_layers
            attention_data["attentions"] = filtered_attentions
            attention_data["num_attention_layers"] = len(filtered_layers)

    renderer = HeatmapRenderer(output_dir=output_dir)
    html_path, json_path = renderer.render(
        attention_data,
        html_filename=html_filename,
        json_filename=json_filename,
    )

    if resolved_export_csv:
        csv_path = json_path.with_suffix(".csv")
        export_attention_csv(attention_data, csv_path)

    return html_path, json_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nemotron Attention Visualizer — interactive attention-head heatmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--model",
        default=_env("MODEL_NAME", "nvidia/Nemotron-Cascade-2-30B-A3B"),
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        default="Explain quantum computing",
        help="Input prompt to analyze",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="PROMPT",
        help="Run multi-prompt comparison instead of single heatmap",
    )
    parser.add_argument(
        "--output-dir",
        default=_env("OUTPUT_DIR", "outputs"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--html",
        default=_env("OUTPUT_HTML", "attention_map.html"),
        help="HTML output filename",
    )
    parser.add_argument(
        "--json",
        default=_env("OUTPUT_JSON", "attention_data.json"),
        help="JSON output filename",
    )
    parser.add_argument(
        "--device",
        default=_env("DEVICE", ""),
        help="Torch device (cuda/cpu/mps, empty=auto)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_env_int("MAX_TOKENS", 128),
        help="Maximum number of input tokens",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=_env_bool("MOCK_MODE"),
        help="Use synthetic attention data (no model download)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="(mock only) Number of attention layers to simulate",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="(mock only) Attention heads per layer to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("MOCK_SEED", 42),
        help="Random seed for mock data",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=_env_bool("EXPORT_CSV"),
        help="Also export per-head stats as CSV",
    )
    parser.add_argument(
        "--fallback-model",
        default=_env("FALLBACK_MODEL", "Qwen/Qwen3-0.6B"),
        help="Small model to try if the primary model fails to load",
    )
    parser.add_argument(
        "--load-retries",
        type=int,
        default=_env_int("LOAD_RETRIES", 2),
        help="Number of retries on transient model load failures",
    )
    parser.add_argument(
        "--layer-filter",
        default=_env("LAYER_FILTER", ""),
        help="Comma-separated layer indices to include (e.g. 0,1,2,3). Empty = all layers.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = _parse_args()

    layer_filter: Optional[List[int]] = None
    if args.layer_filter.strip():
        try:
            layer_filter = [int(x.strip()) for x in args.layer_filter.split(",") if x.strip()]
        except ValueError:
            _rprint(f"[warn] Invalid --layer-filter value: {args.layer_filter!r}. Ignoring.", "yellow")

    t0 = time.time()

    if args.compare:
        _rprint(f"[compare] Comparing {len(args.compare)} prompts...", "cyan")
        out_path = compare_prompts(
            prompts=args.compare,
            mock=args.mock,
            output_dir=args.output_dir,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mock_seed=args.seed,
            device=args.device or None,
        )
        elapsed = time.time() - t0
        _rprint(f"\n[done] Completed in {elapsed:.1f}s", "green")
        _rprint(f"  Comparison HTML: {out_path.resolve()}", "green")
        return

    html_path, json_path = visualize(
        prompt=args.prompt,
        model_name=args.model,
        mock=args.mock,
        output_dir=args.output_dir,
        html_filename=args.html,
        json_filename=args.json,
        device=args.device or None,
        max_tokens=args.max_tokens,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mock_seed=args.seed,
        export_csv=args.export_csv,
        fallback_model=args.fallback_model or None,
        load_retries=args.load_retries,
        layer_filter=layer_filter,
    )
    elapsed = time.time() - t0

    # Show token importance summary in terminal
    try:
        import json as _json
        with open(json_path) as f:
            saved = _json.load(f)
        importance = saved.get("token_importance", [])
        tokens = saved.get("tokens", [])
        if importance and tokens:
            pairs = sorted(zip(tokens, importance), key=lambda x: -x[1])
            top3 = pairs[:3]
            _rprint(
                "\n[insights] Top tokens by importance: " +
                ", ".join(f'"{t}" ({s:.3f})' for t, s in top3),
                "bright_white",
            )
    except Exception:
        pass

    _rprint(f"\n[done] Completed in {elapsed:.1f}s", "green")
    _rprint(f"  HTML heatmap  : {html_path.resolve()}", "green")
    _rprint(f"  Attention JSON: {json_path.resolve()}", "green")
    _rprint("\nOpen the HTML file in your browser to explore attention patterns.", "dim")
    _rprint("Keyboard shortcuts: ← → to change layer, ↑ ↓ to change head, 1-4 to switch views.", "dim")


if __name__ == "__main__":
    main()
