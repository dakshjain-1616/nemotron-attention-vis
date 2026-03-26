#!/usr/bin/env python3
"""
04_full_pipeline.py — End-to-end workflow: the full project capability.

Demonstrates the complete pipeline:
  1. MockAttentionExtractor  — simulate model inference (swap for AttentionExtractor
                               with a real model + HF_TOKEN for production use)
  2. compute_attention_rollout — trace end-to-end attention flow (Abnar & Zuidema 2020)
  3. compute_token_importance  — rank tokens by how much attention they receive
  4. HeatmapRenderer           — produce the interactive HTML heatmap + JSON data file
  5. export_attention_csv      — dump per-layer/head statistics for offline analysis
  6. compare_prompts           — side-by-side HTML for multiple prompts

After running, open outputs/pipeline/*.html in a browser.

Usage:
    python examples/04_full_pipeline.py
    # For a real model (needs GPU + HF_TOKEN env var):
    # HF_TOKEN=hf_... MOCK_MODE=false python examples/04_full_pipeline.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from pathlib import Path

from nemotron_attention_v import (
    MockAttentionExtractor,
    HeatmapRenderer,
    compute_attention_rollout,
    compute_token_importance,
    export_attention_csv,
    compare_prompts,
    visualize,
)

OUT = Path("outputs/pipeline")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Explain quantum computing in simple terms",
    "The capital of France is",
    "Neural networks learn by adjusting their weights",
    "Attention is all you need",
]

MODEL = "nvidia/Nemotron-Cascade-2-30B-A3B"
MOCK  = os.environ.get("MOCK_MODE", "true").lower() != "false"

print(f"{'=' * 60}")
print(f"  Nemotron Attention Visualizer — Full Pipeline")
print(f"  Model  : {MODEL}")
print(f"  Mode   : {'mock (synthetic)' if MOCK else 'real model'}")
print(f"  Output : {OUT}/")
print(f"{'=' * 60}\n")

# ── Step 1 & 2: Extract + analyse each prompt individually ───────────────────
extractor = MockAttentionExtractor(
    model_name=MODEL,
    num_layers=8,
    num_heads=16,
    seed=42,
)

all_importance = {}

for i, prompt in enumerate(PROMPTS, 1):
    t0 = time.time()
    print(f"[{i}/{len(PROMPTS)}] Prompt: \"{prompt}\"")

    # Step 1 — extract attention weights
    data = extractor.extract(prompt)
    tokens     = data["tokens"]
    attentions = data["attentions"]

    # Step 2 — rollout (end-to-end information flow across all layers)
    rollout = compute_attention_rollout(attentions)

    # Step 3 — token importance (mean received attention, all layers + heads)
    importance = compute_token_importance(attentions)
    all_importance[prompt] = dict(zip(tokens, importance.tolist()))

    # Step 4 — render interactive HTML heatmap
    slug = prompt[:30].lower().replace(" ", "_").replace("/", "-")
    renderer = HeatmapRenderer(color_scale="Viridis")
    html_path, json_path = renderer.render(
        data,
        html_filename=str(OUT / f"prompt_{i:02d}.html"),
        json_filename=str(OUT / f"prompt_{i:02d}.json"),
    )

    # Step 5 — export per-head CSV stats
    csv_path = export_attention_csv(data, str(OUT / f"prompt_{i:02d}_stats.csv"))

    elapsed = time.time() - t0
    print(f"         tokens={len(tokens)}  layers={len(attentions)}  "
          f"heads={attentions[0].shape[0]}  ({elapsed:.2f}s)")
    print(f"         HTML → {html_path}")
    print(f"         CSV  → {csv_path}")

    top_tok = max(all_importance[prompt], key=all_importance[prompt].get)
    print(f"         Most-attended token: '{top_tok}'\n")

# ── Step 6 — Multi-prompt comparison ─────────────────────────────────────────
print("Generating side-by-side comparison…")
comparison_path = compare_prompts(
    PROMPTS,
    model_name=MODEL,
    mock=MOCK,
    output_path=str(OUT / "comparison.html"),
)
print(f"Comparison → {comparison_path}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "model": MODEL,
    "mock_mode": MOCK,
    "prompts": len(PROMPTS),
    "outputs": [str(p) for p in sorted(OUT.iterdir())],
    "token_importance": all_importance,
}
summary_path = OUT / "pipeline_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(f"{'=' * 60}")
print(f"  Pipeline complete — {len(list(OUT.iterdir()))} files in {OUT}/")
print(f"  Open {OUT}/comparison.html to see all prompts side-by-side.")
print(f"{'=' * 60}")
