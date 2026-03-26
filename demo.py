#!/usr/bin/env python3
"""
demo.py — Nemotron Attention Visualizer Demo
=============================================
Runs the full pipeline and saves outputs to outputs/.

Mock mode is auto-detected when:
  • MOCK_MODE=true is set in environment, OR
  • HF_TOKEN is not set AND the model is a large gated model

Run:
    python demo.py                     # auto mode (mock if no credentials)
    python demo.py --real              # force real model (needs HF_TOKEN + GPU)
    MOCK_MODE=true python demo.py      # explicit mock
    EXPORT_CSV=true python demo.py     # also export per-head CSV stats
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ─── Load .env if present ─────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Rich setup ───────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False

console = Console() if _RICH else None

VERSION = "1.0.0"


def _env(key: str, default: str = "") -> str:
    """Return an environment variable value or default."""
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    """Parse an integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# ─── Auto-detect mock mode ────────────────────────────────────────────────────

LARGE_GATED_MODELS = {
    "nvidia/Nemotron-Cascade-2-30B-A3B",
    "nvidia/Nemotron-4-340B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
}

MODEL_NAME    = _env("MODEL_NAME", "nvidia/Nemotron-Cascade-2-30B-A3B")
HF_TOKEN      = _env("HF_TOKEN", "")
EXPLICIT_MOCK = _env_bool("MOCK_MODE", False)
FORCE_REAL    = "--real" in sys.argv
EXPORT_CSV    = _env_bool("EXPORT_CSV", False)


def should_use_mock() -> bool:
    """Return True when mock mode should be used."""
    if FORCE_REAL:
        return False
    if EXPLICIT_MOCK:
        return True
    if MODEL_NAME in LARGE_GATED_MODELS and not HF_TOKEN:
        return True
    return False


# ─── Demo configuration ───────────────────────────────────────────────────────

DEMO_PROMPTS = [
    "Explain quantum computing",
    "The capital of France is",
    "Neural networks learn by",
]

OUTPUT_DIR  = Path(_env("OUTPUT_DIR", "outputs"))
NUM_LAYERS  = _env_int("DEMO_NUM_LAYERS", 8)
NUM_HEADS   = _env_int("DEMO_NUM_HEADS", 16)
MOCK_SEED   = _env_int("MOCK_SEED", 42)
COLOR_SCALE = _env("COLOR_SCALE", "Viridis")


def print_banner(mock: bool) -> None:
    """Print a startup banner using Rich if available, else plain ASCII."""
    mode_str = "MOCK (synthetic attention patterns)" if mock else "REAL (HuggingFace model)"
    mode_color = "yellow" if mock else "green"

    if _RICH:
        content = Text.assemble(
            ("Nemotron Attention Visualizer\n", "bold cyan"),
            (f"v{VERSION}  •  NEO Autonomous Build\n\n", "dim"),
            ("Model:  ", "dim"),
            (MODEL_NAME, "bright_white"),
            "\n",
            ("Mode:   ", "dim"),
            (mode_str, mode_color),
        )
        console.print(Panel(content, title="[bold blue]Nemotron Attention Vis[/]", border_style="blue"))
    else:
        model_display = MODEL_NAME[:46] + ".." if len(MODEL_NAME) > 46 else MODEL_NAME
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          Nemotron Attention Visualizer — Demo Runner             ║
║          v{VERSION}  •  NEO Autonomous Build                        ║
║          Model: {model_display:<48}║
║          Mode:  {mode_str:<48}║
╚══════════════════════════════════════════════════════════════════╝
""")


def _rprint(msg: str, style: str = "") -> None:
    """Print a message with optional Rich styling."""
    if _RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _show_token_insights(json_path: Path) -> None:
    """Print top attended-to tokens from the saved JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        tokens = data.get("tokens", [])
        importance = data.get("token_importance", [])
        stats = data.get("stats", {})
        per_layer = data.get("per_layer_stats", [])

        if tokens and importance:
            pairs = sorted(zip(tokens, importance), key=lambda x: -x[1])
            top3 = pairs[:3]
            _rprint(
                "    → Top tokens by importance: " +
                ", ".join(f'"{t}" ({s:.3f})' for t, s in top3),
                "bright_white",
            )

        if stats:
            _rprint(
                f"    → Entropy (mean): {stats.get('entropy_mean', 0):.4f}  "
                f"Peak weight: {stats.get('max', 0):.4f}",
                "dim",
            )

        if per_layer:
            entropies = [s["entropy_mean"] for s in per_layer]
            best_layer = per_layer[entropies.index(min(entropies))]
            _rprint(
                f"    → Most focused layer: Layer {best_layer['layer_idx']} "
                f"(entropy {best_layer['entropy_mean']:.4f})",
                "dim",
            )
    except Exception:
        pass


def run_demo() -> None:
    """Run the full demo pipeline and save outputs."""
    from nemotron_attention_v import visualize, compare_prompts

    mock = should_use_mock()
    print_banner(mock)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    start_wall = time.time()

    # ── Primary prompt (required by spec) ─────────────────────────────────────
    primary_prompt = DEMO_PROMPTS[0]  # "Explain quantum computing"

    _rprint(f"\n[1/{len(DEMO_PROMPTS)}] Primary prompt: \"{primary_prompt}\"")
    t0 = time.time()
    html_path, json_path = visualize(
        prompt=primary_prompt,
        model_name=MODEL_NAME,
        mock=mock,
        output_dir=str(OUTPUT_DIR),
        html_filename="attention_map.html",
        json_filename="attention_data.json",
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mock_seed=MOCK_SEED,
        export_csv=EXPORT_CSV,
    )
    elapsed = time.time() - t0
    _rprint(f"    ✓ {html_path} ({elapsed:.1f}s)", "green")
    _rprint(f"    ✓ {json_path}", "green")
    _show_token_insights(json_path)
    results.append({
        "prompt": primary_prompt,
        "html": str(html_path),
        "json": str(json_path),
        "elapsed_s": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # ── Additional prompts → separate HTML files ───────────────────────────────
    for idx, prompt in enumerate(DEMO_PROMPTS[1:], start=2):
        slug = prompt.lower().replace(" ", "_")[:30]
        _rprint(f"\n[{idx}/{len(DEMO_PROMPTS)}] Additional prompt: \"{prompt}\"")
        t0 = time.time()
        hp, jp = visualize(
            prompt=prompt,
            model_name=MODEL_NAME,
            mock=mock,
            output_dir=str(OUTPUT_DIR),
            html_filename=f"attention_{slug}.html",
            json_filename=f"attention_{slug}.json",
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mock_seed=MOCK_SEED + idx,
            export_csv=EXPORT_CSV,
        )
        elapsed = time.time() - t0
        _rprint(f"    ✓ {hp} ({elapsed:.1f}s)", "green")
        _show_token_insights(jp)
        results.append({
            "prompt": prompt,
            "html": str(hp),
            "json": str(jp),
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ── Multi-prompt comparison HTML ───────────────────────────────────────────
    _rprint(f"\n[{len(DEMO_PROMPTS) + 1}/{len(DEMO_PROMPTS) + 1}] Generating comparison report...")
    comparison_path = None
    if _RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Building comparison HTML...", total=None)
            t0 = time.time()
            try:
                comparison_path = compare_prompts(
                    prompts=DEMO_PROMPTS,
                    model_name=MODEL_NAME,
                    mock=mock,
                    output_dir=str(OUTPUT_DIR),
                    comparison_filename="comparison.html",
                    num_layers=NUM_LAYERS,
                    num_heads=NUM_HEADS,
                    mock_seed=MOCK_SEED,
                )
                elapsed = time.time() - t0
                progress.update(task, completed=True)
            except Exception as e:
                progress.stop()
                _rprint(f"    [warn] Comparison failed (non-fatal): {e}", "yellow")
    else:
        t0 = time.time()
        try:
            comparison_path = compare_prompts(
                prompts=DEMO_PROMPTS,
                model_name=MODEL_NAME,
                mock=mock,
                output_dir=str(OUTPUT_DIR),
                comparison_filename="comparison.html",
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                mock_seed=MOCK_SEED,
            )
            elapsed = time.time() - t0
        except Exception as e:
            _rprint(f"    [warn] Comparison failed (non-fatal): {e}", "yellow")

    if comparison_path:
        _rprint(f"    ✓ {comparison_path}", "green")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model": MODEL_NAME,
        "mock_mode": mock,
        "num_attention_layers": NUM_LAYERS if mock else "from_model",
        "num_heads": NUM_HEADS if mock else "from_model",
        "color_scale": COLOR_SCALE,
        "export_csv": EXPORT_CSV,
        "runs": results,
        "comparison_html": str(comparison_path) if comparison_path else None,
        "total_elapsed_s": round(time.time() - start_wall, 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = OUTPUT_DIR / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _rprint(f"\n[summary] {summary_path}", "dim")

    # ── Final report ──────────────────────────────────────────────────────────
    _print_report(results, mock, comparison_path, summary["total_elapsed_s"])


def _print_report(
    results: List[dict],
    mock: bool,
    comparison_path: Optional[Path],
    total_elapsed: float,
) -> None:
    """Print the final demo summary, using a Rich table when available."""
    if _RICH:
        table = Table(
            title="Demo Complete",
            box=rich_box.ROUNDED,
            border_style="blue",
            show_lines=True,
        )
        table.add_column("Prompt", style="cyan", no_wrap=False)
        table.add_column("HTML", style="green")
        table.add_column("Time", style="bright_white", justify="right")
        table.add_column("Timestamp", style="dim")
        for r in results:
            table.add_row(r["prompt"], r["html"], f"{r['elapsed_s']}s", r["timestamp"])
        console.print()
        console.print(table)

        if comparison_path:
            console.print(f"  [dim]Comparison :[/] [green]{comparison_path}[/]")

        console.print(f"\n  [dim]Total wall-clock time:[/] [bright_white]{total_elapsed:.1f}s[/]")

        if mock:
            console.print(
                "\n  [yellow]NOTE: Running in MOCK mode.[/]\n"
                "  Attention weights are synthetic but structurally realistic.\n"
                "  To use a real model set [bold]HF_TOKEN[/] and [bold]MODEL_NAME[/] in .env\n"
                "  and re-run: [bold]python demo.py --real[/]"
            )
        else:
            console.print("\n  [green]Running with REAL model weights.[/]")

        console.print(
            "\n  Open [bold]outputs/attention_map.html[/] in your browser to explore.\n"
            "  Keyboard shortcuts: [bold]← →[/] layers, [bold]↑ ↓[/] heads, [bold]1-4[/] switch views."
        )
    else:
        print("\n" + "═" * 68)
        print("  DEMO COMPLETE")
        print("═" * 68)
        for r in results:
            print(f"  Prompt : {r['prompt']!r}")
            print(f"  HTML   : {r['html']}")
            print(f"  JSON   : {r['json']}")
            print(f"  Time   : {r['elapsed_s']}s  |  At: {r['timestamp']}")
            print()

        if comparison_path:
            print(f"  Comparison : {comparison_path}")
            print()

        print(f"  Total wall-clock time: {total_elapsed:.1f}s")

        if mock:
            print("\n  NOTE: Running in MOCK mode.")
            print("  Attention weights are synthetic but structurally realistic.")
            print("  To use a real model set HF_TOKEN and MODEL_NAME in .env")
            print("  and re-run: python demo.py --real")
        else:
            print("\n  Running with REAL model weights.")

        print("\n  Open outputs/attention_map.html in your browser to explore.")
        print("  Keyboard shortcuts: ← → layers, ↑ ↓ heads, 1-4 switch views.")
        print("═" * 68)


if __name__ == "__main__":
    run_demo()
