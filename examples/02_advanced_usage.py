#!/usr/bin/env python3
"""
02_advanced_usage.py — Advanced features demonstration.

Shows how to:
  • Use MockAttentionExtractor directly for programmatic access
  • Call compute_attention_rollout() to trace end-to-end attention flow
  • Call compute_token_importance() to rank tokens by received attention
  • Export per-head statistics to CSV
  • Compare multiple prompts side-by-side

Usage:
    python examples/02_advanced_usage.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path

from nemotron_attention_v import (
    MockAttentionExtractor,
    HeatmapRenderer,
    compute_attention_rollout,
    compute_token_importance,
    export_attention_csv,
    compare_prompts,
)

# ── 1. Extract attention data directly ────────────────────────────────────────
extractor = MockAttentionExtractor(
    model_name="nvidia/Nemotron-Cascade-2-30B-A3B",
    num_layers=8,
    num_heads=16,
    seed=42,
)
data = extractor.extract("The capital of France is")

tokens = data["tokens"]
attentions = data["attentions"]   # list[np.ndarray], one per layer

print(f"Tokens  : {tokens}")
print(f"Layers  : {len(attentions)}")
print(f"Shape   : {attentions[0].shape}  (heads, seq, seq)")

# ── 2. Attention rollout ──────────────────────────────────────────────────────
rollout = compute_attention_rollout(attentions)
print(f"\nRollout matrix shape: {rollout.shape}  (seq, seq), rows sum to 1")

# ── 3. Token importance ───────────────────────────────────────────────────────
importance = compute_token_importance(attentions)
ranked = sorted(zip(tokens, importance.tolist()), key=lambda x: -x[1])
print("\nToken importance ranking:")
for tok, score in ranked:
    bar = "█" * int(score * 40)
    print(f"  {tok:20s} {score:.4f}  {bar}")

# ── 4. Render HTML + JSON ─────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
renderer = HeatmapRenderer(color_scale="Plasma")
html_path, json_path = renderer.render(
    data,
    html_filename="outputs/advanced.html",
    json_filename="outputs/advanced.json",
)
print(f"\nRendered → {html_path}")

# ── 5. Export per-head CSV ────────────────────────────────────────────────────
csv_path = export_attention_csv(data, "outputs/advanced_stats.csv")
print(f"CSV stats → {csv_path}")

# ── 6. Multi-prompt comparison ────────────────────────────────────────────────
prompts = [
    "Explain quantum computing",
    "The capital of France is",
    "Neural networks learn by",
]
comparison_path = compare_prompts(
    prompts,
    model_name="nvidia/Nemotron-Cascade-2-30B-A3B",
    mock=True,
    output_path="outputs/advanced_comparison.html",
)
print(f"Comparison → {comparison_path}")
