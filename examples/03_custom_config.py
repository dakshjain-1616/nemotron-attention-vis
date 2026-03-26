#!/usr/bin/env python3
"""
03_custom_config.py — Customising behaviour via environment variables and config.

Demonstrates every major knob exposed through env vars and function parameters:
  • COLOR_SCALE  — choose a Plotly colorscale
  • LAYER_FILTER — visualise only a subset of layers
  • EXPORT_CSV   — write per-head statistics to CSV automatically
  • MOCK_SEED    — reproducible synthetic data
  • OUTPUT_DIR   — redirect all outputs to a custom directory

You can also set these in a .env file (copy .env.example → .env).

Usage:
    python examples/03_custom_config.py
    COLOR_SCALE=Hot LAYER_FILTER=0,2,4 python examples/03_custom_config.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

# ── Override env vars programmatically for this example ───────────────────────
os.environ.setdefault("COLOR_SCALE", "RdPu")
os.environ.setdefault("MOCK_SEED", "7")
os.environ.setdefault("OUTPUT_DIR", "outputs/custom")
os.environ.setdefault("EXPORT_CSV", "true")
# Visualise only layers 0, 2, 4, 6 — useful for large models
os.environ.setdefault("LAYER_FILTER", "0,2,4,6")

from nemotron_attention_v import visualize

Path("outputs/custom").mkdir(parents=True, exist_ok=True)

html_path, json_path = visualize(
    prompt="Neural networks learn by adjusting weights",
    mock=True,
    # All of the following can also be read from env vars:
    output_dir=os.environ["OUTPUT_DIR"],
    html_filename="custom_config.html",
    json_filename="custom_config.json",
    color_scale=os.environ["COLOR_SCALE"],
    export_csv=os.environ.get("EXPORT_CSV", "false").lower() == "true",
    layer_filter=[int(x) for x in os.environ["LAYER_FILTER"].split(",")],
)

print(f"HTML  → {html_path}")
print(f"JSON  → {json_path}")
print()
print("Active configuration:")
for key in ("COLOR_SCALE", "MOCK_SEED", "OUTPUT_DIR", "EXPORT_CSV", "LAYER_FILTER"):
    print(f"  {key}={os.environ.get(key, '<not set>')}")
