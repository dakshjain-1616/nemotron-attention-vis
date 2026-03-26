#!/usr/bin/env python3
"""
01_quick_start.py — Minimal working example.

Runs a single prompt through the visualizer in mock mode (no GPU or HuggingFace
token required) and prints the paths to the generated HTML and JSON files.

Usage:
    python examples/01_quick_start.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nemotron_attention_v import visualize

html_path, json_path = visualize(
    prompt="Explain quantum computing",
    mock=True,
    output_dir="outputs",
    html_filename="quickstart.html",
    json_filename="quickstart.json",
)

print(f"HTML heatmap : {html_path}")
print(f"JSON data    : {json_path}")
print("Open the HTML file in a browser to explore the attention heads interactively.")
