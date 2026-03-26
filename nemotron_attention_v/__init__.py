"""
nemotron_attention_v — Attention visualization toolkit for Nemotron-Cascade-2-30B-A3B.

Public API:
    visualize(prompt, ...)           — extract + render one prompt → (html_path, json_path)
    compare_prompts(prompts, ...)    — multi-prompt side-by-side comparison → html_path
    AttentionExtractor               — loads a real HuggingFace model
    MockAttentionExtractor           — generates synthetic attention patterns (no GPU needed)
    HeatmapRenderer                  — renders attention data → HTML + JSON
    compute_attention_rollout(...)   — Abnar & Zuidema 2020 rollout across layers
    compute_token_importance(...)    — per-token importance score (mean received attention)
    export_attention_csv(...)        — export per-layer/head stats as CSV
"""

from .visualize_attention import (
    visualize,
    compare_prompts,
    AttentionExtractor,
    MockAttentionExtractor,
    HeatmapRenderer,
    compute_attention_rollout,
    compute_token_importance,
    export_attention_csv,
    __version__,
)

__all__ = [
    "visualize",
    "compare_prompts",
    "AttentionExtractor",
    "MockAttentionExtractor",
    "HeatmapRenderer",
    "compute_attention_rollout",
    "compute_token_importance",
    "export_attention_csv",
    "__version__",
]
