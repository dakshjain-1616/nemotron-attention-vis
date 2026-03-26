# Examples

Runnable scripts that demonstrate the Nemotron Attention Visualizer from minimal to full pipeline.
All scripts work out-of-the-box in mock mode (no GPU or HuggingFace token required).

## Scripts

| Script | What it demonstrates |
|--------|----------------------|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example — one prompt, one HTML heatmap, ~15 lines |
| [`02_advanced_usage.py`](02_advanced_usage.py) | Direct API use: `MockAttentionExtractor`, rollout, token importance, CSV export, multi-prompt comparison |
| [`03_custom_config.py`](03_custom_config.py) | Customising behaviour via env vars (`COLOR_SCALE`, `LAYER_FILTER`, `EXPORT_CSV`, `MOCK_SEED`, `OUTPUT_DIR`) |
| [`04_full_pipeline.py`](04_full_pipeline.py) | End-to-end workflow for 4 prompts: extraction → rollout → importance → HTML → CSV → comparison |

## Running

```bash
# From the project root:
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py

# Override env vars inline:
COLOR_SCALE=Hot LAYER_FILTER=0,2,4 python examples/03_custom_config.py

# Use a real model (needs HF_TOKEN and a GPU):
HF_TOKEN=hf_... MOCK_MODE=false python examples/04_full_pipeline.py
```

All outputs land in `outputs/` (or a subdirectory).
Open the generated `.html` files in a browser to explore attention heads interactively.
