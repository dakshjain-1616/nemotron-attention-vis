#!/bin/bash
set -e
cd /root/projects/nemotron-attention-vis-visualizes-attent

echo "=== Step 1: Check installed packages ==="
python3 -c "import sys; print(sys.version)"
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo "numpy missing"
python3 -c "import plotly; print('plotly', plotly.__version__)" 2>/dev/null || echo "plotly missing"
python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "torch missing"

echo ""
echo "=== Step 2: Install missing packages ==="
pip install numpy plotly python-dotenv pytest --quiet 2>&1 | tail -3

echo ""
echo "=== Step 3: Run demo ==="
python3 demo.py

echo ""
echo "=== Step 4: Run tests ==="
python3 -m pytest tests/ -v 2>&1

echo ""
echo "=== Step 5: outputs/ directory contents ==="
ls -la outputs/
