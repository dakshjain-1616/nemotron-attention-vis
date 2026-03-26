#!/usr/bin/env python3
"""
scripts/demo.py — entry point that delegates to demo.py in the project root.
Adds the project root to sys.path so imports work regardless of CWD.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the demo runner from the project root
from demo import run_demo  # noqa: E402

if __name__ == "__main__":
    run_demo()
