#!/usr/bin/env python
"""
Thin entrypoint script for BHOOMIQ - Real Estate Valuation & Market Intelligence Engine.
Execution:
    python run.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in the Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
