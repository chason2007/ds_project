#!/usr/bin/env python
"""
Thin entrypoint script for the Automated Real Estate Valuation Index and Price Prediction Engine.
Execution:
    python run.py
"""

import sys
from pathlib import Path

# Ensure project root is in the Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
