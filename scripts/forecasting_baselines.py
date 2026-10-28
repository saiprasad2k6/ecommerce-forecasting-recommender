#!/usr/bin/env python3
"""CLI wrapper to run the forecasting baseline evaluation pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.forecasting_baselines import main

if __name__ == "__main__":
    raise SystemExit(main())
