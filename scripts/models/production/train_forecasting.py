#!/usr/bin/env python3
"""CLI wrapper for next-day forecasting model training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.production.forecasting import main


if __name__ == "__main__":
    raise SystemExit(main())
