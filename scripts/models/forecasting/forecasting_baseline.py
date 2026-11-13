#!/usr/bin/env python3
"""CLI wrapper to run the forecasting baseline evaluation pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.forecasting.forecasting_baseline import main

if __name__ == "__main__":
    import os
    from src.versioning.workspace import locked
    if os.environ.get("OLIST_MANAGED_PREPARATION"):
        raise SystemExit(main())
    with locked():
        raise SystemExit(main())
