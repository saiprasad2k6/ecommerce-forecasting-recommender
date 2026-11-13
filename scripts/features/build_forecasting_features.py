#!/usr/bin/env python3
"""CLI wrapper to build forecasting features from curated daily demand."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.build_forecasting_features import main


if __name__ == "__main__":
    import os
    from src.versioning.workspace import locked
    if os.environ.get("OLIST_MANAGED_PREPARATION"):
        raise SystemExit(main())
    with locked():
        raise SystemExit(main())
