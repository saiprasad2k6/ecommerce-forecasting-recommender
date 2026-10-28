#!/usr/bin/env python3
"""CLI wrapper to build recommendation features and baseline rankings."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_recommendation_features import main


if __name__ == "__main__":
    raise SystemExit(main())
