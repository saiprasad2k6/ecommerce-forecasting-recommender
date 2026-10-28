#!/usr/bin/env python3
"""CLI wrapper to validate Olist files against the data contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validate_olist_data import main


if __name__ == "__main__":
    raise SystemExit(main())
