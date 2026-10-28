#!/usr/bin/env python3
"""CLI wrapper to clean Olist CSV files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean_olist_data import main


if __name__ == "__main__":
    raise SystemExit(main())
