#!/usr/bin/env python3
"""CLI wrapper to remove local downloads, data products and databases."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.cleanup_data import main


if __name__ == "__main__":
    raise SystemExit(main())
