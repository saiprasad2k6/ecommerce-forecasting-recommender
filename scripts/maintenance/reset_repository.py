#!/usr/bin/env python3
"""CLI wrapper to reset local data, experiments and generated outputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.reset_repository import main


if __name__ == "__main__":
    raise SystemExit(main())
