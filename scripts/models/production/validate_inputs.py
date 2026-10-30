#!/usr/bin/env python3
"""CLI wrapper to validate production training inputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.production.validate_inputs import main


if __name__ == "__main__":
    raise SystemExit(main())
