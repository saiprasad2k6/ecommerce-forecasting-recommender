#!/usr/bin/env python3
"""Validate prepared feature splits and warehouse inputs before training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.datasets import validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data/features/forecasting")
    parser.add_argument("--warehouse", type=Path, default=ROOT / "data/warehouse/olist.duckdb")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    print(json.dumps(validate_inputs(**vars(parser.parse_args())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
