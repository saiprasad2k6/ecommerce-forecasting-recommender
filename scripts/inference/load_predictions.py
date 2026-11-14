#!/usr/bin/env python3
"""Load prediction Parquet batches into DuckDB for SQL analysis."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.prediction_store import main

if __name__ == "__main__":
    raise SystemExit(main())
