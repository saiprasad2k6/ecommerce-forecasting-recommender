#!/usr/bin/env python3
"""Run one registered model job; Airflow controls the job order."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.jobs import main
if __name__ == "__main__":
    raise SystemExit(main())
