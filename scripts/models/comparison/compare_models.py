"""CLI entry point for automated baseline/candidate comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.comparison import main

if __name__ == "__main__":
    raise SystemExit(main())
