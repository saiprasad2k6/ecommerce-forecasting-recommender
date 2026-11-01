"""CLI wrapper for predictions from the active selected model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.batch import main

if __name__ == "__main__":
    raise SystemExit(main())
