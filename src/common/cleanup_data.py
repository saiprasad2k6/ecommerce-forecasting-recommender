"""Remove the repository's local data directory, including downloads and databases."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]


def cleanup_data(root: Path, *, dry_run: bool = True) -> None:
    """Delete only root/data; unlink a symlink without following its target."""
    root = root.resolve()
    data_dir = root / "data"
    print(f"Repository: {root}")
    if not data_dir.exists() and not data_dir.is_symlink():
        print("No local data directory to remove.")
        return
    print(f"Delete: {data_dir}")
    print("Includes raw downloads, processed/curated data, features, warehouses, "
          "and the local MLflow database under data/baselines/.")
    if dry_run:
        print("Dry run: nothing changed. Add --yes to delete local data.")
        return
    if data_dir.is_symlink() or not data_dir.is_dir():
        data_dir.unlink()
    else:
        shutil.rmtree(data_dir)
    print("Local data removed. Download the dataset again to restart the pipeline.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--yes", action="store_true", help="Delete data/, including raw downloads and databases")
    mode.add_argument("--dry-run", action="store_true", help="Preview only (the default)")
    args = parser.parse_args()
    cleanup_data(ROOT, dry_run=not args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
