#!/usr/bin/env python3
"""Validate Olist files in data/raw against the repository data contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = PROJECT_ROOT / "configs" / "schema.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args()


def load_schema(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open(encoding="utf-8") as stream:
        schema = yaml.safe_load(stream) or {}
    required_files = schema.get("required_files", [])
    required_columns = schema.get("required_columns", {})
    if not required_files or not required_columns:
        raise ValueError(f"Schema must define required_files and required_columns: {path}")
    return required_files, required_columns


def validate_file(path: Path, required_columns: list[str]) -> dict[str, int | list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path.name}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual_columns = reader.fieldnames or []
        missing = [column for column in required_columns if column not in actual_columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {missing}; "
                f"found: {actual_columns}"
            )
        row_count = sum(1 for _ in reader)

    if row_count == 0:
        raise ValueError(f"{path.name} contains no data rows")
    return {"rows": row_count, "columns": actual_columns}


def main() -> int:
    args = parse_args()
    required_files, required_columns = load_schema(args.schema)
    results = {
        filename: validate_file(args.input_dir / filename, required_columns.get(filename, []))
        for filename in required_files
    }
    print(json.dumps({"status": "valid", "files": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
