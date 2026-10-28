#!/usr/bin/env python3
"""Download and safely extract the Olist dataset into data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


DATASET = "olistbr/brazilian-ecommerce"
EXPECTED_FILES = {
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_with_cli(dataset: str, output_dir: Path) -> set[str]:
    """Run `kaggle datasets download ... -p data/raw --unzip`."""
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The Kaggle CLI is required. Install it with `python -m pip install kaggle`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Configure ~/.kaggle/kaggle.json or "
            "KAGGLE_API_TOKEN before running this script."
        ) from exc

    extracted = {filename for filename in EXPECTED_FILES if (output_dir / filename).is_file()}
    missing = EXPECTED_FILES - extracted
    if missing:
        raise RuntimeError(f"Kaggle archive is missing expected files: {sorted(missing)}")
    return extracted


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    existing = {path.name for path in output_dir.glob("*.csv")}
    missing = EXPECTED_FILES - existing
    if missing and existing and not args.force:
        raise RuntimeError(
            f"{output_dir} contains a partial dataset. Missing: {sorted(missing)}. "
            "Use --force to replace it."
        )
    if not missing and not args.force:
        print(f"Olist data already exists in {output_dir}; use --force to replace it.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = download_with_cli(args.dataset, output_dir)
    print(f"Downloaded {len(extracted)} Olist CSV files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
