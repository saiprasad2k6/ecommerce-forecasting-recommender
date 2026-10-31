"""Validate existing feature splits and the warehouse before training."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.models.evaluation import sql_path
from src.models.production.common import ROOT
from src.models.production.forecasting import READ_COLUMNS


def validate_inputs(features_dir, warehouse, candidate_limit=100, top_k=10):
    if not 1 <= top_k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")
    metadata = json.loads((features_dir / "metadata.json").read_text())
    ranges = []
    with duckdb.connect() as db:
        for split in ["train", "validation", "test"]:
            path = features_dir / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run olist_end_to_end_pipeline first")
            columns = set(db.sql(f"SELECT * FROM read_parquet('{sql_path(path)}') LIMIT 0").columns)
            missing = set(READ_COLUMNS) - columns
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {sorted(missing)}; rebuild forecasting features"
                )
            minimum, maximum, count = db.sql(f"""
                SELECT MIN(purchase_date), MAX(purchase_date), COUNT(*)
                FROM read_parquet('{sql_path(path)}')
            """).fetchone()
            if not count:
                raise ValueError(f"Empty {split} split")
            actual = {
                "min": pd.Timestamp(minimum).strftime("%Y-%m-%d"),
                "max": pd.Timestamp(maximum).strftime("%Y-%m-%d"),
            }
            if actual != metadata["date_ranges"][split] or count != metadata["rows"][split]:
                raise ValueError(f"{split} data does not match metadata.json; rebuild features")
            ranges.append((minimum, maximum))
    if not ranges[0][1] < ranges[1][0] or not ranges[1][1] < ranges[2][0]:
        raise ValueError("Train/validation/test periods overlap")
    if not warehouse.is_file():
        raise FileNotFoundError(warehouse)
    try:
        with duckdb.connect(str(warehouse), read_only=True) as db:
            db.execute("""SELECT order_id, order_item_id, customer_unique_id, product_id,
                customer_state, product_category_name_english, order_status,
                order_purchase_timestamp, order_delivered_customer_date
                FROM order_items LIMIT 1""")
    except duckdb.IOException as error:
        raise RuntimeError(
            "Cannot open warehouse. Disconnect DBeaver and wait for warehouse-writing tasks "
            "to finish."
        ) from error
    return {"status": "ready", "features_dir": str(features_dir), "warehouse": str(warehouse)}


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
