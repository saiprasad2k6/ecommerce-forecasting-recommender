#!/usr/bin/env python3
"""Build leakage-safe demand forecasting features from curated daily demand.
This will read the partitioned daily_demand files, 
fill missing dates with zero demand per product-region pair, 
create lag and past-only rolling features, 
add calendar fields, 
create a next-day target, 
and write chronological train/validation/test Parquet files under data/features/forecasting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "data" / "curated"
OUTPUT = ROOT / "data" / "features" / "forecasting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=CURATED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    return parser.parse_args()


def daily_demand_query(input_dir: Path) -> str:
    pattern = str(input_dir / "purchase_date=*" / "daily_demand.csv").replace("'", "''")
    return f"""
        WITH daily AS (
            SELECT
                CAST(purchase_date AS DATE) AS purchase_date,
                product_id,
                customer_state,
                SUM(order_item_count) AS demand
            FROM read_csv_auto('{pattern}', union_by_name = true)
            GROUP BY 1, 2, 3
        ), bounds AS (
            SELECT MIN(purchase_date) AS min_date, MAX(purchase_date) AS max_date
            FROM daily
        ), pairs AS (
            SELECT DISTINCT product_id, customer_state
            FROM daily
        ), calendar AS (
            SELECT CAST(day AS DATE) AS purchase_date
            FROM bounds,
            generate_series(min_date, max_date, INTERVAL 1 DAY) AS dates(day)
        ), dense AS (
            SELECT
                calendar.purchase_date,
                pairs.product_id,
                pairs.customer_state,
                COALESCE(daily.demand, 0) AS demand
            FROM calendar
            CROSS JOIN pairs
            LEFT JOIN daily USING (purchase_date, product_id, customer_state)
        )
        SELECT
            purchase_date,
            product_id,
            customer_state,
            demand,
            LAG(demand, 1) OVER series_window AS demand_lag_1,
            LAG(demand, 6) OVER series_window AS demand_lag_6,
            LAG(demand, 7) OVER series_window AS demand_lag_7,
            LAG(demand, 14) OVER series_window AS demand_lag_14,
            LAG(demand, 27) OVER series_window AS demand_lag_27,
            LAG(demand, 28) OVER series_window AS demand_lag_28,
            AVG(demand) OVER past_7 AS rolling_mean_7,
            STDDEV_SAMP(demand) OVER past_7 AS rolling_std_7,
            AVG(demand) OVER past_28 AS rolling_mean_28,
            LEAD(demand, 1) OVER series_window AS target_demand_next_day
        FROM dense
        WINDOW
            series_window AS (
                PARTITION BY product_id, customer_state
                ORDER BY purchase_date
            ),
            past_7 AS (
                PARTITION BY product_id, customer_state
                ORDER BY purchase_date
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ),
            past_28 AS (
                PARTITION BY product_id, customer_state
                ORDER BY purchase_date
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            )
        ORDER BY purchase_date, product_id, customer_state
    """


def build_features(input_dir: Path) -> pd.DataFrame:
    with duckdb.connect() as connection:
        features = connection.sql(daily_demand_query(input_dir)).df()

    features["day_of_week"] = features["purchase_date"].dt.dayofweek
    features["week_of_year"] = features["purchase_date"].dt.isocalendar().week.astype("int16")
    features["month"] = features["purchase_date"].dt.month
    features["quarter"] = features["purchase_date"].dt.quarter
    features["day_of_month"] = features["purchase_date"].dt.day
    features["demand_change_7d"] = features["demand"] - features["demand_lag_7"]
    features["demand_growth_7d"] = (
        features["demand_change_7d"] / features["demand_lag_7"].replace(0, pd.NA)
    )
    required_features = [
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_14",
        "demand_lag_28",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_28",
        "target_demand_next_day",
    ]
    return features.dropna(subset=required_features)


def split_features(
    features: pd.DataFrame, validation_days: int, test_days: int
) -> dict[str, pd.DataFrame]:
    if validation_days <= 0 or test_days <= 0:
        raise ValueError("validation-days and test-days must be positive")

    last_date = features["purchase_date"].max()
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    return {
        "train": features.loc[features["purchase_date"] < validation_start].copy(),
        "validation": features.loc[
            (features["purchase_date"] >= validation_start)
            & (features["purchase_date"] < test_start)
        ].copy(),
        "test": features.loc[features["purchase_date"] >= test_start].copy(),
    }


def write_outputs(
    splits: dict[str, pd.DataFrame], output_dir: Path, validation_days: int, test_days: int
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    metadata = {
        "validation_days": validation_days,
        "test_days": test_days,
        "rows": {name: len(frame) for name, frame in splits.items()},
        "date_ranges": {
            name: {
                "min": frame["purchase_date"].min().strftime("%Y-%m-%d"),
                "max": frame["purchase_date"].max().strftime("%Y-%m-%d"),
            }
            for name, frame in splits.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> int:
    args = parse_args()
    features = build_features(args.input_dir)
    splits = split_features(features, args.validation_days, args.test_days)
    metadata = write_outputs(splits, args.output_dir, args.validation_days, args.test_days)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
