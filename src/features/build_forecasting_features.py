#!/usr/bin/env python3
"""Build leakage-safe demand forecasting features from curated daily demand.
This will read the partitioned daily_demand files, 
fill missing dates with zero demand per product-region pair, 
create lag and past-only rolling features, 
add calendar fields, 
create a next-day target, 
and write chronological train/validation/test and inference Parquet files under data/features/forecasting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.common.config import REPLAY


ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "data" / "curated"
OUTPUT = ROOT / "data" / "features" / "forecasting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=CURATED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--validation-start", default=REPLAY["validation_start"])
    parser.add_argument("--test-start", default=REPLAY["test_start"])
    parser.add_argument("--monitoring-start", default=REPLAY["monitoring_start"])
    parser.add_argument("--monitoring-end", default=REPLAY["monitoring_end"])
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
            SELECT product_id, customer_state, MIN(purchase_date) AS first_purchase
            FROM daily
            GROUP BY product_id, customer_state
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
            WHERE calendar.purchase_date >= pairs.first_purchase
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
    ]
    return features.dropna(subset=required_features)


def split_features(features, validation_start, test_start, monitoring_start, monitoring_end):
    """Separate development labels from replay inputs using next-day target dates."""
    validation_start, test_start, monitoring_start, monitoring_end = map(
        pd.Timestamp, [validation_start, test_start, monitoring_start, monitoring_end]
    )
    if not validation_start < test_start < monitoring_start < monitoring_end:
        raise ValueError("Validation, test and monitoring dates must be chronological")
    target_date = features.purchase_date + pd.Timedelta(days=1)
    labeled = features.target_demand_next_day.notna()
    return {
        "train": features.loc[labeled & target_date.lt(validation_start)].copy(),
        "validation": features.loc[labeled & target_date.ge(validation_start) & target_date.lt(test_start)].copy(),
        "test": features.loc[labeled & target_date.ge(test_start) & target_date.lt(monitoring_start)].copy(),
        # Inference needs features even when tomorrow's demand is not yet known.
        "inference": features.loc[
            features.purchase_date.ge(monitoring_start) & features.purchase_date.lt(monitoring_end)
        ].drop(columns="target_demand_next_day").copy(),
    }


def write_outputs(splits, output_dir, **boundaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"No {name} rows for the configured replay dates")
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    metadata = {
        "boundaries": boundaries,
        "rows": {name: len(frame) for name, frame in splits.items()},
        "date_ranges": {
            name: {
                "min": frame.purchase_date.min().strftime("%Y-%m-%d"),
                "max": frame.purchase_date.max().strftime("%Y-%m-%d"),
            }
            for name, frame in splits.items()
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> int:
    args = parse_args()
    features = build_features(args.input_dir)
    boundaries = {name: getattr(args, name) for name in
                  ["validation_start", "test_start", "monitoring_start", "monitoring_end"]}
    splits = split_features(features, **boundaries)
    metadata = write_outputs(splits, args.output_dir, **boundaries)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
