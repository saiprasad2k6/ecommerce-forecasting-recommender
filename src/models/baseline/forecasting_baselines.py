#!/usr/bin/env python3
"""Demand Forecasting Baseline Models & MLflow Tracking Pipeline.

Implements baseline heuristic models, vectorized out-of-core evaluation via DuckDB,
and MLflow experiment tracking as specified in Section 8 of the Pipeline Guide.

Available baselines:
  - last_day: Demand from previous day (demand_lag_1)
  - same_day_last_week: Weekly seasonal naive (demand_lag_7)
  - moving_average_7: 7-day past rolling mean (rolling_mean_7)
  - moving_average_28: 28-day past rolling mean (rolling_mean_28)
  - seasonal_naive_28: 28-day seasonal naive (demand_lag_28)
  - always_zero: Constant zero demand (diagnostic lower bound)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import duckdb
import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "data" / "features" / "forecasting"
BASELINES = ROOT / "data" / "baselines"
DEFAULT_TRACKING_URI = f"sqlite:///{BASELINES / 'mlflow.db'}"
DEFAULT_EXPERIMENT_NAME = "ecommerce-demand-forecasting"


class BaselineForecaster:
    """Standardized baseline forecaster interface for e-commerce demand forecasting.

    Conforms to the standard predict/evaluate interface for machine learning pipelines.
    """

    METHODS: dict[str, str | None] = {
        "last_day": "demand_lag_1",
        "same_day_last_week": "demand_lag_7",
        "moving_average_7": "rolling_mean_7",
        "moving_average_28": "rolling_mean_28",
        "seasonal_naive_28": "demand_lag_28",
        "always_zero": None,
    }

    def __init__(self, method: str = "moving_average_28") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Valid options: {list(self.METHODS)}")
        self.method = method
        self.feature_col = self.METHODS[method]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate forecasts for the input dataframe."""
        if self.method == "always_zero":
            return pd.Series(0.0, index=df.index, name="forecast", dtype=float)
        if self.feature_col not in df.columns:
            raise KeyError(f"Input dataframe missing required feature column '{self.feature_col}'")
        return df[self.feature_col].fillna(0.0).rename("forecast")

    def evaluate(self, df: pd.DataFrame, target_col: str = "demand") -> dict[str, float]:
        """Compute retail forecasting metrics against actual demand."""
        preds = self.predict(df)
        actuals = df[target_col]
        abs_err = (actuals - preds).abs()
        mae = float(abs_err.mean())
        rmse = float(np.sqrt(((actuals - preds) ** 2).mean()))
        wape = float(abs_err.sum() / actuals.sum()) if actuals.sum() > 0 else 0.0
        pos_mask = actuals > 0
        pos_mae = float(abs_err[pos_mask].mean()) if pos_mask.any() else 0.0
        zero_fraction = float((actuals == 0).mean())
        return {
            "method": self.method,
            "mae": mae,
            "rmse": rmse,
            "wape": wape,
            "positive_demand_mae": pos_mae,
            "zero_demand_fraction": zero_fraction,
        }


def evaluate_baselines_duckdb(parquet_path: Path, split_name: str) -> pd.DataFrame:
    """Evaluate all baseline rules across millions of rows in Parquet using DuckDB."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing feature parquet: {parquet_path}")

    sql_path = str(parquet_path).replace("'", "''")
    query = f"""
    WITH base AS (
        SELECT
            demand AS actual,
            demand_lag_1 AS last_day,
            demand_lag_7 AS same_day_last_week,
            rolling_mean_7 AS moving_average_7,
            rolling_mean_28 AS moving_average_28,
            demand_lag_28 AS seasonal_naive_28,
            0.0 AS always_zero
        FROM read_parquet('{sql_path}')
    )
    SELECT
        '{split_name}' AS split,
        baseline,
        COUNT(*) AS total_rows,
        AVG(ABS(actual - pred)) AS mae,
        SQRT(AVG(POWER(actual - pred, 2))) AS rmse,
        SUM(ABS(actual - pred)) / NULLIF(SUM(actual), 0) AS wape,
        AVG(ABS(actual - pred)) FILTER (WHERE actual > 0) AS positive_demand_mae,
        AVG(CASE WHEN actual = 0 THEN 1.0 ELSE 0.0 END) AS zero_demand_fraction
    FROM (
        SELECT actual, last_day AS pred, 'last_day' AS baseline FROM base
        UNION ALL
        SELECT actual, same_day_last_week, 'same_day_last_week' FROM base
        UNION ALL
        SELECT actual, moving_average_7, 'moving_average_7' FROM base
        UNION ALL
        SELECT actual, moving_average_28, 'moving_average_28' FROM base
        UNION ALL
        SELECT actual, seasonal_naive_28, 'seasonal_naive_28' FROM base
        UNION ALL
        SELECT actual, always_zero, 'always_zero' FROM base
    )
    GROUP BY baseline
    ORDER BY mae
    """
    with duckdb.connect() as con:
        return con.execute(query).df()


def extract_sample_predictions(parquet_path: Path, days: int = 14) -> pd.DataFrame:
    """Extract a multi-day timeline of predictions for the top active product-state series."""
    sql_path = str(parquet_path).replace("'", "''")
    with duckdb.connect() as con:
        top_series = con.execute(f"""
            SELECT product_id, customer_state, SUM(demand) AS total_units
            FROM read_parquet('{sql_path}')
            GROUP BY 1, 2
            ORDER BY total_units DESC
            LIMIT 1
        """).fetchone()

        if top_series is None:
            return pd.DataFrame()

        prod_id, state = top_series[0], top_series[1]
        return con.execute(f"""
            SELECT 
                purchase_date,
                demand AS actual_demand,
                demand_lag_1 AS last_day,
                demand_lag_7 AS same_day_last_week,
                ROUND(rolling_mean_7, 2) AS moving_avg_7,
                ROUND(rolling_mean_28, 2) AS moving_avg_28,
                ROUND(demand_lag_28, 2) AS seasonal_naive_28
            FROM read_parquet('{sql_path}')
            WHERE product_id = ? AND customer_state = ?
            ORDER BY purchase_date
            LIMIT {days}
        """, [prod_id, state]).df()


def run_baseline_pipeline(
    features_dir: Path,
    output_dir: Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Execute evaluation and log all runs into MLflow adhering to MLOps conventions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = features_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # 1. Run DuckDB out-of-core evaluation on validation and test splits
    val_metrics = evaluate_baselines_duckdb(features_dir / "validation.parquet", "validation")
    test_metrics = evaluate_baselines_duckdb(features_dir / "test.parquet", "test")
    combined_metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)

    metrics_csv = output_dir / "forecasting_metrics_all.csv"
    combined_metrics.to_csv(metrics_csv, index=False)

    # 2. Champion Selection (Filter out trivial always_zero)
    non_trivial = val_metrics[val_metrics["baseline"] != "always_zero"]
    champion_baseline = non_trivial.sort_values("mae").iloc[0]["baseline"]

    # 3. Sample active predictions
    sample_preds = extract_sample_predictions(features_dir / "validation.parquet", days=14)
    sample_csv = output_dir / "sample_active_forecasts.csv"
    if not sample_preds.empty:
        sample_preds.to_csv(sample_csv, index=False)

    # 4. MLflow Logging
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logged_runs: list[dict[str, Any]] = []
    for baseline in val_metrics["baseline"].unique():
        is_champion = baseline == champion_baseline
        with mlflow.start_run(run_name=f"baseline__{baseline}") as run:
            # Parameters
            mlflow.log_params({
                "task": "demand_forecasting",
                "model_family": "heuristic_baseline",
                "baseline_name": baseline,
                "window_days": 7 if "7" in baseline else 28 if "28" in baseline else 1 if "last_day" in baseline else 0,
                "pipeline_feature_store": str(features_dir.relative_to(ROOT) if features_dir.is_relative_to(ROOT) else features_dir),
                "validation_range": f"{metadata['date_ranges']['validation']['min']} to {metadata['date_ranges']['validation']['max']}",
                "test_range": f"{metadata['date_ranges']['test']['min']} to {metadata['date_ranges']['test']['max']}",
                "is_champion_baseline": is_champion,
            })

            # Tags
            mlflow.set_tag("pipeline_integrated", "true")
            mlflow.set_tag("selected_on_validation", str(is_champion))
            if is_champion:
                mlflow.set_tag("model_status", "champion_baseline")

            # Metrics
            for split_name, table in [("val", val_metrics), ("test", test_metrics)]:
                row = table.loc[table["baseline"] == baseline].iloc[0]
                mlflow.log_metrics({
                    f"{split_name}_mae": float(row["mae"]),
                    f"{split_name}_rmse": float(row["rmse"]),
                    f"{split_name}_wape": float(row["wape"]),
                    f"{split_name}_positive_mae": float(row["positive_demand_mae"]),
                    f"{split_name}_zero_fraction": float(row["zero_demand_fraction"]),
                })

            # Artifacts
            artifact_path = output_dir / f"metrics_{baseline}.csv"
            subset = combined_metrics.loc[combined_metrics["baseline"] == baseline]
            subset.to_csv(artifact_path, index=False)
            mlflow.log_artifact(str(artifact_path))

            if is_champion and sample_csv.is_file():
                mlflow.log_artifact(str(sample_csv))

            logged_runs.append({
                "baseline": baseline,
                "run_id": run.info.run_id,
                "val_mae": float(val_metrics.loc[val_metrics["baseline"] == baseline, "mae"].iloc[0]),
                "val_wape": float(val_metrics.loc[val_metrics["baseline"] == baseline, "wape"].iloc[0]),
                "val_rmse": float(val_metrics.loc[val_metrics["baseline"] == baseline, "rmse"].iloc[0]),
                "champion": is_champion,
            })

    return {
        "status": "success",
        "champion_baseline": champion_baseline,
        "champion_val_mae": float(val_metrics.loc[val_metrics["baseline"] == champion_baseline, "mae"].iloc[0]),
        "champion_val_wape": float(val_metrics.loc[val_metrics["baseline"] == champion_baseline, "wape"].iloc[0]),
        "runs": logged_runs,
        "metrics_csv": str(metrics_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURES, help="Forecasting features directory")
    parser.add_argument("--output-dir", type=Path, default=BASELINES, help="Artifacts output directory")
    parser.add_argument("--tracking-uri", type=str, default=DEFAULT_TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="MLflow experiment name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("Running Demand Forecasting Baseline Evaluation Pipeline")
    print(f"Features:   {args.features_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Tracking:   {args.tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    print("=" * 70)

    results = run_baseline_pipeline(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )

    print("\nLeaderboard Summary:")
    runs_df = pd.DataFrame(results["runs"]).sort_values("val_mae").reset_index(drop=True)
    print(runs_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"🏆 CHAMPION BASELINE: '{results['champion_baseline']}'")
    print(f"   Validation MAE:  {results['champion_val_mae']:.6f}")
    print(f"   Validation WAPE: {results['champion_val_wape']:.4f}")
    print("=" * 70)
    print(f"Summary metrics saved to: {results['metrics_csv']}")
    print(f"Tracked in MLflow experiment: {args.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
