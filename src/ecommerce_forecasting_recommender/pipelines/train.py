from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecommerce_forecasting_recommender.config import settings
from ecommerce_forecasting_recommender.data.features import demand_features, time_based_split
from ecommerce_forecasting_recommender.data.preprocessing import (
    build_modeling_table,
    save_processed_dataset,
)
from ecommerce_forecasting_recommender.data.validation import validate_raw_data
from ecommerce_forecasting_recommender.models.forecasting import (
    save_forecast_model,
    train_forecast_model,
)
from ecommerce_forecasting_recommender.models.recommender import save_recommender, train_recommender
from ecommerce_forecasting_recommender.models.tracking import log_metrics, optional_mlflow_run


def run_training_pipeline(raw_dir: Path | None = None, processed_dir: Path | None = None, model_dir: Path | None = None) -> dict[str, float]:
    raw_dir = raw_dir or settings.raw_data_dir
    processed_dir = processed_dir or settings.processed_data_dir
    model_dir = model_dir or settings.model_dir

    report = validate_raw_data(raw_dir)
    if report["missing_files"] or report["invalid_columns"]:
        raise ValueError(f"Raw dataset validation failed: {report}")

    merged = build_modeling_table(raw_dir)
    save_processed_dataset(merged, processed_dir / "modeling_table.csv")

    demand_frame = demand_features(merged)
    save_processed_dataset(demand_frame, processed_dir / "demand_features.csv")

    splits = time_based_split(demand_frame)
    if len(splits.train) == 0 or len(splits.validation) == 0:
        raise ValueError("Not enough observations for time-based split")

    forecast = train_forecast_model(splits.train, splits.validation)
    recommender = train_recommender(merged)

    save_forecast_model(forecast.model, model_dir / "forecast_model.pkl")
    save_recommender(recommender, model_dir / "recommender.pkl")

    metrics = {
        "forecast_baseline_mae": forecast.baseline_mae,
        "forecast_model_mae": forecast.model_mae,
        "recommendation_catalog_size": float(len(recommender.popularity_rank)),
    }

    with optional_mlflow_run("ecommerce-forecasting", "train-forecast-and-recs"):
        log_metrics(metrics)

    return metrics


def run_batch_forecast(features_path: Path, model_path: Path, output_path: Path) -> Path:
    from ecommerce_forecasting_recommender.models.forecasting import load_forecast_model

    model = load_forecast_model(model_path)
    batch = pd.read_csv(features_path)
    output = batch.copy()
    output["prediction"] = model.predict(batch[["day_index", "lag_1", "avg_price"]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path
