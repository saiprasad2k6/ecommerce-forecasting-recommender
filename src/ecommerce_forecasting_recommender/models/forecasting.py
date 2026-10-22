from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


@dataclass
class ForecastArtifacts:
    model: LinearRegression
    baseline_mae: float
    model_mae: float


def _naive_baseline(validation: pd.DataFrame) -> np.ndarray:
    return validation["lag_1"].to_numpy()


def train_forecast_model(train: pd.DataFrame, validation: pd.DataFrame) -> ForecastArtifacts:
    features = ["day_index", "lag_1", "avg_price"]
    model = LinearRegression()
    model.fit(train[features], train["demand"])

    baseline_pred = _naive_baseline(validation)
    model_pred = model.predict(validation[features])

    baseline_mae = float(mean_absolute_error(validation["demand"], baseline_pred))
    model_mae = float(mean_absolute_error(validation["demand"], model_pred))
    return ForecastArtifacts(model=model, baseline_mae=baseline_mae, model_mae=model_mae)


def save_forecast_model(model: LinearRegression, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def load_forecast_model(path: Path) -> LinearRegression:
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_demand(model: LinearRegression, day_index: int, lag_1: float, avg_price: float) -> float:
    values = np.array([[day_index, lag_1, avg_price]])
    prediction = model.predict(values)[0]
    return float(max(prediction, 0.0))
