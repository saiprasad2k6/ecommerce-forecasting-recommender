"""Read-only request-time inference using the same approved bundles as batch jobs."""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.common.config import SETTINGS
from src.inference.batch import load_active_model
from src.models.common import ROOT
from src.models.datasets import RecommendationSnapshot, load_history
from src.versioning.workspace import locked
from api.schemas import ModelInfo


class InferenceError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.details = status, code, details


@dataclass(frozen=True)
class InferencePaths:
    models: Path = ROOT / SETTINGS["inference"]["models_dir"]
    forecast_features: Path = ROOT / SETTINGS["inference"]["forecast_features"]
    warehouse: Path = ROOT / SETTINGS["inference"]["warehouse"]


def model_info(task, model):
    return ModelInfo(task=task, **{key: model.serving_metadata[key] for key in (
        "model_name", "model_role", "model_run_id", "training_pipeline_run_id",
    )})


class InferenceService:
    def __init__(self, paths=None):
        self.paths = paths or InferencePaths()

    def models(self):
        with locked():
            return [model_info(task, load_active_model(task, self.paths.models))
                    for task in ("forecasting", "recommendation")]

    def forecast(self, request):
        with locked():
            model = load_active_model("forecasting", self.paths.models)
            if not self.paths.forecast_features.is_file():
                raise InferenceError(503, "features_unavailable", "Prepared forecasting features are unavailable.")
            pairs = pd.DataFrame([item.model_dump() for item in request.items]).astype(object)
            pairs["request_order"] = range(len(pairs))
            with duckdb.connect() as connection:
                connection.register("requested", pairs)
                features = connection.execute("""
                    SELECT f.*, r.request_order
                    FROM read_parquet(?, hive_partitioning=false) f
                    JOIN requested r USING (product_id, customer_state)
                    WHERE CAST(f.purchase_date AS DATE) = CAST(? AS DATE)
                    ORDER BY r.request_order
                """, [str(self.paths.forecast_features), request.as_of]).df()
            found = set(zip(features.product_id, features.customer_state))
            missing = [f"{item.product_id}/{item.customer_state}" for item in request.items
                       if (item.product_id, item.customer_state) not in found]
            if missing:
                raise InferenceError(404, "features_not_found", "No prepared features for the requested date/product/state.", missing)
            if len(features) != len(request.items):
                raise InferenceError(503, "invalid_features", "Prepared features contain duplicate product/state rows.")
            values = np.asarray(model.predict(features.drop(columns=["target_demand_next_day", "request_order"], errors="ignore")))
            if values.shape != (len(features),) or not np.isfinite(values).all() or (values < 0).any():
                raise InferenceError(503, "invalid_prediction", "The active model returned invalid demand predictions.")
            predictions = [dict(item.model_dump(), target_date=request.as_of + timedelta(days=1), prediction=float(value))
                           for item, value in zip(request.items, values)]
            return {"model": model_info("forecasting", model), "predictions": predictions}

    def recommend(self, request):
        with locked():
            model = load_active_model("recommendation", self.paths.models)
            metadata = model.serving_metadata
            limit = metadata.get("candidate_limit", 100)
            k = request.k if request.k is not None else metadata.get("k", 10)
            if not 1 <= k <= limit:
                raise InferenceError(422, "invalid_k", f"k must be between 1 and the active candidate limit ({limit}).")
            if not self.paths.warehouse.is_file():
                raise InferenceError(503, "warehouse_unavailable", "The recommendation warehouse is unavailable.")
            history = load_history(self.paths.warehouse, pd.Timestamp(request.as_of))
            if history.empty:
                raise InferenceError(404, "history_not_found", "No delivered purchase history exists before the requested date.")
            snapshot = RecommendationSnapshot(history, request.as_of, limit)
            products = list(model.predict(snapshot, request.customer_unique_id, k))
            if len(products) > k or len(set(products)) != len(products) or any(product not in snapshot.catalog for product in products):
                raise InferenceError(503, "invalid_prediction", "The active model returned invalid recommendations.")
            return {
                "model": model_info("recommendation", model), "k": k,
                "cold_start": request.customer_unique_id not in snapshot.customer_products,
                "recommendations": [{"rank": index, "product_id": product}
                                    for index, product in enumerate(products, 1)],
            }
