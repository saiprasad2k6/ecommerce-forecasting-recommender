"""Evaluate forecasting and recommendation models on shared validation/test data."""

import duckdb
import numpy as np
import pandas as pd

from src.models.datasets import feature_query


def ranking_metrics(recommendations, purchases, k):
    """Precision: hits / K. Recall: hits / actual purchases."""
    if k < 1:
        raise ValueError("k must be positive")
    hits = len(set(recommendations[:k]) & purchases)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(purchases) if purchases else 0.0,
    }


def evaluate_forecasting(models, features_dir, split, label_before=None):
    """Calculate MAE/RMSE in batches and flag constant predictions."""
    totals = pd.DataFrame(0.0, index=list(models), columns=["absolute_error", "squared_error"])
    lowest_prediction = pd.Series(np.inf, index=totals.index)
    highest_prediction = pd.Series(-np.inf, index=totals.index)
    row_count = 0
    with duckdb.connect() as db:
        query = feature_query(features_dir, split, label_before)
        for batch in db.execute(query).fetch_record_batch(100_000):
            features = batch.to_pandas()
            predictions = pd.DataFrame({name: model.predict(features) for name, model in models.items()})
            errors = predictions.sub(features.target_demand_next_day.to_numpy(), axis="index")
            if not np.isfinite(errors.to_numpy()).all():
                raise ValueError("Forecast targets and predictions must be finite")
            totals["absolute_error"] += errors.abs().sum()
            totals["squared_error"] += errors.pow(2).sum()
            lowest_prediction = np.minimum(lowest_prediction, predictions.min())
            highest_prediction = np.maximum(highest_prediction, predictions.max())
            row_count += len(features)

    if row_count == 0:
        raise ValueError("Cannot evaluate an empty split")
    # Divide after summing all errors: the last batch can be smaller.
    metrics = pd.DataFrame({
        "rows": row_count,
        "mae": totals.absolute_error / row_count,
        "rmse": np.sqrt(totals.squared_error / row_count),
    })
    varying_predictions = highest_prediction - lowest_prediction > 1e-9
    return metrics.to_dict(orient="index"), varying_predictions.to_dict()


def evaluate_recommendations(models, snapshot, labels, k):
    """Average Precision@K and Recall@K equally across purchasing customers."""
    purchases_by_customer = labels.groupby("customer_unique_id").product_id.agg(set)
    if purchases_by_customer.empty:
        raise ValueError("No delivered purchase labels in evaluation period")
    metrics = {}
    for name, model in models.items():
        customer_scores = pd.DataFrame(
            ranking_metrics(model.predict(snapshot, customer, k), purchases, k)
            for customer, purchases in purchases_by_customer.items()
        )
        metrics[name] = customer_scores.mean().to_dict()
        metrics[name]["customers"] = len(purchases_by_customer)
    return metrics
