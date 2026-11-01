"""Evaluate recommendation rules separately from ranker training."""

import argparse
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.common.config import REPLAY, SETTINGS
from src.models.datasets import RecommendationSnapshot, load_history, load_labels
from src.models.evaluation import evaluate_recommendations as evaluate_model
from src.models.common import ROOT, TRACKING_URI, recommendation_context, save_run

METHODS = ["global_popularity", "category_popularity", "purchase_history"]


def predict_baseline(snapshot, method, customer, k=10):
    """Predict from historical popularity, category preference or purchase history."""
    if method == "global_popularity":
        selected = snapshot.global_order[:k]
    elif method == "category_popularity":
        categories = snapshot.customer_categories.get(customer, set())
        matches = snapshot.popular_categories.loc[snapshot.popular_categories.isin(categories)]
        selected = matches.index[:k].tolist() + snapshot.global_order[:k]
    elif method == "purchase_history":
        selected = snapshot.customer_products.get(customer, [])[:k] + snapshot.global_order[:k]
    else:
        raise ValueError(method)
    return pd.Index(selected).drop_duplicates()[:k].tolist()


class BaselineRecommender:
    def __init__(self, method="global_popularity"):
        if method not in METHODS:
            raise ValueError(f"Unknown baseline: {method}")
        self.method = method

    def predict(self, snapshot, customer, k=10):
        return predict_baseline(snapshot, self.method, customer, k)


def run_pipeline(
    warehouse=ROOT / "data/warehouse/olist.duckdb",
    output_dir=ROOT / "data/baselines/recommendations",
    tracking_uri=TRACKING_URI,
    validation_cutoff=REPLAY["validation_start"],
    test_cutoff=REPLAY["test_start"],
    test_end=REPLAY["monitoring_start"],
    candidate_limit=SETTINGS["recommendation"]["candidate_limit"],
    top_k=SETTINGS["recommendation"]["top_k"],
    pipeline_run_id=None,
):
    if not 1 <= top_k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")
    validation_start, test_start, end = map(
        pd.Timestamp, [validation_cutoff, test_cutoff, test_end]
    )
    if not validation_start < test_start < end:
        raise ValueError("Evaluation dates must be chronological")

    pipeline_run_id = pipeline_run_id or uuid4().hex
    context = recommendation_context(
        warehouse, validation_start, test_start, end, candidate_limit, top_k
    )
    models = {name: BaselineRecommender(name) for name in METHODS}

    scores = {}
    for split, start, stop in [
        ("validation", validation_start, test_start),
        ("test", test_start, end),
    ]:
        snapshot = RecommendationSnapshot(
            load_history(warehouse, start), start, candidate_limit
        )
        labels = load_labels(warehouse, start, stop)
        scores[split] = evaluate_model(models, snapshot, labels, top_k)

    if context != recommendation_context(
        warehouse, validation_start, test_start, end, candidate_limit, top_k
    ):
        raise ValueError("Evaluation data changed during baseline evaluation")

    runs = []
    for name, model in models.items():
        report = {
            "task": "recommendation", "role": "baseline", "model": name,
            "validation": scores["validation"][name], "test": scores["test"][name],
            "evaluation_context": context,
        }
        runs.append(save_run(
            model, report, {"k": top_k, "candidate_limit": candidate_limit},
            output_dir, tracking_uri, "ecommerce-recommendations", pipeline_run_id,
        ))
    return {"pipeline_run_id": pipeline_run_id, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse", type=Path, default=ROOT / "data/warehouse/olist.duckdb"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/baselines/recommendations"
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument(
        "--top-k", type=int, default=SETTINGS["recommendation"]["top_k"]
    )
    parser.add_argument(
        "--candidate-limit", type=int, default=SETTINGS["recommendation"]["candidate_limit"]
    )
    parser.add_argument("--validation-cutoff", default=REPLAY["validation_start"])
    parser.add_argument("--test-cutoff", default=REPLAY["test_start"])
    parser.add_argument("--test-end", default=REPLAY["monitoring_start"])
    parser.add_argument("--pipeline-run-id")
    report = run_pipeline(**vars(parser.parse_args()))
    print(f"Pipeline run: {report['pipeline_run_id']}")
    return 0
