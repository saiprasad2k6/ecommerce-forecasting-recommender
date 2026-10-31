"""Evaluate recommendation rules separately from ranker training."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.models.evaluation import RecommendationSnapshot, load_history, load_labels, ranking_metrics, evaluate_recommendations
from src.models.production.common import ROOT, TRACKING_URI, recommendation_context, save_run

METHODS = ["global_popularity", "category_popularity", "purchase_history"]
RECENT_DAYS = 90

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
def unique_top(products, k):
    return list(dict.fromkeys(products))[:k]


def prepare_recommender(history, cutoff):
    snapshot = RecommendationSnapshot(history, cutoff)
    return lambda customer, method, k: predict_baseline(snapshot, method, customer, k)


class BaselineRecommender:
    def __init__(self, method="global_popularity"):
        if method not in METHODS:
            raise ValueError(f"Unknown baseline: {method}")
        self.method = method

    def predict(self, snapshot, customer, k=10):
        return predict_baseline(snapshot, self.method, customer, k)

    recommend = predict


def run_recommendation_pipeline(warehouse_path=ROOT / "data/warehouse/olist.duckdb",
        output_dir=ROOT / "data/baselines/recommendations", tracking_uri=TRACKING_URI,
        experiment_name="ecommerce-recommendations", k=10, candidate_limit=100,
        validation_cutoff="2018-07-01", test_cutoff="2018-08-01", test_end="2018-09-01", comparison_id=None):
    if not 1 <= k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")
    validation_start, test_start, end = map(pd.Timestamp, [validation_cutoff, test_cutoff, test_end])
    if not validation_start < test_start < end:
        raise ValueError("Evaluation dates must be chronological")
    comparison_id = comparison_id or uuid4().hex
    context = recommendation_context(warehouse_path, validation_start, test_start, end, candidate_limit, k)
    models = {name: BaselineRecommender(name) for name in METHODS}
    scores = {}
    for split, start, stop in [("validation", validation_start, test_start), ("test", test_start, end)]:
        snapshot = RecommendationSnapshot(load_history(warehouse_path, start), start, candidate_limit)
        labels = load_labels(warehouse_path, start, stop, stop)
        scores[split] = evaluate_recommendations(models, snapshot, labels, k)
    if context != recommendation_context(warehouse_path, validation_start, test_start, end, candidate_limit, k):
        raise ValueError("Evaluation data changed during baseline evaluation")
    runs = [save_run(model, {"task": "recommendation", "role": "baseline", "model": name,
                 "validation": scores["validation"][name], "test": scores["test"][name], "evaluation_context": context},
                 {"k": k, "candidate_limit": candidate_limit}, output_dir, tracking_uri, experiment_name, comparison_id)
            for name, model in models.items()]
    champion = min(METHODS, key=lambda name: (-scores["validation"][name]["recall_at_k"], -scores["validation"][name]["precision_at_k"], name))
    return {"comparison_id": comparison_id, "champion_baseline": champion, "runs": runs}


run_pipeline = run_recommendation_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", dest="warehouse_path", type=Path, default=ROOT / "data/warehouse/olist.duckdb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/baselines/recommendations")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default="ecommerce-recommendations")
    parser.add_argument("--top-k", dest="k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--validation-cutoff", default="2018-07-01")
    parser.add_argument("--test-cutoff", default="2018-08-01")
    parser.add_argument("--test-end", default="2018-09-01")
    parser.add_argument("--comparison-id")
    print(json.dumps(run_pipeline(**vars(parser.parse_args())), indent=2))
    return 0
