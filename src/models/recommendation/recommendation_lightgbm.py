"""Train and log a LambdaRank candidate, independently of baselines."""

import argparse
from pathlib import Path

from lightgbm import LGBMRanker, early_stopping
import numpy as np
import pandas as pd

from src.common.config import REPLAY, SETTINGS
from src.models.datasets import RecommendationSnapshot, load_history, load_labels
from src.models.evaluation import evaluate_recommendations as evaluate_model
from src.models.common import ROOT, TRACKING_URI, recommendation_context, save_run


class ProductionRecommender:
    """Rank historical candidates using the trained LightGBM estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, snapshot, customer, k=10):
        if not 1 <= k <= snapshot.candidate_limit:
            raise ValueError("k must be between 1 and the snapshot candidate limit")
        frame = snapshot.prepare_features(customer)

        ranked = pd.DataFrame({
            "product_id": frame.index,
            "score": self.estimator.predict(frame),
        })
        ranked = ranked.sort_values(["score", "product_id"], ascending=[False, True])
        return ranked.product_id.head(k).tolist()


def prepare_data(warehouse, start, end, candidate_limit=100):
    """Build monthly ranking groups with purchase outcomes known before end."""
    frames, targets, groups = [], [], []
    boundaries = pd.date_range(start, end, freq=pd.DateOffset(months=1))
    boundaries = boundaries.union(pd.DatetimeIndex([end]))

    for cutoff, label_end in zip(boundaries[:-1], boundaries[1:]):
        history = load_history(warehouse, cutoff)
        snapshot = RecommendationSnapshot(history, cutoff, candidate_limit)
        labels = load_labels(warehouse, cutoff, label_end, end)

        for customer, relevant in (
            labels.groupby("customer_unique_id").product_id.agg(set).items()
        ):
            frame = snapshot.prepare_features(customer).sort_index()
            target = frame.index.isin(relevant).astype("int32")
            # A ranking group needs both purchased and unpurchased candidates.
            if target.any() and not target.all():
                frames.append(frame)
                targets.extend(target)
                groups.append(len(frame))

    if not frames:
        raise ValueError(
            "No informative ranking groups; increase candidates or extend the date range"
        )
    return pd.concat(frames, ignore_index=True), np.asarray(targets), groups


def train_model(train, validation, n_estimators=100, seed=42, top_k=10):
    """Fit the ranker, stopping when validation Precision@K stops improving."""
    features, targets, groups = train
    validation_features, validation_targets, validation_groups = validation

    def validation_precision(actual, predicted, weight, group):
        # Rows are grouped by customer and sorted by product ID to break score ties.
        scores, start = [], 0

        for size in group:
            stop = start + size
            top = np.argsort(-predicted[start:stop], kind="stable")[:top_k]
            scores.append(actual[start:stop][top].sum() / top_k)
            start = stop

        return "precision_at_k", float(np.mean(scores)), True

    estimator = LGBMRanker(
        objective="lambdarank",
        metric="None",
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    estimator.fit(
        features,
        targets,
        group=groups,
        eval_X=validation_features,
        eval_y=validation_targets,
        eval_group=[validation_groups],
        eval_names=["validation"],
        eval_metric=validation_precision,
        callbacks=[early_stopping(20, verbose=False)],
    )
    return ProductionRecommender(estimator)


def run_pipeline(
    warehouse=ROOT / "data/warehouse/olist.duckdb",
    output_dir=ROOT / "data/production/lightgbm/recommendations",
    tracking_uri=TRACKING_URI,
    training_start=REPLAY["training_start"],
    validation_cutoff=REPLAY["validation_start"],
    test_cutoff=REPLAY["test_start"],
    test_end=REPLAY["monitoring_start"],
    candidate_limit=SETTINGS["recommendation"]["candidate_limit"],
    top_k=SETTINGS["recommendation"]["top_k"],
    n_estimators=100,
    seed=42,
    pipeline_run_id=None,
):
    # Validate the date windows and recommendation size.
    start, validation_start, test_start, end = map(
        pd.Timestamp, [training_start, validation_cutoff, test_cutoff, test_end]
    )
    if any(
        date.tzinfo is not None or pd.isna(date)
        for date in [start, validation_start, test_start, end]
    ):
        raise ValueError("Use timezone-naive dates matching warehouse timestamps")
    if not start < validation_start < test_start < end:
        raise ValueError(
            "Training, validation, test cutoffs and test end must be chronological"
        )
    if not 1 <= top_k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")

    context = recommendation_context(
        warehouse, validation_start, test_start, end, candidate_limit, top_k
    )

    # Fit using training data, with validation controlling early stopping.
    train = prepare_data(warehouse, start, validation_start, candidate_limit)
    validation = prepare_data(warehouse, validation_start, test_start, candidate_limit)
    model = train_model(train, validation, n_estimators, seed, top_k)

    # Evaluate validation and test separately after training.
    models = {"lightgbm_ranker": model}
    scores = {}
    for split, cutoff, stop in [
        ("validation", validation_start, test_start),
        ("test", test_start, end),
    ]:
        snapshot = RecommendationSnapshot(
            load_history(warehouse, cutoff), cutoff, candidate_limit
        )
        labels = load_labels(warehouse, cutoff, stop)
        scores[split] = evaluate_model(models, snapshot, labels, top_k)["lightgbm_ranker"]

    if context != recommendation_context(
        warehouse, validation_start, test_start, end, candidate_limit, top_k
    ):
        raise ValueError("Evaluation data changed during training")

    # Save this candidate's metrics and artifact for the comparison job.
    report = {
        "task": "recommendation",
        "role": "candidate",
        "model": "lightgbm_ranker",
        **scores,
        "evaluation_context": context,
    }
    params = {
        "training_start": str(start.date()),
        "seed": seed,
        "n_estimators": n_estimators,
        "k": top_k,
        "candidate_limit": candidate_limit,
    }
    return save_run(
        model,
        report,
        params,
        output_dir,
        tracking_uri,
        "ecommerce-recommendations",
        pipeline_run_id,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse", type=Path, default=ROOT / "data/warehouse/olist.duckdb"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/production/lightgbm/recommendations",
    )
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--training-start", default=REPLAY["training_start"])
    parser.add_argument("--validation-cutoff", default=REPLAY["validation_start"])
    parser.add_argument("--test-cutoff", default=REPLAY["test_start"])
    parser.add_argument("--test-end", default=REPLAY["monitoring_start"])
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=SETTINGS["recommendation"]["candidate_limit"],
    )
    parser.add_argument(
        "--top-k", type=int, default=SETTINGS["recommendation"]["top_k"]
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    report = run_pipeline(**vars(parser.parse_args()))
    print(f"Pipeline run: {report['pipeline_run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
