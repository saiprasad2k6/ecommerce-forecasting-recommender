from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RecommenderArtifacts:
    popularity_rank: pd.DataFrame
    customer_history: dict[str, list[str]]


def train_recommender(frame: pd.DataFrame) -> RecommenderArtifacts:
    popularity = (
        frame.groupby("product_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .rename("score")
        .reset_index()
    )
    customer_history = (
        frame.groupby("customer_id")["product_id"].agg(lambda s: list(dict.fromkeys(s.tolist()))).to_dict()
    )
    return RecommenderArtifacts(popularity_rank=popularity, customer_history=customer_history)


def save_recommender(model: RecommenderArtifacts, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def load_recommender(path: Path) -> RecommenderArtifacts:
    with path.open("rb") as handle:
        return pickle.load(handle)


def recommend_products(model: RecommenderArtifacts, customer_id: str, top_k: int = 5) -> list[str]:
    bought = set(model.customer_history.get(customer_id, []))
    ranked = model.popularity_rank["product_id"].tolist()
    recommendations = [product for product in ranked if product not in bought]
    return recommendations[:top_k]
