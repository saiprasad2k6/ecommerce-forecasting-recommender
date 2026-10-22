from pathlib import Path

from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data
from ecommerce_forecasting_recommender.data.features import demand_features, time_based_split
from ecommerce_forecasting_recommender.data.preprocessing import build_modeling_table
from ecommerce_forecasting_recommender.models.forecasting import (
    predict_demand,
    train_forecast_model,
)
from ecommerce_forecasting_recommender.models.recommender import (
    recommend_products,
    train_recommender,
)


def test_forecast_training_and_prediction(tmp_path: Path) -> None:
    bootstrap_sample_raw_data(tmp_path)
    merged = build_modeling_table(tmp_path)
    daily = demand_features(merged)
    splits = time_based_split(daily, train_ratio=0.5, val_ratio=0.25)
    artifacts = train_forecast_model(splits.train, splits.validation)
    value = predict_demand(artifacts.model, day_index=3, lag_1=1.0, avg_price=20.0)
    assert value >= 0


def test_recommender_top_k(tmp_path: Path) -> None:
    bootstrap_sample_raw_data(tmp_path)
    merged = build_modeling_table(tmp_path)
    model = train_recommender(merged)
    recs = recommend_products(model, customer_id="c1", top_k=2)
    assert len(recs) <= 2
