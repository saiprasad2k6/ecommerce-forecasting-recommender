"""Generate predictions using only the model activated by comparison."""
import argparse
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import duckdb
import joblib
import pandas as pd

from src.models.common import MODEL_FILES, ROOT
from src.models.datasets import RecommendationSnapshot, load_history


def load_active_model(task, models_dir):
    """An active artifact can contain any algorithm or a selected baseline."""
    path = Path(models_dir) / MODEL_FILES[task]
    if not path.is_file():
        raise FileNotFoundError(f"No active {task} model at {path}; run model comparison first")
    model = joblib.load(path)
    metadata = getattr(model, "serving_metadata", {})
    if metadata.get("approved_for_inference") is not True or metadata.get("task") != task:
        raise ValueError(f"{path} has no matching serving approval; publish it through model comparison")
    return model


def run_pipeline(task, as_of, input_path, models_dir=ROOT / "models",
                 output_dir=ROOT / "data/predictions", pipeline_run_id=None):
    """Forecast features are dated as_of; recommendation history is before as_of."""
    as_of = pd.Timestamp(as_of)
    model = load_active_model(task, models_dir)
    selection = model.serving_metadata
    if task == "forecasting":
        with duckdb.connect() as db:
            features = db.execute("""
                SELECT * FROM read_parquet(?)
                WHERE CAST(purchase_date AS DATE) = CAST(? AS DATE)
                ORDER BY product_id, customer_state
            """, [str(input_path), as_of.to_pydatetime()]).df()
        if features.empty:
            raise ValueError(f"No forecasting features for {as_of.date()}")
        predictions = features[["purchase_date", "product_id", "customer_state"]].copy()
        predictions["target_date"] = pd.to_datetime(features.purchase_date) + pd.Timedelta(days=1)
        predictions["prediction"] = model.predict(features)
    else:
        history = load_history(Path(input_path), as_of)
        snapshot = RecommendationSnapshot(history, as_of, selection.get("candidate_limit", 100))
        k = selection.get("k", 10)
        predictions = pd.DataFrame({
            "customer_unique_id": history.customer_unique_id.drop_duplicates().sort_values().tolist(),
        })
        predictions["recommended_products"] = [
            model.predict(snapshot, customer, k) for customer in predictions.customer_unique_id
        ]
        predictions["k"] = k

    predictions["as_of"] = as_of
    predictions["model_name"] = selection["model_name"]
    predictions["model_role"] = selection["model_role"]
    predictions["model_run_id"] = selection["model_run_id"]
    predictions["training_pipeline_run_id"] = selection["training_pipeline_run_id"]
    predictions["pipeline_run_id"] = pipeline_run_id or uuid4().hex
    output = (Path(output_dir) / task / f"as_of={as_of.date()}"
              / f"model_run_id={selection['model_run_id']}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.parquet"
    with tempfile.NamedTemporaryFile(dir=output, suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        predictions.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"task": task, "model_name": selection["model_name"],
            "model_run_id": selection["model_run_id"], "rows": len(predictions), "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(MODEL_FILES), required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--input-path", type=Path, required=True,
                        help="Forecast feature Parquet file, or recommendation DuckDB warehouse")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/predictions")
    parser.add_argument("--pipeline-run-id")
    print(json.dumps(run_pipeline(**vars(parser.parse_args())), indent=2))
    return 0
