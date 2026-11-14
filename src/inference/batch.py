"""Generate predictions using only the model activated from a preserved selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import duckdb
import joblib
import pandas as pd

from src.common.config import SETTINGS
from src.models.common import MODEL_FILES, ROOT
from src.models.datasets import RecommendationSnapshot, load_history


def load_active_model(task, models_dir):
    """An active artifact can contain any algorithm or a selected baseline."""
    path = Path(models_dir) / MODEL_FILES[task]
    if not path.is_file():
        raise FileNotFoundError(f"No active {task} model at {path}; activate an uploaded model selection first")
    model = joblib.load(path)
    metadata = getattr(model, "serving_metadata", {})
    if metadata.get("approved_for_inference") is not True or metadata.get("task") != task:
        raise ValueError(f"{path} has no matching serving approval; activate a preserved model selection")
    return model


def run_pipeline(task, as_of, input_path, models_dir=ROOT / SETTINGS["inference"]["models_dir"],
                 output_dir=ROOT / SETTINGS["inference"]["output_dir"], pipeline_run_id=None):
    from src.versioning.workspace import locked
    with locked():
        return _run_pipeline(task, as_of, input_path, models_dir, output_dir, pipeline_run_id)


def _run_pipeline(task, as_of, input_path, models_dir=ROOT / SETTINGS["inference"]["models_dir"],
                 output_dir=ROOT / SETTINGS["inference"]["output_dir"], pipeline_run_id=None):
    """Forecast features are dated as_of; recommendation history is before as_of."""
    as_of = pd.Timestamp(as_of)
    model = load_active_model(task, models_dir)
    selection = model.serving_metadata
    pipeline_run_id = pipeline_run_id or uuid4().hex
    batch_key = hashlib.sha256(pipeline_run_id.encode()).hexdigest()[:24]
    output = (Path(output_dir) / task / f"as_of={as_of.date()}"
              / f"model_run_id={selection['model_run_id']}" / f"batch_run_key={batch_key}")
    path = output / "predictions.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        if not (existing.pipeline_run_id.eq(pipeline_run_id).all()
                and existing.model_run_id.eq(selection["model_run_id"]).all()
                and pd.to_datetime(existing.as_of).eq(as_of).all()):
            raise ValueError("Existing batch output identity differs; use a new batch run ID")
        return {"task": task, "model_name": selection["model_name"],
                "model_run_id": selection["model_run_id"], "rows": len(existing), "output": str(path)}
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
    predictions["pipeline_run_id"] = pipeline_run_id
    predictions["generated_at"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--models-dir", type=Path, default=ROOT / SETTINGS["inference"]["models_dir"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / SETTINGS["inference"]["output_dir"])
    parser.add_argument("--pipeline-run-id")
    print(json.dumps(run_pipeline(**vars(parser.parse_args())), indent=2))
    return 0
