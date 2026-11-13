"""Activate committed, uploaded model releases with an auditable active record."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib

from src.common.config import SETTINGS
from src.models.common import MODEL_FILES, ROOT, publish_model
from src.versioning.snapshots import digest_tree, lineage, metadata_dir, provenance, require_committed, run_dir, run_key, write_json
from src.versioning.workspace import locked


def activate_run(run_id, models_dir=None):
    with locked():
        return _activate_run(run_id, models_dir)


def _activate_run(run_id, models_dir=None):
    directory = run_dir(run_id)
    metadata = metadata_dir() / "runs" / run_key(run_id)
    pointer = metadata / "artifacts.dvc"
    if not pointer.is_file() or not (metadata / "manifest.json").is_file():
        raise ValueError("Preserve and upload this run before activation")
    receipt = json.loads((metadata / "manifest.json").read_text())
    if not receipt["release"]:
        raise ValueError("Development runs cannot be activated; create a clean release run")
    current = provenance(release=True)
    if current["source_tree_sha256"] != receipt["source_tree_sha256"]:
        raise ValueError("Activation code/configuration differs from the training release")
    if current["environment_sha256"] != receipt["environment_sha256"]:
        raise ValueError("Activation environment differs from the training release")
    chain = lineage(run_id, committed=True)
    if receipt["artifact_pointer_sha256"] != hashlib.sha256(pointer.read_bytes()).hexdigest():
        raise ValueError("Artifact pointer does not match the completed upload")
    if digest_tree(directory) != receipt["artifact_files"]:
        raise ValueError("Preserved run artifacts changed or are missing; restore the snapshot first")
    decisions = json.loads((directory / "selection/decisions.json").read_text())
    models = {task: joblib.load(directory / "selection/models" / MODEL_FILES[task]) for task in decisions}
    paths = ({"forecasting": ROOT / SETTINGS["models"]["forecast_model_path"],
              "recommendation": ROOT / SETTINGS["models"]["recommender_model_path"]}
             if models_dir is None else {task: Path(models_dir) / MODEL_FILES[task] for task in decisions})
    previous = {task: path.read_bytes() if path.exists() else None for task, path in paths.items()}
    publication = {}
    try:
        for task, decision in decisions.items():
            status = publish_model(models[task], paths[task], decision["approved"])
            active = joblib.load(paths[task])
            publication[task] = {"status": status, "active_path": str(paths[task]),
                                 "active_selection": active.serving_metadata,
                                 "active_sha256": hashlib.sha256(paths[task].read_bytes()).hexdigest()}
        record = {"requested_run_id": run_id, "requested_lineage": chain,
                  "activated_at": datetime.now(timezone.utc).isoformat(), "publication": publication}
        write_json(ROOT / "data/activations" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json", record)
    except BaseException:
        # Managed inference uses the same lock, so it cannot see a partial update.
        for task, content in previous.items():
            if content is None:
                paths[task].unlink(missing_ok=True)
            else:
                paths[task].write_bytes(content)
        raise
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--models-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(activate_run(args.pipeline_run_id, args.models_dir), indent=2))
    return 0
