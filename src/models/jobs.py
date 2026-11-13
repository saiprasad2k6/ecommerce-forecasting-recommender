"""Execute one model job using the frozen inputs and parameters of a run."""
import argparse
from importlib import import_module
import json
from pathlib import Path

from src.models import common
from src.models.comparison import MODEL_JOBS
from src.models.datasets import validate_inputs
from src.versioning.snapshots import ROOT, load_version, load_run, run_dir, write_json, validate_run_context


def run_job(run_id, task, model):
    from src.versioning.workspace import training_guard
    with training_guard(run_id):
        return _run_job(run_id, task, model)


def _run_job(run_id, task, model):
    manifest = load_run(run_id)
    validate_run_context(manifest)
    parameters = manifest["resolved_parameters"]
    prepared = load_version("prepared", manifest["prepared_version"], committed=True)
    # Hash only this model task's consumed files, not raw CSVs it never reads.
    resource = "forecasting" if task == "forecasting" else "warehouse"
    from src.versioning.snapshots import digest_tree
    selected = prepared["resources"][resource]
    if digest_tree(ROOT / selected["path"]) != prepared["identity"]["files"][resource]:
        raise ValueError("Prepared inputs changed; restore the selected version")
    features = ROOT / prepared["resources"]["forecasting"]["path"]
    warehouse = ROOT / prepared["resources"]["warehouse"]["path"]
    validate_inputs(features, warehouse, parameters["candidate_limit"], parameters["top_k"])
    directory = run_dir(run_id)
    if (directory / "selection/decisions.json").exists():
        raise ValueError("This run is already compared; use a new run ID to retrain")
    if model == "baselines":
        module = import_module(f"src.models.{task}.{task}_baseline")
        options = {}
        if task == "recommendation":
            options = {"candidate_limit": parameters["candidate_limit"], "top_k": parameters["top_k"],
                       "validation_cutoff": parameters["validation_start"],
                       "test_cutoff": parameters["test_start"], "test_end": parameters["monitoring_start"]}
    else:
        job = MODEL_JOBS[task][model]
        module = import_module("src.models." + job["script"].removesuffix(".py").replace("/", "."))
        options = {flag.replace("-", "_"): parameters[key] for flag, key in job["parameters"].items()}
        options["seed"] = parameters["seed"]
    options["features_dir" if task == "forecasting" else "warehouse"] = features if task == "forecasting" else warehouse
    options.update(output_dir=directory / "training" / task / model, pipeline_run_id=run_id,
                   tracking_uri=common.TRACKING_URI)
    result = module.run_pipeline(**options)
    reports = result["runs"] if "runs" in result else [result]
    for report in reports:
        write_json(directory / "jobs" / task / f"{report['model']}.json", {
            "mlflow_run_id": report["mlflow_run_id"],
            "report": str((Path(report["output_dir"]) / "report.json").relative_to(directory)),
            "model": str((Path(report["output_dir"]) / "model.pkl").relative_to(directory))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--task", required=True, choices=["forecasting", "recommendation"])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(run_job(args.pipeline_run_id, args.task, args.model), indent=2))
    return 0
