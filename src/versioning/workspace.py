"""Cross-DAG ownership for the single shared local workspace.

A persistent lease spans tasks; flock serializes each command. Leases never expire
silently: a crashed workflow must be released explicitly after its workers stop.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from src.versioning import snapshots as versions


def state_path():
    return versions.ROOT / ".dvc/workspace.json"


def state():
    path = state_path()
    return json.loads(path.read_text()) if path.exists() else None


@contextmanager
def locked(owner=None, require_owner=False):
    path = versions.ROOT / ".dvc/workspace.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        current = state()
        if current and current["owner"] != owner:
            raise RuntimeError(f"Workspace reserved by {current['owner']}; finish or explicitly release that workflow")
        if require_owner and (not current or current["owner"] != owner):
            raise RuntimeError("Workspace lease missing; begin the workflow again before retrying")
        yield current


def reserve(owner, kind, run_id, parameters, release=True):
    with locked(owner):
        existing = state()
        if existing:
            if existing["parameters"] != parameters or existing["kind"] != kind or existing["release"] != release:
                raise ValueError("A workspace reservation cannot change parameters")
            return existing
        record = {"owner": owner, "kind": kind, "run_id": run_id,
                  "parameters": parameters, "release": release,
                  "provenance": versions.provenance(release), "completed": []}
        versions.write_json(state_path(), record)
        return record


def release(owner):
    with locked(owner):
        if state_path().exists():
            state_path().unlink()


# The order is part of the preparation contract, not an alternate executor.
PREPARATION_STAGES = ["download_olist_data", "validate_olist_data", "clean_olist_data",
                      "prepare_curated_data", "build_forecasting_features", "build_recommendation_features"]


def execute_preparation(run_id, stage, command):
    """Execute one Airflow/CLI task and promote its outputs only on success."""
    owner = "prepare:" + run_id
    with locked(owner, require_owner=True) as record:
        if stage not in PREPARATION_STAGES:
            raise ValueError(f"Unknown preparation stage: {stage}")
        index = PREPARATION_STAGES.index(stage)
        if index >= 2 and "raw_version" not in record:
            raise ValueError("Publish the validated raw version before transformation")
        if record["completed"][:index] != PREPARATION_STAGES[:index]:
            raise ValueError("Preparation stages must complete in order")
        if "prepared_version" in record:
            raise ValueError("Preparation already published; use a new run ID")
        # Invalidate this stage and every downstream receipt before a retry.
        record["completed"] = record["completed"][:index]
        record["receipts"] = {key: value for key, value in record.get("receipts", {}).items() if key in record["completed"]}
        versions.write_json(state_path(), record)
        settings = versions.SETTINGS["data"]
        outputs = {
            "download_olist_data": {"--output-dir": settings["raw_dir"]},
            "validate_olist_data": {},
            "clean_olist_data": {"--output-dir": settings["processed_dir"]},
            "prepare_curated_data": {"--output-dir": settings["curated_dir"], "--warehouse": settings["warehouse"]},
            "build_forecasting_features": {"--output-dir": settings["forecasting_features"]},
            "build_recommendation_features": {"--output-dir": settings["recommendation_features"]},
        }[stage]
        temporary_root = versions.ROOT / "data/.staging"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            arguments = list(command)
            promotions = []
            for flag, path in outputs.items():
                target = versions.ROOT / path
                staged = Path(temporary) / flag.removeprefix("--") / target.name
                staged.parent.mkdir(parents=True, exist_ok=True)
                # Reuse complete existing downloads unless the caller explicitly requests --force.
                if stage == "download_olist_data" and target.exists() and "--force" not in arguments:
                    staged = target
                if flag in arguments:
                    arguments[arguments.index(flag) + 1] = str(staged)
                else:
                    arguments.extend([flag, str(staged)])
                if staged != target:
                    promotions.append((staged, target))
            environment = {**os.environ, "OLIST_MANAGED_PREPARATION": owner}
            result = subprocess.run(arguments, cwd=versions.ROOT, env=environment, check=True,
                                    stdout=subprocess.PIPE if stage == "validate_olist_data" else None, text=True)
            if stage == "validate_olist_data":
                report = json.loads(result.stdout)
                versions.write_json(versions.ROOT / settings["validation_report"], report)
            # Validate every output before moving any of them.
            for staged, _ in promotions:
                versions.digest_tree(staged)
            backups = []
            try:
                for number, (staged, target) in enumerate(promotions):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup = Path(temporary) / f"previous-{number}"
                    if target.exists():
                        target.rename(backup)
                    backups.append((target, backup))
                    staged.rename(target)
            except BaseException:
                for target, backup in reversed(backups):
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists():
                        target.unlink()
                    if backup.exists():
                        backup.rename(target)
                raise
        record["receipts"][stage] = {"command": list(command),
            "outputs": {path: versions.digest_tree(versions.ROOT / path) for path in outputs.values()}}
        record["completed"].append(stage)
        versions.write_json(state_path(), record)


@contextmanager
def training_guard(run_id):
    managed = (versions.run_dir(run_id) / "manifest.json").exists()
    with locked("train:" + run_id if managed else None, require_owner=managed):
        yield
