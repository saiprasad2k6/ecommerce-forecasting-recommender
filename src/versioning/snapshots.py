"""DVC checkpoints and explicit raw -> prepared -> model lineage.

Data stays in its normal working paths. Only small manifests/pointers are retained
per version; DVC owns content caching. Shared-workspace isolation lives in workspace.py.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
import tempfile

import yaml
from src.common.config import SETTINGS

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def identifier(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_key(run_id):
    return hashlib.sha256(run_id.encode()).hexdigest()[:24]


def run_dir(run_id):
    return ROOT / SETTINGS["versioning"]["runs_dir"] / run_key(run_id)


def load_run(run_id):
    result = json.loads((run_dir(run_id) / "manifest.json").read_text())
    if result["run_id"] != run_id:
        raise ValueError("Run ID mismatch")
    return result


def metadata_dir():
    return ROOT / SETTINGS["versioning"]["metadata_dir"]


def catalog_path(kind, version):
    if kind not in ["raw", "prepared"] or not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected a raw/prepared version ID of 64 lowercase hex characters")
    return metadata_dir() / "catalog" / kind / f"{version}.json"


def load_version(kind, version, committed=False):
    path = catalog_path(kind, version)
    value = json.loads(path.read_text())
    if value["version_id"] != version or value["kind"] != kind:
        raise ValueError("Catalog identity mismatch")
    expected = identifier(value["identity"])
    if expected != version:
        raise ValueError("Catalog contents do not match the version ID")
    if committed:
        require_committed(path)
    return value


def digest_tree(path):
    path = Path(path)
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = {}
    for item in paths:
        if item.is_symlink():
            raise ValueError(f"Versioned inputs cannot contain symlinks: {item}")
        if item.is_file():
            with item.open("rb") as stream:
                result[str(item.relative_to(path)) if path.is_dir() else "."] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not result:
        raise ValueError(f"Missing or empty versioned input: {path}")
    return result


def dvc(*args, output=False):
    command = [str(Path(sys.executable).with_name("dvc")), *map(str, args)]
    if output:
        return subprocess.check_output(command, cwd=ROOT).decode()
    subprocess.run(command, cwd=ROOT, check=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def provenance(release=False):
    dirty = bool(git("status", "--porcelain"))
    if release and dirty:
        raise ValueError("Release workflows require a clean Git checkout; commit code/configuration and published metadata first")
    source_tree = git("ls-tree", "HEAD", "--", "src", "scripts", "pipelines", "configs", "requirements.txt", "requirements-runtime.lock.txt")
    if release:
        validate_runtime_lock()
    packages = sorted(f"{p.metadata['Name']}=={p.version}" for p in distributions() if p.metadata.get("Name"))
    requirement_file = ROOT / "requirements.txt"
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": dirty,
            "git_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "source_tree_sha256": hashlib.sha256(source_tree.encode()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "requirements_sha256": hashlib.sha256(requirement_file.read_bytes()).hexdigest() if requirement_file.exists() else None,
            "installed_packages": packages, "environment_sha256": identifier(packages),
            "container_image_digest": os.environ.get("OLIST_IMAGE_DIGEST")}


def validate_runtime_lock():
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement
    path = ROOT / "requirements-runtime.lock.txt"
    if not path.exists():
        raise ValueError("Release workflows require requirements-runtime.lock.txt")
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if not re.fullmatch(r"==[^*,]+", str(requirement.specifier)):
            raise ValueError("Runtime lock must contain exact versions")
        try:
            actual = version(requirement.name)
        except PackageNotFoundError as error:
            raise ValueError(f"Runtime dependency missing: {requirement.name}") from error
        if actual not in requirement.specifier:
            raise ValueError(f"Runtime differs from lock: {requirement.name}")


def require_committed(path):
    relative = str(Path(path).relative_to(ROOT))
    try:
        content = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Publish metadata to Git before consumption: {relative}") from error
    if content != Path(path).read_bytes():
        raise ValueError(f"Metadata differs from Git HEAD: {relative}")


@contextmanager
def metadata_lock():
    import fcntl
    path = ROOT / ".dvc/snapshot.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def track(target, pointer):
    """Upload first; replace the live pointer only after successful transfer."""
    target, pointer = Path(target), Path(pointer)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with metadata_lock():
        prior = pointer.read_bytes() if pointer.exists() else None
        if pointer.exists():
            pointer.unlink()
        # Seed a single-file stage outside the Git-ignored data/ tree. DVC add
        # updates this existing output instead of creating data/<name>.dvc.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dvc", prefix="pending-", dir=pointer.parent, delete=False) as stream:
            pending = Path(stream.name)
            yaml.safe_dump({"outs": [{"path": os.path.relpath(target, pointer.parent)}]}, stream)
        try:
            dvc("add", target)
            document = yaml.safe_load(pending.read_text())
            dvc("push", pending, "-r", SETTINGS["versioning"]["remote"])
            pending.replace(pointer)
            return document
        except BaseException:
            if prior is not None:
                pointer.write_bytes(prior)
            raise
        finally:
            pending.unlink(missing_ok=True)


def resources(kind):
    settings = SETTINGS["data"]
    if kind == "raw":
        return {"raw": settings["raw_dir"], "validation": settings["validation_report"]}
    return {"forecasting": settings["forecasting_features"], "warehouse": settings["warehouse"]}


def verify_version(value):
    for name, record in value["resources"].items():
        if digest_tree(ROOT / record["path"]) != value["identity"]["files"][name]:
            raise ValueError(f"Working files do not match {value['kind']} version {value['version_id']}: {name}; restore that Git release and run dvc pull")


def verify_pointers(value, committed=False):
    for record in value["resources"].values():
        pointer = ROOT / record["pointer"]
        if yaml.safe_load(pointer.read_text()) != record["dvc"]:
            raise ValueError("Live DVC pointers do not match the requested version; restore its release metadata first")
        if committed:
            require_committed(pointer)


def publish_checkpoint(kind, record):
    files = {name: digest_tree(ROOT / path) for name, path in resources(kind).items()}
    identity = {"files": files}
    if kind == "raw":
        identity["dataset"] = SETTINGS["data"]["dataset"]
    else:
        identity.update(raw_version=record["raw_version"], parameters=record["parameters"],
                        preparation_git_commit=record["provenance"]["git_commit"])
    version = identifier(identity)
    path = catalog_path(kind, version)
    if path.exists():
        value = load_version(kind, version)
        verify_version(value)
    else:
        value = {"kind": kind, "version_id": version, "identity": identity,
                 "producer_run_id": record["run_id"], "observed_at": datetime.now(timezone.utc).isoformat(),
                 "provenance": record["provenance"], "release": record["release"],
                 "preparation_steps": record.get("receipts", {}), "resources": {}}
    for name, target in resources(kind).items():
        pointer = metadata_dir() / kind / f"{name}.dvc"
        document = track(ROOT / target, pointer)
        value["resources"][name] = {"path": target, "pointer": str(pointer.relative_to(ROOT)), "dvc": document}
    # Catch an unmanaged writer even if upload itself succeeded.
    if files != {name: digest_tree(ROOT / target) for name, target in resources(kind).items()}:
        raise ValueError("Inputs changed during publication; checkpoint not published")
    if kind == "prepared":
        value["raw_version"] = record["raw_version"]
    write_json(path, value)
    return version


def publish_raw(run_id):
    from src.versioning import workspace
    with workspace.locked("prepare:" + run_id, require_owner=True) as record:
        if record["completed"][:2] != workspace.PREPARATION_STAGES[:2]:
            raise ValueError("Download and validation must succeed before raw publication")
        if record["completed"] != workspace.PREPARATION_STAGES[:2] and "raw_version" not in record:
            raise ValueError("Publish raw inputs before preparing features")
        if "raw_version" in record:
            verify_version(load_version("raw", record["raw_version"]))
            return record["raw_version"]
        version = publish_checkpoint("raw", record)
        record["raw_version"] = version
        write_json(workspace.state_path(), record)
        return version


def publish_prepared(run_id):
    from src.versioning import workspace
    with workspace.locked("prepare:" + run_id, require_owner=True) as record:
        if record["completed"] != workspace.PREPARATION_STAGES or "raw_version" not in record:
            raise ValueError("All preparation stages and raw publication must succeed first")
        verify_version(load_version("raw", record["raw_version"]))
        if "prepared_version" not in record:
            record["prepared_version"] = publish_checkpoint("prepared", record)
            write_json(workspace.state_path(), record)
        else:
            verify_version(load_version("prepared", record["prepared_version"]))
        return record["prepared_version"]


def training_parameters(parameters):
    defaults = {**SETTINGS["training"], **SETTINGS["replay"], **SETTINGS["recommendation"]}
    unknown = parameters.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown training parameters: {sorted(unknown)}")
    return {**defaults, **parameters}


def initialize_training(run_id, prepared_version, parameters, release=True):
    """Select a published version; never infer the latest dataset or copy it."""
    from src.versioning import workspace
    parameters = training_parameters(parameters)
    value = load_version("prepared", prepared_version, committed=True)
    raw = load_version("raw", value["raw_version"], committed=True)
    if release and (not value["release"] or not raw["release"]):
        raise ValueError("Release training cannot consume development-only checkpoints")
    record = workspace.reserve("train:" + run_id, "training", run_id,
                               {"prepared_version": prepared_version, **parameters}, release)
    with workspace.locked("train:" + run_id, require_owner=True):
        verify_pointers(value, committed=True)
        verify_version(value)
        feature_metadata = json.loads((ROOT / value["resources"]["forecasting"]["path"] / "metadata.json").read_text())
        for key in ["validation_start", "test_start", "monitoring_start", "monitoring_end"]:
            if key in feature_metadata.get("boundaries", {}) and feature_metadata["boundaries"][key] != parameters[key]:
                raise ValueError(f"Prepared features do not match {key}; publish a matching prepared version")
        path = run_dir(run_id) / "manifest.json"
        if path.exists():
            existing = load_run(run_id)
            if existing["prepared_version"] != prepared_version or existing["resolved_parameters"] != parameters or existing["release"] != release:
                raise ValueError("A run ID cannot be reused with different parameters or versions")
            return existing
        value = {"run_id": run_id, "run_key": run_key(run_id), "prepared_version": prepared_version,
                 "raw_version": raw["version_id"], "prepared_manifest": str(catalog_path("prepared", prepared_version).relative_to(ROOT)),
                 "resolved_parameters": parameters, "release": release, **record["provenance"]}
        write_json(path, value)
        return value


def validate_run_context(manifest):
    current = provenance(False)
    source_paths = ["src", "scripts", "pipelines", "configs", "requirements.txt", "requirements-runtime.lock.txt"]
    if manifest["release"] and git("status", "--porcelain", "--", *source_paths):
        raise ValueError("Release source changed during the run")
    for key in ["source_tree_sha256", "environment_sha256", "python"]:
        if current[key] != manifest[key]:
            raise ValueError(f"Run context changed: {key}; start a new run")


def capture_artifacts(run_id):
    from src.versioning import workspace
    with workspace.locked("train:" + run_id, require_owner=True):
        directory = run_dir(run_id)
        manifest = load_run(run_id)
        validate_run_context(manifest)
        if not (directory / "selection/decisions.json").is_file():
            raise ValueError("Compare models before preserving artifacts")
        pointer = metadata_dir() / "runs" / run_key(run_id) / "artifacts.dvc"
        existing_receipt = pointer.parent / "manifest.json"
        if existing_receipt.exists():
            receipt = json.loads(existing_receipt.read_text())
            if digest_tree(directory) != receipt["artifact_files"]:
                raise ValueError("A preserved training run cannot be modified; create a new run ID")
            if hashlib.sha256(pointer.read_bytes()).hexdigest() != receipt["artifact_pointer_sha256"]:
                raise ValueError("Preserved artifact metadata changed")
            dvc("push", pointer, "-r", SETTINGS["versioning"]["remote"])
            return receipt["model_version"]
        document = track(directory, pointer)
        model_version = identifier({"prepared_version": manifest["prepared_version"], "dvc": document})
        receipt = {**manifest, "model_version": model_version, "artifact_dvc_pointer": str(pointer.relative_to(ROOT)),
                   "artifact_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
                   "artifact_files": digest_tree(directory), "uploaded_remote": SETTINGS["versioning"]["remote"]}
        write_json(pointer.parent / "manifest.json", receipt)
        return model_version


def lineage(run_id, committed=True):
    receipt_path = metadata_dir() / "runs" / run_key(run_id) / "manifest.json"
    receipt = json.loads(receipt_path.read_text())
    if committed:
        require_committed(receipt_path)
        require_committed(ROOT / receipt["artifact_dvc_pointer"])
    prepared = load_version("prepared", receipt["prepared_version"], committed)
    raw = load_version("raw", prepared["raw_version"], committed)
    if raw["version_id"] != receipt["raw_version"]:
        raise ValueError("Broken raw -> prepared -> model lineage")
    return {"raw": raw["version_id"], "prepared": prepared["version_id"], "model": receipt["model_version"], "run_id": run_id}


def backup_mlflow():
    from src.models.common import TRACKING_URI
    if not TRACKING_URI.startswith("sqlite:///"):
        raise ValueError("Configure native database backups for a non-SQLite tracking server")
    source = Path(TRACKING_URI.removeprefix("sqlite:///"))
    if not source.is_file():
        raise FileNotFoundError(source)
    folder = ROOT / SETTINGS["versioning"]["backup_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".db")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection, sqlite3.connect(destination) as backup:
        connection.backup(backup)
    return str(destination)


def main():
    from src.versioning import workspace
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["begin-preparation", "execute-preparation", "raw", "prepared", "begin-training", "artifacts", "backup-mlflow", "release-workspace", "lineage"])
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--prepared-version")
    parser.add_argument("--kind", choices=["prepare", "train"])
    parser.add_argument("--stage", choices=workspace.PREPARATION_STAGES)
    parser.add_argument("--development", action="store_true", help="Record a development run; production activation will reject it")
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    command = argv[boundary + 1:] if boundary < len(argv) else []
    args = parser.parse_args(argv[:boundary])
    if args.action != "backup-mlflow" and not args.pipeline_run_id:
        parser.error("--pipeline-run-id is required")
    parameters = {key: value for key, value in json.loads(os.environ.get("OLIST_PARAMETERS", "{}" )).items()
                  if key not in ["release", "prepared_version"]}
    if args.action == "begin-preparation":
        result = workspace.reserve("prepare:" + args.pipeline_run_id, "preparation", args.pipeline_run_id, parameters, not args.development)
    elif args.action == "execute-preparation":
        if not args.stage or not command:
            parser.error("--stage and -- COMMAND are required")
        result = workspace.execute_preparation(args.pipeline_run_id, args.stage, command)
    elif args.action == "raw": result = publish_raw(args.pipeline_run_id)
    elif args.action == "prepared": result = publish_prepared(args.pipeline_run_id)
    elif args.action == "begin-training":
        if not args.prepared_version: parser.error("--prepared-version is required")
        result = initialize_training(args.pipeline_run_id, args.prepared_version, parameters, not args.development)
    elif args.action == "artifacts": result = capture_artifacts(args.pipeline_run_id)
    elif args.action == "lineage": result = lineage(args.pipeline_run_id)
    elif args.action == "backup-mlflow": result = backup_mlflow()
    else:
        if not args.kind: parser.error("--kind is required")
        result = workspace.release(args.kind + ":" + args.pipeline_run_id)
    print(json.dumps(result, indent=2))
    return 0
