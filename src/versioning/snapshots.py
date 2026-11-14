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
SOURCE_PATHS = (
    "src", "scripts", "pipelines", "configs",
    "requirements.txt", "requirements-runtime.lock.txt",
)
SPLIT_DATES = ("validation_start", "test_start", "monitoring_start", "monitoring_end")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace a manifest atomically so readers never see a partial JSON file."""
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
    result = read_json(run_dir(run_id) / "manifest.json")
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
    value = read_json(path)
    if value["version_id"] != version or value["kind"] != kind:
        raise ValueError("Catalog identity mismatch")
    if identifier(value["identity"]) != version:
        raise ValueError("Catalog contents do not match the version ID")
    if committed:
        require_committed(path)
    return value


def digest_tree(path):
    path = Path(path)
    is_directory = path.is_dir()
    paths = sorted(path.rglob("*")) if is_directory else [path]
    result = {}
    for item in paths:
        if item.is_symlink():
            raise ValueError(f"Versioned inputs cannot contain symlinks: {item}")
        if item.is_file():
            with item.open("rb") as stream:
                result[str(item.relative_to(path)) if is_directory else "."] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not result:
        raise ValueError(f"Missing or empty versioned input: {path}")
    return result


def dvc(*args):
    command = [str(Path(sys.executable).with_name("dvc")), *map(str, args)]
    subprocess.run(command, cwd=ROOT, check=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def provenance(release=False):
    dirty = bool(git("status", "--porcelain"))
    if release and dirty:
        raise ValueError("Release workflows require a clean Git checkout; commit code/configuration and published metadata first")
    source_tree = git("ls-tree", "HEAD", "--", *SOURCE_PATHS)
    if release:
        validate_runtime_lock()
    packages = sorted(f"{p.metadata['Name']}=={p.version}" for p in distributions() if p.metadata.get("Name"))
    requirement_file = ROOT / "requirements.txt"
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "source_tree_sha256": hashlib.sha256(source_tree.encode()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements_sha256": (
            hashlib.sha256(requirement_file.read_bytes()).hexdigest()
            if requirement_file.exists() else None
        ),
        "installed_packages": packages,
        "environment_sha256": identifier(packages),
        "container_image_digest": os.environ.get("OLIST_IMAGE_DIGEST"),
    }


def validate_runtime_lock():
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement
    path = ROOT / "requirements-runtime.lock.txt"
    if not path.exists():
        raise ValueError("Release workflows require requirements-runtime.lock.txt")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
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


def resource_hashes(targets):
    """Hash each named workspace resource for publication and verification."""
    return {name: digest_tree(ROOT / path) for name, path in targets.items()}


def checkpoint_identity(kind, files, record):
    """Describe the inputs that determine a raw or prepared version ID."""
    identity = {"files": files}
    if kind == "raw":
        identity["dataset"] = SETTINGS["data"]["dataset"]
    else:
        identity.update(
            raw_version=record["raw_version"],
            parameters=record["parameters"],
            preparation_git_commit=record["provenance"]["git_commit"],
        )
    return identity


def publish_checkpoint(kind, record):
    """Hash and upload the working files, then publish their catalog manifest."""
    targets = resources(kind)
    files = resource_hashes(targets)
    identity = checkpoint_identity(kind, files, record)
    version = identifier(identity)
    path = catalog_path(kind, version)
    if path.exists():
        value = load_version(kind, version)
        verify_version(value)
    else:
        value = {
            "kind": kind,
            "version_id": version,
            "identity": identity,
            "producer_run_id": record["run_id"],
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "provenance": record["provenance"],
            "release": record["release"],
            "preparation_steps": record.get("receipts", {}),
            "resources": {},
        }
    for name, target in targets.items():
        pointer = metadata_dir() / kind / f"{name}.dvc"
        document = track(ROOT / target, pointer)
        value["resources"][name] = {
            "path": target,
            "pointer": str(pointer.relative_to(ROOT)),
            "dvc": document,
        }
    # Catch an unmanaged writer even if upload itself succeeded.
    if files != resource_hashes(targets):
        raise ValueError("Inputs changed during publication; checkpoint not published")
    if kind == "prepared":
        value["raw_version"] = record["raw_version"]
    write_json(path, value)
    return version


def publish_raw(run_id):
    """Publish validated raw inputs, or verify the checkpoint on a retry."""
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
    """Publish completed preparation outputs linked to their verified raw inputs."""
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


def verify_training_inputs(prepared, parameters):
    """Require committed pointers, matching files and matching split dates."""
    verify_pointers(prepared, committed=True)
    verify_version(prepared)
    features = ROOT / prepared["resources"]["forecasting"]["path"]
    boundaries = read_json(features / "metadata.json").get("boundaries", {})
    for key in SPLIT_DATES:
        if key in boundaries and boundaries[key] != parameters[key]:
            raise ValueError(
                f"Prepared features do not match {key}; publish a matching prepared version"
            )


def initialize_training(run_id, prepared_version, parameters, release=True):
    """Select a published version; never infer the latest dataset or copy it."""
    from src.versioning import workspace
    parameters = training_parameters(parameters)
    prepared = load_version("prepared", prepared_version, committed=True)
    raw = load_version("raw", prepared["raw_version"], committed=True)
    if release and (not prepared["release"] or not raw["release"]):
        raise ValueError("Release training cannot consume development-only checkpoints")
    record = workspace.reserve(
        "train:" + run_id, "training", run_id,
        {"prepared_version": prepared_version, **parameters}, release,
    )
    with workspace.locked("train:" + run_id, require_owner=True):
        verify_training_inputs(prepared, parameters)
        path = run_dir(run_id) / "manifest.json"
        if path.exists():
            existing = load_run(run_id)
            expected = {
                "prepared_version": prepared_version,
                "resolved_parameters": parameters,
                "release": release,
            }
            if any(existing[key] != expected_value for key, expected_value in expected.items()):
                raise ValueError("A run ID cannot be reused with different parameters or versions")
            return existing
        manifest = {
            "run_id": run_id,
            "run_key": run_key(run_id),
            "prepared_version": prepared_version,
            "raw_version": raw["version_id"],
            "prepared_manifest": str(catalog_path("prepared", prepared_version).relative_to(ROOT)),
            "resolved_parameters": parameters,
            "release": release,
            **record["provenance"],
        }
        write_json(path, manifest)
        return manifest


def validate_run_context(manifest):
    current = provenance(False)
    if manifest["release"] and git("status", "--porcelain", "--", *SOURCE_PATHS):
        raise ValueError("Release source changed during the run")
    for key in ["source_tree_sha256", "environment_sha256", "python"]:
        if current[key] != manifest[key]:
            raise ValueError(f"Run context changed: {key}; start a new run")


def capture_artifacts(run_id):
    """Upload an evaluated run once; reject changes to an already preserved run."""
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
            receipt = read_json(existing_receipt)
            if digest_tree(directory) != receipt["artifact_files"]:
                raise ValueError("A preserved training run cannot be modified; create a new run ID")
            if hashlib.sha256(pointer.read_bytes()).hexdigest() != receipt["artifact_pointer_sha256"]:
                raise ValueError("Preserved artifact metadata changed")
            dvc("push", pointer, "-r", SETTINGS["versioning"]["remote"])
            return receipt["model_version"]
        document = track(directory, pointer)
        model_version = identifier({"prepared_version": manifest["prepared_version"], "dvc": document})
        receipt = {
            **manifest,
            "model_version": model_version,
            "artifact_dvc_pointer": str(pointer.relative_to(ROOT)),
            "artifact_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
            "artifact_files": digest_tree(directory),
            "uploaded_remote": SETTINGS["versioning"]["remote"],
        }
        write_json(pointer.parent / "manifest.json", receipt)
        return model_version


def lineage(run_id, committed=True):
    """Resolve and validate the model -> prepared -> raw metadata chain."""
    receipt_path = metadata_dir() / "runs" / run_key(run_id) / "manifest.json"
    receipt = read_json(receipt_path)
    if committed:
        require_committed(receipt_path)
        require_committed(ROOT / receipt["artifact_dvc_pointer"])
    prepared = load_version("prepared", receipt["prepared_version"], committed)
    raw = load_version("raw", prepared["raw_version"], committed)
    if raw["version_id"] != receipt["raw_version"]:
        raise ValueError("Broken raw -> prepared -> model lineage")
    return {
        "raw": raw["version_id"],
        "prepared": prepared["version_id"],
        "model": receipt["model_version"],
        "run_id": run_id,
    }


def backup_mlflow():
    """Create a consistent, timestamped backup of the SQLite tracking database."""
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


def parse_arguments(argv):
    """Validate CLI options and separate the preparation command after ``--``."""
    from src.versioning import workspace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[
        "begin-preparation", "execute-preparation", "raw", "prepared",
        "begin-training", "artifacts", "backup-mlflow", "release-workspace", "lineage",
    ])
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--prepared-version")
    parser.add_argument("--kind", choices=["prepare", "train"])
    parser.add_argument("--stage", choices=workspace.PREPARATION_STAGES)
    parser.add_argument("--development", action="store_true",
                        help="Record a development run; production activation will reject it")
    boundary = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:boundary])
    command = argv[boundary + 1:]
    if args.action != "backup-mlflow" and not args.pipeline_run_id:
        parser.error("--pipeline-run-id is required")
    required_option = {
        "execute-preparation": "stage",
        "begin-training": "prepared_version",
        "release-workspace": "kind",
    }.get(args.action)
    if required_option and not getattr(args, required_option):
        parser.error(f"--{required_option.replace('_', '-')} is required")
    if args.action == "execute-preparation" and not command:
        parser.error("-- COMMAND is required")
    return args, command


def environment_parameters():
    """Read workflow parameters, excluding options handled by the CLI."""
    parameters = json.loads(os.environ.get("OLIST_PARAMETERS", "{}"))
    return {
        key: value for key, value in parameters.items()
        if key not in ("release", "prepared_version")
    }


def execute_action(args, command):
    """Dispatch a validated CLI action and return its JSON-serializable result."""
    from src.versioning import workspace

    run_id = args.pipeline_run_id
    parameters = environment_parameters()
    release = not args.development
    match args.action:
        case "begin-preparation":
            return workspace.reserve("prepare:" + run_id, "preparation", run_id, parameters, release)
        case "execute-preparation":
            return workspace.execute_preparation(run_id, args.stage, command)
        case "raw":
            return publish_raw(run_id)
        case "prepared":
            return publish_prepared(run_id)
        case "begin-training":
            return initialize_training(run_id, args.prepared_version, parameters, release)
        case "artifacts":
            return capture_artifacts(run_id)
        case "lineage":
            return lineage(run_id)
        case "backup-mlflow":
            return backup_mlflow()
        case "release-workspace":
            return workspace.release(args.kind + ":" + run_id)


def main():
    args, command = parse_arguments(sys.argv[1:])
    result = execute_action(args, command)
    print(json.dumps(result, indent=2))
    return 0
