"""Checkpoint lineage, transaction failures, concurrency, release gates and recovery."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.common.config import SETTINGS
from src.inference.batch import run_pipeline as infer
from src.models import common, jobs, publication
from src.models.comparison import compare_models
from src.versioning import snapshots, workspace
from test_production_models import forecast_fixture, warehouse_fixture


class SnapshotTests(unittest.TestCase):
    def command(self, *args):
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f"Command failed ({args[0]}): {result.stdout} {result.stderr}")
        return result.stdout

    def commit(self):
        self.command("git", "add", ".")
        self.command("git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "checkpoint")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.command("git", "init", "-q")
        (self.root / ".gitignore").write_text("/data/\n/remote/\n/models/\n/mlruns/\n/mlflow.db*\n/backups/\n/.fixture/\n")
        (self.root / "requirements.txt").write_text("fixture")
        self.dvc = str(Path(sys.executable).with_name("dvc"))
        self.command(self.dvc, "init", "-q")
        with (self.root / ".dvc/.gitignore").open("a") as stream:
            stream.write("\n/workspace.*\n/snapshot.lock\n")
        remote = str(self.root / "remote")
        if os.environ.get("OLIST_TEST_RUSTFS") == "1":
            remote = "s3://olist-dvc/verification/" + self.root.name
            source = Path(__file__).resolve().parents[1] / ".dvc/config.local"
            private = self.root / ".dvc/config.local"
            private.write_text(source.read_text().replace('remote "rustfs"', 'remote "local"'))
            private.chmod(0o600)
        self.command(self.dvc, "remote", "add", "-d", "local", remote)
        self.command(self.dvc, "config", "cache.type", "reflink,copy")
        shutil.copy2(Path(__file__).resolve().parents[1] / "requirements-runtime.lock.txt", self.root / "requirements-runtime.lock.txt")
        self.commit()
        self.tracking = f"sqlite:///{self.root / 'mlflow.db'}"
        for obj, attribute, value in [(snapshots, "ROOT", self.root), (jobs, "ROOT", self.root),
                                      (publication, "ROOT", self.root), (common, "ROOT", self.root),
                                      (common, "TRACKING_URI", self.tracking)]:
            patcher = patch.object(obj, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(SETTINGS["versioning"], remote="local")
        patcher.start()
        self.addCleanup(patcher.stop)
        _, metadata = forecast_fixture(self.root / ".fixture/features")
        warehouse_fixture(self.root / ".fixture/warehouse.duckdb")
        self.parameters = {"max_train_rows": 100, "forecast_estimators": 5, "ranking_estimators": 5,
                           "candidate_limit": 5, "top_k": 3, **metadata["boundaries"]}

    def execute(self, stage):
        # Small subprocesses stand in for the expensive real preparation jobs;
        # output promotion and receipt tracking are the real implementation.
        code = '''
import json, shutil, sys
from pathlib import Path
stage = sys.argv[1]
flags = dict(zip(sys.argv[2::2], sys.argv[3::2]))
if stage == "validate_olist_data":
    print(json.dumps({"status": "valid", "rows": 1}))
else:
    output = Path(flags["--output-dir"])
    if stage == "build_forecasting_features":
        shutil.copytree(".fixture/features", output)
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "input.csv").write_text("id\\n1\\n")
    if stage == "prepare_curated_data":
        shutil.copy2(".fixture/warehouse.duckdb", flags["--warehouse"])
'''
        workspace.execute_preparation("prep", stage, [sys.executable, "-c", code, stage])

    def prepare(self, release=True):
        workspace.reserve("prepare:prep", "preparation", "prep", {}, release)
        for stage in workspace.PREPARATION_STAGES:
            self.execute(stage)
            if stage == "validate_olist_data":
                raw = snapshots.publish_raw("prep")
        prepared = snapshots.publish_prepared("prep")
        self.assertEqual(snapshots.publish_prepared("prep"), prepared)
        workspace.release("prepare:prep")
        return raw, prepared

    def train(self, prepared, release=True):
        snapshots.initialize_training("example", prepared, self.parameters, release)
        for task, model in [("forecasting", "baselines"), ("recommendation", "baselines"),
                            ("forecasting", "lightgbm"), ("recommendation", "lightgbm_ranker")]:
            jobs.run_job("example", task, model)
        return compare_models("example", self.tracking)

    def test_lineage_restore_from_empty_cache_and_infer_without_mlflow(self):
        raw, prepared = self.prepare()
        with self.assertRaisesRegex(ValueError, "Publish metadata"):
            snapshots.initialize_training("example", prepared, self.parameters)
        self.commit()
        decisions = self.train(prepared)
        manifest = snapshots.load_run("example")
        self.assertEqual(manifest["raw_version"], raw)
        self.assertFalse((self.root / "data/snapshots").exists(), "No custom full-copy snapshot tree")
        self.assertFalse((self.root / "models").exists(), "Comparison must not activate models")
        from mlflow.tracking import MlflowClient
        tags = MlflowClient(tracking_uri=self.tracking).get_run(decisions["forecasting"]["candidate_run_id"]).data.tags
        self.assertEqual(tags["prepared_version"], prepared)
        self.assertEqual(tags["raw_version"], raw)
        with self.assertRaisesRegex(ValueError, "parameters"):
            snapshots.initialize_training("example", prepared, {**self.parameters, "seed": 999})
        model_version = snapshots.capture_artifacts("example")
        self.assertEqual(snapshots.capture_artifacts("example"), model_version)
        backup = snapshots.backup_mlflow()
        with sqlite3.connect(backup) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        workspace.release("train:example")
        with self.assertRaisesRegex(ValueError, "clean Git|Publish metadata"):
            publication.activate_run("example")
        self.commit()
        self.assertEqual(snapshots.lineage("example"), {"raw": raw, "prepared": prepared, "model": model_version, "run_id": "example"})
        self.command(self.dvc, "cache", "dir", str(self.root / ".fixture/fresh-cache"), "--local")
        shutil.rmtree(self.root / "data")
        self.command(self.dvc, "pull")
        # Delete live MLflow history: restored bundles and manifests are sufficient.
        (self.root / "mlflow.db").unlink(missing_ok=True)
        with patch("mlflow.search_runs", side_effect=AssertionError("Restoration must not need MLflow")):
            published = publication.activate_run("example")
            path = self.root / SETTINGS["data"]["forecasting_features"] / "test.parquet"
            date = pd.read_parquet(path).purchase_date.min()
            predictions = infer("forecasting", date, path, self.root / "models", self.root / "data/predictions")
        self.assertGreater(predictions["rows"], 0)
        self.assertEqual(published["forecasting"]["active_selection"]["prepared_version"], prepared)
        before = {p.name: p.read_bytes() for p in (self.root / "models").iterdir()}
        publication.activate_run("example")
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "models").iterdir()})
        (snapshots.run_dir("example") / "selection/models/forecast_model.pkl").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed"):
            publication.activate_run("example")
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "models").iterdir()})

    def test_cross_workflow_lock_stage_failure_and_missing_lineage(self):
        workspace.reserve("prepare:prep", "preparation", "prep", {}, True)
        with self.assertRaisesRegex(RuntimeError, "reserved"):
            workspace.reserve("train:other", "training", "other", {}, True)
        with self.assertRaisesRegex(RuntimeError, "reserved"):
            workspace.release("train:other")
        with self.assertRaisesRegex(ValueError, "order|validated raw"):
            self.execute("clean_olist_data")
        self.execute("download_olist_data")
        original = (self.root / "data/raw/input.csv").read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            workspace.execute_preparation("prep", "download_olist_data", [sys.executable, "-c", "raise SystemExit(1)"])
        self.assertEqual((self.root / "data/raw/input.csv").read_bytes(), original)
        self.assertEqual(workspace.state()["completed"], [])
        with self.assertRaisesRegex(ValueError, "All preparation"):
            snapshots.publish_prepared("prep")
        workspace.release("prepare:prep")
        self.assertIsNone(workspace.state())

    def test_failed_upload_does_not_publish_a_version(self):
        workspace.reserve("prepare:prep", "preparation", "prep", {}, True)
        self.execute("download_olist_data")
        self.execute("validate_olist_data")
        original = snapshots.dvc
        def fail_push(*args, **kwargs):
            if args[0] == "push":
                raise RuntimeError("upload failed")
            return original(*args, **kwargs)
        with patch.object(snapshots, "dvc", side_effect=fail_push):
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                snapshots.publish_raw("prep")
        self.assertNotIn("raw_version", workspace.state())
        self.assertFalse(list((self.root / "snapshots/catalog/raw").glob("*.json")))
        self.assertFalse((self.root / "snapshots/raw/raw.dvc").exists())
        # A retry completes the same reservation after storage recovers.
        self.assertEqual(len(snapshots.publish_raw("prep")), 64)

    def test_raw_parent_and_prepared_content_changes_are_detected(self):
        workspace.reserve("prepare:prep", "preparation", "prep", {}, True)
        for stage in workspace.PREPARATION_STAGES:
            self.execute(stage)
            if stage == "validate_olist_data":
                snapshots.publish_raw("prep")
        (self.root / "data/raw/input.csv").write_text("changed raw files")
        with self.assertRaisesRegex(ValueError, "do not match raw"):
            snapshots.publish_prepared("prep")
        workspace.release("prepare:prep")

    def test_process_cannot_enter_another_workflows_workspace(self):
        workspace.reserve("prepare:prep", "preparation", "prep", {}, True)
        code = "from pathlib import Path; from src.versioning import snapshots,workspace; snapshots.ROOT=Path(__import__('sys').argv[1]); workspace.reserve('train:other','training','other',{},False)"
        result = subprocess.run([sys.executable, "-c", code, str(self.root)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reserved", result.stderr)
        self.assertEqual(workspace.state()["owner"], "prepare:prep")

    def test_dirty_release_and_development_activation_are_rejected(self):
        (self.root / "uncommitted.py").write_text("pass")
        with self.assertRaisesRegex(ValueError, "clean Git"):
            workspace.reserve("prepare:prep", "preparation", "prep", {}, True)
        raw, prepared = self.prepare(release=False)
        self.commit()
        with self.assertRaisesRegex(ValueError, "development-only"):
            snapshots.initialize_training("example", prepared, self.parameters, release=True)
        self.train(prepared, release=False)
        snapshots.capture_artifacts("example")
        workspace.release("train:example")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Development"):
            publication.activate_run("example")
        self.assertFalse((self.root / "models").exists())


if __name__ == "__main__":
    unittest.main()
