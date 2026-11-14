"""CLI validation and Airflow parameter handling without external storage."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from src.versioning import snapshots


class SnapshotCliTests(unittest.TestCase):
    def test_preparation_command_preserves_argument_boundaries(self):
        command = ["/path with spaces/python", "script.py", "--customer-id", "customer; example"]
        args, actual = snapshots.parse_arguments([
            "execute-preparation", "--pipeline-run-id", "prep-1",
            "--stage", "build_recommendation_features", "--", *command,
        ])
        self.assertEqual(actual, command)
        self.assertEqual(args.pipeline_run_id, "prep-1")
        args, command = snapshots.parse_arguments(["backup-mlflow"])
        self.assertIsNone(args.pipeline_run_id)
        self.assertEqual(command, [])

    def test_missing_options_fail_before_execution(self):
        invalid = [
            ["raw"],
            ["begin-training", "--pipeline-run-id", "train-1"],
            ["release-workspace", "--pipeline-run-id", "prep-1"],
            ["execute-preparation", "--pipeline-run-id", "prep-1"],
            ["execute-preparation", "--pipeline-run-id", "prep-1",
             "--stage", "download_olist_data"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    snapshots.parse_arguments(argv)
                self.assertEqual(error.exception.code, 2)

    def test_training_receives_only_model_parameters_and_development_flag(self):
        argv = ["snapshot.py", "begin-training", "--pipeline-run-id", "train-1",
                "--prepared-version", "a" * 64, "--development"]
        parameters = {"release": False, "prepared_version": "a" * 64, "top_k": 3}
        output = io.StringIO()
        with patch.object(snapshots.sys, "argv", argv), \
             patch.dict(snapshots.os.environ, OLIST_PARAMETERS=json.dumps(parameters)), \
             patch.object(snapshots, "initialize_training", return_value={"run_id": "train-1"}) as train, \
             redirect_stdout(output):
            self.assertEqual(snapshots.main(), 0)
        train.assert_called_once_with("train-1", "a" * 64, {"top_k": 3}, False)
        self.assertEqual(json.loads(output.getvalue()), {"run_id": "train-1"})

    def test_all_actions_keep_their_arguments_and_json_output(self):
        """CLI refactoring must preserve dispatch and printed results."""
        from src.versioning import workspace

        run_id = "run-1"
        prepared = "a" * 64
        command = ["python", "download.py", "--output-dir", "path with spaces"]
        cases = [
            (["begin-preparation"], workspace, "reserve",
             ("prepare:run-1", "preparation", run_id, {"top_k": 3}, True)),
            (["begin-preparation", "--development"], workspace, "reserve",
             ("prepare:run-1", "preparation", run_id, {"top_k": 3}, False)),
            (["execute-preparation", "--stage", "download_olist_data", "--", *command],
             workspace, "execute_preparation", (run_id, "download_olist_data", command)),
            (["raw"], snapshots, "publish_raw", (run_id,)),
            (["prepared"], snapshots, "publish_prepared", (run_id,)),
            (["begin-training", "--prepared-version", prepared], snapshots,
             "initialize_training", (run_id, prepared, {"top_k": 3}, True)),
            (["artifacts"], snapshots, "capture_artifacts", (run_id,)),
            (["lineage"], snapshots, "lineage", (run_id,)),
            (["backup-mlflow"], snapshots, "backup_mlflow", ()),
            (["release-workspace", "--kind", "prepare"], workspace, "release",
             ("prepare:run-1",)),
            (["release-workspace", "--kind", "train"], workspace, "release",
             ("train:run-1",)),
        ]
        for options, module, name, expected in cases:
            with self.subTest(options=options):
                # Insert before the optional separator so the child command is untouched.
                argv = ["snapshot.py", options[0], "--pipeline-run-id", run_id, *options[1:]]
                output = io.StringIO()
                result = None if name == "release" else {"result": name}
                with patch.object(snapshots.sys, "argv", argv), \
                     patch.dict(snapshots.os.environ, OLIST_PARAMETERS='{"top_k":3}'), \
                     patch.object(module, name, return_value=result) as operation, \
                     redirect_stdout(output):
                    self.assertEqual(snapshots.main(), 0)
                operation.assert_called_once_with(*expected)
                self.assertEqual(output.getvalue(), json.dumps(result, indent=2) + "\n")
