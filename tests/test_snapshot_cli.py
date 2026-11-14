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
