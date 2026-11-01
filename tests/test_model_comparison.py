"""Independent MLflow jobs, matching evaluation groups and release decisions."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib

from src.models.comparison import compare_models, compare_model_results, load_model_results, TASKS
from src.models.common import save_run
from src.models.forecasting.forecasting_baseline import BaselineForecaster
from src.models.recommendation.recommendation_baseline import BaselineRecommender
from test_production_models import isolated_tracking, forecast_fixture, warehouse_fixture
from src.models.forecasting import forecasting_lightgbm as forecasting
from src.models.recommendation import recommendation_lightgbm as recommendation
from src.models.forecasting.forecasting_baseline import run_pipeline as run_forecasting_baselines
from src.models.recommendation.recommendation_baseline import run_pipeline as run_recommendation_baselines


class ComparisonTests(unittest.TestCase):
    def log_group(self, root, tracking, group="group", approved=True, mismatch=False, holdout_passed=True):
        for task, config in TASKS.items():
            candidates = config["candidates"]
            for name in [*config["baselines"], *candidates]:
                is_candidate = name in candidates
                if task == "forecasting":
                    scores = {"mae": 1.0 if is_candidate and approved else 2.0,
                              "rmse": 1.0 if is_candidate and approved else 2.0, "rows": 10}
                    model = BaselineForecaster("last_day" if is_candidate else name)
                else:
                    score = 0.8 if is_candidate and approved else 0.5
                    scores = {"precision_at_k": score, "recall_at_k": score, "customers": 10}
                    model = BaselineRecommender("global_popularity" if is_candidate else name)
                report = {"task": task, "model": name, "role": "candidate" if is_candidate else "baseline",
                    "validation": scores, "test": copy.copy(scores), "prediction_varies": True,
                    "evaluation_context": {"evaluation_id": "mismatch" if mismatch and is_candidate and task == "recommendation" else task}}
                if is_candidate and not holdout_passed:
                    report["test"] = ({"mae": 3.0, "rmse": 3.0, "rows": 10} if task == "forecasting"
                                      else {"precision_at_k": 0.1, "recall_at_k": 0.1, "customers": 10})
                save_run(model, report, {}, root / "runs", tracking, config["experiment"], group)

    def test_approval_then_rejection_retains_active_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with isolated_tracking(root) as tracking:
                self.log_group(root, tracking)
                approved = compare_models("group", tracking, root / "models", root / "comparisons")
                for task, report in approved.items():
                    self.assertTrue(report["approved"])
                    self.assertEqual(report["publication"], "published_trained_model")
                    self.assertTrue(Path(report["model_path"]).is_file())
                previous = {p.name: p.read_bytes() for p in (root / "models").iterdir()}
                self.log_group(root, tracking, "rejected", approved=False)
                rejected = compare_models("rejected", tracking, root / "models", root / "comparisons")
                for report in rejected.values():
                    self.assertFalse(report["approved"])
                    self.assertEqual(report["publication"], "retained_existing_model")
                self.assertEqual(previous, {p.name: p.read_bytes() for p in (root / "models").iterdir()})
                # A first deployment of the same rejected candidates gets real baseline artifacts.
                fallback = compare_models("rejected", tracking, root / "fallback", root / "comparisons")
                for report in fallback.values():
                    self.assertEqual(report["publication"], "published_baseline_fallback")
                    model = joblib.load(report["model_path"])
                    self.assertEqual(model.method, report["baseline"])

    def test_missing_or_mismatched_groups_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with isolated_tracking(root) as tracking:
                self.log_group(root, tracking, mismatch=True)
                with self.assertRaisesRegex(ValueError, "Mismatched evaluation"):
                    compare_models("group", tracking, root / "models", root / "comparisons")
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    compare_models("unrelated", tracking, root / "models", root / "comparisons")
                self.assertFalse((root / "models").exists())

    def test_holdout_can_veto_validation_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with isolated_tracking(root) as tracking:
                self.log_group(root, tracking, holdout_passed=False)
                decisions = compare_models("group", tracking, root / "models", root / "comparisons")
                for report in decisions.values():
                    self.assertTrue(report["validation_approved"])
                    self.assertFalse(report["holdout_passed"])
                    self.assertFalse(report["approved"])
                    self.assertEqual(report["publication"], "published_baseline_fallback")

    def test_multiple_candidates_select_validation_winner_without_test_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with isolated_tracking(root) as tracking, \
                 patch.dict(TASKS["forecasting"], candidates=["lightgbm", "challenger"]), \
                 patch.dict(TASKS["recommendation"], candidates=["lightgbm_ranker", "challenger_ranker"]):
                self.log_group(root, tracking)
                for task, name in [("forecasting", "challenger"), ("recommendation", "challenger_ranker")]:
                    result = load_model_results(task, "group")[name]
                    if task == "forecasting":
                        result["validation"] = {"mae": 0.5, "rmse": 0.5, "rows": 10}
                        result["test"] = {"mae": 3.0, "rmse": 3.0, "rows": 10}
                        model = BaselineForecaster("last_day")
                    else:
                        result["validation"] = {"precision_at_k": 0.9, "recall_at_k": 0.9, "customers": 10}
                        result["test"] = {"precision_at_k": 0.1, "recall_at_k": 0.1, "customers": 10}
                        model = BaselineRecommender("global_popularity")
                    save_run(model, result, {}, root / "runs", tracking, TASKS[task]["experiment"], "group")

                decisions = compare_models("group", tracking, root / "models", root / "comparisons")
                for task, name in [("forecasting", "challenger"), ("recommendation", "challenger_ranker")]:
                    decision = decisions[task]
                    self.assertEqual(decision["candidate"], name)
                    self.assertEqual(decision["validation_ranking"][0], name)
                    self.assertEqual(len(decision["candidate_run_ids"]), 2)
                    self.assertTrue(decision["validation_approved"])
                    self.assertFalse(decision["holdout_passed"])
                    self.assertFalse(decision["approved"])
                    self.assertEqual(decision["publication"], "published_baseline_fallback")

                results = load_model_results("forecasting", "group")
                results["challenger"]["validation"]["rmse"] = 4.0
                decision = compare_model_results("forecasting", results)
                self.assertEqual(decision["candidate"], "lightgbm")
                self.assertNotIn("challenger", decision["validation_eligible"])
                results["challenger"]["validation"]["rmse"] = 0.5
                results["challenger"]["prediction_varies"] = False
                self.assertEqual(compare_model_results("forecasting", results)["candidate"], "lightgbm")

                # Every configured candidate must finish before either model can publish.
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                experiment = client.get_experiment_by_name(TASKS["forecasting"]["experiment"])
                for run in client.search_runs([experiment.experiment_id]):
                    if run.data.tags.get("model") == "challenger":
                        client.delete_run(run.info.run_id)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    compare_models("group", tracking, root / "missing_models", root / "comparisons")
                self.assertFalse((root / "missing_models").exists())

    def test_real_sample_jobs_then_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_fixture(root / "features")
            warehouse_fixture(root / "olist.duckdb")
            with isolated_tracking(root) as tracking:
                run_forecasting_baselines(root / "features", root / "forecast_baselines", tracking, pipeline_run_id="sample")
                run_recommendation_baselines(root / "olist.duckdb", root / "rec_baselines", tracking,
                                           top_k=3, candidate_limit=5, pipeline_run_id="sample")
                forecast = forecasting.run_pipeline(root / "features", root / "forecast", tracking,
                    max_train_rows=100, n_estimators=5, pipeline_run_id="sample")
                rec = recommendation.run_pipeline(root / "olist.duckdb", root / "rec", tracking,
                    candidate_limit=5, top_k=3, n_estimators=5, pipeline_run_id="sample")
                self.assertEqual(forecast["role"], "candidate")
                self.assertEqual(rec["role"], "candidate")
                self.assertEqual(forecast["pipeline_run_id"], "sample")
                self.assertEqual(rec["pipeline_run_id"], "sample")
                self.assertFalse((root / "models").exists())
                decisions = compare_models("sample", tracking, root / "models", root / "comparisons")
                self.assertEqual(decisions["forecasting"]["candidate_run_id"], forecast["mlflow_run_id"])
                self.assertEqual(decisions["recommendation"]["candidate_run_id"], rec["mlflow_run_id"])
                # The weekly fixture is perfectly forecast by a seasonal baseline.
                self.assertFalse(decisions["forecasting"]["approved"])
                self.assertEqual(decisions["forecasting"]["publication"], "published_baseline_fallback")
                for task, decision in decisions.items():
                    active = joblib.load(decision["model_path"])
                    self.assertTrue(active.serving_metadata["approved_for_inference"])
                    expected = decision["candidate_run_id"] if decision["approved"] else decision["baseline_run_id"]
                    self.assertEqual(active.serving_metadata["model_run_id"], expected)


if __name__ == "__main__":
    unittest.main()
