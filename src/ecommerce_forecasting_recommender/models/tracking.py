from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)


@contextmanager
def optional_mlflow_run(experiment_name: str, run_name: str) -> Iterator[object | None]:
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError:
        LOGGER.warning("MLflow not installed, continuing without tracking")
        yield None
        return

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except MlflowException as exc:  # pragma: no cover - best effort integration
        LOGGER.warning("MLflow unavailable, continuing without tracking: %s", exc)
        yield None


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError:
        return
    try:
        mlflow.log_metrics(metrics)
    except MlflowException:
        return


def log_model_artifact(path: str, artifact_path: str) -> None:
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError:
        return
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except MlflowException:
        return
