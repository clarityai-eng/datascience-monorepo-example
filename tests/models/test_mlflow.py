import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from config.base import BaseConfig
from models.mlflow_wrappers import MLFlowConfig, mlflow_bulk_log, mlflow_log_config, mlflow_log_dataframe


@pytest.fixture(scope="session")
def mlruns(tmp_path_factory: pytest.TempPathFactory) -> Path:  # type: ignore
    """Fixture for local mlflow artifacts file storage."""
    mlruns_path = tmp_path_factory.mktemp("mlruns")
    mlflow.set_tracking_uri(str(mlruns_path))
    mlflow.set_experiment("test")
    yield mlruns_path
    mlflow.end_run()


def test_mlflow_log_dataframe(mlruns: Path):
    with mlflow.start_run() as run:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mlflow_log_dataframe(df, "test", artifact_path="datasets")
        assert (mlruns / run.info.experiment_id / run.info.run_id / "artifacts" / "datasets" / "test.parquet").exists()

    with mlflow.start_run() as run:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mlflow_log_dataframe(df, "train", artifact_path="various/datasets")
        assert (
            mlruns / run.info.experiment_id / run.info.run_id / "artifacts" / "various/datasets" / "train.parquet"
        ).exists()


def test_mlflow_bulk_log(mlruns: Path):
    model_params = {"alpha": "0.1", "l1_ratio": "2.0"}  # NOTE: params will be stored as strings
    model_metrics = {"rmse": 0.5, "r2": 0.8}
    with mlflow.start_run() as run:
        mlflow_bulk_log(model_params=model_params, model_metrics=model_metrics)
        assert mlflow.get_run(run.info.run_id).data.params == model_params
        assert mlflow.get_run(run.info.run_id).data.metrics == model_metrics


def test_mlflow_log_config(mlruns: Path):
    class TestConfig(BaseConfig):
        foo: str
        mlflow: MLFlowConfig

    config = TestConfig(foo="not_default", mlflow=MLFlowConfig(experiment_name="experiment_test"))

    with mlflow.start_run() as run:
        artifact_path = "various/config"
        mlflow_log_config(config, artifact_path=artifact_path)
        logged_artifact_json = (
            mlruns / run.info.experiment_id / run.info.run_id / "artifacts" / artifact_path / "config.json"
        )
        assert logged_artifact_json.exists()
        logged_config = json.loads(logged_artifact_json.read_text())
        assert logged_config["foo"] == config.foo
        assert logged_config["mlflow"]["experiment_name"] == config.mlflow.experiment_name
        assert logged_config["mlflow"]["flavor"] == config.mlflow.flavor
        assert logged_config["mlflow"]["run_name"] == config.mlflow.run_name
