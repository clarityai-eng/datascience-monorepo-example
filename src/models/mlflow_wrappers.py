"""Helpers and configuration utilities for MLFlow."""
import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict

import mlflow
import pandas as pd
from pydantic import SecretStr

from config.base import BaseConfig, BaseModelValidator


class MLFlowConfig(BaseModelValidator):
    experiment_name: str
    run_name: str = f'run-{datetime.now().isoformat(timespec="seconds").replace(":", "-")}'
    tracking_uri: SecretStr = SecretStr(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    flavor: str = "sklearn"
    tags: Dict[str, str] | None = None


def mlflow_log_dataframe(df: pd.DataFrame, artifact_name: str, artifact_path: str = "datasets"):
    """Log a pandas dataframe to mlflow as an artifact.

    Args:
        df (pd.DataFrame): dataframe to save as an artifact
        artifact_name (str): name of the artifact in mlflow
        artifact_path (str, optional): path to the artifact in mlflow. Defaults to None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, f"{artifact_name}.parquet")
        df.to_parquet(local_path)
        mlflow.log_artifact(local_path, artifact_path)


def mlflow_bulk_log(
    model_params: Dict[str, Any] | None = None,
    model_metrics: Dict[str, float] | None = None,
    datasets: Dict[str, pd.DataFrame] | None = None,
):
    """Bulk log standard model parameters, metrics and artifacts to mlflow.

    Args:
        model_params (Dict[str, Any]): model parameters to log
        model_metrics (Dict[str, float]): model metrics to log
        datasets (Dict[str, pd.DataFrame]): pandas dataframes to log as artifacts
    """
    if model_params is not None:
        for p, v in model_params.items():
            mlflow.log_param(p, v)
    if model_metrics is not None:
        for m, v in model_metrics.items():
            mlflow.log_metric(m, v)
    if datasets is not None:
        for df_name, df in datasets.items():
            mlflow_log_dataframe(df, df_name, artifact_path="datasets")


def mlflow_log_config(config: BaseConfig, artifact_path: str = "config"):
    """Log a pydantic config model to mlflow.

    Args:
        config (BaseConfig): pydantic config to log
        artifact_path (str, optional): path to the artifact in mlflow. Defaults to "config".
    """
    mlflow.log_dict(json.loads(config.json()), artifact_file=os.path.join(artifact_path, "config.json"))
