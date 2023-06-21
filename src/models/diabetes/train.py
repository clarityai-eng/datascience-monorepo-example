import argparse

import mlflow
import pandas as pd
from loguru import logger
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression

from config import BaseConfig, cli
from config.logging import logger_wraps
from models.mlflow_wrappers import MLFlowConfig, mlflow_bulk_log, mlflow_log_config
from models.train import TrainResults, score_estimator_regression


class Config(BaseConfig):
    model_name: str = "diabetes"

    src_x_train: str
    src_y_train: str

    src_x_test: str
    src_y_test: str

    mlflow: MLFlowConfig = MLFlowConfig(experiment_name="diabetes")


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = LinearRegression()

    @logger_wraps()
    def run(self):
        X_train = pd.read_parquet(self.config.src_x_train)
        y_train = pd.read_parquet(self.config.src_y_train)["target"]
        X_test = pd.read_parquet(self.config.src_x_test)
        y_test = pd.read_parquet(self.config.src_y_test)["target"]

        self.model.fit(X_train, y_train)

        train_metrics = {f"train_{k}": v for k, v in score_estimator_regression(self.model, X_train, y_train).items()}
        test_metrics = {f"test_{k}": v for k, v in score_estimator_regression(self.model, X_test, y_test).items()}

        train_results = TrainResults(
            model=self.model,
            model_signature=infer_signature(X_test, y_test),
            params=self.model.get_params(),
            metrics={**train_metrics, **test_metrics},
            datasets={
                "X_train": X_train,
                "y_train": y_train.to_frame(),
                "X_test": X_test,
                "y_test": y_test.to_frame(),
            },
        )
        return train_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cli.add_model(parser, Config)
    args = parser.parse_args()
    config = Config(**vars(args))
    mlflow.set_tracking_uri(config.mlflow.tracking_uri.get_secret_value())
    mlflow.set_experiment(config.mlflow.experiment_name)

    task = Trainer(config)

    with mlflow.start_run(run_name=config.model_name):
        mlflow_log_config(config)
        results = task.run()
        mlflow_bulk_log(model_metrics=results.metrics, model_params=results.params)
        model_info = mlflow.sklearn.log_model(results.model, "model")
        logger.info(f"Model saved to {model_info.model_uri}")
