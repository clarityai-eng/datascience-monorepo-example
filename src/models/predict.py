import argparse
import warnings
from datetime import datetime
from typing import List, Literal, Mapping

import mlflow
import pandas as pd
import sklearn
from joblib import parallel_backend
from loguru import logger
from mlflow.types.schema import Schema as MLFlowInputSchema
from numpy.typing import ArrayLike
from pydantic import constr
from tqdm.auto import tqdm

from config import BaseConfig, cli
from config.logging import log_time, logger_wraps
from datasets import options
from models.mlflow_wrappers import MLFlowConfig


def load_mlflow_model_with_flavour(
    model_uri: str, flavour: Literal["sklearn", "pyfunc"]
) -> mlflow.pyfunc.PyFuncModel | sklearn.base.BaseEstimator:
    match flavour:
        case "sklearn":
            return mlflow.sklearn.load_model(model_uri)
        case "pyfunc":
            return mlflow.pyfunc.load_model(model_uri)
        case _:
            raise ValueError(f"Flavour {flavour} not supported")


class Config(BaseConfig):
    src_features: str
    src_model: constr(regex=options.MODEL_URI_REGEX)

    flavour: str = "sklearn"
    parallel_backend: str = "threading"
    n_jobs: int = -1
    batch_predictions: bool = False
    batch_size: int = 10000
    progress_bar: bool = True

    dst_y_hat: str

    mlflow: MLFlowConfig = MLFlowConfig(experiment_name="predict")

    execution_date: str = datetime.now().isoformat(timespec="seconds") + "Z"


class Predictor:
    def __init__(self, config: Config):
        self.config = config

    def run(self, model: mlflow.pyfunc.PyFuncModel | sklearn.base.BaseEstimator, X: pd.DataFrame) -> pd.DataFrame:
        if self.config.batch_predictions:
            y_hat = self.predict_in_batches(model, X, self.config.batch_size)
        else:
            y_hat = self.predict(model, X)
        df_y_hat = y_hat.to_frame(name="y_hat")
        return df_y_hat

    @logger_wraps()
    @log_time(level="INFO", unit="seconds")
    def predict_in_batches(
        self,
        model: mlflow.pyfunc.PyFuncModel | sklearn.base.BaseEstimator,
        X: pd.DataFrame,
        batch_size: int = 10000,
    ) -> pd.Series:
        y_hat_batches: List[pd.Series] = []
        for i in tqdm(range(0, len(X), batch_size), disable=not self.config.progress_bar):
            batch_begin = i
            batch_end = i + batch_size
            X_batch = X.iloc[batch_begin:batch_end]
            y_hat_batches.append(self.predict(model, X_batch))
        y_hat = pd.concat(y_hat_batches, ignore_index=True)
        return y_hat

    @logger_wraps(level="DEBUG")
    @log_time(level="DEBUG", unit="seconds")
    def predict(self, model: mlflow.pyfunc.PyFuncModel | sklearn.base.BaseEstimator, X: pd.DataFrame) -> pd.Series:
        y_hat: pd.Series
        match model:
            case mlflow.pyfunc.PyFuncModel():
                y_hat = self.predict_mlflow_pyfunc_model(model, X)
            case sklearn.base.BaseEstimator():
                y_hat = self.predict_sklearn_model(model, X)
            case _:
                raise ValueError(f"Unknown model type {type(model)}")
        y_hat = pd.Series(y_hat, name="y_hat", index=X.index)
        return y_hat

    @logger_wraps(level="DEBUG")
    def predict_sklearn_model(self, model: sklearn.base.BaseEstimator, X: pd.DataFrame) -> ArrayLike:
        with parallel_backend(self.config.parallel_backend, n_jobs=self.config.n_jobs):
            return model.predict(X)

    @logger_wraps(level="DEBUG")
    def predict_mlflow_pyfunc_model(self, model: mlflow.pyfunc.PyFuncModel, X: pd.DataFrame) -> ArrayLike:
        input_schema = model.metadata.get_input_schema()
        dtypes = self.get_pandas_dtypes_from_input_schema(input_schema)
        X = X.astype(dtypes).replace({pd.NA: None})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            y_hat = model.predict(X)
        return y_hat

    @staticmethod
    def get_pandas_dtypes_from_input_schema(input_schema: MLFlowInputSchema) -> Mapping[str, str]:
        if input_schema is None:
            logger.warning("Input schema is None. Using default dtypes")
            return {}
        return {name: type_ for name, type_ in zip(input_schema.input_names(), input_schema.pandas_types())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cli.add_model(parser, Config)
    args = parser.parse_args()
    config = Config(**vars(args))
    logger.info(config)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri.get_secret_value())

    predictor = Predictor(config)

    logger.info(f"Reading data from {config.src_features}")
    features = pd.read_parquet(config.src_features)
    logger.info(f"Loading model from {config.src_model}")
    model = load_mlflow_model_with_flavour(model_uri=config.src_model, flavour=config.flavour)

    df_y = predictor.run(model, features)
    logger.info(f"Writing y_hat to {config.dst_y_hat}")
    df_y.y_hat.to_frame().to_parquet(config.dst_y_hat)
