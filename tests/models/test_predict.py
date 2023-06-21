import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from models.predict import Config, Predictor, load_mlflow_model_with_flavour


def test_load_mlflow_model_with_flavour(mlruns: str, fitted_model: Pipeline):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(fitted_model, "model")

    model_sklearn = load_mlflow_model_with_flavour(model_info.model_uri, "sklearn")
    assert isinstance(model_sklearn, BaseEstimator)
    model_pyfunc = load_mlflow_model_with_flavour(model_info.model_uri, "pyfunc")
    assert isinstance(model_pyfunc, mlflow.pyfunc.PyFuncModel)


class TestPredict:
    @pytest.fixture
    def config(self, mlruns) -> Config:
        return Config(
            src_features="dummy",
            src_model="models:/test-model/1",
            dst_y_hat="dummy_y_hat",
            mlflow={
                "tracking_uri": mlruns,
                "experiment_name": "test",
                "run_name": "test-1",
            },
        )

    def test_predict_sklearn(self, X: pd.DataFrame, fitted_model: Pipeline, y_hat: pd.Series, config: Config):
        predictor = Predictor(config)

        data = X.copy()
        df_y_hat = predictor.run(fitted_model, data)

        assert len(df_y_hat) == len(data)
        assert (~df_y_hat["y_hat"].isna()).all()

        y_hat_expected = y_hat
        y_hat_result = df_y_hat["y_hat"]
        assert np.isclose(y_hat_expected, y_hat_result).all()

    def test_predict_mlflow(self, X: pd.DataFrame, fitted_model: Pipeline, y_hat: pd.Series, config: Config):
        mlflow.set_experiment(config.mlflow.experiment_name)
        with mlflow.start_run():
            mlflow.sklearn.log_model(fitted_model, "model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_mlflow = mlflow.pyfunc.load_model(model_uri)
        config = config.copy(update={"src_model": model_uri})

        predictor = Predictor(config)

        data = X.copy()
        df_y_hat = predictor.run(model_mlflow, data)

        assert len(df_y_hat) == len(data)
        assert (~df_y_hat["y_hat"].isna()).all()

        y_hat_expected = y_hat
        y_hat_result = df_y_hat["y_hat"]
        assert np.isclose(y_hat_expected, y_hat_result).all()

    def test_predict_batches(self, X: pd.DataFrame, fitted_model: Pipeline, y_hat: pd.Series, config: Config):
        config_batches = config.copy(update={"batch_size": 10, "batch_predictions": True})
        predictor = Predictor(config_batches)

        data = X.copy()
        df_y_hat = predictor.run(fitted_model, data)

        assert len(df_y_hat) == len(data)
        assert (~df_y_hat["y_hat"].isna()).all()

        y_hat_expected = y_hat
        y_hat_result = df_y_hat["y_hat"]
        assert np.isclose(y_hat_expected, y_hat_result).all()

    def test_predict_unknown_model(self, X: pd.DataFrame, config: Config):
        data = X.copy()
        predictor = Predictor(config)

        with pytest.raises(ValueError):
            predictor.run("dummy", data)
