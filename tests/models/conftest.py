from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

np.random.seed(42)


@pytest.fixture(scope="class")
def mlruns() -> str:  # type: ignore
    """MLFlow backend storage in an in memory sqlite database"""
    mlruns_path = "sqlite://"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("test")
    yield mlruns_path
    mlflow.end_run()


@pytest.fixture(scope="class")
def mlruns_file(tmp_path_factory: pytest.TempPathFactory) -> Path:  # type: ignore
    """MLFlow backend storage in an temporal sqlite file database"""
    mlruns_db = tmp_path_factory.mktemp("mlruns").joinpath("mlruns.db")
    mlruns_db.touch()
    mlruns_path = f"sqlite:///{mlruns_db}"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("test")
    yield mlruns_path
    mlflow.end_run()


@pytest.fixture(scope="session")
def data_company_regression() -> pd.DataFrame:
    X, y = datasets.make_regression(n_samples=500, n_features=4, random_state=42)
    data = pd.DataFrame(X, columns=["revenue", "feature_1", "feature_2", "feature_3"])
    data["id"] = list(range(len(data)))
    data["target"] = y
    return data


@pytest.fixture(scope="session")
def X(data_company_regression: pd.DataFrame) -> pd.DataFrame:
    return data_company_regression.loc[:, data_company_regression.columns != "target"]


@pytest.fixture(scope="session")
def y(data_company_regression: pd.DataFrame) -> pd.Series:
    return data_company_regression["target"]


@pytest.fixture
def data_company_regression_train_test_split(X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_frame(),
        "y_test": y_test.to_frame(),
    }


@pytest.fixture(scope="session")
def model() -> Pipeline:
    model = make_pipeline(
        make_column_transformer(("passthrough", make_column_selector(dtype_include="number"))),
        linear_model.LinearRegression(),
    )
    return model


@pytest.fixture(scope="session")
def fitted_model(model: Pipeline, X: pd.DataFrame, y: pd.DataFrame) -> Pipeline:
    model_cloned = clone(model)
    model_cloned.fit(X, y)
    return model_cloned


@pytest.fixture(scope="session")
def y_hat(X: pd.DataFrame, fitted_model: linear_model.LinearRegression) -> pd.Series:
    return pd.Series(fitted_model.predict(X))
