from typing import Any, Dict

import pandas as pd
from mlflow.models import ModelSignature
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.metrics.regression import smape


class TrainResults(BaseModel):
    """Pydantic Model to encapsulate common results in model training"""

    model: BaseEstimator = Field(repr=False)
    model_signature: ModelSignature = Field(repr=False)
    params: Dict[str, Any]
    metrics: Dict[str, float]
    datasets: Dict[str, pd.DataFrame] | None = Field(repr=False)
    plots: Dict[str, Any] | None = Field(repr=False)

    class Config:
        arbitrary_types_allowed = True  # NOTE: to be able to put the model in the results


def score_estimator_regression(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series):
    """For a given y and y_hat, generate a set of metrics scores."""
    y_hat: pd.Series[float] = pd.Series(estimator.predict(X), name="y_hat", index=X.index, dtype="float64")
    negative_y_filter = (y >= 0.0) & (y_hat >= 0.0)
    return {
        "mae": mean_absolute_error(y, y_hat),
        "mse": mean_squared_error(y, y_hat),
        "smape": smape(y[negative_y_filter], y_hat[negative_y_filter]),
    }
