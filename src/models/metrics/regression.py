"""Metrics to assess performance on regression task."""

from typing import Any

import numpy as np


def smape(y_true: Any, y_pred: Any) -> float:
    """Symmetric mean absolute percentage error: accuracy measure based on percentage (or relative) errors."""
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
