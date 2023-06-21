import numpy as np

from models.metrics.regression import smape


def test_symmetric_mean_absolute_error():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert np.isclose(smape(y_true, y_pred), 0.0)

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([0, 0, 0, 0, 0])
    assert smape(y_true, y_pred) >= 200

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 4])
    assert smape(y_true, y_pred) == smape(y_pred, y_true)

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred_high = np.array([0, 2, 3, 4, 5])
    y_pred_low = np.array([1, 2, 3, 4, 6])
    assert smape(y_true, y_pred_low) < smape(y_true, y_pred_high)
