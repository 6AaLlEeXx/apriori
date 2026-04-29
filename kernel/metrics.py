from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


FloatArray = NDArray[np.float64]


def _to_1d(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array


def pearson_corr(left: FloatArray, right: FloatArray) -> float:
    x = _to_1d(left)
    y = _to_1d(right)
    if x.size == 0 or y.size == 0:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(left: FloatArray, right: FloatArray) -> float:
    x = rankdata(_to_1d(left), method="average")
    y = rankdata(_to_1d(right), method="average")
    return pearson_corr(x, y)


def rmse(left: FloatArray, right: FloatArray) -> float:
    x = _to_1d(left)
    y = _to_1d(right)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(left: FloatArray, right: FloatArray) -> float:
    x = _to_1d(left)
    y = _to_1d(right)
    return float(np.mean(np.abs(x - y)))


def sign_accuracy(left: FloatArray, right: FloatArray) -> float:
    x = np.sign(_to_1d(left))
    y = np.sign(_to_1d(right))
    return float(np.mean(x == y))


def evaluate_predictions(
    true_delta: FloatArray,
    pred_delta: FloatArray,
    true_adapter_score: FloatArray,
    pred_adapter_score: FloatArray,
) -> dict[str, Any]:
    return {
        "delta": {
            "pearson": pearson_corr(true_delta, pred_delta),
            "spearman": spearman_corr(true_delta, pred_delta),
            "rmse": rmse(true_delta, pred_delta),
            "mae": mae(true_delta, pred_delta),
            "sign_accuracy": sign_accuracy(true_delta, pred_delta),
        },
        "adapter_score": {
            "pearson": pearson_corr(true_adapter_score, pred_adapter_score),
            "spearman": spearman_corr(true_adapter_score, pred_adapter_score),
            "rmse": rmse(true_adapter_score, pred_adapter_score),
            "mae": mae(true_adapter_score, pred_adapter_score),
        },
    }
