from __future__ import annotations

import numpy as np

from kernel.metrics import evaluate_predictions


def test_evaluate_predictions_returns_expected_keys() -> None:
    true_delta = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pred_delta = np.array([1.1, 1.9, 2.8], dtype=np.float64)
    true_adapter = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    pred_adapter = np.array([0.4, 1.2, 1.6], dtype=np.float64)
    payload = evaluate_predictions(
        true_delta=true_delta,
        pred_delta=pred_delta,
        true_adapter_score=true_adapter,
        pred_adapter_score=pred_adapter,
    )
    assert "delta" in payload
    assert "adapter_score" in payload
    assert "pearson" in payload["delta"]
    assert "sign_accuracy" in payload["delta"]
