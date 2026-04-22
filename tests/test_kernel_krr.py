from __future__ import annotations

import numpy as np

from lora.kernel.krr import (
    fit_krr_dual,
    fit_krr_nystrom,
    predict_krr_dual,
    predict_krr_nystrom,
)


def test_dual_krr_recovers_training_targets_on_identity_kernel() -> None:
    kernel = np.eye(4, dtype=np.float64)
    targets = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float64)
    model = fit_krr_dual(kernel, targets, ridge_lambda=1e-6)
    pred = predict_krr_dual(model, kernel).reshape(-1)
    assert np.allclose(pred, targets, atol=1e-4)


def test_nystrom_krr_shapes_are_consistent() -> None:
    kernel = np.array(
        [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    model = fit_krr_nystrom(
        k_train_landmarks=kernel,
        k_landmarks=kernel,
        targets=targets,
        ridge_lambda=1e-3,
        rank=2,
    )
    pred = predict_krr_nystrom(model, kernel).reshape(-1)
    assert pred.shape == targets.shape
    assert np.isfinite(pred).all()
