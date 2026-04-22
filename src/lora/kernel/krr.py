from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


_SAFE_CLIP = 1e12
_JITTER_REL = 1e-12


def _safe_matmul(left: FloatArray, right: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = left @ right
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=_SAFE_CLIP, neginf=-_SAFE_CLIP)
    return out


def _safe_solve(system: FloatArray, rhs: FloatArray) -> FloatArray:
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.isfinite(solution).all():
        solution = np.nan_to_num(
            solution,
            nan=0.0,
            posinf=_SAFE_CLIP,
            neginf=-_SAFE_CLIP,
        )
    return np.asarray(solution, dtype=np.float64)


def _ensure_targets(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[:, None]
    return array


@dataclass
class DualKRRModel:
    alpha: FloatArray
    ridge_lambda: float


@dataclass
class NystromKRRModel:
    landmark_indices: IntArray
    feature_proj: FloatArray
    beta: FloatArray
    ridge_lambda: float
    rank: int


def fit_krr_dual(
    k_train: FloatArray,
    targets: FloatArray,
    ridge_lambda: float,
) -> DualKRRModel:
    k_train = np.asarray(k_train, dtype=np.float64)
    targets = _ensure_targets(targets)
    n_train = k_train.shape[0]
    reg = 0.5 * (k_train + k_train.T) + (n_train * ridge_lambda) * np.eye(
        n_train,
        dtype=np.float64,
    )
    jitter = _JITTER_REL * max(float(np.trace(reg) / max(n_train, 1)), 1.0)
    reg = reg + jitter * np.eye(n_train, dtype=np.float64)
    alpha = _safe_solve(reg, targets)
    return DualKRRModel(alpha=alpha, ridge_lambda=ridge_lambda)


def predict_krr_dual(model: DualKRRModel, k_test_train: FloatArray) -> FloatArray:
    return _safe_matmul(np.asarray(k_test_train, dtype=np.float64), model.alpha)


def select_landmarks(n_train: int, count: int, seed: int) -> IntArray:
    if count <= 0 or count > n_train:
        raise ValueError(
            f"Landmark count must be between 1 and n_train={n_train}, got {count}."
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n_train), size=count, replace=False))


def fit_krr_nystrom(
    k_train_landmarks: FloatArray,
    k_landmarks: FloatArray,
    targets: FloatArray,
    ridge_lambda: float,
    rank: int | None = None,
    landmark_indices: IntArray | None = None,
) -> NystromKRRModel:
    k_train_landmarks = np.asarray(k_train_landmarks, dtype=np.float64)
    k_landmarks = np.asarray(k_landmarks, dtype=np.float64)
    targets = _ensure_targets(targets)
    n_train = int(k_train_landmarks.shape[0])
    n_landmarks = int(k_train_landmarks.shape[1])
    rank_value = n_landmarks if rank is None else rank
    rank = max(1, min(rank_value, n_landmarks))

    eigvals, eigvecs = np.linalg.eigh(0.5 * (k_landmarks + k_landmarks.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:rank]
    eigvecs = eigvecs[:, order][:, :rank]
    floor = max(1e-12, 1e-10 * float(max(np.max(eigvals), 1.0)))
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, floor))
    feature_proj = eigvecs * inv_sqrt[None, :]
    phi_train = _safe_matmul(k_train_landmarks, feature_proj)
    reg = _safe_matmul(phi_train.T, phi_train) + (n_train * ridge_lambda) * np.eye(
        rank,
        dtype=np.float64,
    )
    jitter = _JITTER_REL * max(float(np.trace(reg) / max(rank, 1)), 1.0)
    reg = reg + jitter * np.eye(rank, dtype=np.float64)
    beta = _safe_solve(reg, _safe_matmul(phi_train.T, targets))
    if landmark_indices is None:
        landmark_indices = np.arange(n_landmarks)
    return NystromKRRModel(
        landmark_indices=np.asarray(landmark_indices, dtype=np.int64),
        feature_proj=feature_proj,
        beta=beta,
        ridge_lambda=ridge_lambda,
        rank=rank,
    )


def predict_krr_nystrom(
    model: NystromKRRModel,
    k_test_landmarks: FloatArray,
) -> FloatArray:
    phi_test = _safe_matmul(
        np.asarray(k_test_landmarks, dtype=np.float64), model.feature_proj
    )
    return _safe_matmul(phi_test, model.beta)
