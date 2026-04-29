from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[reportMissingTypeStubs]
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    pairwise_distances_argmin_min,
)


FloatMatrix = NDArray[np.float32]


def _dedupe_and_fill_nearest_indices(
    *,
    nearest_indices: NDArray[np.int64],
    features: FloatMatrix,
    labels: NDArray[np.int32],
    centers: FloatMatrix,
    count: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for index in nearest_indices:
        selected_index = int(index)
        if selected_index in seen:
            continue
        seen.add(selected_index)
        selected.append(selected_index)

    if len(selected) < count:
        assigned_centers = centers[labels]
        assigned_distances = np.linalg.norm(features - assigned_centers, axis=1)
        for index in np.argsort(assigned_distances):
            selected_index = int(index)
            if selected_index in seen:
                continue
            seen.add(selected_index)
            selected.append(selected_index)
            if len(selected) == count:
                break

    return sorted(selected)


def select_kmeans_indices(
    features: FloatMatrix,
    count: int,
    *,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> list[int]:
    if count <= 0:
        return []
    if features.ndim != 2:
        raise ValueError("K-means selector expects a 2D feature matrix.")
    sample_count = int(features.shape[0])
    if sample_count <= count:
        return list(range(sample_count))

    kmeans = KMeans(
        n_clusters=count,
        random_state=seed,
        n_init=cast(Any, n_init),
        max_iter=max_iter,
    )
    labels = np.asarray(kmeans.fit_predict(features), dtype=np.int32)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    nearest_indices, _ = pairwise_distances_argmin_min(centers, features)
    return _dedupe_and_fill_nearest_indices(
        nearest_indices=np.asarray(nearest_indices, dtype=np.int64),
        features=features,
        labels=labels,
        centers=centers,
        count=count,
    )


def select_rows_by_feature_matrix(
    rows: list[dict[str, Any]],
    features: FloatMatrix,
    *,
    max_example: int | None,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> list[dict[str, Any]]:
    if max_example is None or len(rows) <= max_example:
        return list(rows)
    selected_indices = select_kmeans_indices(
        features,
        max_example,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    return [rows[index] for index in selected_indices]
