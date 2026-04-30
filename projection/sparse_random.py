from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.random_projection import SparseRandomProjection  # type: ignore[reportMissingTypeStubs]


def project(
    features: NDArray[np.float32],
    *,
    seed: int = 42,
    n_components: int | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError("Sparse random projection expects a 2D feature matrix.")
    input_dim = int(features.shape[1])
    output_dim = int(n_components or min(1024, max(1, input_dim)))
    projector = SparseRandomProjection(
        n_components=cast(Any, output_dim),
        random_state=seed,
        dense_output=True,
    )
    projected = np.asarray(
        projector.fit_transform(features),
        dtype=np.float32,
    )
    return projected, {
        "name": "sparse_random",
        "input_dim": input_dim,
        "output_dim": int(projected.shape[1]),
        "n_components": output_dim,
        "density": cast(Any, projector).density_,
    }


def project_chunked(
    features: NDArray[np.float32],
    *,
    seed: int = 42,
    n_components: int | None = None,
    chunk_size: int = 16,
    transform_chunk: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError("Sparse random projection expects a 2D feature matrix.")
    sample_count = int(features.shape[0])
    input_dim = int(features.shape[1])
    output_dim = int(n_components or min(1024, max(1, input_dim)))
    chunk_size = max(1, int(chunk_size))
    projector = SparseRandomProjection(
        n_components=cast(Any, output_dim),
        random_state=seed,
        dense_output=True,
    )
    projector.fit(np.zeros((1, input_dim), dtype=np.float32))
    projected = np.empty((sample_count, output_dim), dtype=np.float32)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        chunk = np.asarray(features[start:end], dtype=np.float32)
        if transform_chunk is not None:
            chunk = np.asarray(transform_chunk(chunk), dtype=np.float32)
        projected[start:end] = np.asarray(projector.transform(chunk), dtype=np.float32)
        if progress_callback is not None:
            progress_callback(end, sample_count)
    return projected, {
        "name": "sparse_random",
        "input_dim": input_dim,
        "output_dim": int(projected.shape[1]),
        "n_components": output_dim,
        "density": cast(Any, projector).density_,
        "chunked": True,
        "chunk_size": chunk_size,
    }
