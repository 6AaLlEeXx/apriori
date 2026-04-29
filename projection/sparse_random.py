from __future__ import annotations

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
