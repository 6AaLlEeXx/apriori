from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def project(
    features: NDArray[np.float32],
    *,
    seed: int = 42,
    n_components: int | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    del seed, n_components
    return features, {
        "name": "identity",
        "input_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "output_dim": int(features.shape[1]) if features.ndim == 2 else 0,
    }
