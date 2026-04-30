from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


DEFAULT_THRESHOLD = 0.01


def transform(
    features: NDArray[np.float32],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> NDArray[np.float32]:
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("thresholded_sign threshold must be non-negative.")

    transformed = np.zeros_like(features, dtype=np.float32)
    transformed[features > threshold] = 1.0
    transformed[features < -threshold] = -1.0
    return transformed
