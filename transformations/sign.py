from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def transform(features: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.sign(features).astype(np.float32, copy=False)
