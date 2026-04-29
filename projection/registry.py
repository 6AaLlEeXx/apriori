from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def resolve_projection(name: str) -> Callable[..., tuple[FloatMatrix, dict[str, Any]]]:
    normalized = _normalize_name(name)
    if normalized == "identity":
        from projection.identity import project

        return project
    if normalized in {"sparse_random", "sparse_random_projection"}:
        from projection.sparse_random import project

        return project
    raise ValueError(f"Unknown selector projection: {name}")


def apply_projection(
    features: FloatMatrix,
    name: str | None = None,
    *,
    seed: int = 42,
    n_components: int | None = None,
) -> tuple[FloatMatrix, dict[str, Any]]:
    projection_name = name or "identity"
    projected, metadata = resolve_projection(projection_name)(
        features,
        seed=seed,
        n_components=n_components,
    )
    projected = np.asarray(projected, dtype=np.float32)
    if projected.ndim != 2:
        raise ValueError(
            f"Selector projection `{projection_name}` returned a non-2D matrix."
        )
    return projected, metadata
