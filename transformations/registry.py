from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]
Transformation = Callable[[FloatMatrix], FloatMatrix]


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def resolve_transformation(name: str) -> Transformation:
    normalized = _normalize_name(name)
    if normalized == "identity":
        from transformations.identity import transform

        return transform
    if normalized == "sign":
        from transformations.sign import transform

        return transform
    raise ValueError(f"Unknown selector transformation: {name}")


def normalize_transformation_names(names: Sequence[str] | str | None) -> list[str]:
    if names is None:
        return ["identity"]
    if isinstance(names, str):
        values = [part.strip() for part in names.split(",")]
    else:
        values = [str(part).strip() for part in names]
    return [value for value in values if value] or ["identity"]


def apply_transformations(
    features: FloatMatrix,
    names: Sequence[str] | str | None = None,
) -> FloatMatrix:
    transformed = features
    for name in normalize_transformation_names(names):
        transformed = np.asarray(
            resolve_transformation(name)(transformed),
            dtype=np.float32,
        )
        if transformed.ndim != 2:
            raise ValueError(
                f"Selector transformation `{name}` returned a non-2D matrix."
            )
    return transformed
