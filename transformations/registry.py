from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]
Transformation = Callable[..., FloatMatrix]


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
    if normalized == "thresholded_sign":
        from transformations.thresholded_sign import transform

        return transform
    raise ValueError(f"Unknown selector transformation: {name}")


def normalize_transformation_names(names: Sequence[str] | str | None) -> list[str]:
    if names is None:
        return ["identity"]
    if isinstance(names, str):
        values = [part.strip() for part in names.split(",")]
    else:
        values = [
            part.strip()
            for value in names
            for part in str(value).split(",")
        ]
    return [_normalize_name(value) for value in values if value] or ["identity"]


def normalize_transformation_params(
    params: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not params:
        return {}
    return {
        _normalize_name(name): dict(value)
        for name, value in params.items()
    }


def apply_transformations(
    features: FloatMatrix,
    names: Sequence[str] | str | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> FloatMatrix:
    transformed = features
    normalized_params = normalize_transformation_params(params)
    for name in normalize_transformation_names(names):
        transformed = np.asarray(
            resolve_transformation(name)(
                transformed,
                **normalized_params.get(name, {}),
            ),
            dtype=np.float32,
        )
        if transformed.ndim != 2:
            raise ValueError(
                f"Selector transformation `{name}` returned a non-2D matrix."
            )
    return transformed
