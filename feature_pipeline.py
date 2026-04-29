from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from kernel.data import PairRecord
from projection import apply_projection
from transformations import apply_transformations, normalize_transformation_names


FloatMatrix = NDArray[np.float32]


class FeatureBackend(Protocol):
    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]: ...


def build_pair_records(
    rows: list[dict[str, Any]],
    split: str = "train",
) -> list[PairRecord]:
    records: list[PairRecord] = []
    for index, row in enumerate(rows):
        if "prompt" not in row:
            raise ValueError(f"Selector row is missing `prompt`: row {index}")
        if "completion" not in row:
            raise ValueError(f"Selector row is missing `completion`: row {index}")
        records.append(
            PairRecord(
                pair_id=f"{split}-{index:06d}",
                split=split,
                prompt="" if row["prompt"] is None else str(row["prompt"]),
                completion=(
                    "" if row["completion"] is None else str(row["completion"])
                ),
            )
        )
    return records


def extract_feature_matrix(
    records: list[PairRecord],
    backend: FeatureBackend,
    cache_path: str | Path | None = None,
) -> FloatMatrix:
    if not records:
        return np.empty((0, 0), dtype=np.float32)

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.ndim == 2 and cached.shape[0] == len(records):
            return cached

    first = np.asarray(backend.extract_feature(records[0]), dtype=np.float32).reshape(
        -1
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), first.shape[0]),
        )
    else:
        matrix = np.empty((len(records), first.shape[0]), dtype=np.float32)

    matrix[0] = first
    for index, record in enumerate(records[1:], start=1):
        feature = np.asarray(backend.extract_feature(record), dtype=np.float32).reshape(
            -1
        )
        if feature.shape[0] != first.shape[0]:
            raise ValueError(
                "Feature backend returned inconsistent dimensions: "
                f"expected {first.shape[0]}, got {feature.shape[0]}"
            )
        matrix[index] = feature

    if path is not None:
        del matrix
        return np.load(path, mmap_mode="r")
    return matrix


def _feature_cache_path(context: dict[str, Any]) -> Path | None:
    explicit_path = context.get("feature_cache_path")
    if explicit_path:
        return Path(str(explicit_path))
    run_dir = context.get("run_dir")
    if run_dir:
        return Path(str(run_dir)) / "selector_features" / "lora_ntk_train.npy"
    return None


def build_lora_ntk_backend(context: dict[str, Any]) -> FeatureBackend:
    from kernel.features import LoRANTKFeatureBackend

    base_model = context.get("base_model")
    if not base_model:
        raise ValueError("LoRA NTK feature extraction requires `context.base_model`.")
    leaf_filter = str(context.get("leaf_filter", "lora_b_only"))

    adapter_path = context.get("adapter_path")
    if adapter_path:
        return LoRANTKFeatureBackend(
            base_model=str(base_model),
            adapter_path=str(adapter_path),
            leaf_filter=leaf_filter,
        )

    mlx_args = context.get("mlx_args")
    if not isinstance(mlx_args, dict):
        raise ValueError("LoRA NTK feature extraction requires `context.mlx_args`.")
    return LoRANTKFeatureBackend.from_mlx_args(
        base_model=str(base_model),
        mlx_args=mlx_args,
        leaf_filter=leaf_filter,
    )


def prepare_feature_matrix(
    features: FloatMatrix,
    *,
    context: dict[str, Any],
) -> FloatMatrix:
    transformations = context.get("selector_transformations")
    projection = str(context.get("selector_projection", "identity"))
    projection_components = context.get("selector_projection_components")
    if projection_components is not None:
        projection_components = int(projection_components)

    transformation_names = normalize_transformation_names(transformations)
    transformed = apply_transformations(features, transformation_names)
    projected, projection_metadata = apply_projection(
        transformed,
        projection,
        seed=int(context.get("seed", 42)),
        n_components=projection_components,
    )
    context["selector_feature_pipeline"] = {
        "raw_feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "transformations": transformation_names,
        "transformed_feature_dim": (
            int(transformed.shape[1]) if transformed.ndim == 2 else 0
        ),
        "projection": projection_metadata,
    }
    return projected


def prepare_lora_ntk_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    backend: FeatureBackend | None = None,
) -> FloatMatrix:
    records = build_pair_records(rows)
    feature_backend = backend or build_lora_ntk_backend(context)
    features = extract_feature_matrix(
        records,
        feature_backend,
        cache_path=_feature_cache_path(context),
    )
    return prepare_feature_matrix(features, context=context)
