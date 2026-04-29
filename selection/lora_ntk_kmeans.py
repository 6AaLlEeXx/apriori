from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[reportMissingTypeStubs]
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    pairwise_distances_argmin_min,
)

from kernel.data import PairRecord


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


def _feature_cache_path(context: dict[str, Any]) -> Path | None:
    explicit_path = context.get("feature_cache_path")
    if explicit_path:
        return Path(str(explicit_path))
    run_dir = context.get("run_dir")
    if run_dir:
        return Path(str(run_dir)) / "selector_features" / "lora_ntk_train.npy"
    return None


def _build_lora_ntk_backend(context: dict[str, Any]) -> FeatureBackend:
    from kernel.features import LoRANTKFeatureBackend

    base_model = context.get("base_model")
    if not base_model:
        raise ValueError("LoRA NTK k-means selector requires `context.base_model`.")
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
        raise ValueError("LoRA NTK k-means selector requires `context.mlx_args`.")
    return LoRANTKFeatureBackend.from_mlx_args(
        base_model=str(base_model),
        mlx_args=mlx_args,
        leaf_filter=leaf_filter,
    )


def select_lora_ntk_kmeans_rows(
    rows: list[dict[str, Any]],
    *,
    max_example: int | None,
    context: dict[str, Any] | None,
    backend: FeatureBackend | None = None,
) -> list[dict[str, Any]]:
    if max_example is None or len(rows) <= max_example:
        return list(rows)
    if max_example <= 0:
        return []

    context = dict(context or {})
    seed = int(context.get("seed", 42))
    n_init = int(context.get("kmeans_n_init", 10))
    max_iter = int(context.get("kmeans_max_iter", 300))
    records = build_pair_records(rows)
    backend = backend or _build_lora_ntk_backend(context)
    features = extract_feature_matrix(
        records,
        backend,
        cache_path=_feature_cache_path(context),
    )
    return select_rows_by_feature_matrix(
        rows,
        features,
        max_example=max_example,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
    )


def select_samples(
    rows: list[dict[str, Any]],
    max_example: int | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return select_lora_ntk_kmeans_rows(
        rows,
        max_example=max_example,
        context=context,
    )
