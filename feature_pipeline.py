from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from kernel.data import PairRecord
from paths import resolve_project_path
from projection import apply_projection
from transformations import apply_transformations, normalize_transformation_names


FloatMatrix = NDArray[np.float32]
FEATURE_CACHE_VERSION = 1
DEFAULT_MAX_UNPROJECTED_KMEANS_BYTES = 4 * 1024**3


class FeatureBackend(Protocol):
    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]: ...


def _selector_debug_enabled(context: dict[str, Any]) -> bool:
    value = context.get("selector_debug", False)
    if isinstance(value, str):
        return value.lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _debug(context: dict[str, Any], message: str) -> None:
    if _selector_debug_enabled(context):
        print(f"[selector] {message}", flush=True)


def _normalize_projection_name(name: str | None) -> str:
    return (name or "identity").strip().lower().replace("-", "_")


def _matrix_nbytes(features: FloatMatrix) -> int:
    if features.ndim != 2:
        return 0
    return (
        int(features.shape[0])
        * int(features.shape[1])
        * np.dtype(np.float32).itemsize
    )


def _max_unprojected_kmeans_bytes(context: dict[str, Any]) -> int:
    value = context.get(
        "selector_max_kmeans_feature_bytes",
        DEFAULT_MAX_UNPROJECTED_KMEANS_BYTES,
    )
    return int(value)


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.2f} GiB"


def _validate_unprojected_feature_size(
    features: FloatMatrix,
    *,
    context: dict[str, Any],
) -> None:
    byte_count = _matrix_nbytes(features)
    limit = _max_unprojected_kmeans_bytes(context)
    if limit <= 0 or byte_count <= limit:
        return
    raise ValueError(
        "Unprojected k-means selector features are too large for in-memory "
        "clustering: "
        f"shape={tuple(features.shape)}, size={_format_gib(byte_count)}, "
        f"limit={_format_gib(limit)}. Use "
        "`--selector-projection sparse_random --selector-projection-components "
        "<dim>` or lower the data/selector size."
    )


def _load_cached_feature_matrix(
    path: Path,
    expected_rows: int,
) -> FloatMatrix | None:
    try:
        cached = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if cached.ndim == 2 and cached.shape[0] == expected_rows:
        return cached
    return None


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
    backend: FeatureBackend | None,
    cache_path: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_feature_bytes: int | None = None,
) -> FloatMatrix:
    if not records:
        return np.empty((0, 0), dtype=np.float32)

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        cached = _load_cached_feature_matrix(path, len(records))
        if cached is not None:
            return cached
        if backend is None:
            raise ValueError(f"Feature cache shape does not match records: {path}")

    if backend is None:
        raise ValueError("Feature backend is required when no valid cache exists.")

    first = np.asarray(backend.extract_feature(records[0]), dtype=np.float32).reshape(
        -1
    )
    expected_bytes = len(records) * first.shape[0] * first.dtype.itemsize
    if max_feature_bytes is not None and expected_bytes > max_feature_bytes:
        raise ValueError(
            "LoRA-NTK feature matrix is too large to materialize without "
            "projection: "
            f"shape=({len(records)}, {first.shape[0]}), "
            f"size={_format_gib(expected_bytes)}, "
            f"limit={_format_gib(max_feature_bytes)}. Use "
            "`--selector-projection sparse_random --selector-projection-components "
            "<dim>`."
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
    if progress_callback is not None:
        progress_callback(1, len(records))
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
        if progress_callback is not None:
            progress_callback(index + 1, len(records))

    if path is not None:
        del matrix
        return np.load(path, mmap_mode="r")
    return matrix


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_feature_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                {
                    "prompt": row.get("prompt"),
                    "completion": row.get("completion"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _lora_ntk_cache_payload(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    mlx_args = context.get("mlx_args")
    if not isinstance(mlx_args, dict):
        mlx_args = {}
    return {
        "version": FEATURE_CACHE_VERSION,
        "backend": "lora_ntk",
        "base_model": str(context.get("base_model", "")),
        "seed": int(context.get("seed", 42)),
        "leaf_filter": str(context.get("leaf_filter", "lora_b_only")),
        "fine_tune_type": str(mlx_args.get("fine_tune_type", "lora")),
        "num_layers": mlx_args.get("num_layers"),
        "lora_parameters": mlx_args.get("lora_parameters"),
        "rows": {
            "count": len(rows),
            "sha256": _hash_feature_rows(rows),
        },
    }


def _shared_feature_cache_path(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> Path | None:
    cache_root = context.get("shared_feature_cache_root")
    if not cache_root:
        return None
    payload = _lora_ntk_cache_payload(rows, context)
    cache_key = _json_hash(payload)
    root = resolve_project_path(str(cache_root))
    return root / "lora_ntk" / f"{cache_key}.npy"


def _feature_cache_path(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[Path | None, str]:
    explicit_path = context.get("feature_cache_path")
    if explicit_path:
        return resolve_project_path(str(explicit_path)), "explicit"
    shared_path = _shared_feature_cache_path(rows, context)
    if shared_path is not None:
        return shared_path, "shared"
    run_dir = context.get("run_dir")
    if run_dir:
        return Path(str(run_dir)) / "selector_features" / "lora_ntk_train.npy", "run"
    return None, "none"


def build_lora_ntk_backend(context: dict[str, Any]) -> FeatureBackend:
    from kernel.features import LoRANTKFeatureBackend

    base_model = context.get("base_model")
    if not base_model:
        raise ValueError("LoRA NTK feature extraction requires `context.base_model`.")
    leaf_filter = str(context.get("leaf_filter", "lora_b_only"))
    seed = int(context.get("seed", 42))

    adapter_path = context.get("adapter_path")
    if adapter_path:
        return LoRANTKFeatureBackend(
            base_model=str(base_model),
            adapter_path=str(adapter_path),
            leaf_filter=leaf_filter,
            seed=seed,
        )

    mlx_args = context.get("mlx_args")
    if not isinstance(mlx_args, dict):
        raise ValueError("LoRA NTK feature extraction requires `context.mlx_args`.")
    return LoRANTKFeatureBackend.from_mlx_args(
        base_model=str(base_model),
        mlx_args=mlx_args,
        leaf_filter=leaf_filter,
        seed=seed,
    )


def _project_sparse_random_chunked(
    features: FloatMatrix,
    *,
    context: dict[str, Any],
    transformation_names: list[str],
    projection_components: int | None,
) -> tuple[FloatMatrix, dict[str, Any], float, float]:
    from projection.sparse_random import project_chunked

    transform_seconds = 0.0

    def transform_chunk(chunk: FloatMatrix) -> FloatMatrix:
        nonlocal transform_seconds
        start = perf_counter()
        transformed_chunk = apply_transformations(chunk, transformation_names)
        transform_seconds += perf_counter() - start
        return transformed_chunk

    def report_progress(done: int, total: int) -> None:
        if not _selector_debug_enabled(context):
            return
        interval = max(1, int(context.get("selector_debug_interval", 0) or 0))
        if interval <= 1:
            interval = max(1, total // 10)
        if done == total or done % interval == 0:
            _debug(context, f"sparse random projection {done}/{total}")

    projection_start = perf_counter()
    projected, projection_metadata = project_chunked(
        features,
        seed=int(context.get("seed", 42)),
        n_components=projection_components,
        chunk_size=int(context.get("selector_projection_chunk_size", 16)),
        transform_chunk=transform_chunk,
        progress_callback=report_progress,
    )
    projection_seconds = perf_counter() - projection_start
    return projected, projection_metadata, transform_seconds, projection_seconds


def prepare_feature_matrix(
    features: FloatMatrix,
    *,
    context: dict[str, Any],
) -> FloatMatrix:
    transformations = context.get("selector_transformations")
    projection = str(context.get("selector_projection", "identity"))
    normalized_projection = _normalize_projection_name(projection)
    projection_components = context.get("selector_projection_components")
    if projection_components is not None:
        projection_components = int(projection_components)

    transformation_names = normalize_transformation_names(transformations)
    if normalized_projection in {"sparse_random", "sparse_random_projection"}:
        projected, projection_metadata, transform_seconds, projection_seconds = (
            _project_sparse_random_chunked(
                features,
                context=context,
                transformation_names=transformation_names,
                projection_components=projection_components,
            )
        )
        transformed_feature_dim = int(features.shape[1]) if features.ndim == 2 else 0
    else:
        _validate_unprojected_feature_size(features, context=context)
        transform_start = perf_counter()
        transformed = apply_transformations(features, transformation_names)
        transform_seconds = perf_counter() - transform_start
        projection_start = perf_counter()
        projected, projection_metadata = apply_projection(
            transformed,
            projection,
            seed=int(context.get("seed", 42)),
            n_components=projection_components,
        )
        projection_seconds = perf_counter() - projection_start
        transformed_feature_dim = (
            int(transformed.shape[1]) if transformed.ndim == 2 else 0
        )
    context["selector_feature_pipeline"] = {
        "raw_feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "transformations": transformation_names,
        "transformed_feature_dim": transformed_feature_dim,
        "projection": projection_metadata,
        "timing": {
            "transformation_seconds": transform_seconds,
            "projection_seconds": projection_seconds,
        },
    }
    projection_name = projection_metadata.get("name", projection)
    output_dim = int(projected.shape[1]) if projected.ndim == 2 else 0
    _debug(
        context,
        "feature pipeline "
        f"raw_dim={context['selector_feature_pipeline']['raw_feature_dim']} "
        f"transformations={transformation_names} "
        f"projection={projection_name} output_dim={output_dim}",
    )
    return projected


def prepare_lora_ntk_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    backend: FeatureBackend | None = None,
) -> FloatMatrix:
    records = build_pair_records(rows)
    projection = _normalize_projection_name(
        str(context.get("selector_projection", "identity"))
    )
    max_feature_bytes = (
        _max_unprojected_kmeans_bytes(context)
        if projection == "identity"
        else None
    )
    cache_path, cache_scope = _feature_cache_path(rows, context)
    extraction_start = perf_counter()
    features = (
        _load_cached_feature_matrix(cache_path, len(records))
        if cache_path is not None and cache_path.exists()
        else None
    )
    cache_hit = features is not None
    if cache_hit:
        _debug(
            context,
            f"LoRA-NTK raw feature cache hit rows={len(records)} path={cache_path}",
        )
    else:
        cache_description = str(cache_path) if cache_path is not None else "disabled"
        _debug(
            context,
            "LoRA-NTK raw feature cache miss "
            f"rows={len(records)} scope={cache_scope} path={cache_description}",
        )
    if features is None:
        _debug(context, "building LoRA-NTK feature backend")
        feature_backend = backend or build_lora_ntk_backend(context)

        def report_progress(done: int, total: int) -> None:
            interval = max(1, int(context.get("selector_debug_interval", 0) or 0))
            if interval <= 1:
                interval = max(1, total // 10)
            if done == 1 or done == total or done % interval == 0:
                _debug(context, f"extracting LoRA-NTK features {done}/{total}")

        features = extract_feature_matrix(
            records,
            feature_backend,
            cache_path=cache_path,
            progress_callback=report_progress,
            max_feature_bytes=max_feature_bytes,
        )
        if cache_path is not None:
            _debug(context, f"LoRA-NTK raw feature cache written path={cache_path}")
    context["selector_feature_extraction_seconds"] = (
        perf_counter() - extraction_start
    )
    context["selector_feature_cache"] = {
        "path": str(cache_path) if cache_path is not None else None,
        "scope": cache_scope,
        "hit": cache_hit,
    }
    _debug(
        context,
        "LoRA-NTK raw feature stage finished "
        f"seconds={context['selector_feature_extraction_seconds']:.2f}",
    )
    return prepare_feature_matrix(features, context=context)
