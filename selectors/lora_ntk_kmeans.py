from __future__ import annotations

from time import perf_counter
from typing import Any

from feature_pipeline import FeatureBackend, prepare_lora_ntk_feature_matrix
from selector_algorithms import select_rows_by_feature_matrix


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

    context = context if context is not None else {}
    seed = int(context.get("seed", 42))
    n_init = int(context.get("kmeans_n_init", 10))
    max_iter = int(context.get("kmeans_max_iter", 300))
    max_feature_bytes = int(
        context.get("selector_max_kmeans_feature_bytes", 4 * 1024**3)
    )
    total_start = perf_counter()
    features = prepare_lora_ntk_feature_matrix(
        rows,
        context=context,
        backend=backend,
    )
    kmeans_start = perf_counter()
    selected = select_rows_by_feature_matrix(
        rows,
        features,
        max_example=max_example,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        max_feature_bytes=max_feature_bytes,
    )
    kmeans_seconds = perf_counter() - kmeans_start
    context["selector_timing"] = {
        "total_seconds": perf_counter() - total_start,
        "feature_extraction_seconds": context.get(
            "selector_feature_extraction_seconds",
        ),
        "transformation_seconds": (
            context.get("selector_feature_pipeline", {})
            .get("timing", {})
            .get("transformation_seconds")
        ),
        "projection_seconds": (
            context.get("selector_feature_pipeline", {})
            .get("timing", {})
            .get("projection_seconds")
        ),
        "kmeans_seconds": kmeans_seconds,
    }
    return selected


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
