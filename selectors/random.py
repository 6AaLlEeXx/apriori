from __future__ import annotations

from typing import Any
import random
from time import perf_counter


def select_samples(
    rows: list[dict[str, Any]],
    subset_size: int | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start = perf_counter()
    context = context if context is not None else {}
    if subset_size is None or len(rows) <= subset_size:
        context["selector_timing"] = {"total_seconds": perf_counter() - start}
        return list(rows)
    if subset_size <= 0:
        context["selector_timing"] = {"total_seconds": perf_counter() - start}
        return []

    seed = int(context.get("seed", 42))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), subset_size))
    selected = [rows[index] for index in indices]
    context["selector_timing"] = {"total_seconds": perf_counter() - start}
    return selected
