from __future__ import annotations

from typing import Any
import random
from time import perf_counter


def select_samples(
    rows: list[dict[str, Any]],
    max_example: int | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start = perf_counter()
    context = context if context is not None else {}
    if max_example is None or len(rows) <= max_example:
        context["selector_timing"] = {"total_seconds": perf_counter() - start}
        return list(rows)
    if max_example <= 0:
        context["selector_timing"] = {"total_seconds": perf_counter() - start}
        return []

    seed = int(context.get("seed", 42))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_example))
    selected = [rows[index] for index in indices]
    context["selector_timing"] = {"total_seconds": perf_counter() - start}
    return selected
