from __future__ import annotations

from typing import Any
import random


def select_samples(
    rows: list[dict[str, Any]],
    max_example: int | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if max_example is None or len(rows) <= max_example:
        return list(rows)
    if max_example <= 0:
        return []

    seed = int((context or {}).get("seed", 42))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_example))
    return [rows[index] for index in indices]
