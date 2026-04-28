from __future__ import annotations

from typing import Any


def select_samples(rows: list[dict[str, Any]], max_example: int | None = None):
    """Default selector: return the rows unchanged."""
    return rows
