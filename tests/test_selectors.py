from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from transformations import apply_transformations


_RANDOM_SELECTOR_PATH = Path(__file__).resolve().parents[1] / "selectors" / "random.py"
_RANDOM_SPEC = importlib.util.spec_from_file_location(
    "test_random_selector",
    _RANDOM_SELECTOR_PATH,
)
assert _RANDOM_SPEC is not None
assert _RANDOM_SPEC.loader is not None
_RANDOM_SELECTOR = importlib.util.module_from_spec(_RANDOM_SPEC)
_RANDOM_SPEC.loader.exec_module(_RANDOM_SELECTOR)


def test_random_selector_uses_seed_and_max_examples() -> None:
    rows = [{"id": index} for index in range(10)]

    first = _RANDOM_SELECTOR.select_samples(
        rows,
        max_example=4,
        context={"seed": 123},
    )
    second = _RANDOM_SELECTOR.select_samples(
        rows,
        max_example=4,
        context={"seed": 123},
    )

    assert first == second
    assert len(first) == 4
    assert first != rows[:4]


def test_random_selector_returns_all_rows_when_limit_is_absent() -> None:
    rows = [{"id": index} for index in range(3)]
    context: dict[str, object] = {}

    selected = _RANDOM_SELECTOR.select_samples(rows, context=context)

    assert selected == rows
    assert "selector_timing" in context


def test_random_selector_returns_empty_for_non_positive_limit() -> None:
    rows = [{"id": index} for index in range(3)]

    selected = _RANDOM_SELECTOR.select_samples(rows, max_example=0)

    assert selected == []


def test_thresholded_sign_uses_default_threshold() -> None:
    features = np.asarray(
        [[-0.02, -0.005, 0.0, 0.005, 0.02]],
        dtype=np.float32,
    )

    transformed = apply_transformations(features, ["thresholded_sign"])

    assert np.asarray(transformed).tolist() == [[-1.0, 0.0, 0.0, 0.0, 1.0]]


def test_thresholded_sign_accepts_custom_threshold() -> None:
    features = np.asarray(
        [[-0.2, -0.05, 0.05, 0.2]],
        dtype=np.float32,
    )

    transformed = apply_transformations(
        features,
        ["thresholded-sign"],
        params={"thresholded-sign": {"threshold": 0.1}},
    )

    assert np.asarray(transformed).tolist() == [[-1.0, 0.0, 0.0, 1.0]]
