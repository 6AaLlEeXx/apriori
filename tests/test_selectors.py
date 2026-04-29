from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feature_pipeline import (
    build_pair_records,
    extract_feature_matrix,
    prepare_feature_matrix,
)
from kernel.data import PairRecord
from selector_algorithms import select_rows_by_feature_matrix


_SELECTOR_PATH = (
    Path(__file__).resolve().parents[1] / "selectors" / "lora_ntk_kmeans.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_lora_ntk_kmeans_selector",
    _SELECTOR_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_SELECTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SELECTOR)

select_lora_ntk_kmeans_rows = _SELECTOR.select_lora_ntk_kmeans_rows

_RANDOM_SELECTOR_PATH = Path(__file__).resolve().parents[1] / "selectors" / "random.py"
_RANDOM_SPEC = importlib.util.spec_from_file_location(
    "test_random_selector",
    _RANDOM_SELECTOR_PATH,
)
assert _RANDOM_SPEC is not None
assert _RANDOM_SPEC.loader is not None
_RANDOM_SELECTOR = importlib.util.module_from_spec(_RANDOM_SPEC)
_RANDOM_SPEC.loader.exec_module(_RANDOM_SELECTOR)


class FakeBackend:
    def __init__(self, features: dict[str, list[float]]) -> None:
        self.features = features
        self.calls = 0

    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]:
        self.calls += 1
        return np.asarray(self.features[record.prompt], dtype=np.float32)


def test_build_pair_records_validates_prompt_completion() -> None:
    records = build_pair_records(
        [{"prompt": "p", "completion": "c"}, {"prompt": None, "completion": None}]
    )

    assert records[0] == PairRecord(
        pair_id="train-000000",
        split="train",
        prompt="p",
        completion="c",
    )
    assert records[1].prompt == ""
    assert records[1].completion == ""


def test_select_rows_by_feature_matrix_picks_cluster_representatives() -> None:
    rows = [{"id": index} for index in range(4)]
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)

    selected = select_rows_by_feature_matrix(
        rows,
        features,
        max_example=2,
        seed=7,
    )
    selected_ids = {int(row["id"]) for row in selected}

    assert len(selected_ids) == 2
    assert selected_ids & {0, 1}
    assert selected_ids & {2, 3}


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


def test_lora_ntk_kmeans_selector_uses_backend_features() -> None:
    rows: list[dict[str, Any]] = [
        {"prompt": "a", "completion": "x"},
        {"prompt": "b", "completion": "x"},
        {"prompt": "c", "completion": "x"},
        {"prompt": "d", "completion": "x"},
    ]
    backend = FakeBackend(
        {
            "a": [0.0],
            "b": [0.2],
            "c": [9.8],
            "d": [10.0],
        }
    )

    selected = select_lora_ntk_kmeans_rows(
        rows,
        max_example=2,
        context={"seed": 11},
        backend=backend,
    )
    selected_prompts = {str(row["prompt"]) for row in selected}

    assert len(selected_prompts) == 2
    assert selected_prompts & {"a", "b"}
    assert selected_prompts & {"c", "d"}
    assert backend.calls == len(rows)


def test_prepare_feature_matrix_applies_transformations_and_projection() -> None:
    features = np.asarray(
        [[-2.0, 0.0, 4.0], [5.0, -0.1, 0.0], [1.0, 2.0, -3.0]],
        dtype=np.float32,
    )
    context = {
        "seed": 3,
        "selector_transformations": ["sign"],
        "selector_projection": "sparse_random",
        "selector_projection_components": 2,
    }

    prepared = prepare_feature_matrix(features, context=context)

    assert prepared.shape == (3, 2)
    assert context["selector_feature_pipeline"]["transformations"] == ["sign"]
    assert context["selector_feature_pipeline"]["raw_feature_dim"] == 3
    assert (
        context["selector_feature_pipeline"]["projection"]["name"]
        == "sparse_random"
    )
    assert context["selector_feature_pipeline"]["projection"]["output_dim"] == 2


def test_extract_feature_matrix_reuses_cache(tmp_path: Path) -> None:
    records = build_pair_records(
        [{"prompt": "a", "completion": "x"}, {"prompt": "b", "completion": "x"}]
    )
    cache_path = tmp_path / "features.npy"
    backend = FakeBackend({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    first = extract_feature_matrix(records, backend, cache_path=cache_path)
    second = extract_feature_matrix(records, backend, cache_path=cache_path)

    assert np.asarray(first).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert np.asarray(second).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert backend.calls == len(records)
