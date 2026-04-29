from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernel.data import PairRecord
from selection.lora_ntk_kmeans import (
    build_pair_records,
    extract_feature_matrix,
    select_lora_ntk_kmeans_rows,
    select_rows_by_feature_matrix,
)


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
