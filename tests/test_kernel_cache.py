from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from numpy.typing import NDArray

from kernel.config import KernelRunConfig
from kernel.data import PairRecord
from kernel.run import _extract_features_to_npy, _load_features, _score_splits_with_cache
from kernel.scoring import PairTokens


class FakeScorer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def score_record(self, record: PairRecord) -> tuple[float, PairTokens]:
        self.calls.append(record.pair_id)
        return (
            float(len(record.prompt)),
            PairTokens(
                token_ids=np.asarray([], dtype=np.int32),
                prompt_offset=1,
                sequence_length=3,
                completion_targets=2,
            ),
        )


class FakeFeatureBackend:
    def __init__(self) -> None:
        self.calls = 0

    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]:
        self.calls += 1
        return np.asarray([float(len(record.prompt)), 1.0], dtype=np.float32)


def _records() -> dict[str, list[PairRecord]]:
    return {
        split: [
            PairRecord(
                pair_id=f"{split}-000000",
                split=split,
                prompt=f"{split} prompt",
                completion="completion",
            )
        ]
        for split in ("train", "valid", "test")
    }


def _kernel_config(tmp_path: Path) -> KernelRunConfig:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    return KernelRunConfig(
        base_model="models/test",
        output_root=str(tmp_path / "kernel"),
        adapter_path=str(adapter_dir),
    )


def _write_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir()
    adapter_dir.joinpath("adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": 1,
                "lora_parameters": {"rank": 1, "scale": 1.0},
            }
        )
    )
    adapter_dir.joinpath("adapters.safetensors").write_bytes(b"adapter")


def test_kernel_score_cache_reuses_scored_splits(tmp_path: Path) -> None:
    config = _kernel_config(tmp_path)
    records = _records()
    calls: list[str] = []

    def scorer_factory(_base_model: str, _adapter_path: str | None) -> FakeScorer:
        return FakeScorer(calls)

    first = _score_splits_with_cache(
        config,
        records,
        adapter_path=None,
        scorer_factory=scorer_factory,
    )
    second = _score_splits_with_cache(
        config,
        records,
        adapter_path=None,
        scorer_factory=scorer_factory,
    )

    assert len(calls) == 3
    assert first == second
    assert sorted((tmp_path / "kernel" / "cache" / "scores").glob("*.jsonl"))


def test_kernel_feature_cache_reuses_extracted_features(tmp_path: Path) -> None:
    config = _kernel_config(tmp_path)
    records = _records()["test"]
    first_backend = FakeFeatureBackend()
    second_backend = FakeFeatureBackend()

    first_path, first_dim = _extract_features_to_npy(
        config=config,
        backend=first_backend,
        records=records,
    )
    second_path, second_dim = _extract_features_to_npy(
        config=config,
        backend=second_backend,
        records=records,
    )

    assert first_path == second_path
    assert first_dim == 2
    assert second_dim == 2
    assert first_backend.calls == 1
    assert second_backend.calls == 0
    assert first_path.is_relative_to(tmp_path / "kernel" / "cache" / "features")


def test_kernel_feature_cache_is_shared_across_matching_adapter_configs(
    tmp_path: Path,
) -> None:
    left_adapter = tmp_path / "left-adapter"
    right_adapter = tmp_path / "right-adapter"
    _write_adapter(left_adapter)
    _write_adapter(right_adapter)
    left_config = KernelRunConfig(
        base_model="models/test",
        output_root=str(tmp_path / "kernel"),
        adapter_path=str(left_adapter),
    )
    right_config = KernelRunConfig(
        base_model="models/test",
        output_root=str(tmp_path / "kernel"),
        adapter_path=str(right_adapter),
    )
    records = _records()["test"]
    first_backend = FakeFeatureBackend()
    second_backend = FakeFeatureBackend()

    first_path, _ = _extract_features_to_npy(
        config=left_config,
        backend=first_backend,
        records=records,
    )
    second_path, _ = _extract_features_to_npy(
        config=right_config,
        backend=second_backend,
        records=records,
    )

    assert first_path == second_path
    assert first_backend.calls == 1
    assert second_backend.calls == 0


def test_kernel_feature_transform_reuses_raw_feature_cache(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    raw_config = KernelRunConfig(
        base_model="models/test",
        output_root=str(tmp_path / "kernel"),
        adapter_path=str(adapter),
        backend_args={"leaf_filter": "lora_b_only"},
    )
    transformed_config = KernelRunConfig(
        base_model="models/test",
        output_root=str(tmp_path / "kernel"),
        adapter_path=str(adapter),
        backend_args={
            "leaf_filter": "lora_b_only",
            "feature_transform": "thresholded_sign",
            "threshold": 0.1,
        },
    )
    records = _records()["test"]
    raw_backend = FakeFeatureBackend()
    transformed_backend = FakeFeatureBackend()

    raw_path, _ = _extract_features_to_npy(
        config=raw_config,
        backend=raw_backend,
        records=records,
    )
    transformed_path, _ = _extract_features_to_npy(
        config=transformed_config,
        backend=transformed_backend,
        records=records,
    )

    assert raw_path == transformed_path
    assert raw_backend.calls == 1
    assert transformed_backend.calls == 0


def test_kernel_load_features_applies_thresholded_sign(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.asarray([[-0.2, -0.05, 0.05, 0.2]], dtype=np.float32))
    config = KernelRunConfig(
        backend_args={
            "feature_transform": "thresholded_sign",
            "threshold": 0.1,
        },
    )

    features = _load_features(path, config)

    assert np.asarray(features).tolist() == [[-1.0, 0.0, 0.0, 1.0]]
