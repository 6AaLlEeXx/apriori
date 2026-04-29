from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pytest

from kernel.adapter_compare import (
    build_adapter_comparison_rows,
    render_adapter_comparison_report,
    run_adapter_comparison,
    summarize_adapter_comparison,
)
from kernel.data import PairRecord
from kernel.scoring import PairTokens


class FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score_record(self, record: PairRecord) -> tuple[float, PairTokens]:
        return (
            self.scores[record.pair_id],
            PairTokens(
                token_ids=np.asarray([], dtype=np.int32),
                prompt_offset=2,
                sequence_length=6,
                completion_targets=4,
            ),
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_summarize_adapter_comparison_measures_delta_agreement() -> None:
    records = [
        PairRecord("test-000000", "test", "p0", "c0"),
        PairRecord("test-000001", "test", "p1", "c1"),
    ]
    tokens = [
        PairTokens(
            np.asarray([], dtype=np.int32),
            prompt_offset=1,
            sequence_length=4,
            completion_targets=3,
        ),
        PairTokens(
            np.asarray([], dtype=np.int32),
            prompt_offset=1,
            sequence_length=4,
            completion_targets=3,
        ),
    ]
    rows = build_adapter_comparison_rows(
        records,
        base_scores=[0.0, 0.0],
        full_scores=[1.0, -1.0],
        subset_scores=[0.8, -0.7],
        tokens=tokens,
    )

    summary = summarize_adapter_comparison(rows)

    assert summary["num_examples"] == 2
    assert summary["metrics"]["delta"]["sign_accuracy"] == 1.0
    assert summary["metrics"]["delta"]["pearson"] == pytest.approx(1.0)
    assert summary["mean_full_score_delta"] == pytest.approx(0.0)
    assert summary["mean_subset_score_delta"] == pytest.approx(0.05)


def test_run_adapter_comparison_writes_scores_summary_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_jsonl(
        data_dir / "test.jsonl",
        [
            {"prompt": "p0", "completion": "c0"},
            {"prompt": "p1", "completion": "c1"},
        ],
    )
    full_adapter = tmp_path / "adapters" / "full"
    subset_adapter = tmp_path / "adapters" / "subset"
    full_adapter.mkdir(parents=True)
    subset_adapter.mkdir(parents=True)

    def scorer_factory(_base_model: str, adapter_path: str | None) -> FakeScorer:
        if adapter_path is None:
            return FakeScorer({"test-000000": 0.0, "test-000001": 0.0})
        if Path(adapter_path) == full_adapter:
            return FakeScorer({"test-000000": 1.0, "test-000001": -1.0})
        if Path(adapter_path) == subset_adapter:
            return FakeScorer({"test-000000": 0.9, "test-000001": -0.8})
        raise AssertionError(f"unexpected adapter path: {adapter_path}")

    output_dir = tmp_path / "comparison"
    paths, summary = run_adapter_comparison(
        base_model="models/base",
        data_dir=data_dir,
        full_adapter_path=full_adapter,
        subset_adapter_path=subset_adapter,
        split="test",
        limit=0,
        output_dir=output_dir,
        scorer_factory=scorer_factory,
    )

    score_rows = [
        json.loads(line) for line in paths.scores_path.read_text().splitlines()
    ]
    saved_summary = json.loads(paths.summary_path.read_text())
    report = paths.report_path.read_text()

    assert paths.output_dir == output_dir
    assert len(score_rows) == 2
    assert score_rows[0]["full_score_delta"] == 1.0
    assert score_rows[0]["subset_score_delta"] == 0.9
    assert summary["metrics"]["delta"]["sign_accuracy"] == 1.0
    assert saved_summary["num_examples"] == 2
    assert "# Adapter Comparison" in report


def test_render_adapter_comparison_report_includes_core_metrics() -> None:
    report = render_adapter_comparison_report(
        {
            "base_model": "model",
            "split": "test",
            "num_examples": 2,
            "full_adapter_path": "full",
            "subset_adapter_path": "subset",
            "metrics": {
                "delta": {
                    "pearson": 0.9,
                    "spearman": 0.8,
                    "rmse": 0.1,
                    "mae": 0.05,
                    "sign_accuracy": 1.0,
                },
                "adapter_score": {"pearson": 0.95, "rmse": 0.2},
            },
        }
    )

    assert "Delta Pearson" in report
    assert "0.9000" in report
    assert "Sign Accuracy" in report
