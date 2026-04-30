from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pytest

from kernel.adapter_compare import (
    build_adapter_comparison_rows,
    collect_adapter_comparison_summaries,
    infer_selection_method,
    make_adapter_comparison_report,
    render_adapter_comparison_report,
    render_adapter_comparison_summary_report,
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
    assert summary["mean_delta_gap"] == pytest.approx(0.05)


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
    run_root = tmp_path / "results"
    full_run = run_root / "runs" / "full"
    subset_run = run_root / "runs" / "subset"
    full_run.mkdir(parents=True)
    subset_run.mkdir(parents=True)
    (full_run / "summary.json").write_text(
        json.dumps(
            {
                "run_name": "full",
                "dataset_name": "toy",
                "task": "generic",
                "adapter_dir": str(full_adapter),
            }
        )
    )
    (subset_run / "summary.json").write_text(
        json.dumps(
            {
                "run_name": "subset",
                "dataset_name": "toy",
                "task": "generic",
                "adapter_dir": str(subset_adapter),
                "sampling": {
                    "selector_path": "selectors/lora_ntk_kmeans.py",
                    "max_example": 2,
                    "selected_train_examples": 2,
                    "selector_context": {
                        "selector_transformations": ["sign"],
                        "selector_projection": "sparse_random",
                        "selector_feature_pipeline": {
                            "transformations": ["sign"],
                            "projection": {
                                "name": "sparse_random",
                                "output_dim": 4,
                            },
                        },
                    },
                },
            }
        )
    )

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
        output_root=run_root,
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
    assert summary["dataset_name"] == "toy"
    assert summary["method"] == "kmeans+sparse_random+sign"
    assert summary["selector_details"]["projection_dim"] == 4
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


def test_infer_selection_method_covers_experiment_variants() -> None:
    assert (
        infer_selection_method({"selector_path": "selectors/random.py"})
        == "random"
    )
    assert (
        infer_selection_method(
            {
                "selector_path": "selectors/lora_ntk_kmeans.py",
                "selector_context": {
                    "selector_transformations": ["identity"],
                    "selector_projection": "identity",
                },
            }
        )
        == "kmeans"
    )
    assert (
        infer_selection_method(
            {
                "selector_path": "selectors/lora_ntk_kmeans.py",
                "selector_context": {
                    "selector_transformations": ["sign"],
                    "selector_projection": "identity",
                },
            }
        )
        == "kmeans+sign"
    )
    assert (
        infer_selection_method(
            {
                "selector_path": "selectors/lora_ntk_kmeans.py",
                "selector_context": {
                    "selector_transformations": ["identity"],
                    "selector_projection": "sparse_random",
                },
            }
        )
        == "kmeans+sparse_random"
    )
    assert (
        infer_selection_method(
            {
                "selector_path": "selectors/lora_ntk_kmeans.py",
                "selector_context": {
                    "selector_transformations": ["sign"],
                    "selector_projection": "sparse_random",
                },
            }
        )
        == "kmeans+sparse_random+sign"
    )


def test_make_adapter_comparison_report_aggregates_methods(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    first = output_root / "comparisons" / "first"
    second = output_root / "comparisons" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "summary.json").write_text(
        json.dumps(
            {
                "dataset_name": "toy",
                "base_model": "models/base",
                "method": "random",
                "split": "test",
                "num_examples": 2,
                "subset_run_name": "toy-random",
                "selector_details": {"max_examples": 2},
                "metrics": {
                    "delta": {
                        "pearson": 0.2,
                        "rmse": 0.5,
                        "sign_accuracy": 0.5,
                    },
                    "adapter_score": {"pearson": 0.3, "rmse": 0.4},
                },
                "mean_delta_gap": 0.1,
            }
        )
    )
    (second / "summary.json").write_text(
        json.dumps(
            {
                "dataset_name": "toy",
                "base_model": "models/base",
                "method": "kmeans+sparse_random+sign",
                "split": "test",
                "num_examples": 2,
                "subset_run_name": "toy-kmeans",
                "selector_details": {"max_examples": 2, "projection_dim": 4},
                "metrics": {
                    "delta": {
                        "pearson": 0.9,
                        "rmse": 0.1,
                        "sign_accuracy": 1.0,
                    },
                    "adapter_score": {"pearson": 0.95, "rmse": 0.2},
                },
                "mean_delta_gap": 0.01,
            }
        )
    )

    output_path = tmp_path / "report.md"
    make_adapter_comparison_report(
        output_root=output_root,
        output_path=output_path,
    )

    summaries = collect_adapter_comparison_summaries(output_root=output_root)
    report = output_path.read_text()
    rendered = render_adapter_comparison_summary_report(summaries)

    assert len(summaries) == 2
    assert "random" in report
    assert "kmeans+sparse_random+sign" in report
    assert "Adapter RMSE" in rendered
    assert "Visualizations" in report
    assert (tmp_path / "assets" / "adapter_comparisons" / "method_delta_pearson.svg").exists()
    assert (tmp_path / "assets" / "adapter_comparisons" / "method_adapter_rmse.svg").exists()
