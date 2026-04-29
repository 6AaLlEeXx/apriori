from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
import gc
import json

import numpy as np

from kernel.data import PairRecord, load_pair_split, maybe_subset_pairs
from kernel.metrics import evaluate_predictions
from kernel.scoring import ModelScorer, PairTokens
from paths import DEFAULT_RESULTS_ROOT, resolve_project_path


class Scorer(Protocol):
    def score_record(self, record: PairRecord) -> tuple[float, PairTokens]: ...


ScorerFactory = Callable[[str, str | None], Scorer]


@dataclass(frozen=True)
class AdapterComparisonPaths:
    output_dir: Path
    scores_path: Path
    summary_path: Path
    report_path: Path


def default_scorer_factory(base_model: str, adapter_path: str | None) -> Scorer:
    return ModelScorer(base_model, adapter_path=adapter_path, lazy=True)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _default_output_dir(
    *,
    output_root: str | Path,
    split: str,
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return resolve_project_path(output_root) / "comparisons" / f"{timestamp}__{split}"


def prepare_adapter_comparison_paths(
    output_dir: str | Path | None,
    *,
    split: str,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> AdapterComparisonPaths:
    resolved_output_dir = (
        resolve_project_path(output_dir)
        if output_dir is not None
        else _default_output_dir(output_root=output_root, split=split)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=False)
    return AdapterComparisonPaths(
        output_dir=resolved_output_dir,
        scores_path=resolved_output_dir / "scores.jsonl",
        summary_path=resolved_output_dir / "summary.json",
        report_path=resolved_output_dir / "report.md",
    )


def _score_records(
    scorer: Scorer,
    records: list[PairRecord],
) -> tuple[list[float], list[PairTokens]]:
    scores: list[float] = []
    tokens: list[PairTokens] = []
    for record in records:
        score, pair_tokens = scorer.score_record(record)
        scores.append(float(score))
        tokens.append(pair_tokens)
    return scores, tokens


def build_adapter_comparison_rows(
    records: list[PairRecord],
    *,
    base_scores: list[float],
    full_scores: list[float],
    subset_scores: list[float],
    tokens: list[PairTokens],
) -> list[dict[str, Any]]:
    if not (
        len(records)
        == len(base_scores)
        == len(full_scores)
        == len(subset_scores)
        == len(tokens)
    ):
        raise ValueError("Comparison inputs must have the same length.")

    rows: list[dict[str, Any]] = []
    for record, base_score, full_score, subset_score, pair_tokens in zip(
        records,
        base_scores,
        full_scores,
        subset_scores,
        tokens,
    ):
        full_delta = float(full_score - base_score)
        subset_delta = float(subset_score - base_score)
        rows.append(
            {
                "pair_id": record.pair_id,
                "split": record.split,
                "base_score": float(base_score),
                "full_adapter_score": float(full_score),
                "subset_adapter_score": float(subset_score),
                "full_score_delta": full_delta,
                "subset_score_delta": subset_delta,
                "delta_error": subset_delta - full_delta,
                "delta_sign_match": bool(
                    np.sign(full_delta) == np.sign(subset_delta)
                ),
                "prompt_offset": pair_tokens.prompt_offset,
                "sequence_length": pair_tokens.sequence_length,
                "completion_targets": pair_tokens.completion_targets,
            }
        )
    return rows


def summarize_adapter_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_delta = np.asarray(
        [row["full_score_delta"] for row in rows],
        dtype=np.float64,
    )
    subset_delta = np.asarray(
        [row["subset_score_delta"] for row in rows],
        dtype=np.float64,
    )
    full_scores = np.asarray(
        [row["full_adapter_score"] for row in rows],
        dtype=np.float64,
    )
    subset_scores = np.asarray(
        [row["subset_adapter_score"] for row in rows],
        dtype=np.float64,
    )
    return {
        "num_examples": len(rows),
        "metrics": evaluate_predictions(
            true_delta=full_delta,
            pred_delta=subset_delta,
            true_adapter_score=full_scores,
            pred_adapter_score=subset_scores,
        ),
        "mean_full_score_delta": (
            float(np.mean(full_delta)) if len(full_delta) else None
        ),
        "mean_subset_score_delta": (
            float(np.mean(subset_delta)) if len(subset_delta) else None
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def render_adapter_comparison_report(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    delta = metrics.get("delta", {})
    adapter_score = metrics.get("adapter_score", {})
    lines = [
        "# Adapter Comparison",
        "",
        f"- Base model: `{summary.get('base_model', '-')}`",
        f"- Split: `{summary.get('split', '-')}`",
        f"- Examples: `{summary.get('num_examples', 0)}`",
        f"- Full adapter: `{summary.get('full_adapter_path', '-')}`",
        f"- Subset adapter: `{summary.get('subset_adapter_path', '-')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Delta Pearson | {_format_float(delta.get('pearson'))} |",
        f"| Delta Spearman | {_format_float(delta.get('spearman'))} |",
        f"| Delta RMSE | {_format_float(delta.get('rmse'))} |",
        f"| Delta MAE | {_format_float(delta.get('mae'))} |",
        f"| Delta Sign Accuracy | {_format_float(delta.get('sign_accuracy'))} |",
        f"| Adapter Score Pearson | {_format_float(adapter_score.get('pearson'))} |",
        f"| Adapter Score RMSE | {_format_float(adapter_score.get('rmse'))} |",
        f"| Mean Full Delta | {_format_float(summary.get('mean_full_score_delta'))} |",
        f"| Mean Subset Delta | {_format_float(summary.get('mean_subset_score_delta'))} |",
    ]
    return "\n".join(lines) + "\n"


def run_adapter_comparison(
    *,
    base_model: str,
    data_dir: str | Path,
    full_adapter_path: str | Path,
    subset_adapter_path: str | Path,
    split: str = "test",
    limit: int = 512,
    seed: int = 42,
    output_dir: str | Path | None = None,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
    scorer_factory: ScorerFactory = default_scorer_factory,
) -> tuple[AdapterComparisonPaths, dict[str, Any]]:
    data_dir = resolve_project_path(data_dir)
    full_adapter_path = resolve_project_path(full_adapter_path)
    subset_adapter_path = resolve_project_path(subset_adapter_path)
    records = load_pair_split(data_dir / f"{split}.jsonl", split=split)
    records = maybe_subset_pairs(records, limit=limit, seed=seed)
    if not records:
        raise ValueError(f"No records selected for adapter comparison: {data_dir}")

    paths = prepare_adapter_comparison_paths(
        output_dir,
        split=split,
        output_root=output_root,
    )

    base_scorer = scorer_factory(base_model, None)
    base_scores, tokens = _score_records(base_scorer, records)
    del base_scorer
    gc.collect()

    full_scorer = scorer_factory(base_model, str(full_adapter_path))
    full_scores, _ = _score_records(full_scorer, records)
    del full_scorer
    gc.collect()

    subset_scorer = scorer_factory(base_model, str(subset_adapter_path))
    subset_scores, _ = _score_records(subset_scorer, records)
    del subset_scorer
    gc.collect()

    rows = build_adapter_comparison_rows(
        records,
        base_scores=base_scores,
        full_scores=full_scores,
        subset_scores=subset_scores,
        tokens=tokens,
    )
    summary = {
        "created_at": _now_iso(),
        "base_model": base_model,
        "data_dir": str(data_dir),
        "split": split,
        "limit": limit,
        "seed": seed,
        "full_adapter_path": str(full_adapter_path),
        "subset_adapter_path": str(subset_adapter_path),
        "scores_path": str(paths.scores_path),
        **summarize_adapter_comparison(rows),
    }

    write_jsonl(paths.scores_path, rows)
    write_json(paths.summary_path, summary)
    paths.report_path.write_text(render_adapter_comparison_report(summary))
    return paths, summary
