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
from paths import DEFAULT_REPORTS_ROOT
from reporting import generate_adapter_comparison_plots, markdown_plot_section


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
        "mean_delta_gap": (
            float(np.mean(subset_delta - full_delta)) if len(subset_delta) else None
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


def _load_json_if_valid(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _same_path(left: str | Path, right: str | Path) -> bool:
    return resolve_project_path(left).resolve() == resolve_project_path(right).resolve()


def find_run_summary_for_adapter(
    adapter_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any] | None:
    output_root = resolve_project_path(output_root)
    for summary_path in sorted((output_root / "runs").glob("*/summary.json")):
        summary = _load_json_if_valid(summary_path)
        if not summary:
            continue
        candidate = summary.get("adapter_dir")
        if candidate and _same_path(str(candidate), adapter_path):
            return summary
    return None


def _basename(path: Any) -> str:
    return Path(str(path or "")).name


def _selector_details(sampling: dict[str, Any] | None) -> dict[str, Any]:
    sampling = sampling or {}
    context = sampling.get("selector_context", {})
    if not isinstance(context, dict):
        context = {}
    pipeline = context.get("selector_feature_pipeline", {})
    if not isinstance(pipeline, dict):
        pipeline = {}
    projection = pipeline.get("projection", {})
    if not isinstance(projection, dict):
        projection = {}

    transformations = pipeline.get(
        "transformations",
        context.get("selector_transformations", ["identity"]),
    )
    if isinstance(transformations, str):
        transformations = [transformations]
    transformations = list(transformations or ["identity"])
    projection_name = projection.get(
        "name",
        context.get("selector_projection", "identity"),
    )
    timing = context.get("selector_timing", {})
    if not isinstance(timing, dict):
        timing = {}

    return {
        "selector": _basename(sampling.get("selector_path")) or "-",
        "max_examples": sampling.get("max_example"),
        "transformations": transformations,
        "projection": projection_name,
        "raw_feature_dim": pipeline.get("raw_feature_dim"),
        "transformed_feature_dim": pipeline.get("transformed_feature_dim"),
        "projection_dim": projection.get(
            "output_dim",
            context.get("selector_projection_components"),
        ),
        "selected_train_examples": sampling.get("selected_train_examples"),
        "original_train_examples": sampling.get("original_train_examples"),
        "selector_total_seconds": timing.get("total_seconds"),
        "feature_extraction_seconds": timing.get("feature_extraction_seconds"),
        "transformation_seconds": timing.get("transformation_seconds"),
        "projection_seconds": timing.get("projection_seconds"),
        "kmeans_seconds": timing.get("kmeans_seconds"),
    }


def infer_selection_method(
    sampling: dict[str, Any] | None,
    *,
    fallback: str | None = None,
) -> str:
    details = _selector_details(sampling)
    selector = str(details["selector"])
    transformations = {str(value) for value in details["transformations"]}
    projection = str(details["projection"])

    if selector == "random.py":
        return "random"
    if selector == "lora_ntk_kmeans.py":
        has_sign = "sign" in transformations
        has_projection = projection not in {"", "-", "identity", "None", "none"}
        if has_projection and has_sign:
            return f"kmeans+{projection}+sign"
        if has_projection:
            return f"kmeans+{projection}"
        if has_sign:
            return "kmeans+sign"
        return "kmeans"
    if fallback:
        return fallback
    if selector and selector != "-":
        return selector.removesuffix(".py")
    return "-"


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
        f"- Method: `{summary.get('method', '-')}`",
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
        f"| Adapter Score Spearman | {_format_float(adapter_score.get('spearman'))} |",
        f"| Adapter Score RMSE | {_format_float(adapter_score.get('rmse'))} |",
        f"| Adapter Score MAE | {_format_float(adapter_score.get('mae'))} |",
        f"| Mean Full Delta | {_format_float(summary.get('mean_full_score_delta'))} |",
        f"| Mean Subset Delta | {_format_float(summary.get('mean_subset_score_delta'))} |",
        f"| Mean Delta Gap | {_format_float(summary.get('mean_delta_gap'))} |",
    ]
    return "\n".join(lines) + "\n"


def run_adapter_comparison(
    *,
    base_model: str,
    data_dir: str | Path,
    full_adapter_path: str | Path,
    subset_adapter_path: str | Path,
    dataset_name: str | None = None,
    task: str | None = None,
    method: str | None = None,
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
    full_run = find_run_summary_for_adapter(full_adapter_path, output_root=output_root)
    subset_run = find_run_summary_for_adapter(
        subset_adapter_path,
        output_root=output_root,
    )
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
        "dataset_name": dataset_name
        or (subset_run or {}).get("dataset_name")
        or (full_run or {}).get("dataset_name"),
        "task": task or (subset_run or {}).get("task") or (full_run or {}).get("task"),
        "data_dir": str(data_dir),
        "split": split,
        "limit": limit,
        "seed": seed,
        "method": method
        or infer_selection_method((subset_run or {}).get("sampling")),
        "full_run_name": (full_run or {}).get("run_name"),
        "subset_run_name": (subset_run or {}).get("run_name"),
        "full_adapter_path": str(full_adapter_path),
        "subset_adapter_path": str(subset_adapter_path),
        "subset_sampling": (subset_run or {}).get("sampling"),
        "selector_details": _selector_details((subset_run or {}).get("sampling")),
        "scores_path": str(paths.scores_path),
        **summarize_adapter_comparison(rows),
    }

    write_jsonl(paths.scores_path, rows)
    write_json(paths.summary_path, summary)
    paths.report_path.write_text(render_adapter_comparison_report(summary))
    return paths, summary


def collect_adapter_comparison_summaries(
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, Any]]:
    output_root = resolve_project_path(output_root)
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(
        (output_root / "comparisons").glob("*/summary.json"),
        reverse=True,
    ):
        summary = _load_json_if_valid(summary_path)
        if not summary:
            continue
        summary["comparison_dir"] = str(summary_path.parent)
        summaries.append(summary)
    return summaries


def render_adapter_comparison_summary_report(
    summaries: list[dict[str, Any]],
    *,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    base_models: list[str] | None = None,
    plot_markdown: str = "",
) -> str:
    dataset_filter = {str(value) for value in datasets or []}
    method_filter = {str(value) for value in methods or []}
    model_filter = {str(value) for value in base_models or []}

    rows = []
    for summary in summaries:
        dataset = str(summary.get("dataset_name") or "-")
        method = str(summary.get("method") or "-")
        model = str(summary.get("base_model") or "-")
        if dataset_filter and dataset not in dataset_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        if model_filter and model not in model_filter:
            continue
        rows.append(summary)

    rows.sort(
        key=lambda item: (
            str(item.get("dataset_name") or "-"),
            str(item.get("base_model") or "-"),
            int((item.get("selector_details") or {}).get("max_examples") or 0),
            str(item.get("method") or "-"),
            str(item.get("created_at") or ""),
        )
    )

    lines = [
        "# Adapter Comparison Summary",
        "",
        "| Dataset | Base Model | Method | Max N | Raw Dim | Projection Dim | Selector Sec | Split | Examples | Delta Pearson | Delta RMSE | Sign Acc | Adapter Pearson | Adapter RMSE | Mean Delta Gap | Subset Run |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in rows:
        metrics = summary.get("metrics", {})
        delta = metrics.get("delta", {})
        adapter_score = metrics.get("adapter_score", {})
        selector_details = summary.get("selector_details", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary.get("dataset_name") or "-"),
                    str(summary.get("base_model") or "-").rsplit("/", 1)[-1],
                    str(summary.get("method") or "-"),
                    str(selector_details.get("max_examples") or "-"),
                    str(selector_details.get("raw_feature_dim") or "-"),
                    str(selector_details.get("projection_dim") or "-"),
                    _format_float(selector_details.get("selector_total_seconds")),
                    str(summary.get("split") or "-"),
                    str(summary.get("num_examples") or "-"),
                    _format_float(delta.get("pearson")),
                    _format_float(delta.get("rmse")),
                    _format_float(delta.get("sign_accuracy")),
                    _format_float(adapter_score.get("pearson")),
                    _format_float(adapter_score.get("rmse")),
                    _format_float(summary.get("mean_delta_gap")),
                    f"`{summary.get('subset_run_name') or '-'}`",
                ]
            )
            + " |"
        )

    if len(lines) == 4:
        lines.extend(["", "_No adapter comparisons found yet._"])
    report = "\n".join(lines) + "\n"
    if plot_markdown:
        report += plot_markdown
    return report


def make_adapter_comparison_report(
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
    output_path: str | Path = f"{DEFAULT_REPORTS_ROOT}/adapter_comparisons.md",
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    base_models: list[str] | None = None,
    plots: bool = True,
    assets_dir: str | Path | None = None,
) -> Path:
    summaries = collect_adapter_comparison_summaries(output_root=output_root)
    output_path = resolve_project_path(output_path)
    plot_markdown = ""
    if plots:
        plot_dir = (
            resolve_project_path(assets_dir)
            if assets_dir is not None
            else output_path.parent / "assets" / "adapter_comparisons"
        )
        artifacts = generate_adapter_comparison_plots(summaries, plot_dir)
        plot_markdown = markdown_plot_section(
            artifacts,
            report_path=output_path,
        )
    report = render_adapter_comparison_summary_report(
        summaries,
        datasets=datasets,
        methods=methods,
        base_models=base_models,
        plot_markdown=plot_markdown,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path
