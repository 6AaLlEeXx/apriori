from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from kernel.metrics import mae, pearson_corr, rmse, sign_accuracy, spearman_corr
from paths import (
    DEFAULT_KERNEL_RESULTS_ROOT,
    DEFAULT_REPORTS_ROOT,
    resolve_project_path,
)
from reporting import generate_kernel_prediction_plots, markdown_plot_section


def _dataset_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _model_key(value: Any) -> str:
    return str(value or "-").strip()


def _format_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _nested(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _delta_metrics(true_delta: list[float], pred_delta: list[float]) -> dict[str, float]:
    true_array = np.asarray(true_delta, dtype=np.float64)
    pred_array = np.asarray(pred_delta, dtype=np.float64)
    return {
        "pearson": pearson_corr(true_array, pred_array),
        "spearman": spearman_corr(true_array, pred_array),
        "rmse": rmse(true_array, pred_array),
        "mae": mae(true_array, pred_array),
        "sign_accuracy": sign_accuracy(true_array, pred_array),
    }


def _prediction_deltas(rows: list[dict[str, Any]]) -> list[float]:
    return [
        number
        for row in rows
        for number in [_number(row.get("score_delta"))]
        if number is not None
    ]


def _attach_kernel_baseline_metrics(summary: dict[str, Any]) -> None:
    eval_payload = summary.get("eval")
    if not isinstance(eval_payload, dict):
        eval_payload = {}
        summary["eval"] = eval_payload

    run_dir = Path(str(summary.get("run_dir") or ""))
    predictions_dir = run_dir / "predictions"
    train_rows: list[dict[str, Any]] = []
    try:
        train_rows = _read_jsonl(predictions_dir / "train.jsonl")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    train_deltas = _prediction_deltas(train_rows)

    baseline_payload: dict[str, Any] = {}
    if isinstance(eval_payload.get("baseline"), dict):
        baseline_payload.update(eval_payload["baseline"])
    if train_deltas:
        baseline_payload.setdefault("strategy", "train_mean_delta")
        baseline_payload.setdefault("train_mean_delta", sum(train_deltas) / len(train_deltas))
    baseline_mean = _number(baseline_payload.get("train_mean_delta"))

    for split in ("train", "valid", "test"):
        split_eval = eval_payload.get(split)
        if not isinstance(split_eval, dict):
            split_eval = {}
            eval_payload[split] = split_eval

        existing_baseline = split_eval.get("baseline")
        if isinstance(existing_baseline, dict):
            baseline_payload[split] = existing_baseline
            continue

        if baseline_mean is None:
            continue
        try:
            split_rows = _read_jsonl(predictions_dir / f"{split}.jsonl")
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        split_deltas = _prediction_deltas(split_rows)
        if not split_deltas:
            continue
        split_baseline = {
            "delta": _delta_metrics(
                split_deltas,
                [baseline_mean for _ in split_deltas],
            )
        }
        split_eval["baseline"] = split_baseline
        baseline_payload[split] = split_baseline

    if baseline_payload:
        summary["baseline"] = baseline_payload
        for split in ("valid", "test"):
            baseline_rmse = _number(
                _nested(baseline_payload, f"{split}.delta.rmse")
            )
            krr_rmse = _number(_nested(eval_payload, f"{split}.delta.rmse"))
            if baseline_rmse is not None:
                summary[f"{split}_baseline_delta_rmse"] = baseline_rmse
            if baseline_rmse is not None and krr_rmse is not None:
                summary[f"{split}_delta_rmse_gain"] = baseline_rmse - krr_rmse


def collect_kernel_run_summaries(
    output_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
) -> list[dict[str, Any]]:
    output_root = resolve_project_path(output_root)
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(
        (output_root / "runs").glob("*/summary.json"), reverse=True
    ):
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        summary["run_dir"] = str(summary_path.parent)
        eval_path = summary_path.parent / "eval.json"
        if eval_path.exists():
            try:
                summary["eval"] = json.loads(eval_path.read_text())
            except json.JSONDecodeError:
                summary["eval"] = {}
        _attach_kernel_baseline_metrics(summary)
        summaries.append(summary)
    return summaries


def _train_size(summary: dict[str, Any]) -> int:
    split_sizes = summary.get("split_sizes", {})
    return int(split_sizes.get("train", 0) or 0)


def _selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _train_size(summary),
        float(summary.get("test_delta_pearson") or float("-inf")),
        str(summary.get("run_name") or ""),
    )


def select_primary_kernel_runs(
    summaries: list[dict[str, Any]],
    datasets: list[str] | None = None,
    backends: list[str] | None = None,
    base_models: list[str] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    dataset_filter = {_dataset_key(value) for value in datasets or []}
    backend_filter = {str(value) for value in backends or []}
    model_filter = {_model_key(value) for value in base_models or []}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}

    for summary in summaries:
        if str(summary.get("status", "completed")).lower() != "completed":
            continue
        dataset_name = str(summary.get("dataset_name", ""))
        dataset_key = _dataset_key(dataset_name)
        backend = str(summary.get("backend", ""))
        model = _model_key(summary.get("base_model"))
        if dataset_filter and dataset_key not in dataset_filter:
            continue
        if backend_filter and backend not in backend_filter:
            continue
        if model_filter and model not in model_filter:
            continue
        key = (model, dataset_key, backend)
        previous = selected.get(key)
        if previous is None or _selection_key(summary) > _selection_key(previous):
            selected[key] = summary

    return selected


def _short_model_name(base_model: Any) -> str:
    if not base_model:
        return "-"
    return str(base_model).rsplit("/", 1)[-1]


def render_kernel_report(
    summaries: list[dict[str, Any]],
    *,
    plot_markdown: str = "",
) -> str:
    lines = [
        "# LoRA Kernel Runs",
        "",
        "| Run | Base Model | Dataset | Backend | Train N | Feature Dim | Valid Delta Pearson | Test Delta Pearson | Test Delta RMSE | Baseline RMSE | RMSE Gain |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        test_rmse = _nested(row, "eval.test.delta.rmse")
        lines.append(
            "| "
            + f"`{row.get('run_name', '-')}`"
            + " | "
            + _short_model_name(row.get("base_model"))
            + " | "
            + str(row.get("dataset_name", "-"))
            + " | "
            + str(row.get("backend", "-"))
            + " | "
            + str(_train_size(row))
            + " | "
            + str(row.get("feature_dim", "-"))
            + " | "
            + _format_float(row.get("valid_delta_pearson"))
            + " | "
            + _format_float(row.get("test_delta_pearson"))
            + " | "
            + _format_float(test_rmse)
            + " | "
            + _format_float(row.get("test_baseline_delta_rmse"))
            + " | "
            + _format_float(row.get("test_delta_rmse_gain"))
            + " |"
        )

    if len(lines) == 4:
        lines.extend(["", "_No tracked kernel runs found yet._"])
    report = "\n".join(lines) + "\n"
    if plot_markdown:
        report += plot_markdown
    return report


def make_kernel_report(
    output_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
    output_path: str | Path = f"{DEFAULT_REPORTS_ROOT}/lora_kernel_runs.md",
    plots: bool = True,
    assets_dir: str | Path | None = None,
) -> Path:
    rows = collect_kernel_run_summaries(output_root=output_root)
    output_path = resolve_project_path(output_path)
    plot_markdown = ""
    if plots:
        plot_dir = (
            resolve_project_path(assets_dir)
            if assets_dir is not None
            else output_path.parent / "assets" / "kernel"
        )
        artifacts = generate_kernel_prediction_plots(rows, plot_dir)
        plot_markdown = markdown_plot_section(
            artifacts,
            report_path=output_path,
        )
    report = render_kernel_report(rows, plot_markdown=plot_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path


def render_kernel_comparison_report(
    summaries: list[dict[str, Any]],
    datasets: list[str] | None = None,
    backends: list[str] | None = None,
    base_models: list[str] | None = None,
) -> str:
    selected = select_primary_kernel_runs(
        summaries=summaries,
        datasets=datasets,
        backends=backends,
        base_models=base_models,
    )

    lines = [
        "# Kernel Comparison",
        "",
        "Selection rule: for each base-model/dataset/backend group, choose the run with the largest train split; break ties by higher test delta Pearson, then by newer run name.",
        "",
        "| Base Model | Dataset | Backend | Train N | Feature Dim | Test Delta Pearson | Test Delta Spearman | Test RMSE | Baseline RMSE | RMSE Gain | Test Sign Acc | Test Adapter Pearson | Run |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    explicit_grid = bool(datasets or backends or base_models)
    if explicit_grid:
        model_values = base_models or sorted({key[0] for key in selected})
        dataset_values = datasets or sorted(
            {
                str(summary.get("dataset_name", key[1]))
                for key, summary in selected.items()
            }
        )
        backend_values = backends or sorted({key[2] for key in selected})
        if not model_values:
            model_values = ["-"]
        if not dataset_values:
            dataset_values = ["-"]
        if not backend_values:
            backend_values = ["-"]
        render_keys = [
            (_model_key(model), dataset, _dataset_key(dataset), backend)
            for model in model_values
            for dataset in dataset_values
            for backend in backend_values
        ]
    else:
        render_keys = [
            (
                key[0],
                str(summary.get("dataset_name", key[1])),
                key[1],
                key[2],
            )
            for key, summary in selected.items()
        ]
        render_keys.sort(
            key=lambda item: (_short_model_name(item[0]), item[1], item[3])
        )

    for model, dataset, dataset_key, backend in render_keys:
        summary = selected.get((model, dataset_key, backend))
        if summary is None:
            lines.append(
                "| "
                + _short_model_name(model)
                + " | "
                + dataset
                + " | "
                + backend
                + " | - | - | - | - | - | - | - | - | - | missing |"
            )
            continue

        eval_payload = summary.get("eval", {})
        test_metrics = eval_payload.get("test", {})
        delta = test_metrics.get("delta", {})
        adapter = test_metrics.get("adapter_score", {})
        baseline_rmse = summary.get("test_baseline_delta_rmse")
        rmse_gain = summary.get("test_delta_rmse_gain")
        lines.append(
            "| "
            + _short_model_name(summary.get("base_model"))
            + " | "
            + str(summary.get("dataset_name", dataset))
            + " | "
            + backend
            + " | "
            + str(_train_size(summary))
            + " | "
            + str(summary.get("feature_dim", "-"))
            + " | "
            + _format_float(delta.get("pearson"))
            + " | "
            + _format_float(delta.get("spearman"))
            + " | "
            + _format_float(delta.get("rmse"))
            + " | "
            + _format_float(baseline_rmse)
            + " | "
            + _format_float(rmse_gain)
            + " | "
            + _format_float(delta.get("sign_accuracy"))
            + " | "
            + _format_float(adapter.get("pearson"))
            + " | "
            + f"`{summary.get('run_name', '-')}`"
            + " |"
        )

    if len(lines) == 6:
        lines.extend(["", "_No completed kernel runs found yet._"])

    return "\n".join(lines) + "\n"


def make_kernel_comparison_report(
    output_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
    output_path: str | Path = f"{DEFAULT_REPORTS_ROOT}/lora_kernel_comparison.md",
    datasets: list[str] | None = None,
    backends: list[str] | None = None,
    base_models: list[str] | None = None,
) -> Path:
    rows = collect_kernel_run_summaries(output_root=output_root)
    report = render_kernel_comparison_report(
        summaries=rows,
        datasets=datasets,
        backends=backends,
        base_models=base_models,
    )
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path
