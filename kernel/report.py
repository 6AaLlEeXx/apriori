from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from paths import (
    DEFAULT_KERNEL_RESULTS_ROOT,
    DEFAULT_REPORTS_ROOT,
    resolve_project_path,
)


def _dataset_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _model_key(value: Any) -> str:
    return str(value or "-").strip()


def _format_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


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
        eval_path = summary_path.parent / "eval.json"
        if eval_path.exists():
            try:
                summary["eval"] = json.loads(eval_path.read_text())
            except json.JSONDecodeError:
                summary["eval"] = {}
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


def render_kernel_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# LoRA Kernel Runs",
        "",
        "| Run | Base Model | Dataset | Backend | Train N | Feature Dim | Valid Delta Pearson | Test Delta Pearson |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
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
            + " |"
        )

    if len(lines) == 4:
        lines.extend(["", "_No tracked kernel runs found yet._"])
    return "\n".join(lines) + "\n"


def make_kernel_report(
    output_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
    output_path: str | Path = f"{DEFAULT_REPORTS_ROOT}/lora_kernel_runs.md",
) -> Path:
    rows = collect_kernel_run_summaries(output_root=output_root)
    report = render_kernel_report(rows)
    output_path = resolve_project_path(output_path)
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
        "| Base Model | Dataset | Backend | Train N | Feature Dim | Test Delta Pearson | Test Delta Spearman | Test RMSE | Test Sign Acc | Test Adapter Pearson | Run |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                + " | - | - | - | - | - | - | - | missing |"
            )
            continue

        eval_payload = summary.get("eval", {})
        test_metrics = eval_payload.get("test", {})
        delta = test_metrics.get("delta", {})
        adapter = test_metrics.get("adapter_score", {})
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
