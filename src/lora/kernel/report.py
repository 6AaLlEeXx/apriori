from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from lora.paths import (
    DEFAULT_KERNEL_RESULTS_ROOT,
    DEFAULT_REPORTS_ROOT,
    resolve_project_path,
)


def _dataset_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


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
) -> dict[tuple[str, str], dict[str, Any]]:
    dataset_filter = {_dataset_key(value) for value in datasets or []}
    backend_filter = {str(value) for value in backends or []}
    selected: dict[tuple[str, str], dict[str, Any]] = {}

    for summary in summaries:
        dataset_name = str(summary.get("dataset_name", ""))
        dataset_key = _dataset_key(dataset_name)
        backend = str(summary.get("backend", ""))
        if dataset_filter and dataset_key not in dataset_filter:
            continue
        if backend_filter and backend not in backend_filter:
            continue
        key = (dataset_key, backend)
        previous = selected.get(key)
        if previous is None or _selection_key(summary) > _selection_key(previous):
            selected[key] = summary

    return selected


def render_kernel_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# LoRA Kernel Runs",
        "",
        "| Run | Dataset | Backend | Train N | Feature Dim | Valid Delta Pearson | Test Delta Pearson |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + f"`{row.get('run_name', '-')}`"
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
    datasets: list[str],
    backends: list[str],
) -> str:
    selected = select_primary_kernel_runs(
        summaries=summaries,
        datasets=datasets,
        backends=backends,
    )

    lines = [
        "# Kernel Comparison",
        "",
        "Selection rule: for each dataset/backend pair, choose the run with the largest train split; break ties by higher test delta Pearson, then by newer run name.",
        "",
        "| Dataset | Backend | Train N | Feature Dim | Test Delta Pearson | Test Delta Spearman | Test RMSE | Test Sign Acc | Test Adapter Pearson | Run |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for dataset in datasets:
        dataset_key = _dataset_key(dataset)
        for backend in backends:
            summary = selected.get((dataset_key, backend))
            if summary is None:
                lines.append(
                    "| "
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

    return "\n".join(lines) + "\n"


def make_kernel_comparison_report(
    output_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
    output_path: str | Path = f"{DEFAULT_REPORTS_ROOT}/lora_kernel_comparison.md",
    datasets: list[str] | None = None,
    backends: list[str] | None = None,
) -> Path:
    datasets = datasets or ["dolly", "gsm8k", "sql_create_context"]
    backends = backends or ["frozen_pair", "lora_ntk"]
    rows = collect_kernel_run_summaries(output_root=output_root)
    report = render_kernel_comparison_report(
        summaries=rows,
        datasets=datasets,
        backends=backends,
    )
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path
