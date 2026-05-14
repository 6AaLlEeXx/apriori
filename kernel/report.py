from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from kernel.metrics import mae, pearson_corr, rmse, sign_accuracy, spearman_corr
from paths import DEFAULT_KERNEL_RESULTS_ROOT, resolve_project_path


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
    try:
        train_rows = _read_jsonl(predictions_dir / "train.jsonl")
    except (FileNotFoundError, json.JSONDecodeError):
        train_rows = []
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
    kernel_results_root: str | Path = DEFAULT_KERNEL_RESULTS_ROOT,
) -> list[dict[str, Any]]:
    kernel_results_root = resolve_project_path(kernel_results_root)
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(
        (kernel_results_root / "runs").glob("*/summary.json"),
        reverse=True,
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
