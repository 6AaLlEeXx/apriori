from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import re


MetricGetter = Callable[[dict[str, Any]], float | None]


@dataclass(frozen=True)
class KernelPlotData:
    experiment_names: list[str]
    train_sizes: list[int]
    features: list[str]
    selected: dict[tuple[str, int, str], dict[str, Any]]
    split: str


def slug(value: Any) -> str:
    text = str(value or "plot").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "plot"


def nested(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def kernel_feature_label(summary: dict[str, Any]) -> str:
    direct = summary.get("feature_transform_label")
    if direct:
        return str(direct)
    transform = summary.get("feature_transform")
    if isinstance(transform, dict) and transform.get("label"):
        return str(transform["label"])
    eval_payload = summary.get("eval")
    if isinstance(eval_payload, dict):
        eval_transform = eval_payload.get("feature_transform")
        if isinstance(eval_transform, dict) and eval_transform.get("label"):
            return str(eval_transform["label"])
    return "raw"


def kernel_feature_key(summary: dict[str, Any]) -> str:
    label = kernel_feature_label(summary)
    normalized = label.lower().strip()
    if normalized in {"", "identity", "none", "raw"}:
        return "raw"
    if normalized.startswith("thresholded_sign"):
        return "thresholded_sign"
    if normalized.startswith("sign"):
        return "sign"
    return normalized


def kernel_train_size(summary: dict[str, Any]) -> int | None:
    split_sizes = summary.get("split_sizes") or {}
    if isinstance(split_sizes, dict):
        value = number(split_sizes.get("train"))
        if value is not None:
            return int(value)
    value = number(summary.get("train_limit"))
    return int(value) if value is not None else None


def prediction_points(
    summary: dict[str, Any] | None,
    *,
    split: str,
) -> list[tuple[float, float]]:
    if summary is None:
        return []
    run_dir = summary.get("run_dir")
    if not run_dir:
        return []
    try:
        rows = read_jsonl(Path(str(run_dir)) / "predictions" / f"{split}.jsonl")
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    points: list[tuple[float, float]] = []
    for row in rows:
        actual = number(row.get("score_delta"))
        predicted = number(row.get("predicted_score_delta"))
        if actual is not None and predicted is not None:
            points.append((actual, predicted))
    return points


def feature_label(feature: str) -> str:
    labels = {
        "raw": "LoRA-NTK",
        "sign": "Sign LoRA-NTK",
        "thresholded_sign": "Thresholded-sign LoRA-NTK",
    }
    return labels.get(feature, feature.replace("_", " "))


def rmse_gain(summary: dict[str, Any]) -> float | None:
    value = number(summary.get("test_delta_rmse_gain"))
    if value is not None:
        return value
    krr = number(nested(summary, "eval.test.delta.rmse"))
    baseline = number(nested(summary, "baseline.test.delta.rmse"))
    if baseline is None:
        baseline = number(summary.get("test_baseline_delta_rmse"))
    if krr is None or baseline is None:
        return None
    return baseline - krr


def baseline_rmse(summary: dict[str, Any]) -> float | None:
    value = number(summary.get("test_baseline_delta_rmse"))
    if value is not None:
        return value
    return number(nested(summary, "baseline.test.delta.rmse"))


def krr_rmse(summary: dict[str, Any]) -> float | None:
    return number(nested(summary, "eval.test.delta.rmse"))


def matches_adapter(summary: dict[str, Any], adapter_contains: str | None) -> bool:
    if not adapter_contains:
        return True
    needle = adapter_contains.lower()
    haystacks = [
        str(summary.get("run_name") or ""),
        str(summary.get("adapter_path") or ""),
    ]
    return any(needle in value.lower() for value in haystacks)


def available_train_sizes(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    adapter_contains: str | None,
) -> list[int]:
    values: set[int] = set()
    for _, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not matches_adapter(summary, adapter_contains):
                continue
            train_size = kernel_train_size(summary)
            if train_size is not None:
                values.add(train_size)
    return sorted(values)


def available_features(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    adapter_contains: str | None,
) -> list[str]:
    values: set[str] = set()
    for _, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not matches_adapter(summary, adapter_contains):
                continue
            values.add(kernel_feature_key(summary))
    preferred = ["raw", "thresholded_sign", "sign"]
    ordered = [feature for feature in preferred if feature in values]
    ordered.extend(sorted(values - set(ordered)))
    return ordered


def select_summaries(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    *,
    train_sizes: list[int],
    features: list[str],
    adapter_contains: str | None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    wanted_train_sizes = set(train_sizes)
    wanted_features = set(features)
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for experiment, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not matches_adapter(summary, adapter_contains):
                continue
            train_size = kernel_train_size(summary)
            feature = kernel_feature_key(summary)
            if train_size not in wanted_train_sizes or feature not in wanted_features:
                continue
            key = (experiment, train_size, feature)
            previous = selected.get(key)
            run_name = str(summary.get("run_name") or "")
            if previous is None or run_name > str(previous.get("run_name") or ""):
                selected[key] = summary
    return selected


def prepare_kernel_plot_data(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    *,
    train_sizes: list[int] | None = None,
    features: list[str] | None = None,
    adapter_contains: str | None = None,
    split: str = "test",
) -> KernelPlotData:
    resolved_train_sizes = train_sizes or available_train_sizes(
        experiments,
        adapter_contains,
    )
    resolved_features = features or available_features(experiments, adapter_contains)
    if not resolved_train_sizes:
        resolved_train_sizes = [0]
    if not resolved_features:
        resolved_features = ["raw"]
    return KernelPlotData(
        experiment_names=[label for label, _ in experiments],
        train_sizes=resolved_train_sizes,
        features=resolved_features,
        selected=select_summaries(
            experiments,
            train_sizes=resolved_train_sizes,
            features=resolved_features,
            adapter_contains=adapter_contains,
        ),
        split=split,
    )


def metric_series(
    data: KernelPlotData,
    experiment: str,
    value_getter: MetricGetter,
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for feature in data.features:
        values: list[tuple[int, float]] = []
        for train_size in sorted(data.train_sizes):
            summary = data.selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = value_getter(summary)
            if value is not None:
                values.append((train_size, value))
        if values:
            series[feature] = values
    return series


def baseline_series(data: KernelPlotData, experiment: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for train_size in sorted(data.train_sizes):
        for feature in data.features:
            summary = data.selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = baseline_rmse(summary)
            if value is not None:
                values.append((train_size, value))
                break
    return values
