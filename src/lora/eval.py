from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import re

from lora.paths import resolve_project_path


MetricFunction = Callable[[str, str], float]

WORD_RE = re.compile(r"\w+")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True)
class EvaluationConfig:
    name: str = "generic"
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "token_f1"])
    primary_metric: str = "token_f1"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_sql(text: str) -> str:
    text = normalize_text(text)
    text = text.rstrip(";")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gsm8k_answer_accuracy(prediction: str, reference: str) -> float:
    pred_number = extract_last_number(prediction)
    ref_number = extract_last_number(reference)
    if pred_number is None or ref_number is None:
        return 0.0
    return float(pred_number == ref_number)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sql_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_sql(prediction) == normalize_sql(reference))


METRIC_REGISTRY: dict[str, MetricFunction] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "gsm8k_answer_accuracy": gsm8k_answer_accuracy,
    "sql_exact_match": sql_exact_match,
    "rouge_l_f1": rouge_l_f1,
    "conala_exact_match": exact_match,
}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with resolve_project_path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_evaluation_config(
    raw: dict[str, Any] | EvaluationConfig | None,
) -> EvaluationConfig:
    if raw is None:
        config = EvaluationConfig()
    elif isinstance(raw, EvaluationConfig):
        config = raw
    else:
        unknown = sorted(set(raw) - {"name", "metrics", "primary_metric"})
        if unknown:
            raise ValueError(f"Unknown evaluation config keys: {unknown}")
        metrics = raw.get("metrics", ["exact_match", "token_f1"])
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("`evaluation.metrics` must be a non-empty list.")
        config = EvaluationConfig(
            name=str(raw.get("name", "generic")),
            metrics=[str(metric) for metric in metrics],
            primary_metric=str(raw.get("primary_metric", metrics[0])),
        )

    missing = sorted(set(config.metrics) - set(METRIC_REGISTRY))
    if missing:
        raise ValueError(
            f"Unknown evaluation metrics: {missing}. "
            f"Available metrics: {sorted(METRIC_REGISTRY)}"
        )
    if config.primary_metric not in config.metrics:
        raise ValueError("`evaluation.primary_metric` must be listed in metrics.")
    return config


def evaluate_predictions(
    records: list[dict[str, Any]],
    evaluation: dict[str, Any] | EvaluationConfig | None = None,
) -> dict[str, Any]:
    config = build_evaluation_config(evaluation)
    if not records:
        raise ValueError("Predictions file is empty.")

    predictions = [str(record["prediction"]) for record in records]
    references = [str(record["reference"]) for record in records]
    metrics = {
        metric_name: _mean(
            [
                METRIC_REGISTRY[metric_name](prediction, reference)
                for prediction, reference in zip(predictions, references)
            ]
        )
        for metric_name in config.metrics
    }

    return {
        "evaluation": config.name,
        "num_examples": len(records),
        "metric_name": config.primary_metric,
        "metric_value": metrics[config.primary_metric],
        "metrics": metrics,
    }
