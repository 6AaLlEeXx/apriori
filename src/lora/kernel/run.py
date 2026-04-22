from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import json

import numpy as np
from numpy.typing import NDArray

from lora.kernel.backends import create_feature_backend
from lora.kernel.config import (
    KernelRunConfig,
    KernelRunPaths,
    build_kernel_metadata,
    build_kernel_run_name,
    now_iso,
    prepare_kernel_run,
    resolve_kernel_run_config,
    save_kernel_run_config,
    write_json,
)
from lora.kernel.data import PairRecord, load_pair_split, maybe_subset_pairs
from lora.kernel.krr import (
    fit_krr_dual,
    fit_krr_nystrom,
    predict_krr_dual,
    predict_krr_nystrom,
    select_landmarks,
)
from lora.kernel.metrics import evaluate_predictions
from lora.kernel.scoring import ModelScorer
from lora.paths import resolve_project_path


Array = NDArray[Any]
FloatArray = NDArray[np.float64]


def _load_split_records(
    config: KernelRunConfig,
) -> dict[str, list[PairRecord]]:
    data_dir = resolve_project_path(config.data_dir)
    raw = {
        "train": load_pair_split(data_dir / "train.jsonl", split="train"),
        "valid": load_pair_split(data_dir / "valid.jsonl", split="valid"),
        "test": load_pair_split(data_dir / "test.jsonl", split="test"),
    }
    return {
        "train": maybe_subset_pairs(
            raw["train"], limit=config.train_limit, seed=config.seed
        ),
        "valid": maybe_subset_pairs(
            raw["valid"],
            limit=config.valid_limit,
            seed=config.seed + 1,
        ),
        "test": maybe_subset_pairs(
            raw["test"], limit=config.test_limit, seed=config.seed + 2
        ),
    }


def _score_records(
    scorer: ModelScorer,
    records: list[PairRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        score, tokens = scorer.score_record(record)
        rows.append(
            {
                "pair_id": record.pair_id,
                "split": record.split,
                "score": score,
                "prompt_offset": tokens.prompt_offset,
                "sequence_length": tokens.sequence_length,
                "completion_targets": tokens.completion_targets,
            }
        )
    return rows


def _index_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): row for row in rows}


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_features_to_npy(
    backend: Any,
    records: list[PairRecord],
    path: str | Path,
) -> tuple[Path, int]:
    path = Path(path)
    if not records:
        raise ValueError("Cannot extract features for an empty split.")
    first = backend.extract_feature(records[0])
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(records), first.shape[0]),
    )
    matrix[0] = first
    for index, record in enumerate(records[1:], start=1):
        matrix[index] = backend.extract_feature(record)
    feature_dim = int(first.shape[0])
    del matrix
    return path, feature_dim


def _load_features(path: str | Path) -> Array:
    return np.load(Path(path), mmap_mode="r")


def _kernel_from_features(left: Array, right: Array) -> FloatArray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        kernel = (
            np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64).T
        )
    if not np.isfinite(kernel).all():
        kernel = np.nan_to_num(kernel, nan=0.0, posinf=1e12, neginf=-1e12)
    return kernel


def _predict_targets(
    config: KernelRunConfig,
    train_features: Array,
    train_targets: FloatArray,
) -> tuple[Any, dict[str, Any]]:
    method = config.kernel.method.lower()
    if method == "dual":
        k_train = _kernel_from_features(train_features, train_features)
        model = fit_krr_dual(
            k_train=k_train,
            targets=train_targets,
            ridge_lambda=config.kernel.ridge_lambda,
        )
        payload = {
            "method": "dual",
            "ridge_lambda": config.kernel.ridge_lambda,
        }
        return model, payload

    if method == "nystrom":
        landmark_count = min(config.kernel.num_landmarks, len(train_features))
        rank = min(config.kernel.rank, landmark_count)
        landmark_indices = select_landmarks(
            n_train=len(train_features),
            count=landmark_count,
            seed=config.seed,
        )
        train_landmarks = train_features[landmark_indices]
        k_train_landmarks = _kernel_from_features(train_features, train_landmarks)
        k_landmarks = _kernel_from_features(train_landmarks, train_landmarks)
        model = fit_krr_nystrom(
            k_train_landmarks=k_train_landmarks,
            k_landmarks=k_landmarks,
            targets=train_targets,
            ridge_lambda=config.kernel.ridge_lambda,
            rank=rank,
            landmark_indices=landmark_indices,
        )
        payload = {
            "method": "nystrom",
            "ridge_lambda": config.kernel.ridge_lambda,
            "rank": rank,
            "num_landmarks": landmark_count,
            "landmark_indices": landmark_indices.tolist(),
        }
        return model, payload

    raise ValueError(f"Unsupported kernel method: {config.kernel.method}")


def _predict_split(
    config: KernelRunConfig,
    fitted_model: Any,
    fit_payload: dict[str, Any],
    train_features: Array,
    split_features: Array,
) -> FloatArray:
    method = config.kernel.method.lower()
    if method == "dual":
        k_split_train = _kernel_from_features(split_features, train_features)
        return predict_krr_dual(fitted_model, k_split_train).reshape(-1)
    if method == "nystrom":
        landmark_indices = np.asarray(fit_payload["landmark_indices"], dtype=np.int64)
        train_landmarks = train_features[landmark_indices]
        k_split_landmarks = _kernel_from_features(split_features, train_landmarks)
        return predict_krr_nystrom(fitted_model, k_split_landmarks).reshape(-1)
    raise ValueError(f"Unsupported kernel method: {config.kernel.method}")


def _build_prediction_rows(
    records: list[PairRecord],
    base_scores: dict[str, dict[str, Any]],
    adapter_scores: dict[str, dict[str, Any]],
    pred_delta: FloatArray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        base_score = float(base_scores[record.pair_id]["score"])
        adapter_score = float(adapter_scores[record.pair_id]["score"])
        delta = adapter_score - base_score
        predicted_delta = float(pred_delta[index])
        rows.append(
            {
                "pair_id": record.pair_id,
                "split": record.split,
                "base_score": base_score,
                "adapter_score": adapter_score,
                "score_delta": delta,
                "predicted_score_delta": predicted_delta,
                "predicted_adapter_score": base_score + predicted_delta,
            }
        )
    return rows


def _split_eval_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_delta = np.asarray([row["score_delta"] for row in rows], dtype=np.float64)
    pred_delta = np.asarray(
        [row["predicted_score_delta"] for row in rows],
        dtype=np.float64,
    )
    true_adapter = np.asarray([row["adapter_score"] for row in rows], dtype=np.float64)
    pred_adapter = np.asarray(
        [row["predicted_adapter_score"] for row in rows],
        dtype=np.float64,
    )
    return evaluate_predictions(
        true_delta=true_delta,
        pred_delta=pred_delta,
        true_adapter_score=true_adapter,
        pred_adapter_score=pred_adapter,
    )


def _build_report_markdown(
    config: KernelRunConfig,
    paths: KernelRunPaths,
    eval_payload: dict[str, Any],
    feature_dim: int,
) -> str:
    lines = [
        "# Kernel Run",
        "",
        f"- Run: `{paths.run_name}`",
        f"- Dataset: `{config.dataset_name}`",
        f"- Backend: `{config.backend}`",
        f"- Target: `{config.target}`",
        f"- Kernel method: `{config.kernel.method}`",
        f"- Feature dimension: `{feature_dim}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Delta Pearson | Delta Spearman | Delta RMSE | Adapter Pearson | Adapter Spearman | Adapter RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "valid", "test"):
        metrics = eval_payload[split]
        lines.append(
            "| "
            + split
            + " | "
            + f"{metrics['delta']['pearson']:.4f}"
            + " | "
            + f"{metrics['delta']['spearman']:.4f}"
            + " | "
            + f"{metrics['delta']['rmse']:.4f}"
            + " | "
            + f"{metrics['adapter_score']['pearson']:.4f}"
            + " | "
            + f"{metrics['adapter_score']['spearman']:.4f}"
            + " | "
            + f"{metrics['adapter_score']['rmse']:.4f}"
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_kernel_experiment(
    config: KernelRunConfig,
    config_source: str | Path,
    run_name: str | None = None,
) -> KernelRunPaths:
    runtime_config = resolve_kernel_run_config(config)
    if runtime_config.target != "score_delta":
        raise ValueError("Only `score_delta` is currently implemented for kernel runs.")
    run_name = run_name or build_kernel_run_name(runtime_config)
    paths = prepare_kernel_run(runtime_config, run_name)
    save_kernel_run_config(runtime_config, paths.resolved_config_path)
    metadata = build_kernel_metadata(
        config=runtime_config,
        paths=paths,
        config_source=config_source,
        status="running",
    )
    write_json(paths.metadata_path, metadata)

    records = _load_split_records(runtime_config)

    score_rows: dict[str, list[dict[str, Any]]] = {}
    base_scores_by_split: dict[str, list[dict[str, Any]]] = {}
    adapter_scores_by_split: dict[str, list[dict[str, Any]]] = {}
    base_scorer = ModelScorer(runtime_config.base_model, adapter_path=None, lazy=True)
    for split in ("train", "valid", "test"):
        base_scores_by_split[split] = _score_records(base_scorer, records[split])
    del base_scorer
    gc.collect()

    adapter_scorer = ModelScorer(
        runtime_config.base_model,
        adapter_path=runtime_config.adapter_path,
        lazy=True,
    )
    for split in ("train", "valid", "test"):
        adapter_scores_by_split[split] = _score_records(adapter_scorer, records[split])
    del adapter_scorer
    gc.collect()

    for split in ("train", "valid", "test"):
        base_rows = base_scores_by_split[split]
        adapter_rows = adapter_scores_by_split[split]
        merged_rows: list[dict[str, Any]] = []
        for base_row, adapter_row in zip(base_rows, adapter_rows):
            merged_rows.append(
                {
                    "pair_id": base_row["pair_id"],
                    "split": split,
                    "base_score": base_row["score"],
                    "adapter_score": adapter_row["score"],
                    "score_delta": adapter_row["score"] - base_row["score"],
                    "prompt_offset": base_row["prompt_offset"],
                    "sequence_length": base_row["sequence_length"],
                    "completion_targets": base_row["completion_targets"],
                }
            )
        score_rows[split] = merged_rows
        _write_jsonl(paths.scores_dir / f"{split}.jsonl", merged_rows)

    feature_backend = create_feature_backend(runtime_config)
    feature_paths: dict[str, Path] = {}
    feature_dim = 0
    for split in ("train", "valid", "test"):
        path, feature_dim = _extract_features_to_npy(
            backend=feature_backend,
            records=records[split],
            path=paths.features_dir / f"{runtime_config.backend}_{split}.npy",
        )
        feature_paths[split] = path
    del feature_backend
    gc.collect()

    train_features = _load_features(feature_paths["train"])
    targets = np.asarray(
        [row["score_delta"] for row in score_rows["train"]],
        dtype=np.float64,
    )
    fitted_model, fit_payload = _predict_targets(
        config=runtime_config,
        train_features=train_features,
        train_targets=targets,
    )
    eval_payload: dict[str, Any] = {
        "backend": runtime_config.backend,
        "target": runtime_config.target,
        "feature_dim": feature_dim,
        "timestamp": now_iso(),
    }
    prediction_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        split_features = _load_features(feature_paths[split])
        pred_delta = _predict_split(
            config=runtime_config,
            fitted_model=fitted_model,
            fit_payload=fit_payload,
            train_features=train_features,
            split_features=split_features,
        )
        base_index = {
            row["pair_id"]: {"score": row["base_score"]} for row in score_rows[split]
        }
        adapter_index = {
            row["pair_id"]: {"score": row["adapter_score"]} for row in score_rows[split]
        }
        prediction_rows = _build_prediction_rows(
            records=records[split],
            base_scores=base_index,
            adapter_scores=adapter_index,
            pred_delta=pred_delta,
        )
        prediction_rows_by_split[split] = prediction_rows
        _write_jsonl(paths.predictions_dir / f"{split}.jsonl", prediction_rows)
        eval_payload[split] = _split_eval_payload(prediction_rows)

    eval_payload["fit"] = fit_payload

    report = _build_report_markdown(
        config=runtime_config,
        paths=paths,
        eval_payload=eval_payload,
        feature_dim=feature_dim,
    )
    paths.report_path.write_text(report)

    summary = {
        "run_name": paths.run_name,
        "status": "completed",
        "dataset_name": runtime_config.dataset_name,
        "backend": runtime_config.backend,
        "target": runtime_config.target,
        "adapter_path": runtime_config.adapter_path,
        "feature_dim": feature_dim,
        "split_sizes": {
            split: len(records[split]) for split in ("train", "valid", "test")
        },
        "fit": fit_payload,
        "valid_delta_pearson": eval_payload["valid"]["delta"]["pearson"],
        "test_delta_pearson": eval_payload["test"]["delta"]["pearson"],
        "valid_adapter_pearson": eval_payload["valid"]["adapter_score"]["pearson"],
        "test_adapter_pearson": eval_payload["test"]["adapter_score"]["pearson"],
        "report_path": str(paths.report_path),
    }
    write_json(paths.eval_path, eval_payload)
    write_json(paths.summary_path, summary)

    metadata["status"] = "completed"
    metadata["finished_at"] = now_iso()
    write_json(paths.metadata_path, metadata)
    return paths
