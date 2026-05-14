from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol
import gc
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from transformations import (
    apply_transformations,
    normalize_transformation_names,
    normalize_transformation_params,
)
from kernel.config import (
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
from kernel.data import PairRecord, load_pair_split, maybe_subset_pairs
from kernel.features import create_feature_backend
from kernel.krr import (
    fit_krr_dual,
    fit_krr_nystrom,
    predict_krr_dual,
    predict_krr_nystrom,
    select_landmarks,
)
from kernel.metrics import evaluate_predictions
from kernel.scoring import ModelScorer
from paths import resolve_project_path


Array = NDArray[Any]
FloatArray = NDArray[np.float64]
KERNEL_CACHE_VERSION = 1
FEATURE_TRANSFORM_BACKEND_ARG_KEYS = {
    "feature_transform",
    "feature_transformations",
    "feature_transform_params",
    "feature_transformation_params",
    "threshold",
}


class Scorer(Protocol):
    def score_record(self, record: PairRecord) -> tuple[float, Any]: ...


ScorerFactory = Callable[[str, str | None], Scorer]


def _split_limits(config: KernelRunConfig) -> dict[str, tuple[int, int]]:
    return {
        "train": (config.krr_fit_examples, config.seed),
        "valid": (config.validation_examples, config.seed + 1),
        "test": (config.test_examples, config.seed + 2),
    }


def validate_kernel_run_inputs(config: KernelRunConfig) -> None:
    prepared_data_dir = resolve_project_path(config.prepared_data_dir)
    for split, (limit, seed) in _split_limits(config).items():
        split_path = prepared_data_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Kernel data split not found: {split_path}")
        if split_path.stat().st_size == 0:
            raise ValueError(f"Kernel data split is empty: {split_path}")

        records = load_pair_split(split_path, split=split)
        if not records:
            raise ValueError(f"Kernel data split has no JSONL records: {split_path}")
        sampled = maybe_subset_pairs(records, limit=limit, seed=seed)
        if not sampled:
            raise ValueError(f"Kernel data split has no sampled records: {split_path}")
        for record in sampled:
            if not record.prompt.strip():
                raise ValueError(
                    f"Kernel data row has empty `prompt`: "
                    f"{split_path} ({record.pair_id})"
                )
            if not record.completion.strip():
                raise ValueError(
                    f"Kernel data row has empty `completion`: "
                    f"{split_path} ({record.pair_id})"
                )

    adapter_dir = resolve_project_path(config.adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter path is not a directory: {adapter_dir}")
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config}")


def _load_split_records(
    config: KernelRunConfig,
) -> dict[str, list[PairRecord]]:
    prepared_data_dir = resolve_project_path(config.prepared_data_dir)
    raw = {
        "train": load_pair_split(prepared_data_dir / "train.jsonl", split="train"),
        "valid": load_pair_split(prepared_data_dir / "valid.jsonl", split="valid"),
        "test": load_pair_split(prepared_data_dir / "test.jsonl", split="test"),
    }
    return {
        split: maybe_subset_pairs(records, limit=limit, seed=seed)
        for split, records in raw.items()
        for limit, seed in [_split_limits(config)[split]]
    }


def _score_records(
    scorer: Scorer,
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


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(path),
    }


def _records_payload(records: list[PairRecord]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                {
                    "pair_id": record.pair_id,
                    "split": record.split,
                    "prompt": record.prompt,
                    "completion": record.completion,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "count": len(records),
        "sha256": digest.hexdigest(),
        "pair_ids": [record.pair_id for record in records],
    }


def _adapter_config_payload(adapter_path: str | Path) -> dict[str, Any]:
    adapter_dir = resolve_project_path(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return {"exists": False}
    payload = json.loads(config_path.read_text())
    return {
        "exists": True,
        "sha256": _file_sha256(config_path),
        "config": payload,
    }


def _adapter_score_payload(adapter_path: str | Path | None) -> dict[str, Any]:
    if adapter_path is None:
        return {"kind": "base"}
    adapter_dir = resolve_project_path(adapter_path)
    files = {
        name: _file_fingerprint(adapter_dir / name)
        for name in ("adapter_config.json", "adapters.safetensors")
    }
    return {
        "kind": "adapter",
        "path": str(adapter_dir),
        "files": files,
    }


def _kernel_cache_root(config: KernelRunConfig) -> Path:
    return resolve_project_path(config.kernel_results_root) / "cache"


def _score_cache_path(
    config: KernelRunConfig,
    records: list[PairRecord],
    *,
    adapter_path: str | Path | None,
) -> Path:
    payload = {
        "version": KERNEL_CACHE_VERSION,
        "kind": "score",
        "base_model": config.base_model,
        "adapter": _adapter_score_payload(adapter_path),
        "records": _records_payload(records),
    }
    return _kernel_cache_root(config) / "scores" / f"{_json_hash(payload)}.jsonl"


def _feature_cache_path(
    config: KernelRunConfig,
    records: list[PairRecord],
) -> Path:
    extraction_backend_args = {
        key: value
        for key, value in dict(config.feature_backend_args).items()
        if key not in FEATURE_TRANSFORM_BACKEND_ARG_KEYS
    }
    payload = {
        "version": KERNEL_CACHE_VERSION,
        "kind": "features",
        "feature_backend": config.feature_backend,
        "base_model": config.base_model,
        "seed": config.seed,
        "feature_backend_args": extraction_backend_args,
        "adapter_config": _adapter_config_payload(config.adapter_path),
        "records": _records_payload(records),
    }
    feature_backend = config.feature_backend.lower().replace("-", "_")
    return (
        _kernel_cache_root(config)
        / "features"
        / feature_backend
        / f"{_json_hash(payload)}.npy"
    )


def _feature_transform_names(config: KernelRunConfig) -> list[str]:
    raw = config.feature_backend_args.get(
        "feature_transformations",
        config.feature_backend_args.get("feature_transform"),
    )
    return normalize_transformation_names(raw)


def _feature_transform_params(config: KernelRunConfig) -> dict[str, dict[str, Any]]:
    raw = config.feature_backend_args.get(
        "feature_transformation_params",
        config.feature_backend_args.get("feature_transform_params"),
    )
    params = normalize_transformation_params(raw if isinstance(raw, dict) else None)
    if "threshold" in config.feature_backend_args:
        thresholded_params = dict(params.get("thresholded_sign", {}))
        thresholded_params.setdefault(
            "threshold",
            config.feature_backend_args["threshold"],
        )
        params["thresholded_sign"] = thresholded_params
    return params


def _feature_transform_label(
    names: list[str],
    params: dict[str, dict[str, Any]],
) -> str:
    active = [name for name in names if name != "identity"]
    if not active:
        return "raw"
    labels: list[str] = []
    for name in active:
        label = name
        if name == "thresholded_sign":
            threshold = params.get("thresholded_sign", {}).get("threshold")
            if threshold is not None:
                label = f"{name}(threshold={threshold})"
        labels.append(label)
    return "+".join(labels)


def _feature_transform_payload(config: KernelRunConfig) -> dict[str, Any]:
    names = _feature_transform_names(config)
    params = _feature_transform_params(config)
    return {
        "names": names,
        "params": params,
        "label": _feature_transform_label(names, params),
    }


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


def _score_cache_valid(
    rows: list[dict[str, Any]],
    records: list[PairRecord],
) -> bool:
    return [str(row.get("pair_id")) for row in rows] == [
        record.pair_id for record in records
    ]


def _load_cached_scores(
    path: str | Path,
    records: list[PairRecord],
) -> list[dict[str, Any]] | None:
    try:
        rows = _read_jsonl(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if _score_cache_valid(rows, records):
        return rows
    return None


def _load_cached_feature_matrix(
    path: str | Path,
    expected_rows: int,
) -> Array | None:
    try:
        features = np.load(Path(path), mmap_mode="r")
    except (FileNotFoundError, OSError, ValueError):
        return None
    if features.ndim == 2 and features.shape[0] == expected_rows:
        return features
    return None


def _cache_log(message: str) -> None:
    print(f"[kernel-cache] {message}", flush=True)


def _score_splits_with_cache(
    config: KernelRunConfig,
    records: dict[str, list[PairRecord]],
    *,
    adapter_path: str | None,
    scorer_factory: ScorerFactory = ModelScorer,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    missing: list[tuple[str, Path]] = []
    label = "base" if adapter_path is None else Path(adapter_path).name

    for split in ("train", "valid", "test"):
        cache_path = _score_cache_path(
            config,
            records[split],
            adapter_path=adapter_path,
        )
        cached = _load_cached_scores(cache_path, records[split])
        if cached is not None:
            _cache_log(f"score hit model={label} split={split} path={cache_path}")
            rows_by_split[split] = cached
        else:
            _cache_log(f"score miss model={label} split={split} path={cache_path}")
            missing.append((split, cache_path))

    if missing:
        scorer = scorer_factory(config.base_model, adapter_path)
        for split, cache_path in missing:
            rows = _score_records(scorer, records[split])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(cache_path, rows)
            _cache_log(
                f"score written model={label} split={split} path={cache_path}"
            )
            rows_by_split[split] = rows
        del scorer
        gc.collect()

    return rows_by_split


def _index_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): row for row in rows}


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_features_to_npy(
    config: KernelRunConfig,
    backend: Any,
    records: list[PairRecord],
) -> tuple[Path, int]:
    if not records:
        raise ValueError("Cannot extract features for an empty split.")
    path = _feature_cache_path(config, records)
    cached = _load_cached_feature_matrix(path, len(records))
    if cached is not None:
        feature_dim = int(cached.shape[1])
        _cache_log(f"feature hit split={records[0].split} path={path}")
        return path, feature_dim

    _cache_log(f"feature miss split={records[0].split} path={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    _cache_log(f"feature written split={records[0].split} path={path}")
    return path, feature_dim


def _load_features(path: str | Path, config: KernelRunConfig | None = None) -> Array:
    features = np.load(Path(path), mmap_mode="r")
    if config is None:
        return features
    names = _feature_transform_names(config)
    if all(name == "identity" for name in names):
        return features
    return apply_transformations(
        np.asarray(features, dtype=np.float32),
        names,
        params=_feature_transform_params(config),
    )


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
    method = config.krr.method.lower()
    if method == "dual":
        k_train = _kernel_from_features(train_features, train_features)
        model = fit_krr_dual(
            k_train=k_train,
            targets=train_targets,
            ridge_lambda=config.krr.ridge_lambda,
        )
        payload = {
            "method": "dual",
            "ridge_lambda": config.krr.ridge_lambda,
        }
        return model, payload

    if method == "nystrom":
        landmark_count = min(config.krr.num_landmarks, len(train_features))
        rank = min(config.krr.rank, landmark_count)
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
            ridge_lambda=config.krr.ridge_lambda,
            rank=rank,
            landmark_indices=landmark_indices,
        )
        payload = {
            "method": "nystrom",
            "ridge_lambda": config.krr.ridge_lambda,
            "rank": rank,
            "num_landmarks": landmark_count,
            "landmark_indices": landmark_indices.tolist(),
        }
        return model, payload

    raise ValueError(f"Unsupported KRR method: {config.krr.method}")


def _predict_split(
    config: KernelRunConfig,
    fitted_model: Any,
    fit_payload: dict[str, Any],
    train_features: Array,
    split_features: Array,
) -> FloatArray:
    method = config.krr.method.lower()
    if method == "dual":
        k_split_train = _kernel_from_features(split_features, train_features)
        return predict_krr_dual(fitted_model, k_split_train).reshape(-1)
    if method == "nystrom":
        landmark_indices = np.asarray(fit_payload["landmark_indices"], dtype=np.int64)
        train_landmarks = train_features[landmark_indices]
        k_split_landmarks = _kernel_from_features(split_features, train_landmarks)
        return predict_krr_nystrom(fitted_model, k_split_landmarks).reshape(-1)
    raise ValueError(f"Unsupported KRR method: {config.krr.method}")


def _build_prediction_rows(
    records: list[PairRecord],
    base_scores: dict[str, dict[str, Any]],
    adapter_scores: dict[str, dict[str, Any]],
    pred_delta: FloatArray,
    baseline_delta: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        base_score = float(base_scores[record.pair_id]["score"])
        adapter_score = float(adapter_scores[record.pair_id]["score"])
        delta = adapter_score - base_score
        predicted_delta = float(pred_delta[index])
        row = {
            "pair_id": record.pair_id,
            "split": record.split,
            "base_score": base_score,
            "adapter_score": adapter_score,
            "score_delta": delta,
            "predicted_score_delta": predicted_delta,
            "predicted_adapter_score": base_score + predicted_delta,
        }
        if baseline_delta is not None:
            row["baseline_score_delta"] = baseline_delta
            row["baseline_adapter_score"] = base_score + baseline_delta
        rows.append(row)
    return rows


def _split_eval_payload(
    rows: list[dict[str, Any]],
    *,
    delta_key: str = "predicted_score_delta",
    adapter_key: str = "predicted_adapter_score",
) -> dict[str, Any]:
    true_delta = np.asarray([row["score_delta"] for row in rows], dtype=np.float64)
    pred_delta = np.asarray(
        [row[delta_key] for row in rows],
        dtype=np.float64,
    )
    true_adapter = np.asarray([row["adapter_score"] for row in rows], dtype=np.float64)
    pred_adapter = np.asarray(
        [row[adapter_key] for row in rows],
        dtype=np.float64,
    )
    return evaluate_predictions(
        true_delta=true_delta,
        pred_delta=pred_delta,
        true_adapter_score=true_adapter,
        pred_adapter_score=pred_adapter,
    )


def run_kernel_experiment(
    config: KernelRunConfig,
    config_source: str | Path,
    run_name: str | None = None,
) -> KernelRunPaths:
    runtime_config = resolve_kernel_run_config(config)
    if runtime_config.prediction_target != "score_delta":
        raise ValueError("Only `score_delta` is currently implemented for kernel runs.")
    validate_kernel_run_inputs(runtime_config)
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
    base_scores_by_split = _score_splits_with_cache(
        runtime_config,
        records,
        adapter_path=None,
    )
    adapter_scores_by_split = _score_splits_with_cache(
        runtime_config,
        records,
        adapter_path=runtime_config.adapter_path,
    )

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
            config=runtime_config,
            backend=feature_backend,
            records=records[split],
        )
        feature_paths[split] = path
    del feature_backend
    gc.collect()

    train_features = _load_features(feature_paths["train"], runtime_config)
    targets = np.asarray(
        [row["score_delta"] for row in score_rows["train"]],
        dtype=np.float64,
    )
    baseline_delta = float(np.mean(targets))
    fitted_model, fit_payload = _predict_targets(
        config=runtime_config,
        train_features=train_features,
        train_targets=targets,
    )
    eval_payload: dict[str, Any] = {
        "feature_backend": runtime_config.feature_backend,
        "prediction_target": runtime_config.prediction_target,
        "feature_dim": feature_dim,
        "feature_transform": _feature_transform_payload(runtime_config),
        "timestamp": now_iso(),
    }
    prediction_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        split_features = _load_features(feature_paths[split], runtime_config)
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
            baseline_delta=baseline_delta,
        )
        prediction_rows_by_split[split] = prediction_rows
        _write_jsonl(paths.predictions_dir / f"{split}.jsonl", prediction_rows)
        eval_payload[split] = _split_eval_payload(prediction_rows)
        eval_payload[split]["baseline"] = _split_eval_payload(
            prediction_rows,
            delta_key="baseline_score_delta",
            adapter_key="baseline_adapter_score",
        )

    eval_payload["fit"] = fit_payload
    eval_payload["baseline"] = {
        "strategy": "train_mean_delta",
        "train_mean_delta": baseline_delta,
    }

    summary = {
        "run_name": paths.run_name,
        "status": "completed",
        "dataset_name": runtime_config.dataset_name,
        "task": runtime_config.task,
        "base_model": runtime_config.base_model,
        "prepared_data_dir": runtime_config.prepared_data_dir,
        "feature_backend": runtime_config.feature_backend,
        "prediction_target": runtime_config.prediction_target,
        "feature_transform": eval_payload["feature_transform"],
        "feature_transform_label": eval_payload["feature_transform"]["label"],
        "adapter_path": runtime_config.adapter_path,
        "feature_dim": feature_dim,
        "split_sizes": {
            split: len(records[split]) for split in ("train", "valid", "test")
        },
        "fit": fit_payload,
        "valid_delta_pearson": eval_payload["valid"]["delta"]["pearson"],
        "test_delta_pearson": eval_payload["test"]["delta"]["pearson"],
        "valid_baseline_delta_rmse": eval_payload["valid"]["baseline"]["delta"][
            "rmse"
        ],
        "test_baseline_delta_rmse": eval_payload["test"]["baseline"]["delta"]["rmse"],
        "valid_delta_rmse_gain": (
            eval_payload["valid"]["baseline"]["delta"]["rmse"]
            - eval_payload["valid"]["delta"]["rmse"]
        ),
        "test_delta_rmse_gain": (
            eval_payload["test"]["baseline"]["delta"]["rmse"]
            - eval_payload["test"]["delta"]["rmse"]
        ),
        "valid_adapter_pearson": eval_payload["valid"]["adapter_score"]["pearson"],
        "test_adapter_pearson": eval_payload["test"]["adapter_score"]["pearson"],
        "baseline": eval_payload["baseline"],
    }
    write_json(paths.eval_path, eval_payload)
    write_json(paths.summary_path, summary)

    metadata["status"] = "completed"
    metadata["finished_at"] = now_iso()
    write_json(paths.metadata_path, metadata)
    return paths
