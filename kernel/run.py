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
        "train": (config.train_limit, config.seed),
        "valid": (config.valid_limit, config.seed + 1),
        "test": (config.test_limit, config.seed + 2),
    }


def _for_all_splits_dup_test(splits : dict[str, list[PairRecord]], format : Any) -> bool:
    """
        Check for duplicate records over all split comulatively. That is,
        if there are A=B for any A and B from any splits even if different,
        those are considered to be duplicates.  

        Args:
            splits : records lists by splits to check the duplication for.
            format : function PairRecord -> str that prepares a PairRecord.
                     For instance, format can return concatenation of record's
                     prompt and completion if full match is the duplicate 
                     criterion.

        Returns:
            True if no dulicates are found and False otherwise
    """
    full_list = []
    #merge all the splits into one list
    for key, val in splits.items():
        full_list.extend([format(v) for v in val])

    #deduplicate with set() and check if the length change. If duplicates exist
    #the length becomes smaller
    res = (len(set(full_list)) == len(full_list))
    del full_list
    gc.collect()
    return res


def _across_splits_dup_test(splits : dict[str, list[PairRecord]], format : Any) -> bool:
    """
        Check for duplicate records only across different splits. That is,
        if there are A=B such that A and B are in train,
        those are not considered duplicates. But is A=B are such that
        A is in test and B is in train, then they are considered as duplicates. 

        Args:
            splits : records lists by splits to check the duplication for.
            format : function PairRecord -> str that prepares a PairRecord.
                     For instance, format can return concatenation of record's
                     prompt and completion if full match is the duplicate 
                     criterion.

        Returns:
            True if no dulicates are found and False otherwise
    """
    full_set = set()
    num = 0

    #for each split, deduplicate the split and add the deduplicated
    #split's length len(subset) to num to restore the total
    #deduped_train + deduped_test + deduped_val
    #then add the deduplicated set subset to the total set full_set, 
    # to get the cross deduplication 

    for key, val in splits.items():
        subset = set([format(v) for v in val])
        num += len(subset)
        full_set.union(subset)

    res = (len(full_set) == num)
    del full_set
    gc.collect()
    return res


def _within_splits_dup_test(splits : dict[str, list[PairRecord]], format : Any) -> bool:
    """
        Check for duplicate records only within each split. That is,
        if there are A=B such that A is in train and B is in test,
        those are not considered duplicates.  

        Args:
            splits : records lists by splits to check the duplication for.
            format : function PairRecord -> str that prepares a PairRecord.
                     For instance, format can return concatenation of record's
                     prompt and completion if full match is the duplicate 
                     criterion.

        Returns:
            True if no dulicates are found and False otherwise
    """
    full_list = []

    #merge all splits into one list and add the split label to
    #the record to make records from different splits automatically
    #diferent, even if cross-split duplicates exist

    for key, val in splits.items():
        full_list.extend([key + format(v) for v in val])

    res = (len(set(full_list)) == len(full_list))
    del full_list
    gc.collect()
    return res
        

def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _resolve_dup_test(dup_test: dict[str,str]) -> Any:
    """
        Takes test specificatons 'dup_test' of the form:
            dup_test = {'compare_method' : METHOD,
                        'splits_strategy' : STRATEGY}

        Returns the test function. 

        Test function is applyed in validate_kernel_run_inputs to
        records splits to check if, for example, there are same
        prompt-completion pairs in train and test data.
    """
    label = _normalize_name(dup_test["compare_method"])
    strategy = _normalize_name(dup_test["splits_strategy"])

    if not label in {"full", "prompt", "completion"}:
        raise ValueError(f"Provided duplicates test name ({label}) is not supported.")
    if not strategy in {"across", "all", "within"}:
        raise ValueError(f"Provided duplicates test strategy ({strategy}) is not supported.")

    apply_format = lambda record : str(record.prompt)+str(record.completion)

    if label == "prompt":
        apply_format = lambda record : str(record.prompt)
    if label == "completion":
        apply_format = lambda record : str(record.completion)

    apply_strategy = _across_splits_dup_test

    if strategy == "all":
        apply_strategy = _for_all_splits_dup_test
    if strategy == "within":
        apply_strategy = _within_splits_dup_test

    return lambda x : apply_strategy(x, apply_format)


def validate_kernel_run_inputs(
        config: KernelRunConfig, 
        dup_test: dict[str,str] = {"compare_method" : "full",
                                   "splits_strategy" : "across"}
        ) -> None:

    test = _resolve_dup_test(dup_test) #run resolve in the start in case an error is thrown
    record_splits = dict() #collect all sampled splits in here

    data_dir = resolve_project_path(config.data_dir)
    for split, (limit, seed) in _split_limits(config).items():
        split_path = data_dir / f"{split}.jsonl"
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
        
        record_splits[split] = sampled

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

    if not test(record_splits):
        raise ValueError(f"Duplicate records are found.")

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
    data_dir = resolve_project_path(config.data_dir)
    raw = {
        "train": load_pair_split(data_dir / "train.jsonl", split="train"),
        "valid": load_pair_split(data_dir / "valid.jsonl", split="valid"),
        "test": load_pair_split(data_dir / "test.jsonl", split="test"),
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
    """
        Utility. Takes a list of PairRecords data representation and
        applies 'scorer' entrie-wise to get scores. Returns list of dictionaries
        with keys
         - 'score' : the computed score value
         - 'pair_id' & 'split' : corresponding sample's id
         - other metadata

        for instance, 'scorer' might compute cross-entropy loss
        scores of base or fine-tuned model.
    """
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
    return resolve_project_path(config.output_root) / "cache"


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
        for key, value in dict(config.backend_args).items()
        if key not in FEATURE_TRANSFORM_BACKEND_ARG_KEYS
    }
    payload = {
        "version": KERNEL_CACHE_VERSION,
        "kind": "features",
        "backend": config.backend,
        "base_model": config.base_model,
        "seed": config.seed,
        "backend_args": extraction_backend_args,
        "adapter_config": _adapter_config_payload(config.adapter_path),
        "records": _records_payload(records),
    }
    backend = config.backend.lower().replace("-", "_")
    return (
        _kernel_cache_root(config)
        / "features"
        / backend
        / f"{_json_hash(payload)}.npy"
    )


def _feature_transform_names(config: KernelRunConfig) -> list[str]:
    raw = config.backend_args.get(
        "feature_transformations",
        config.backend_args.get("feature_transform"),
    )
    return normalize_transformation_names(raw)


def _feature_transform_params(config: KernelRunConfig) -> dict[str, dict[str, Any]]:
    raw = config.backend_args.get(
        "feature_transformation_params",
        config.backend_args.get("feature_transform_params"),
    )
    params = normalize_transformation_params(raw if isinstance(raw, dict) else None)
    if "threshold" in config.backend_args:
        thresholded_params = dict(params.get("thresholded_sign", {}))
        thresholded_params.setdefault("threshold", config.backend_args["threshold"])
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
    """
        Uses 'scorer_factory' to genearte an instance of scorer. Applies the scorer
        to compute loss scores for the given records list with
        the model specified in 'config' and the model's adapter in 'adapter_path'.

        For instance, for the default 'scorer_factore', the cross-entropy loss delta
        scores are computed.

        This function makes use of caching, identifying the relevant cache path
        by context hashing. If the scores for the given records are alredy available,
        returns the cached data. If some are missing, computes only for the missing
        records.

        ### How is it used?

        The adapter_loss - base_loss scores used to fit the kernel ridge regression model,
        that predicts the end loss score change. To avoid extra compuation, caching is used.
    """
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
    """
        Applies backend to extract kernel features required for kernel ridge regression
        predictions. The backend would typically generate gradient vectors w.r.t.
        LoRA B parameters of a given model and adapter path (passed to the backend factory).

        This functions ensures that the features are computed. If teh cached values are
        available - does nothing. If pre-computed data is missing - extracts the features and
        stores them as cached data. That is the computation is performed at most once.

        Returns the cached data path. 
    """
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
    """
        Loads the gradient features found in the provided path as 
        an 2 dimensional nd array. Then applies a feature transformation (e.g. apply sign(x))
        with transformation parameters specified in 'config'.

        ### How is this used?

        The raw gardient features can be used for kernel predictions as is, but
        applying transformations like sing(x) or sparse projetcion allows for
        drastic reduction in memory and compute load.
    """
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
    r"""
    Returns k(left, right) = <left, right> where <*,*> is the scaler product.
    Takes care of nan, clipping inf to some big but finite value and 
    an undefined value to 0.
    """

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
    """
        Follows 'config' specifications to construct the kernel regresion model.
        Return the model approximation type in 'method' (e.g. dual or nystrom),
        'ridge_lambda' value (for a different ridge lambda the kernel model needs
        to be generated again) and the model itself.
    """
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
    """
        Takes fit model and adjacent data from _predict_targets as 'fitted_model' and 'fit_payload'.
        'train_features' are the same as in _predict_targets split_features is just
        2d nd array of some split kernel features. 'config' is used to infer the type of
        kernel approximation method used.
    """
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
    baseline_delta: float | None = None,
) -> list[dict[str, Any]]:
    """Just some formatting function. Organizes provided data into expected
        list of dictionaries format."""
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
    """
        Runs evaluation results in 'rows' through a battery of tests and returns 
        teh corresponding metrics.
        'delta_key' and 'adapter_key' specify the corresponding 'rows' keys
        where the evaluation results to be found.
    """
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


def _build_report_markdown(
    config: KernelRunConfig,
    paths: KernelRunPaths,
    eval_payload: dict[str, Any],
    feature_dim: int,
) -> str:
    """
        Args:
            config : kernel specs
            paths : emited after run, holds paths to the files with results
            eval_payload : dictionaory holding evaluation results. Generated inside the
                           run_kernel_experiment function
            feature_dim : dimension of used kernel features. Reported in the generated
                          report

        This functioned is called in run_kernel)experiment to generate and store a
        markdown report containing the main experiment data.
    """
    lines = [
        "# Kernel Run",
        "",
        f"- Run: `{paths.run_name}`",
        f"- Dataset: `{config.dataset_name}`",
        f"- Backend: `{config.backend}`",
        f"- Target: `{config.target}`",
        f"- Kernel method: `{config.kernel.method}`",
        f"- Feature dimension: `{feature_dim}`",
        f"- Feature transform: `{eval_payload.get('feature_transform', {}).get('label', 'raw')}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Delta Pearson | Delta Spearman | Delta RMSE | Baseline RMSE | RMSE Gain | Adapter Pearson | Adapter Spearman | Adapter RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "valid", "test"):
        metrics = eval_payload[split]
        baseline = metrics.get("baseline", {})
        baseline_delta = baseline.get("delta", {})
        baseline_rmse = baseline_delta.get("rmse")
        rmse_gain = (
            baseline_rmse - metrics["delta"]["rmse"]
            if baseline_rmse is not None
            else None
        )
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
            + (f"{baseline_rmse:.4f}" if baseline_rmse is not None else "-")
            + " | "
            + (f"{rmse_gain:.4f}" if rmse_gain is not None else "-")
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
        "backend": runtime_config.backend,
        "target": runtime_config.target,
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
        "task": runtime_config.task,
        "base_model": runtime_config.base_model,
        "data_dir": runtime_config.data_dir,
        "backend": runtime_config.backend,
        "target": runtime_config.target,
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
        "report_path": str(paths.report_path),
    }
    write_json(paths.eval_path, eval_payload)
    write_json(paths.summary_path, summary)

    metadata["status"] = "completed"
    metadata["finished_at"] = now_iso()
    write_json(paths.metadata_path, metadata)
    return paths
