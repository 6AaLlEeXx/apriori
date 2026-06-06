from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Any
from dataclasses import dataclass, replace
from mlops import RunPaths, save_lora_run_config

import json

from mlops import (
    build_mlx_config_payload,
    build_run_metadata,
    build_run_name,
    build_summary,
    build_test_command,
    build_train_command,
    LoraRunConfig,
    load_lora_run_config,
    prepare_sampled_data_dir,
    prepare_run,
    save_lora_run_config,
    run_command_with_logging,
    write_command,
    write_metadata,
    write_mlx_runtime_config,
    write_summary,
)
from paths import resolve_project_path

from kernel.run import (
    KernelRunPaths, KernelRunConfig,
    _score_cache_path, _load_cached_scores,
    _score_records, _cache_log,
    _write_jsonl, ScorerFactory,
    resolve_kernel_run_config, validate_kernel_run_inputs,
    build_kernel_run_name, prepare_kernel_run,
    save_kernel_run_config, build_kernel_metadata,
    load_pair_split, create_feature_backend, _extract_features_to_npy,
    _load_features, _predict_targets, write_json
)

from kernel.data import PairRecord
from kernel.scoring import ModelScorer
from kernel.krr import NystromKRRModel, DualKRRModel

SELECTOR_PATHS = {
                    "random": "selectors/random.py",
                    "identity": "selectors/identity.py",
                    "lora_ntk_kmeans": "selector/lora_ntk_kmeans.py"
                 }

DEFAULT_LORA_DIRECTORY = "estimator/lora_models"
DEFAULT_CONFIGS_ROOT = "estimator/configs"

from pathlib import Path
import gc

import numpy as np

KRR = NystromKRRModel | DualKRRModel

SUBSET_ITERS_POLICIES = {
    "match_full_exposure",
    "match_full_passes",
    "per_example",
}


@dataclass
class lora_ft_JobCard:
    #path to the lora run config file
    config: str = "/Users/malyshevviktor/Desktop/Apriory/apriori/configs/dolly.yaml"
    #name of the base model
    base_model: str = "mlx-community/SmolLM2-1.7B-Instruct"
    #give your run a name!
    run_name: str = "name"
    #path to a sample selector module
    sample_selector: str = "/Users/malyshevviktor/Desktop/Apriory/apriori/selectors/random.py"
    # this is passed to selector, defines how big the subset will be
    max_examples: int = 256
    # name of applied trasformation, like 'thresholded_sign'
    selector_transformation: str|None = None
    #name of used feature projector, 'identity' is no projection is needed
    selector_projection: str = "identity"
    #if a non-trivial projection is used, specify the end dimesionality
    selector_projection_components: int | None = None
    #if thresholed sign is used, specify the threshold
    selector_thresholded_sign_threshold: float = 0.01
    #proccess features in chunks when projectiong
    selector_projection_chunk_size: int = 16
    #fix a limit to the memory usage if kmeans sampling is used
    selector_max_kmeans_feature_gb: float = 4.0
    #does some additional prints after selector is assembled
    selector_debug: bool = False
    #do you want to talk to me? No?
    verbose: bool = False
    #just print out the plan, no training
    dry_run: bool = False
    #do you want to run test on the ft-ed model after its trained?
    skip_test: bool = False



def _apply_subset_training_policy(
    config: LoraRunConfig,
    sampling_meta: dict[str, Any],
) -> tuple[LoraRunConfig, dict[str, Any] | None]:
    """
    Essentially it just rescales base_iters given for mlx-lora configuration
    so that it to scale of subset / set when subset training is used
    """
    policy = str(config.subset_training.get("iters_policy", "none")).strip().lower()
    if policy in {"", "none", "fixed"}:
        return config, None
    if policy not in SUBSET_ITERS_POLICIES:
        raise ValueError(
            "`subset_training.iters_policy` must be one of "
            f"{sorted(SUBSET_ITERS_POLICIES | {'none', 'fixed'})}; got {policy!r}."
        )

    base_iters = int(config.mlx_args.get("iters") or 0)
    original_examples = int(sampling_meta.get("original_train_examples") or 0)
    selected_examples = int(sampling_meta.get("selected_train_examples") or 0)
    if base_iters <= 0 or original_examples <= 0 or selected_examples <= 0:
        return config, None

    min_iters = int(config.subset_training.get("min_iters", 1) or 1)
    if min_iters < 1:
        raise ValueError("`subset_training.min_iters` must be at least 1.")
    cap_at_full = bool(config.subset_training.get("cap_at_full_iters", True))

    #scaled_iters normalize number of iterations w.r.t. the subset size
    #so, for exampled, if |subset| = 1/2*|original| then devide the 
    # basic mlx specified iters by 2
    scaled_iters = max(
        min_iters,
        ceil(base_iters * selected_examples / original_examples),
    )
    if cap_at_full:
        scaled_iters = min(base_iters, scaled_iters)

    mlx_args = dict(config.mlx_args)
    mlx_args["iters"] = scaled_iters
    details = {
        "iters_policy": policy,
        "base_iters": base_iters,
        "effective_iters": scaled_iters,
        "original_train_examples": original_examples,
        "selected_train_examples": selected_examples,
        "min_iters": min_iters,
        "cap_at_full_iters": cap_at_full,
    }
    return replace(config, mlx_args=mlx_args), details



def sampled_data_from_configs(args: lora_ft_JobCard, run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """
        Nothing special, just take all the configs associated with a lora run and
        generate a specified training sunset.

        Returns the sampled data path plus metadata.
    """
    config = load_lora_run_config(args.config)
    train_data_dir, sampling_meta = prepare_sampled_data_dir(
            source_data_dir=config.data_dir,
            run_dir=run_dir,
            sample_selector=args.sample_selector,
            max_example=args.max_examples,
            selector_context={
                "base_model": config.base_model,
                "mlx_args": config.mlx_args,
                "seed": config.mlx_args.get("seed", 42),
                "selector_transformations": (
                    args.selector_transformation or ["identity"]
                ),
                "selector_transformation_params": {
                    "thresholded_sign": {
                        "threshold": args.selector_thresholded_sign_threshold,
                    }
                },
                "selector_projection": args.selector_projection,
                "selector_projection_components": (
                    args.selector_projection_components
                ),
                "selector_projection_chunk_size": (
                    args.selector_projection_chunk_size
                ),
                "selector_max_kmeans_feature_bytes": int(
                    args.selector_max_kmeans_feature_gb * 1024**3
                ),
                "shared_feature_cache_root": str(
                    resolve_project_path(config.output_root)
                    / "selector_feature_cache"
                ),
                "selector_debug": args.selector_debug,
            },
        )
    
    return train_data_dir, sampling_meta


def fine_tune_on_random_subset(args: lora_ft_JobCard) -> tuple[LoraRunConfig,RunPaths] | None:
    """
        Runs LoRA fine-tuning job specified by ft_job_args.
    """

    config = load_lora_run_config(args.config)
    if args.base_model is not None:
        config = replace(config, base_model=args.base_model)
    run_name = args.run_name or build_run_name(config)
    paths = prepare_run(config, run_name)

    train_data_dir = config.data_dir
    sampling_meta = None
    subset_training_meta = None
    verbose = (args.verbose!=0)
    if verbose: print("Verbose mode ACTIVATED...")

    if args.sample_selector:
        if verbose: print("We cooking some selectors here...")
        train_data_dir, sampling_meta = sampled_data_from_configs(args, paths.run_dir)
        if verbose: print("Nicely done!")

        if args.selector_debug:
            print(
                "[selector] selected "
                f"{sampling_meta['selected_train_examples']}/"
                f"{sampling_meta['original_train_examples']} training rows; "
                f"sampled_data_dir={sampling_meta['sampled_data_dir']}",
                flush=True,
            )
        config, subset_training_meta = _apply_subset_training_policy(
            config,
            sampling_meta,
        )

    save_lora_run_config(config, paths.resolved_config_path)
    metadata = build_run_metadata(
        config=config,
        paths=paths,
        config_source=args.config,
        status="prepared",
    )
    if sampling_meta is not None:
        metadata["sampling"] = sampling_meta
    if subset_training_meta is not None:
        metadata["subset_training_effective"] = subset_training_meta
    metadata["train_data_dir"] = str(train_data_dir)
    write_metadata(paths.metadata_path, metadata)
    write_mlx_runtime_config(
        paths.mlx_config_path,
        build_mlx_config_payload(config),
    )

    train_command = build_train_command(config, paths, data_dir=train_data_dir)
    write_command(paths.command_path, train_command)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status="prepared",
        ),
    )

    print(f"Run directory: {paths.run_dir}")
    print(f"Adapter directory: {paths.adapter_dir}")
    print(f"Command: {paths.command_path}")

    if args.dry_run:
        return

    metadata["status"] = "running"
    write_metadata(paths.metadata_path, metadata)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status="running",
        ),
    )

    train_exit_code = run_command_with_logging(
        command=train_command,
        log_path=paths.train_log,
        metrics_path=paths.metrics_path,
        source_phase="train",
    )

    status = "completed" if train_exit_code == 0 else "failed"
    test_exit_code: int | None = None

    if verbose and status=="failed": print("Bro, we've failed!...")

    if train_exit_code == 0 and config.test_after_train and not args.skip_test:
        test_command = build_test_command(config, paths, data_dir=train_data_dir)
        test_exit_code = run_command_with_logging(
            command=test_command,
            log_path=paths.test_log,
            metrics_path=paths.metrics_path,
            source_phase="test",
        )
        if test_exit_code != 0:
            status = "failed"

    metadata["status"] = status
    metadata["train_exit_code"] = train_exit_code
    metadata["test_exit_code"] = test_exit_code
    write_metadata(paths.metadata_path, metadata)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status=status,
            train_exit_code=train_exit_code,
            test_exit_code=test_exit_code,
        ),
    )

    if train_exit_code != 0:
        raise SystemExit(train_exit_code)
    if test_exit_code not in {None, 0}:
        raise SystemExit(test_exit_code)
    
    return config, paths

def red_bold_print(msg: str):
    print(f"\033[1m\033[91m{msg}\033[0m")

def _score_with_cache(
    config: KernelRunConfig,
    records: list[PairRecord],
    *,
    adapter_path: str | None,
    scorer_factory: ScorerFactory = ModelScorer,
) -> list[dict[str, Any]]:
    label = "base" if adapter_path is None else Path(adapter_path).name

    cache_path = _score_cache_path(
        config,
        records,
        adapter_path=adapter_path,
    )
    cached = _load_cached_scores(cache_path, records)
    if cached is not None:
        print(f"[Score Cache] Precomputed score found, path={cache_path}")
        train_rows = cached
    else:
        red_bold_print(f"[Score Cache] Missing precomputed scores, path={cache_path}")

        scorer = scorer_factory(config.base_model, adapter_path)
        
        rows = _score_records(scorer, records)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(cache_path, rows)
        print(f"[Score Cache] Score computed and written to path={cache_path}")
        train_rows = rows

        del scorer
        gc.collect()

    return train_rows


def _instantiate_kernel(
    config: KernelRunConfig,
    config_source: str | Path,
    run_name: str | None = None,
) -> dict[str, Any]:
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
    
    dir = resolve_project_path(config.data_dir)
    records = load_pair_split(dir / "train.jsonl", split="train")

    base_rows = _score_with_cache(
        runtime_config,
        records,
        adapter_path=None,
    )
    adapter_rows = _score_with_cache(
        runtime_config,
        records,
        adapter_path=runtime_config.adapter_path,
    )

    merged_rows: list[dict[str, Any]] = []
    for base_row, adapter_row in zip(base_rows, adapter_rows):
        merged_rows.append(
            {
                "pair_id": base_row["pair_id"],
                "split": "train",
                "base_score": base_row["score"],
                "adapter_score": adapter_row["score"],
                "score_delta": adapter_row["score"] - base_row["score"],
                "prompt_offset": base_row["prompt_offset"],
                "sequence_length": base_row["sequence_length"],
                "completion_targets": base_row["completion_targets"],
            }
        )
    score_rows_train = merged_rows
    _write_jsonl(paths.scores_dir / f"train.jsonl", merged_rows)

    feature_backend = create_feature_backend(runtime_config)
    
    feature_paths_train, _ = _extract_features_to_npy(
            config=runtime_config,
            backend=feature_backend,
            records=records,
        )
        
    del feature_backend
    gc.collect()

    train_features = _load_features(feature_paths_train, runtime_config)
    targets = np.asarray(
        [row["score_delta"] for row in score_rows_train],
        dtype=np.float64,
    )
    
    fitted_model, fit_payload = _predict_targets(
        config=runtime_config,
        train_features=train_features,
        train_targets=targets,
    )
    
    return {"model": fitted_model, "payload": fit_payload, "train_features": train_features, "paths": paths}

def get_lora_config_path_from_lora_key(lora_key: str) -> Path:
    return resolve_project_path(DEFAULT_CONFIGS_ROOT)/f"{lora_key}.yaml"

def prepare_lora_job_card(card: lora_ft_JobCard, lora_key: str) -> tuple[lora_ft_JobCard, LoraRunConfig]:
    config = lora_run_config_from_lora_job_card(card)
    hash_name = lora_key

    card.dry_run = False
    card.run_name = hash_name
    card.sample_selector = str(resolve_project_path(SELECTOR_PATHS["random"]))
    card.skip_test = True

    config.output_root = str(resolve_project_path(DEFAULT_LORA_DIRECTORY))
    config.test_after_train = False

    lora_config_path = get_lora_config_path_from_lora_key(hash_name)
    save_lora_run_config(config, lora_config_path)
    card.config = str(lora_config_path)

    return card, config
    
def load_pair_idx(path: str | Path, idx: list[int], split="train") -> list[PairRecord]:
    records: list[PairRecord] = []
    idx.sort()
    if idx[0] < 0:
        raise ValueError("Indices must be non-negative")
    current = 0
    real_index = -1
    
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            else:
                real_index += 1
            if not real_index == idx[current]:
                continue
            else:
                current = min(current+1, len(idx)-1)
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Kernel data row must be an object: {path}:{index + 1}"
                )
            if "prompt" not in payload:
                raise ValueError(
                    f"Kernel data row is missing `prompt`: {path}:{index + 1}"
                )
            if "completion" not in payload:
                raise ValueError(
                    f"Kernel data row is missing `completion`: {path}:{index + 1}"
                )
            records.append(
                PairRecord(
                    pair_id=f"{split}-{index:06d}",
                    split=split,
                    prompt="" if payload["prompt"] is None else str(payload["prompt"]),
                    completion=(
                        ""
                        if payload["completion"] is None
                        else str(payload["completion"])
                    ),
                )
            )
    return records
    

def lora_run_config_from_lora_job_card(card: lora_ft_JobCard)->LoraRunConfig:
    config = load_lora_run_config(card.config)
    return config


def get_run_dir_from_lora_key(lora_key: str, existing: bool = True) -> Path:
   lora_path = resolve_project_path(DEFAULT_LORA_DIRECTORY)
   run_path = lora_path / "runs" / lora_key
   if not run_path.exists() and existing:
      raise FileNotFoundError(f"Runs directory not found: {run_path}")
   return run_path

def get_data_train_path_from_lora_key(lora_key: str, existing: bool = True) -> Path:
   lora_path = resolve_project_path(DEFAULT_LORA_DIRECTORY)
   train_path = lora_path / "runs" / lora_key / "data"
   if not train_path.exists() and existing:
      raise FileNotFoundError(f"Train jsonl file not found: {train_path}")
   return train_path

def get_metadata_path_from_lora_key(lora_key: str, existing: bool = True) -> Path:
    metadata_path = get_run_dir_from_lora_key(lora_key) / "metadata.json"
    if not metadata_path.exists() and existing:
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    return metadata_path

def get_summary_path_from_lora_key(lora_key: str, existing: bool = True) -> Path:
    summary_path = get_run_dir_from_lora_key(lora_key) / "summary.json"
    if not summary_path.exists() and existing:
        raise FileNotFoundError(f"Summary not found at {summary_path}")
    return summary_path

def get_command_path_from_lora_key(lora_key: str, existing: bool = True) -> Path:
    command_path = get_run_dir_from_lora_key(lora_key) / "command.json"
    if not command_path.exists() and existing:
        raise FileNotFoundError(f"Command not found at {command_path}")
    return command_path

def get_adapter_path_from_lora_key(lora_key: str, existing: bool = True) -> Path:
    lora_path = resolve_project_path(DEFAULT_LORA_DIRECTORY)
    adapter_path = lora_path / "adapters" / lora_key
    if not adapter_path.exists() and existing:
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    return adapter_path

def _validate_lora_key(lora_key: str) -> Any:
    GOOD_STATUS = {"completed"}
    ADAPTER_FILES = {"adapter_config.json", "adapters.safetensors"}
    RUN_FILES = {
                "metadata.json", "summary.json", 
                "command.txt", "mlx_config.yaml", 
                "resolved_config.yaml", "metrics.jsonl", 
                }

    lora_path = resolve_project_path(DEFAULT_LORA_DIRECTORY)

    if not lora_path.exists():
        raise FileNotFoundError(f"Path doesn't exist: {lora_path}")
    if not lora_path.is_dir():
        raise FileNotFoundError(f"Lora path is not a directory: {lora_path}")
    
    adapter_dir = get_adapter_path_from_lora_key(lora_key)

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter path is not a directory: {adapter_dir}")

    adapter_files = set(f.name for f in adapter_dir.iterdir() if f.is_file())
    if not ADAPTER_FILES.issubset(adapter_files):
        raise FileNotFoundError(f"Missing adapter files: {ADAPTER_FILES - adapter_files} in {adapter_dir}")
    
    run_dir = get_run_dir_from_lora_key(lora_key)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Runs path is not a directory: {run_dir}")

    run_files = set(f.name for f in run_dir.iterdir() if f.is_file())
    if not run_files.issubset(RUN_FILES):
        raise FileNotFoundError(f"Missing runs files: {RUN_FILES - run_files} in {run_dir}")
    
    meta_path = get_metadata_path_from_lora_key(lora_key)
    get_status = json.loads(meta_path.read_text()).get("status", None)
    if (get_status == None) or (not get_status in GOOD_STATUS):
        raise ValueError(f"The provided run is not completed. Metadata: {meta_path}")

