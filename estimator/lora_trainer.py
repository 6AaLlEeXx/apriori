from __future__ import annotations

import hashlib
from math import ceil
from typing import Any
from dataclasses import dataclass, replace, asdict
from mlops import RunPaths, save_lora_run_config, load_lora_run_config

import json

from mlops import (
    build_mlx_config_payload,
    build_run_metadata,
    build_summary,
    build_train_command,
    LoraRunConfig,
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
    load_pair_split, 
)

from kernel.data import PairRecord
from kernel.krr import NystromKRRModel, DualKRRModel

SELECTOR_PATHS = {
                    "random": "selectors/random.py",
                    "identity": "selectors/identity.py",
                    "lora_ntk_kmeans": "selector/lora_ntk_kmeans.py"
                 }

DEFAULT_LORA_DIRECTORY = "estimator/lora_models"
DEFAULT_CONFIGS_ROOT = "estimator/configs"
DEFAULT_LORA_RUN_CONFIG = "configs/dolly_smoke.yaml"

from pathlib import Path
KRR = NystromKRRModel | DualKRRModel

SUBSET_ITERS_POLICIES = {
    "match_full_exposure",
    "match_full_passes",
    "per_example",
}

DEFAULT_SELECTOR_PATH = "selectors/random.py"

# @dataclass
# class LoraRunConfig:
#     base_model: str = "mlx-community/SmolLM2-1.7B-Instruct"
#     mlx_command: str = "mlx_lm.lora"
#     mlx_args: dict[str, Any] = field(default_factory=dict)
#     subset_training: dict[str, Any] = field(default_factory=dict)
#     extra_args: list[str] = field(default_factory=list)


@dataclass
class SelectorConfig():
    sample_selector : str = DEFAULT_SELECTOR_PATH
    max_examples : int = 256
    selector_transformation : list[str] | None = None
    selector_projection : str = 'identity'
    selector_projection_components : Any = None
    selector_thresholded_sign_threshold : float = 0.01
    selector_projection_chunk_size : int = 16
    selector_max_kmeans_feature_gb : float = 4.0
    selector_debug : bool = False


def combined_key(**keys)->str:
    key = ""
    for k, v in keys.items():
        key = f"{key}{k}={v};"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

class LoraRunHash:
    def __init__(self, lora_run_config: LoraRunConfig, selector_args: SelectorConfig) -> None:
        self.__computed_lora_run_hash = self.__lora_run_hash(
                                                    config=lora_run_config,
                                                    selector_args=selector_args,
                                                 )
        self.__computed_lora_init_hash = self.__lora_init_hash(lora_run_config)

    @property
    def lora_run_hash(self):
        return self.__computed_lora_run_hash

    @property
    def lora_init_hash(self):
        return self.__computed_lora_init_hash

    @property
    def config_path(self):
        return self.__config_path_from_lora_hash(self.lora_run_hash)

    @staticmethod
    def __config_path_from_lora_hash(lora_run_hash: str) -> Path:
        return resolve_project_path(DEFAULT_CONFIGS_ROOT)/f"{lora_run_hash}.yaml"

    @staticmethod
    def __lora_trainset_hash(config: LoraRunConfig, selector_args: dict[str, Any]) -> str:
        records_key = LoraRunHash.__dataset_hash(config.data_dir)
        selector_key = LoraRunHash.__selector_hash(selector_args, config)

        encoded = f"selector={selector_key}; data={records_key}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def __train_hyperparameters_hash(lora_config: LoraRunConfig) -> str:
        # We need to hash the following parametres:
        #     base_model: str
        #     mlx_command: str
        #     mlx_args: dict[str, Any]
        #     subset_training: dict[str, Any]
        #     extra_args: list[str]

        encoded = f'''  base_model={lora_config.base_model};
                        mlx_command={lora_config.mlx_command};
                        mlx_args={json.dumps(lora_config.mlx_args)};
                        subset_training={json.dumps(lora_config.subset_training)};
                        extra_args={",".join(lora_config.extra_args)};
                    '''.encode("utf-8")
        
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def __lora_init_hash(lora_config: LoraRunConfig) -> str:
        config = lora_config.to_dict()
        
        base_model = config.get("base_model", None)
        if base_model is None:
            raise ValueError("Base model is not defined")
        
        mlx_args = config.get("mlx_args", None)
        if mlx_args is None or not isinstance(mlx_args,dict):
            raise ValueError("Expected mlx arguments are missing")
        seed = mlx_args.get("seed", None)
        if seed is None:
            raise ValueError("Seed is not provided")
        lora_parameters = mlx_args.get("lora_parameters", None)
        if lora_parameters is None or not isinstance(lora_parameters, dict):
            raise ValueError("LoRA parameters are missing")
        
        lora_parameters = json.dumps(
                                        lora_parameters,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                        default=str,
                                    )
        
        encoded = f"base_model={base_model}; seed={seed}; lora_parameters={lora_parameters}".encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def __dataset_hash(data_dir: str | Path) -> str:
        def _records_payload(records: list[PairRecord]) -> str:
            digest = hashlib.sha256()
            for record in records:
                digest.update(
                    json.dumps(
                        {
                            "prompt": record.prompt,
                            "completion": record.completion,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
            return digest.hexdigest()
        
        data_dir = resolve_project_path(Path(data_dir))
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not data_dir.is_dir():
            raise ValueError(f"Data directory is not a directory: {data_dir}")
        
        records = load_pair_split(data_dir/"train.jsonl", "data_key_data")
        key = _records_payload(records)
        
        return key

    @staticmethod
    def __selector_hash(selector_args: dict[str, Any], config: LoraRunConfig) -> str:
        name = Path(selector_args['sample_selector']).stem

        def _hash_random_selector() -> str:
            seed = config.mlx_args.get("seed", 42)
            max_examples = selector_args["max_examples"]
            encoded = f"random_selector_seed={seed} and random_selector_max_examples={max_examples}".encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        def _hash_identity_selector() -> str:
            encoded = f"do nothing".encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        if name == "random":
            return _hash_random_selector()
        elif name == "identity":
            return _hash_identity_selector()
        else:
            raise ValueError(f"LoRa Trainer doesn't support selector with name {name}.")

    @staticmethod
    def __lora_run_hash(config: LoraRunConfig, selector_args: SelectorConfig) -> str:
        """
            Encodes records and lora parameters(ignoring some) into a hash. 
        """
        payload_key = LoraRunHash.__train_hyperparameters_hash(config)
        trainset_key = LoraRunHash.__lora_trainset_hash(config, asdict(selector_args))
        encoded = f"{payload_key}_{trainset_key}".encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

class LoraTrainer:
    def __init__(self, lora_config: LoraRunConfig, selector_args: SelectorConfig) -> None:
        self.__lora_config = lora_config
        self.__selector_args = selector_args

        self.__lora_hash = LoraRunHash(self.__lora_config, self.__selector_args)
        self.__lora_config.output_root = str(resolve_project_path(DEFAULT_LORA_DIRECTORY))
        self.__config_path = self.__lora_hash.config_path
        self.__run_name = self.__lora_hash.lora_run_hash

    @classmethod
    def smoke_run(cls) -> dict[str, Any]:
        lora_config = load_lora_run_config(DEFAULT_LORA_RUN_CONFIG)
        selector_args = SelectorConfig()
        trainer = cls(lora_config, selector_args)
        res = trainer.train()

        return res                               

    def train(self) -> dict[str, Any]:
        save_lora_run_config(self.__lora_config, self.__config_path)
        paths = self.__get_paths()
        
        if paths['adapter'].exists() or paths['run'].exists():
            self.__validate()
            print(f"Found existing run for {self.__run_name}, loading metadata and summary")
        else:
            print(f"No existing run found for {self.__run_name}, starting fine-tuning job")
            self.__fine_tune()

        return {
                'paths' : paths, 
                'lora_hash' : self.__lora_hash, 
                'base_model' : self.__lora_config.base_model,
                }

    def __get_paths(self):
        return self.__paths_from_lora_hash(self.__run_name)

    @staticmethod
    def __paths_from_lora_hash(lora_run_hash: str):
        lora_path = resolve_project_path(DEFAULT_LORA_DIRECTORY)
        run_path = lora_path / "runs" / lora_run_hash

        paths = {
                    'lora': lora_path,
                    'run': run_path,
                    'train': lora_path / "runs" / lora_run_hash / "data",
                    'adapter': lora_path / "adapters" / lora_run_hash,
                    'metadata': run_path / "metadata.json",
                    'summary': run_path / "summary.json",
                    'command': run_path / "command.json",
                }

        return paths

    def __validate(self) -> Any:
        GOOD_STATUS = {"completed"}
        ADAPTER_FILES = {"adapter_config.json", "adapters.safetensors"}
        RUN_FILES = {
                    "metadata.json", "summary.json", 
                    "command.txt", "mlx_config.yaml", 
                    "resolved_config.yaml", "metrics.jsonl", 
                    }

        paths = self.__get_paths()

        if not paths['lora'].exists():
            raise FileNotFoundError(f"Path doesn't exist: {paths['lora']}")
        if not paths['lora'].is_dir():
            raise FileNotFoundError(f"Lora path is not a directory: {paths['lora']}")

        if not paths['adapter'].exists():
            raise FileNotFoundError(f"Path doesn't exist: {paths['adapter']}")
        if not paths['adapter'].is_dir():
            raise FileNotFoundError(f"Adapter path is not a directory: {paths['adapter']}")

        adapter_files = set(f.name for f in paths['adapter'].iterdir() if f.is_file())
        if not ADAPTER_FILES.issubset(adapter_files):
            raise FileNotFoundError(f"Missing adapter files: {ADAPTER_FILES - adapter_files} in {paths['adapter']}")

        if not paths['run'].exists():
                raise FileNotFoundError(f"Path doesn't exist: {paths['run']}")
        if not paths['run'].is_dir():
            raise FileNotFoundError(f"Runs path is not a directory: {paths['run']}")

        run_files = set(f.name for f in paths['run'].iterdir() if f.is_file())
        if not run_files.issubset(RUN_FILES):
            raise FileNotFoundError(f"Missing run files: {RUN_FILES - run_files} in {paths['run']}")
        
        get_status = json.loads(paths['metadata'].read_text()).get("status", None)
        if (get_status is None) or (not get_status in GOOD_STATUS):
            raise ValueError(f"The provided run is not completed. Metadata: {paths['metadata']}")


    def __apply_subset_training_policy(self, sampling_meta: dict[str, Any]) -> tuple[LoraRunConfig, dict[str, Any] | None]:
        """
        Essentially it just rescales base_iters given for mlx-lora configuration
        so that it to scale of subset / set when subset training is used
        """
        policy = str(self.__lora_config.subset_training.get("iters_policy", "none")).strip().lower()
        if policy in {"", "none", "fixed"}:
            return self.__lora_config, None
        if policy not in SUBSET_ITERS_POLICIES:
            raise ValueError(
                "`subset_training.iters_policy` must be one of "
                f"{sorted(SUBSET_ITERS_POLICIES | {'none', 'fixed'})}; got {policy!r}."
            )

        base_iters = int(self.__lora_config.mlx_args.get("iters") or 0)
        original_examples = int(sampling_meta.get("original_train_examples") or 0)
        selected_examples = int(sampling_meta.get("selected_train_examples") or 0)
        if base_iters <= 0 or original_examples <= 0 or selected_examples <= 0:
            return self.__lora_config, None

        min_iters = int(self.__lora_config.subset_training.get("min_iters", 1) or 1)
        if min_iters < 1:
            raise ValueError("`subset_training.min_iters` must be at least 1.")
        cap_at_full = bool(self.__lora_config.subset_training.get("cap_at_full_iters", True))

        scaled_iters = max(
            min_iters,
            ceil(base_iters * selected_examples / original_examples),
        )
        if cap_at_full:
            scaled_iters = min(base_iters, scaled_iters)

        mlx_args = dict(self.__lora_config.mlx_args)
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
        return replace(self.__lora_config, mlx_args=mlx_args), details


    def __sampled_data_from_configs(self, run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
        """
            Nothing special, just take all the configs associated with a lora run and
            generate a specified training sunset.

            Returns the sampled data path plus metadata.
        """
        train_data_dir, sampling_meta = prepare_sampled_data_dir(
                source_data_dir=self.__lora_config.data_dir,
                run_dir=run_dir,
                sample_selector=self.__selector_args.sample_selector,
                max_example=self.__selector_args.max_examples,
                selector_context={
                    "base_model": self.__lora_config.base_model,
                    "mlx_args": self.__lora_config.mlx_args,
                    "seed": self.__lora_config.mlx_args.get("seed", 42),
                    "selector_transformations": (
                        self.__selector_args.selector_transformation or ["identity"]
                    ),
                    "selector_transformation_params": {
                        "thresholded_sign": {
                            "threshold": self.__selector_args.selector_thresholded_sign_threshold,
                        }
                    },
                    "selector_projection": self.__selector_args.selector_projection,
                    "selector_projection_components": (
                        self.__selector_args.selector_projection_components
                    ),
                    "selector_projection_chunk_size": (
                        self.__selector_args.selector_projection_chunk_size
                    ),
                    "selector_max_kmeans_feature_bytes": int(
                        self.__selector_args.selector_max_kmeans_feature_gb * 1024**3
                    ),
                    "shared_feature_cache_root": str(
                        resolve_project_path(self.__lora_config.output_root)
                        / "selector_feature_cache"
                    ),
                    "selector_debug": self.__selector_args.selector_debug,
                },
            )
        
        return train_data_dir, sampling_meta

    def __fine_tune(self) -> RunPaths | None:
        def write() -> None:
            write_metadata(paths.metadata_path, metadata)
    
            write_summary(
                paths.summary_path,
                build_summary(
                    config=self.__lora_config,
                    paths=paths,
                    metadata=metadata,
                    status=metadata['status'],
                ),
            )

        paths = prepare_run(self.__lora_config, self.__run_name)
        metadata = build_run_metadata(
                                                    config=self.__lora_config,
                                                    paths=paths,
                                                    config_source=self.__config_path,
                                                    status="prepared",
                                                    )
        train_data_dir = self.__lora_config.data_dir
        sampling_meta = None
        subset_training_meta = None

        if self.__selector_args.sample_selector:
            train_data_dir, sampling_meta = self.__sampled_data_from_configs(paths.run_dir)
            self.__lora_config, subset_training_meta = self.__apply_subset_training_policy(sampling_meta)

        save_lora_run_config(self.__lora_config, paths.resolved_config_path)
    
        if sampling_meta is not None:
            metadata["sampling"] = sampling_meta
        if subset_training_meta is not None:
            metadata["subset_training_effective"] = subset_training_meta
        metadata["train_data_dir"] = str(train_data_dir)

        write_mlx_runtime_config(
                                    paths.mlx_config_path,
                                    build_mlx_config_payload(self.__lora_config),
                                )
        write()

        train_command = build_train_command(self.__lora_config, paths, data_dir=train_data_dir)
        write_command(paths.command_path, train_command)

        metadata["status"] = "running"
        write()

        train_exit_code = run_command_with_logging(
            command=train_command,
            log_path=paths.train_log,
            metrics_path=paths.metrics_path,
            source_phase="train",
        )

        status = "completed" if train_exit_code == 0 else "failed"

        metadata["status"] = status
        metadata["train_exit_code"] = train_exit_code
        write()

        if train_exit_code != 0:
            raise SystemExit(train_exit_code)

        return paths