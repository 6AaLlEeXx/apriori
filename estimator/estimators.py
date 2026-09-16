from __future__ import annotations
from dataclasses import dataclass, asdict, field
import gc
import hashlib
from pathlib import Path
from typing import Any, Literal
import numpy as np
import json
from numpy.typing import NDArray
from paths import resolve_project_path
from mlx_lm import load
from pydantic import BaseModel, Field

from kernel.data import PairRecord, load_pair_split
from kernel.config import save_kernel_run_config, load_kernel_run_config, KernelMethodConfig, KernelRunConfig
from kernel.run import ScorerFactory
import kernel.run as kerun

from kernel.features import create_feature_backend
from kernel.scoring import ModelScorer, score_pair
from kernel.krr import NystromKRRModel, DualKRRModel
from mlops import LoraRunConfig, load_lora_run_config

from estimator.lora_trainer import combined_key, SelectorConfig, LoraTrainer
from estimator.caching import (FeatureRepresentor, resolve_representor, FeatureExtractor,
                               ShardedFeatureFetch, IndexedFeatureFetch, RepresentorName)

KRR = DualKRRModel | NystromKRRModel

DEFAULT_KERNEL_CONFIG_PATH = Path("estimator/kernels/configs")
DEFAULT_FEATURE_CACHE_PATH = Path("estimator/feature_cache")
DEFAULT_KERNEL_OUT_PATH = Path("estimator/kernels/outs")
DEFAULT_KERNEL_WEIGHTSS_PATH = Path("estimator/kernels/weights")


@dataclass
class KernelDataCard:
    data_dir: str = ""
    key: str = ""
    name: str = "benchmark"
    num_records: int | None = None

    def to_dict(self):
        return asdict(self)
    
    def __len__(self):
        if self.num_records is None:
            self.num_records = len(load_pair_split(self.data_dir, "Validate"))
        return self.num_records


class KenerelEstimatorPaths(BaseModel):
    kernel_config_path: str | Path
    weights_path: str | Path
    hash: str
    lora_key: str
    lora_init_key: str

    output_root: str | Path
    data_dir: str | Path

    translator: dict[str, Any] = Field(default_factory=lambda: {"name": "float32", "parameters": {}})

    
@dataclass
class KernelEsimatorPayload:
    backend: FeatureExtractor
    representor: FeatureRepresentor
    config: KernelRunConfig
    model: KRR
    payload: dict[str, Any]
    train_features: NDArray
    backend_key: str


COMPATIBLE_TRANSFORMS = {"identity": {"multitransform", "float32", "float64", "sign_transform", "thresholded_sign_transform"},
                         "sign": {"multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                         "thresholded_sign": {"multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                }


def __save_train_features_as_npy(features: NDArray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)


def __save_kernel_class_as_npz(config: KRR, path: Path | str) -> None: 
    """
        Saves kernel dataclass 'config' as .npz numpy dictionary.
    """  
    if isinstance(config, NystromKRRModel):
        np.savez(
                path, 
                landmark_indices = config.landmark_indices,
                feature_proj = config.feature_proj,
                beta = config.beta,
                ridge_lambda = config.ridge_lambda,
                rank = config.rank,
                )
    elif isinstance(config, DualKRRModel):
        np.savez(
                path, 
                alpha = config.alpha,
                ridge_lambda = config.ridge_lambda,
                )

def save_model(model: KRR, payload: dict[str,Any], train_features: NDArray, path: Path | str) -> None:
    """
        Takes model dataclass object and payload as in _predict_targets.
        Returns nothing, saves everything for the later use.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)

    kernel_path = path/"kernel.npz"
    __save_kernel_class_as_npz(model, kernel_path)

    context = {
               "method": payload["method"],
               "kernel_path": str(kernel_path)
               }
    
    context_path = path/("payload.json")
    context_path.write_text(json.dumps(context, indent=2))

    train_features_path = path/"train_features.npy"
    __save_train_features_as_npy(train_features, train_features_path)


def load_model(path: Path | str) -> dict[str, Any]:
    """
        Takes .npz path to kenel's weights and returns model class
        object and payload as in _predict_targets.
    """
    path = Path(path)
    payload_path = path / "payload.json"

    context = json.loads(payload_path.read_text())
    if not isinstance(context, dict):
        raise ValueError(f"Unexpected payload format in {payload_path}. Context must be a dictionary.")

    method = context.get("method", None)
    if method is None:
        raise ValueError("Method is not specified in the payload!")

    loaded = np.load(path/"kernel.npz")
    train_features = np.load(path/"train_features.npy")

    if method=="nystrom":
        model = NystromKRRModel(landmark_indices=loaded["landmark_indices"],
                               feature_proj=loaded["feature_proj"],
                               beta=loaded["beta"],
                               ridge_lambda=loaded["ridge_lambda"],
                               rank=loaded["rank"],
                              )
        payload = {
            "method": "nystrom",
            "ridge_lambda": model.ridge_lambda,
            "rank": model.rank,
            "num_landmarks": len(model.landmark_indices),
            "landmark_indices": model.landmark_indices.tolist(),
        }

        return {"model": model, "train_features": train_features, "payload": payload}

    elif method=="dual":
        model = DualKRRModel(alpha=loaded["alpha"],
                            ridge_lambda=loaded["ridge_lambda"],
                            )
        payload = {
            "method": "dual",
            "ridge_lambda": model.ridge_lambda,
        }

        return {"model": model, "train_features": train_features, "payload": payload}

        
    raise ValueError(f"Unsupported kernel method: {method}")

def json_hash(dict_config: dict[str, Any]) -> str:
    json_str = json.dumps(dict_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

def normalize_name(name: str | None)->str:
    if name is None:
        return ""
    return name.lower().strip().replace("-", "_")

def _prepare_feature_transforms(config: KernelRunConfig)->KernelRunConfig:
    names = kerun._feature_transform_names(config)
    if len(names) > 1:
        raise ValueError("Chained transformations are not supported.")
    if len(names)==0:
        config.backend_args = {"leaf_filter": "lora_b_only"}
    if len(names)==1:
        args = {"leaf_filter": config.backend_args.get("leaf_filter", "lora_b_only")}
        params = kerun._feature_transform_params(config).get(names[0],{})
        name = normalize_name(names[0])
        if name == "identity":
            args["feature_transform"] = "identity"
            config.backend_args = args
        elif name == "sign":
            args["feature_transform"] = "sign"
            config.backend_args = args
        elif name == "thresholded_sign":
            args["feature_transform"] = "thresholded_sign"
            args["threshold"] = params.get("threshold", 0.)
            config.backend_args = args
        else:
            raise ValueError(f"Unsupported transformation name: {name}")
        
    return config


class BaselineEstimatorPaths(BaseModel):
    base_model: str
    adapter_path: str | Path
    
    key: str | None = None
    lora_key: str | None = None #unique lora run key, in general, might differ from 'key' 

    name: str = "baseline_estimator"  #doesnt have to be unique
    description: str = "simple baseline loss estimator" #doesnt affect anything

    lora_summary: str | Path | None = None
    lora_metadata: str | Path | None = None
    lora_command: str | Path | None = None   


class BaselineEstimator:
    def __init__(self, paths: BaselineEstimatorPaths, lazy = True) -> None:
        self._estimator_paths = paths
        self._model = None
        self._tokenizer = None
        self._loaded = False

        if not lazy:
            self._load()
            self._loaded = True

    def __call__(self, records: list[PairRecord]) -> list[float]:
        if not self._loaded:
            self._load()
            self._loaded = True

        loss = self._estimate_loss(records=records)
        return loss

    def _estimate_loss(self, records: list[PairRecord]) -> list[float]:
        estimated_loss = [score_pair(self._model, self._tokenizer, r)[0] for r in records]

        return estimated_loss

    def _load(self) -> None:
        loaded = load(
                                self._estimator_paths.base_model,
                                adapter_path=str(self._estimator_paths.adapter_path),
                                lazy=False,
                            )
                
        self._model = loaded[0]
        self._tokenizer = loaded[1]
        self._loaded = True


class EstimatorLoadError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class KernelEstimator:
    def __init__(   
                    self, 
                    config: KenerelEstimatorPaths, 
                    lazy: bool = True, 
                    representor_name: RepresentorName = 'float32',
                    representor_args: dict[str, Any] = {},
                    chache: Literal["indexed", "sharded"] = "sharded",
                    data_card: KernelDataCard | None = None,
                    block_size: int = 512,
                    max_blocks: int | None = 64, 
                    cyclic : bool = True,
                    append_only : bool = True,
                 ) -> None:
        self.__estimator_config = config

        self.__loaded_backend: FeatureExtractor | None = None
        self.__loaded_config: KernelRunConfig | None = None
        self.__loaded_model: str | None = None
        self.__loaded_payload: Any = None
        self.__loaded_train_features: Any = None
        self.__loaded_backend_key: str | None = None
        self.__loaded_representor_name: RepresentorName = representor_name
        self.__loaded_representor_args: dict[str, Any] = representor_args

        self.__cache = chache
        self.__sharded_block_size = block_size
        self.__sharded_append_only = append_only
        self.__sharded_max_blocks = max_blocks
        self.__sharded_cyclic = cyclic
        self.__data_card = data_card

        self.__loaded = False
        self.__lazy = lazy

    def __post_init__(self) -> None:
        if not self.__lazy: self.__load()

    def __call__(self, records: list[PairRecord] = [], idx: list[int] = []):
        if not self.__loaded: self.__load()

        if self.__cache == 'sharded':
            if not records:
                raise ValueError("Must specify the list of records to use sharded caching")
            loss = self.__estimate_loss_sharded(records=records)
        else:
            if not idx:
                raise ValueError("Must specify the list of records id to use indexed caching")
            if self.__data_card is None:
                raise ValueError("Must specify dataset to use caching by index")
            if not records:
                records = [load_pair_split(self.__data_card.data_dir, "loss_eval")[i] for i in idx]
            loss = self.__estimate_loss_indexed(idx=idx, records=records, dataset=self.__data_card)
        return loss

    def __load(self) -> None:
        kernel_config = _prepare_feature_transforms(load_kernel_run_config(self.__estimator_config.kernel_config_path))
        kernel_payload = load_model(self.__estimator_config.weights_path)
        backend = FeatureExtractor(kernel_config)

        is_multitransform = self.__loaded_representor_name == 'multitransform'
        backend_key = combined_key(
                                    lora_init_key=self.__estimator_config.lora_init_key, 
                                    backend_key=backend.transform_hash if not is_multitransform else backend.striped_hash,
                                )

        self.__loaded_backend = backend
        self.__loaded_config = kernel_config
        self.__loaded_model = kernel_payload['model']
        self.__loaded_payload = kernel_payload['payload']
        self.__loaded_train_features = kernel_payload['train_features']
        self.__loaded_backend_key = backend_key
        self.__loaded = True

    def __ensure_backend_key(self) -> str:
        if self.__loaded_backend_key is None:
            raise EstimatorLoadError("Backend hash was not loaded")
        return self.__loaded_backend_key

    def __ensure_backend(self) -> FeatureExtractor:
        if self.__loaded_backend is None:
            raise EstimatorLoadError("LoRa backend is has not been loaded")
        return self.__loaded_backend

    def __ensure_kernel_config(self) -> KernelRunConfig:
        if self.__loaded_config is None:
            raise EstimatorLoadError("LoRa backend is has not been loaded")
        return self.__loaded_config

    def __representor_hash(self) -> str:
        if self.__loaded_representor_args.get('dim', None) is None:
            raise ValueError(f"Input dimension is not defined")
        return resolve_representor(self.__loaded_representor_name, **self.__loaded_representor_args).hash()

    def __estimate_loss_indexed(self, idx: list[int], records: list[PairRecord], dataset: KernelDataCard):
        self.__loaded_representor_args['dim'] = self.__ensure_backend().ensure_output_dim(records[0])

        hash = combined_key(backend_key=self.__ensure_backend_key(), 
                            representor_key=self.__representor_hash(), 
                            dataset_key=dataset.key,
                            cache_name = "indexed",
                            )
        
        cache_config_path = Path(DEFAULT_FEATURE_CACHE_PATH)/f'{hash}'

        if cache_config_path.exists():
            caching_master = IndexedFeatureFetch.load(cache_config_path/'represented_meta.json')
        else:
            caching_master = IndexedFeatureFetch.get(
                                                        root = cache_config_path,
                                                        input_dim = self.__ensure_backend().dim or 0,
                                                        feature_num = len(dataset),
                                                        representor_name = self.__loaded_representor_name,
                                                        representor_args = self.__loaded_representor_args,                                        
                                                    )
            caching_master.save(cache_config_path/'represented_meta.json')

        features = np.stack(caching_master.fetch(self.__ensure_backend(), idx, records))
        
        pred_delta = kerun._predict_split(
                config=self.__ensure_kernel_config(),
                fitted_model=self.__loaded_model,
                fit_payload=self.__loaded_payload,
                train_features=self.__loaded_train_features,
                split_features=features,
            )
        
        return pred_delta
  
    def __estimate_loss_sharded(self, records: list[PairRecord]):
        self.__loaded_representor_args['dim'] = self.__ensure_backend().ensure_output_dim(records[0])
       
        hash = combined_key(    
                                backend_key=self.__loaded_backend_key, 
                                representor_key=self.__representor_hash(),
                                cache_name = "sharded",
                                dim = self.__ensure_backend().dim or 0,
                                size = self.__sharded_block_size,
                                max_blocks = self.__sharded_block_size,
                                cyclic = self.__sharded_cyclic,
                            )
        cache_config_path = Path(DEFAULT_FEATURE_CACHE_PATH)/f'{hash}'
        
        if cache_config_path.exists():
            caching_master = ShardedFeatureFetch.load(cache_config_path/'represented_meta.json')
        else:
            caching_master = ShardedFeatureFetch.get(
                                                        root=cache_config_path,
                                                        input_dim = self.__ensure_backend().dim or 0,
                                                        size = self.__sharded_block_size,
                                                        max_blocks = self.__sharded_max_blocks,
                                                        cyclic_writes = self.__sharded_cyclic,
                                                        representor_name =  self.__loaded_representor_name,
                                                        representor_args =  self.__loaded_representor_args,
                                                        append_only = self.__sharded_append_only,
                                                    )
            caching_master.save(cache_config_path/'represented_meta.json')

        features = np.stack(caching_master.fetch(self.__ensure_backend(), records))
        
        pred_delta = kerun._predict_split(
                                            config=self.__ensure_kernel_config(),
                                            fitted_model=self.__loaded_model,
                                            fit_payload=self.__loaded_payload,
                                            train_features=self.__loaded_train_features,
                                            split_features=features,
                                        )
                                    
        return pred_delta


@dataclass(kw_only=True)
class KernelArgs():
    seed: int = 42
    limit: int = 128
    kernel: KernelMethodConfig = field(default_factory=KernelMethodConfig)
    backend_args: dict[str, Any] = field(default_factory=dict)


class Trainer:
    def __init__(self, lora_config: LoraRunConfig, selector_args: SelectorConfig) -> None:
        self._lora_config = lora_config
        self._selector_args = selector_args
        self._lora_trainer = LoraTrainer(lora_config=lora_config,
                                         selector_args=selector_args,
                                         )
        self._lora_payload: dict[str, Any] | None = None

    @classmethod
    def smoke_kernel_estimator(cls) -> KenerelEstimatorPaths:
        DEFAULT_LORA_RUN_CONFIG = "configs/dolly_smoke.yaml"

        lora_config = load_lora_run_config(DEFAULT_LORA_RUN_CONFIG)
        selector_args = SelectorConfig()
        kernel_args = KernelArgs()

        trainer = cls(lora_config, selector_args)
        res = trainer.train_kernel(kernel_args)

        return res 

    @classmethod
    def smoke_baseline_estimator(cls) -> BaselineEstimatorPaths:
        DEFAULT_LORA_RUN_CONFIG = "configs/dolly_smoke.yaml"

        lora_config = load_lora_run_config(DEFAULT_LORA_RUN_CONFIG)
        selector_args = SelectorConfig()

        trainer = cls(lora_config, selector_args)
        res = trainer.train_baseline()

        return res 

    @staticmethod
    def __kernel_run_hash(config: KernelRunConfig) -> str:
        def records_payload(records: list[PairRecord]) -> str:
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
        GET_KERNEL_SPECS = ["backend", "target", "seed", "train_limit", "kernel", "backend_args"]

        dict_config = config.to_dict()

        data_path = resolve_project_path(config.data_dir)/"train.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"No training data path found at {data_path}")

        records = load_pair_split(data_path, "train")
        records_key = records_payload(records)

        specs = ""
        for key in GET_KERNEL_SPECS:
            val = dict_config.get(key, None)
            if val is None:
                raise ValueError(f"Some information is missing: {key}")
            if isinstance(val, dict):
                payload = json_hash(val)
            else:
                payload = f"{val}"
            specs = f"{specs}; {key}={payload}"
        specs_key = hashlib.sha256(specs.encode("utf-8")).hexdigest()

        key = hashlib.sha256(f"{specs_key}_{records_key}".encode("utf-8")).hexdigest()

        return key

    @staticmethod
    def __score_with_cache(
        config: KernelRunConfig,
        records: list[PairRecord],
        *,
        adapter_path: str | None,
        scorer_factory: ScorerFactory = ModelScorer,
    ) -> list[dict[str, Any]]:
        cache_path = kerun._score_cache_path(
            config,
            records,
            adapter_path=adapter_path,
        )
        cached = kerun._load_cached_scores(cache_path, records)
        if cached is not None:
            print(f"[Score Cache] Precomputed score found, path={cache_path}")
            train_rows = cached
        else:
            print(f"[Score Cache] Missing precomputed scores, path={cache_path}")

            scorer = scorer_factory(config.base_model, adapter_path)
            
            rows = kerun._score_records(scorer, records)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            kerun._write_jsonl(cache_path, rows)
            print(f"[Score Cache] Score computed and written to path={cache_path}")
            train_rows = rows

            del scorer
            gc.collect()

        return train_rows

    @staticmethod
    def __instantiate_kernel(
        config: KernelRunConfig,
        config_source: str | Path,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        runtime_config = kerun.resolve_kernel_run_config(config)
        if runtime_config.target != "score_delta":
            raise ValueError("Only `score_delta` is currently implemented for kernel runs.")
        kerun.validate_kernel_run_inputs(runtime_config)
        run_name = run_name or kerun.build_kernel_run_name(runtime_config)
        paths = kerun.prepare_kernel_run(runtime_config, run_name)
        save_kernel_run_config(runtime_config, paths.resolved_config_path)
        metadata = kerun.build_kernel_metadata(
            config=runtime_config,
            paths=paths,
            config_source=config_source,
            status="running",
        )
        kerun.write_json(paths.metadata_path, metadata)
        
        dir = resolve_project_path(config.data_dir)
        records = load_pair_split(dir / "train.jsonl", split="train")

        base_rows = Trainer.__score_with_cache(
            runtime_config,
            records,
            adapter_path=None,
        )
        adapter_rows =  Trainer.__score_with_cache(
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
        kerun._write_jsonl(paths.scores_dir / f"train.jsonl", merged_rows)

        feature_backend = create_feature_backend(runtime_config)
        
        feature_paths_train, _ = kerun._extract_features_to_npy(
                config=runtime_config,
                backend=feature_backend,
                records=records,
            )
            
        del feature_backend
        gc.collect()

        train_features = kerun._load_features(feature_paths_train, runtime_config)
        targets = np.asarray(
            [row["score_delta"] for row in score_rows_train],
            dtype=np.float64,
        )
        
        fitted_model, fit_payload = kerun._predict_targets(
            config=runtime_config,
            train_features=train_features,
            train_targets=targets,
        )
        
        return {"model": fitted_model, "payload": fit_payload, "train_features": train_features, "paths": paths}

    @staticmethod
    def __get_kernel_paths(kernel_key: str) -> dict[str, Path]:
        paths = {'weights' : resolve_project_path(DEFAULT_KERNEL_WEIGHTSS_PATH/f'{kernel_key}'),
                 'config' : resolve_project_path(DEFAULT_KERNEL_CONFIG_PATH)/f"{kernel_key}.json",
                 'out' : resolve_project_path(DEFAULT_KERNEL_OUT_PATH)}
        return paths

    def train_baseline(self) -> BaselineEstimatorPaths:
        payload = self._lora_trainer.train()
        self._lora_payload = payload

        baseline_paths = BaselineEstimatorPaths(
                                                base_model=payload['base_model'],
                                                adapter_path=payload['paths']['adapter'],
                                                key=payload['lora_hash'].lora_run_hash,
                                                lora_key=payload['lora_hash'].lora_run_hash,
                                                lora_command=payload['paths']['command'],
                                                lora_metadata=payload['paths']['metadata'],
                                                lora_summary=payload['paths']['summary'],
                                                )

        return baseline_paths

    def train_kernel(self, kernel_args: KernelArgs) -> KenerelEstimatorPaths:
        if self._lora_payload is None:
            self.train_baseline()

        kernel_config = self.__get_estimator(kernel_args=kernel_args)
        return kernel_config

    def __get_estimator(self, kernel_args: KernelArgs) -> KenerelEstimatorPaths:
        if self._lora_payload is None:
            raise ValueError("LoRa training payload is not prepared")
        lora_paths = self._lora_payload['paths']
        lora_hash = self._lora_payload['lora_hash']
        lora_key = lora_hash.lora_run_hash
        lora_init_key = lora_hash.lora_init_hash

        kernel_config = KernelRunConfig(
                                        dataset_name="KRR Estimator Run Data",
                                        task="estimate loss",
                                        base_model=self._lora_payload['base_model'],
                                        data_dir=lora_paths['train'],
                                        adapter_path=str(lora_paths['adapter']),
                                        seed=kernel_args.seed,
                                        train_limit=kernel_args.limit,
                                        test_limit=kernel_args.limit,
                                        valid_limit=kernel_args.limit,
                                        kernel=kernel_args.kernel,
                                        backend_args=kernel_args.backend_args,
                                        run_tags=["KRR", "loss estimator", "estimator"],
                                        notes="Part of the krr loss estimator routine",
                                        )
                                        
        kernel_config = _prepare_feature_transforms(kernel_config)
        estimator_key = combined_key(lora_key=lora_key, kernel_key=self.__kernel_run_hash(kernel_config))
        kernel_paths = self.__get_kernel_paths(estimator_key)

        # Ensure all paths are str:
        #   data_dir: str
        #   adapter_path: str
        #   output_root: str
        #   source_config: dict[str, str]
 
        kernel_config.output_root = str(kernel_paths['out'])
        kernel_config.data_dir = str(kernel_config.data_dir)
        kernel_config.adapter_path = str(kernel_config.adapter_path)
        kernel_config.source_config = {key : str(val) for key, val in kernel_config.source_config.items()}

        save_kernel_run_config(kernel_config, kernel_paths['config'])
        kerun.validate_kernel_run_inputs(kernel_config)

        if not kernel_paths['weights'].exists():
            kernel_payload =  Trainer.__instantiate_kernel(kernel_config, kernel_paths['config'], estimator_key)
            save_model(
                        model=kernel_payload["model"],
                        payload=kernel_payload["payload"],
                        train_features=kernel_payload["train_features"],
                        path = kernel_paths['weights'],
                    )

        return KenerelEstimatorPaths(
                                    kernel_config_path=str(kernel_paths['config']),
                                    weights_path=str(kernel_paths['weights']),
                                    lora_key=lora_key,
                                    lora_init_key=lora_init_key,
                                    hash=estimator_key,
                                    output_root=str(kernel_paths['out']),
                                    data_dir=str(lora_paths['train']),
                                    )
