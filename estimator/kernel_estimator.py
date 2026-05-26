from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import hashlib
import json
from numpy.typing import NDArray

from estimator.keys import generate_apriori_lora_key, get_kernel_run_key, get_kernel_estimator_key, kernelANDdataset_key
from paths import resolve_project_path

from kernel.data import PairRecord
from kernel.config import save_kernel_run_config, load_kernel_run_config
from kernel.run import _predict_split, KernelRunConfig
from kernel.krr import NystromKRRModel, DualKRRModel

from estimator.jobs import (
                            get_adapter_path_from_lora_key, 
                            get_run_dir_from_lora_key, 
                            get_data_train_path_from_lora_key, 
                            standard_lora_job_card, 
                            lora_ft_JobCard, 
                            fine_tune_on_random_subset, 
                            _instantiate_kernel,
                            _validate_lora_key,
)

from estimator.kernel_caching import (
    _fetch_features_by_idx, 
    get_cache_meta, compare_cache, 
    CacheMeta, _prepare_cache_matrix, 
    write_cache_meta,
)
from estimator.extractor import FeatureExtractor, representor_parameters_from_backend
from kernel.run import _feature_transform_names, _feature_transform_params

KRR = DualKRRModel | NystromKRRModel


@dataclass
class KernelDataCard:
    data_dir: str = ""
    key: str = ""
    name: str = "benchmark"

    def to_dict(self):
        return asdict(self)

@dataclass
class KenerelEstimatorCard:
    kernel_config_path: str | Path
    weights_path: str | Path
    estimator_key: str
    lora_key: str

    output_root: str | Path
    data_dir: str | Path

    translator: str | None = None

    def to_dict(self)->dict[str, Any]:
        iamdict = asdict(self)

        iamdict["kernel_config_path"] = str(iamdict["kernel_config_path"])
        iamdict["weights_path"] = str(iamdict["weights_path"])
        iamdict["output_root"] = str(iamdict["output_root"])
        iamdict["data_dir"] = str(iamdict["data_dir"])
   
        return asdict(self)
    
@dataclass
class KernelEsimatorPayload:
    backend: FeatureExtractor
    config: KernelRunConfig
    model: KRR
    payload: dict[str, Any]
    train_features: NDArray
    key: str
    translator: str | None = None


def _save_train_features_as_npy(features: NDArray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)


def _save_kernel_class_as_npz(config: KRR, path: Path | str) -> None: 
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

def _save_model(model: KRR, payload: dict[str,Any], train_features: NDArray, path: Path | str) -> None:
    """
        Takes model dataclass object and payload as in _predict_targets,
        also path.npz is specified.
        Returns nothing, just saves everything for the later use,
        recommended to be used with hash-path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)

    kernel_path = path/"kernel.npz"
    _save_kernel_class_as_npz(model, kernel_path)

    context = {
               "method": payload["method"],
               "kernel_path": str(kernel_path)
               }
    
    context_path = path/("payload.json")
    context_path.write_text(json.dumps(context, indent=2))

    train_features_path = path/"train_features.npy"
    _save_train_features_as_npy(train_features, train_features_path)


def _load_model(path: Path | str) -> dict[str, Any]:
    """
        Takes .npz path to kenel's weights and returns model class
        object and payload as in _predict_targets.
    """
    path = Path(path)
    payload_path = path / "payload.json"

    context = json.loads(payload_path.read_text())
    if not isinstance(context, dict):
        raise ValueError(f"Unexpected payload format in {payload_path}")

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


def get_chache_meta_path_in_root(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Cache root doesn't exist at {path}")
    meta_path = path/"meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Didn't find the metadata at {meta_path}")
    return meta_path


DEFAULT_KERNEL_CONFIG_PATH = "estimator/kernels/configs"
DEFAULT_FEATURE_CACHE_PATH = "estimator/feature_cache"
DEFAULT_KERNEL_OUT_PATH = "estimator/kernels/outs"

def get_cache_root(key: str) -> Path:
    return resolve_project_path(DEFAULT_FEATURE_CACHE_PATH)/key

def get_cache_meta_path(key: str) -> Path:
    return get_cache_root(key)/"meta.json"

def get_cache_matrix_path(key: str) -> Path:
    return get_cache_root(key)/"features.npy"

def get_kernel_weights_path(key: str) -> Path:
    weights_path = resolve_project_path(f"estimator/kernels/weights/{key}")
    return weights_path

def get_kernel_config_path(key: str) -> Path:
    config_path = resolve_project_path(DEFAULT_KERNEL_CONFIG_PATH)/f"{key}.json"
    return config_path

def get_kernel_out_path(key: str) -> Path:
    out_path = resolve_project_path(DEFAULT_KERNEL_OUT_PATH)/key
    return out_path

SUPPORTED_TRANSFORMS = {"identity", "sign", "threshlded_sign"}

def is_translator_a_feature_trasformation(translator_name: str)->bool:
    return True

def normalize_name(name: str | None)->str:
    if name is None:
        return ""
    return name.lower().strip().replace("-", "_")


def standardize_kernel_backend_args(config: KernelRunConfig)->KernelRunConfig:
    names = _feature_transform_names(config)
    if len(names) > 1:
        raise ValueError("Chained transformations are not supported. Stop it!")
    if len(names)==0:
        config.backend_args = {"leaf_filter": "lora_b_only"}
    if len(names)==1:
        new_args = {"leaf_filter": config.backend_args.get("leaf_filter", "lora_b_only")}
        params = _feature_transform_params(config).get(names[0],None)
        if params is not None:
            new_args.update(params)
        config.backend_args = new_args
    return config

SUPPORTED_REPRESENTORS = {"identity", "sign", "threshlded_sign","float32", "float64", "sign_int8", "sign_int16", "sign_int32", "sign_int64"}
USERDEFINED_ARGS = {"thresholded_sign": {"threshold"}, "sign_int8": {"dim"}, "sign_int16": {"dim"}, "sign_int32": {"dim"}, "sign_int64": {"dim"}}

def unpack_translator_context(context: dict[str, Any])->tuple[str,dict[str,Any]]:
    if not set(context.keys())=={"name", "parameters"}:
        raise ValueError("Unexpected translator context! I'm so surprised!!!")
    name = normalize_name(context.get("name", None))
    parameters = context.get("parameters", None)
    if not isinstance(parameters,dict):
        raise TypeError("Parameters must be a dictionary")

    if not name in SUPPORTED_REPRESENTORS:
        raise ValueError(f"We don't support {name} representor")
    req = USERDEFINED_ARGS.get(name, {})
    if not set(parameters.keys()).issubset(req):
        raise ValueError(f"Missing parameters for {name} representor")
    resolved_parameters = {k:v for k,v in parameters.items() if k in req}

    return name, resolved_parameters



def get_estimator(card: lora_ft_JobCard, kernel_config: KernelRunConfig, translator: str | None = None)-> KenerelEstimatorCard:
    lora_key = generate_apriori_lora_key(card)
    resolved_card, _ = standard_lora_job_card(card, lora_key)
    adapter_path = get_adapter_path_from_lora_key(lora_key, existing=False)
    run_path = get_run_dir_from_lora_key(lora_key, existing=False)

    if adapter_path.exists() or run_path.exists():
        _validate_lora_key(lora_key)
        print(f"Found existing run for {lora_key}, loading metadata and summary")
    else:
        print(f"No existing run found for {lora_key}, starting fine-tuning job")
        fine_tune_on_random_subset(resolved_card)

    kernel_key = get_kernel_run_key(kernel_config)
    estimator_key = get_kernel_estimator_key(lora_key, kernel_key)

    weight_path = get_kernel_weights_path(estimator_key)
    kernel_config.adapter_path = str(adapter_path)
    kernel_path = get_kernel_config_path(estimator_key)
    kernel_config.data_dir = str(get_data_train_path_from_lora_key(lora_key))
    save_kernel_run_config(kernel_config, kernel_path)
    kernel_config.output_root = str(get_kernel_out_path(estimator_key))

    if weight_path.exists():
        print("Found a precomputed kernel. Loading")
        kernel_payload = _load_model(weight_path)
    else:
        print("No cached kernel found. Running the kernel instantiation job")
        kernel_payload = _instantiate_kernel(kernel_config, kernel_path, estimator_key)
        _save_model(model=kernel_payload["model"],
                    payload=kernel_payload["payload"],
                    train_features=kernel_payload["train_features"],
                    path = weight_path,
                    )

    return KenerelEstimatorCard(
                                kernel_config_path=kernel_path,
                                weights_path=weight_path,
                                lora_key=lora_key,
                                estimator_key=estimator_key,
                                output_root=str(get_kernel_out_path(estimator_key)),
                                data_dir=str(get_data_train_path_from_lora_key(lora_key)),
                                translator = translator,
    )


def prepare_estimator(card: KenerelEstimatorCard):
    kernel_config = load_kernel_run_config(resolve_project_path(card.kernel_config_path))
    backend = FeatureExtractor(kernel_config)
    kernel_payload = _load_model(card.weights_path)

    payload = KernelEsimatorPayload(
        backend=backend,
        config=kernel_config,
        model=kernel_payload["model"],
        payload=kernel_payload["payload"],
        train_features=kernel_payload["train_features"],
        key = card.estimator_key,
        translator=card.translator,
    )

    return payload


def resolve_default_starage_type(feature_transform_name: str) -> str:
    DEFAUL_STORAGE_TYPES = {"identity" : "float32", "sign" : "sign_int8", "threshlded_sign" : "sign_int8"}

    store_type = DEFAUL_STORAGE_TYPES.get(feature_transform_name, "float32")
    
    return store_type


def estimate_loss(payload: KernelEsimatorPayload, 
                  records: list[tuple[int, PairRecord]],
                  dataset: KernelDataCard,
                  ):
    
    config = payload.config
    backend = payload.backend
    fitted_model = payload.model
    fit_payload = payload.payload
    train_features = payload.train_features
    model = config.base_model
    adapter_path = config.adapter_path
    transform_name = backend.transform_name
    transform_params = backend.transform_params
    storage_type = payload.translator if payload.translator is not None else resolve_default_starage_type(transform_name)

    if any([v is None for v in backend.get_backend_statistics().values()]):
        backend.extract_feature(records[0][1])
    dim = backend.get_backend_statistics().get("dim", 0)

    key = kernelANDdataset_key(payload.key, storage_type, dataset.key)

    if get_cache_meta_path(key).exists():
        meta_a = get_cache_meta(get_cache_meta_path(key))
        meta_b = CacheMeta(key=key,
                           feature_transform_name=transform_name,
                           feature_transform_params=transform_params,
                           num_records=len(records),
                           feature_dim=dim,
                           model=model,
                           adapter_path=adapter_path,
                           storage_type=storage_type,
                           computed_idx=[],
                           representor_args=representor_parameters_from_backend(storage_type, backend),
                           )
        compare_cache(meta_a, meta_b, error_log=f"Cache key: {key}", success_log=f"Cache key: {key}")
    else:
        write_cache_meta(get_cache_meta_path(key), CacheMeta(
            key = key,
            feature_transform_name = transform_name,
            feature_transform_params = transform_params,
            num_records = len(records),
            feature_dim = dim,
            model = model,
            adapter_path = adapter_path,
            storage_type = storage_type,
            representor_args = representor_parameters_from_backend(storage_type, backend),
        ))

    _prepare_cache_matrix(get_cache_matrix_path(key), expected_feature_d=dim, expected_records_n=len(records))
    features = _fetch_features_by_idx(get_cache_meta_path(key),get_cache_matrix_path(key), records, backend)

    pred_delta = _predict_split(
            config=config,
            fitted_model=fitted_model,
            fit_payload=fit_payload,
            train_features=train_features,
            split_features=features,
        )
    
    return pred_delta