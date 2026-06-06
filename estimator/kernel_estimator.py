from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import numpy as np
import json
from numpy.typing import NDArray

from estimator.keys import generate_apriori_lora_key, get_kernel_run_key, combined_key, generate_lora_init_key, get_backend_key
from paths import resolve_project_path

from kernel.data import PairRecord, load_pair_split
from kernel.config import save_kernel_run_config, load_kernel_run_config
from kernel.run import _predict_split, KernelRunConfig, validate_kernel_run_inputs
from kernel.krr import NystromKRRModel, DualKRRModel

from estimator.jobs import (
                            get_adapter_path_from_lora_key, 
                            get_run_dir_from_lora_key, 
                            get_data_train_path_from_lora_key, 
                            prepare_lora_job_card, 
                            lora_ft_JobCard, 
                            fine_tune_on_random_subset, 
                            _instantiate_kernel,
                            validate_lora_by_key,
)

from estimator.kernel_caching import (
    _fetch_features_by_idx, 
    get_cache_meta, compare_cache, 
    CacheMeta, _prepare_cache_matrix, 
    write_cache_meta,
)
from estimator.extractor import FeatureExtractor#, representor_parameters_from_backend
from kernel.run import _feature_transform_names, _feature_transform_params
from estimator.representor import FeatureRepresentor, resolve_representor

KRR = DualKRRModel | NystromKRRModel


def green_bold_print(text: str):
    print(f"\033[1;32m{text}\033[0m")


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
            self.num_records = len(load_pair_split(self.data_dir, "hui"))
        return self.num_records

@dataclass
class KenerelEstimatorCard:
    kernel_config_path: str | Path
    weights_path: str | Path
    estimator_key: str
    lora_key: str
    lora_init_key: str

    output_root: str | Path
    data_dir: str | Path

    translator: dict[str, Any] = field(default_factory=lambda: {"name": "float32", "parameters": {}})

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
    representor: FeatureRepresentor
    config: KernelRunConfig
    model: KRR
    payload: dict[str, Any]
    train_features: NDArray
    backend_key: str


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
    out_path = resolve_project_path(DEFAULT_KERNEL_OUT_PATH)
    return out_path

SUPPORTED_TRANSFORMS = {"identity", "sign", "threshlded_sign"}

def is_translator_a_feature_trasformation(translator_name: str)->bool:
    return True

def normalize_name(name: str | None)->str:
    if name is None:
        return ""
    return name.lower().strip().replace("-", "_")


def prepare_kernel_config(config: KernelRunConfig)->KernelRunConfig:
    names = _feature_transform_names(config)
    if len(names) > 1:
        raise ValueError("Chained transformations are not supported. Stop it!")
    if len(names)==0:
        config.backend_args = {"leaf_filter": "lora_b_only"}
    if len(names)==1:
        args = {"leaf_filter": config.backend_args.get("leaf_filter", "lora_b_only")}
        params = _feature_transform_params(config).get(names[0],{})
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


SUPPORTED_REPRESENTORS = {"multitransform", "float32", "float64", "sign_int8", "sign_int16", "sign_int32", "sign_int64"}
USERDEFINED_ARGS = {}

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


def validate_translator_context(translator: dict[str, Any]):
    return unpack_translator_context(translator)

def get_estimator(card: lora_ft_JobCard, kernel_config: KernelRunConfig, translator: dict[str, Any] = {"name": "float32", "parameters": {}})-> KenerelEstimatorCard:
    green_bold_print("[Get Kernel Estimator Card] Starting the kernel card generation sequence>>>")
    
    #validation and data preparation
    kernel_config = prepare_kernel_config(kernel_config)
    validate_translator_context(translator)
    lora_key = generate_apriori_lora_key(card)
    lora_init_key = generate_lora_init_key(card)
    resolved_card, _ = prepare_lora_job_card(card, lora_key)

    adapter_path = get_adapter_path_from_lora_key(lora_key, existing=False)
    run_path = get_run_dir_from_lora_key(lora_key, existing=False)

    if adapter_path.exists() or run_path.exists():
        #we chech for the integraty of all files that will be used
        validate_lora_by_key(lora_key)
        print(f"[Get Kernel Estimator Card] Found existing run for lora key {lora_key[:10]}...")
    else:
        #if no lora is found we just train it
        print(f"[Get Kernel Estimator Card] No existing run found for lora key {lora_key[:10]}..., starting fine-tuning job")
        fine_tune_on_random_subset(resolved_card)

    print(f"[Get Kernel Estimator Card] The fine-tuned lora model is ready")
    #this is the key that is unique to the kernel estimator as a whole and
    #it is not sensetive to things like storage type or caching strategy
    estimator_key = combined_key(lora_key=lora_key, kernel_key=get_kernel_run_key(kernel_config))
    weight_path = get_kernel_weights_path(estimator_key)
    #it is expected that kernel uses the same adapter as the lora it is based on
    kernel_config.adapter_path = str(adapter_path)
    kernel_path = get_kernel_config_path(estimator_key)
    #set the kernel train data = lora train data it is the same as how we do it in the paper
    #keeping lora and kernel data the same gives cleaner signal and allows
    #to reuse lora gradients computed during FT for kernel features later on!
    kernel_config.data_dir = str(get_data_train_path_from_lora_key(lora_key))
    #the ready for work kernek is stored in working files
    save_kernel_run_config(kernel_config, kernel_path)
    kernel_config.output_root = str(get_kernel_out_path(estimator_key))
    #after the kernel config preparation is finished we validate its correctness
    validate_kernel_run_inputs(kernel_config)
    green_bold_print(f"[Get Kernel Estimator Card] Kernel-run config is ready, initialising kernel weights. Data path is set to {kernel_config.data_dir}. Adapter path is set to {kernel_config.adapter_path}. Output root is set to {kernel_config.output_root}")

    #now lets see if the kernel weights are already in place
    #for example, if we only change the validation data but not the estimator, then
    #we can just use cache. Also, we could use the same trained lora model bur for different
    #transformation settings, so that lora is loaded from memory but weights are recomputed
    if weight_path.exists():
        print(f"[Get Kernel Estimator Card] Found precomputed kernel weights. Loading")
    else:
        #in case weights are not precomputed we generate them and save right away
        #notice that _instantiate_kernel will use precomputed gradient features if the representor is set
        #to multitransform, because caching is transformation-agnostic
        print(f"[Get Kernel Estimator Card] No precomputed kernel weights found. Running the kernel instantiation job")
        kernel_payload = _instantiate_kernel(kernel_config, kernel_path, estimator_key)
        print(f"[Get Kernel Estimator Card] Kernel weights are ready. Saving the kernel weights")
        _save_model(model=kernel_payload["model"],
                    payload=kernel_payload["payload"],
                    train_features=kernel_payload["train_features"],
                    path = weight_path,
                    )
        print(f"[Get Kernel Estimator Card] Kernel weights are saved at {weight_path}")
    green_bold_print(f"[Get Kernel Estimator Card] The kernel estimator card is ready for estimator key {estimator_key[:10]}...")
   
    return KenerelEstimatorCard(
                                kernel_config_path=kernel_path,
                                weights_path=weight_path,
                                lora_key=lora_key,
                                lora_init_key=lora_init_key,
                                estimator_key=estimator_key,
                                output_root=str(get_kernel_out_path(estimator_key)),
                                data_dir=str(get_data_train_path_from_lora_key(lora_key)),
                                translator=translator,
    )


def prepare_estimator(card: KenerelEstimatorCard):
    green_bold_print(f"[Prepare Kernel Estimator] Starting the payload preparation sequence>>>")
    
    #run preparatons and validations for kernel config
    kernel_config = prepare_kernel_config(load_kernel_run_config(card.kernel_config_path))
    validate_kernel_run_inputs(kernel_config)
    validate_translator_context(card.translator)

    #loads the kernel weights down the given path
    kernel_payload = _load_model(card.weights_path)
    translator_name, params = unpack_translator_context(card.translator)
    #we also add transform params in case when multitransform representor is used
    #then from the multitransform in resolve_representor the transform params are used to configure
    params.update(kernel_config.backend_args)
    representor = resolve_representor(translator_name, **params)

    #when the multitarnsform is used we need to fix backend to a simple grad features
    #because the backend is used to define the features that get stored
    backend_args = kernel_config.backend_args
    if translator_name == "multitransform":
        print(f"[Prepare Kernel Estimator] Multitransform representor is used, fixing the backend to use raw gradient features without transformations for caching compatibility")
        kernel_config.backend_args = {"leaf_filter" : kernel_config.backend_args.get("leaf_filter", "lora_b_only")}
    #we pass config to defie feature_extractor, an object incapsulating the backend
    #together with all the transformations applied together
    green_bold_print(f"[Prepare Kernel Estimator] Initializing the feature extractor backend. Transformations: {kernel_config.backend_args.get('feature_transform', 'identity')}") 
    backend = FeatureExtractor(kernel_config)
    #backend key is used to identify backend as an independent entity as it
    #often times is applied with different representors and datasets, and we keep this key constant across these cases
    backend_key = combined_key(lora_init_key=card.lora_init_key, backend_key=get_backend_key(kernel_config))
    kernel_config.backend_args = backend_args
    print(f"[Prepare Kernel Estimator] Backend is ready. Backend key is {backend_key[:10]}...")

    payload = KernelEsimatorPayload(
        backend=backend,
        config=kernel_config,
        model=kernel_payload["model"],
        payload=kernel_payload["payload"],
        train_features=kernel_payload["train_features"],
        backend_key = backend_key,
        representor = representor,
    )

    return payload


def estimate_loss(payload: KernelEsimatorPayload, 
                  records: list[tuple[int, PairRecord]],
                  dataset: KernelDataCard,
                  ):
    
    green_bold_print("[Loss Estimation] Starting the loss estimation sequence>>>")

    config = payload.config
    backend = payload.backend
    representor = payload.representor
    fitted_model = payload.model
    fit_payload = payload.payload
    train_features = payload.train_features
    model = config.base_model
    adapter_path = config.adapter_path
    transform_name = backend.transform_name
    transform_params = backend.transform_params

    #representor needs to be fitted for some examplary vector
    #parameters, such as padding, need to be determined during fitting
    if not representor.ready or backend.dim is None:
        green_bold_print(f"[Loss Estimation] The representor is not fitted yet, fitting on the first record...")
        representor.fit(backend.extract_feature(records[0][1]))
    dim = backend.dim if backend.dim is not None else 0

    #now we compute the cache key, which presents information that determines the features,
    #the format and the dataset for which we need to store features
    #notice that it is not sensetive to kernel training information, so, for example
    #we can train diffrent kernel estimators for different training datasest and reuse the same
    #grad features for some fixed validation set across them
    key = combined_key(backend_key=payload.backend_key, representor_key=representor.get_key(), dataset_key=dataset.key)
    
    computed_idx = []

    #check if metadata is already stored for the key
    if get_cache_meta_path(key).exists():
        print(f"[Loss Estimation] Found existing cache metadata for key: {key[:10]}..., validating the cache metadata...")
        meta_a = get_cache_meta(get_cache_meta_path(key))
        meta_b = CacheMeta(key=key,
                           feature_transform_name=transform_name,
                           feature_transform_params=transform_params,
                           num_records=len(dataset),
                           feature_dim=dim,
                           model=model,
                           adapter_path=adapter_path,
                           representor_name=representor.init_name,
                           computed_idx=[],
                           representor_args=representor.args,
                           )
        compare_cache(meta_a, meta_b, error_log=f"Cache key: {key[:10]}...", success_log=f"Cache key: {key}")
        computed_idx = meta_a.computed_idx
    else:
        print(f"[Loss Estimation] No cache metadata found for key: {key[:10]}...")

    #metadata is always overwritten. If new metadata is compatable with the one that is stored,
    #no value error is raised and the code proceedes so that new updated meta is prioritised
    #for example, if old and new meta both store multitransform parameters that encode for different
    #transformatons, it doesnt raise value error and the parameters just get updated to the newest setting
    #the idea is to raise error only when the cache content itself could be altered
    print(f"[Loss Estimation] Writing new cache metadata for key: {key[:10]}...")
    write_cache_meta(get_cache_meta_path(key), CacheMeta(
        key = key,
        feature_transform_name = transform_name,
        feature_transform_params = transform_params,
        num_records = len(dataset),
        feature_dim = dim,
        model = model,
        adapter_path = adapter_path,
        representor_name = representor.init_name,
        representor_args = representor.args,
        computed_idx=computed_idx,
    ))

    green_bold_print(f"[Loss Estimation] Metadata is ready. Preparing the feature cache matrix.")
    #here we make sure that the cache matrix is ready. If it exists we validate it, if it doesnt, we create one
    _prepare_cache_matrix(get_cache_matrix_path(key), expected_feature_d=dim, expected_records_n=len(dataset))
    features = _fetch_features_by_idx(get_cache_meta_path(key),get_cache_matrix_path(key), records, backend)
    print(f"[Loss Estimation] Features are ready. Starting the loss prediction.")

    pred_delta = _predict_split(
            config=config,
            fitted_model=fitted_model,
            fit_payload=fit_payload,
            train_features=train_features,
            split_features=features,
        )
    
    return pred_delta