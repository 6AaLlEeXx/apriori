
import hashlib
from estimator.jobs import lora_ft_JobCard, lora_run_config_from_lora_job_card
from mlops import LoraRunConfig
from kernel.data import PairRecord, load_pair_split
import json
from paths import resolve_project_path
from typing import Any
from pathlib import Path
from estimator.configs import flatten_ignoring_keys
from kernel.config import KernelRunConfig


def get_lora_trainset_key(card: lora_ft_JobCard) -> str:
    config = lora_run_config_from_lora_job_card(card)
    records_key = get_dataset_key(config.data_dir)
    selector_key = get_selector_key(card)

    encoded = f"selector={selector_key}; data={records_key}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def generate_apriori_lora_key(card: lora_ft_JobCard) -> str:
    """
        Encodes records and lora parameters(ignoring some) into a hash. 
    """
    payload_key = generate_lora_init_key(card)
    trainset_key = get_lora_trainset_key(card)
    encoded = f"{payload_key}_{trainset_key}".encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def generate_lora_init_key(card: lora_ft_JobCard | LoraRunConfig) -> str:
    if isinstance(card, LoraRunConfig):
        config = card.to_dict()
    else:
        config = lora_run_config_from_lora_job_card(card).to_dict()
    
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


def get_dataset_key(data_dir: str | Path) -> str:
    data_dir = resolve_project_path(Path(data_dir))
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Data directory is not a directory: {data_dir}")
    
    records = load_pair_split(data_dir/"train.jsonl", "data_key_data")
    key = _records_payload(records)
    
    return key


def get_selector_key(card: lora_ft_JobCard) -> str:
    name = Path(card.sample_selector).stem

    def _hash_random_selector(card: lora_ft_JobCard) -> str:
        config = lora_run_config_from_lora_job_card(card)
        seed = config.mlx_args["seed"]
        encoded = f"random_selector_seed={seed}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _hash_identity_selector() -> str:
        encoded = f"do nothing, chill".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    if name == "random":
        return _hash_random_selector(card)
    elif name == "identity":
        return _hash_identity_selector()
    else:
        raise ValueError(f"Unsupported selector {name}.")


IGNOR_LORA_ATR = {
    "task", "output_root", "test_after_train", 
    "run_tags", "name", "metrics", "primary_metric", 
    "steps_per_report", "steps_per_eval", "val_batches", 
    "save_every", "notes",
    }


def _traverse_lora_config(config: LoraRunConfig) -> dict[str, Any]:
    dictonfig = config.to_dict()
    return flatten_ignoring_keys(dictonfig, IGNOR_LORA_ATR)

def _json_hash(dict_config: dict[str, Any]) -> str:
    json_str = json.dumps(dict_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def get_kernel_run_key(config: KernelRunConfig) -> str:
    GET_KERNEL_SPECS = ["backend", "target", "seed", "train_limit", "kernel", "backend_args"]

    dict_config = config.to_dict()

    data_dir = dict_config.get("data_dir", None)
    if data_dir is None or data_dir.strip() == "":
        raise ValueError("Data directory must be provided")

    data_path = resolve_project_path(config.data_dir)/"train.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"No training data path found at {data_path}")
    records = load_pair_split(data_path, "train")
    records_key = _records_payload(records)

    
    specs = ""
    for key in GET_KERNEL_SPECS:
        val = dict_config.get(key, None)
        if val is None:
            raise ValueError(f"Some information is missing: {key}")
        if isinstance(val, dict):
            payload = _json_hash(val)
        else:
            payload = f"{val}"
        specs = f"{specs}; {key}={payload}"
    specs_key = hashlib.sha256(specs.encode("utf-8")).hexdigest()

    key = hashlib.sha256(f"{specs_key}_{records_key}".encode("utf-8")).hexdigest()

    return key


def get_backend_key(config: KernelRunConfig) -> str:
    GET_KERNEL_SPECS = ["backend", "backend_args"]

    dict_config = config.to_dict()
    
    specs = ""
    for key in GET_KERNEL_SPECS:
        val = dict_config.get(key, None)
        if val is None:
            raise ValueError(f"Some information is missing: {key}")
        if isinstance(val, dict):
            payload = _json_hash(val)
        else:
            payload = f"{val}"
        specs = f"{specs}; {key}={payload}"

    return hashlib.sha256(specs.encode("utf-8")).hexdigest()


def combined_key(**keys)->str:
    key = ""
    for k, v in keys.items():
        key = f"{key}{k}={v};"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()