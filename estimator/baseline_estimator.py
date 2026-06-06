from __future__ import annotations

from mlx_lm import load
from transformers import Any
from estimator.keys import generate_apriori_lora_key
from kernel.data import load_pair_split
from kernel.scoring import score_pair
from estimator.jobs import (
                            fine_tune_on_random_subset, 
                            lora_ft_JobCard, 
                            prepare_lora_job_card,
                            load_pair_idx,
                            validate_lora_by_key,
                            get_adapter_path_from_lora_key,
                            get_command_path_from_lora_key,
                            get_run_dir_from_lora_key,
                            get_metadata_path_from_lora_key,
                            get_summary_path_from_lora_key
)

from pathlib import Path
from dataclasses import dataclass, asdict
import json


@dataclass
class BaselineDataCard:
    data_dir: str | None = None
    key: str | None = None
    name: str = "benchmark"


@dataclass
class BaselineEstimator:
    # thsi is the defining property, the KEY is how a particular 
    # instance of an estimator is refered to concistently
    key: str | None = None
    lora_key: str | None = None #unique lora run key, in general, might differ from 'key' 

    name: str = "baseline_estimator"  #doesnt have to be unique
    description: str = "The simplest baseline loss estimator" #doesnt affect anything

    # where to look for lora details
    lora_summary: str | None = None
    lora_metadata: str | None = None
    lora_command: str | None = None

    model: str | None = None
    adapter: str | None = None


    def to_dict(self):
        return asdict(self)        

    def get_adapter_path(self) -> Path:
        if self.lora_key is None:
            raise ValueError("Lora key is not set")
        adapter_path = get_adapter_path_from_lora_key(self.lora_key)

        self.adapter = str(adapter_path)
        return Path(adapter_path)

    def get_mlx_model_name(self) -> str:
        adapter_config_path = self.get_adapter_path()/"adapter_config.json"
        
        if not adapter_config_path.exists():
            raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}")
        
        adapter_config = json.loads(adapter_config_path.read_text())

        if not isinstance(adapter_config, dict):
            raise ValueError(f"Adapter config must be a dictionary: {adapter_config_path}")
        model_name = adapter_config.get("model", None)

        if model_name is None:
            raise ValueError(f"No mode is specified: {adapter_config_path}")
        self.model = model_name
        
        return model_name


def get_estimator_from_jobcard(card: lora_ft_JobCard):
    lora_key = generate_apriori_lora_key(card)
    resolved_card, _ = prepare_lora_job_card(card, lora_key)
    adapter_path = get_adapter_path_from_lora_key(lora_key, existing=False)
    run_path = get_run_dir_from_lora_key(lora_key, existing=False)
    
    if adapter_path.exists() or run_path.exists():
        validate_lora_by_key(lora_key)
        print(f"Found existing run for {lora_key}, loading metadata and summary")
    else:
        print(f"No existing run found for {lora_key}, starting fine-tuning job")
        fine_tune_on_random_subset(resolved_card)

    print(f"The baseline estimator is ready for {lora_key}")
    return BaselineEstimator(
                                key = lora_key,
                                lora_key = lora_key,
                                lora_command = str(get_command_path_from_lora_key(lora_key)),
                                lora_summary = str(get_summary_path_from_lora_key(lora_key)),
                                lora_metadata = str(get_metadata_path_from_lora_key(lora_key)),
                            )


def estimate_loss(
        estimator: BaselineEstimator,
        dataset: BaselineDataCard,
        record_idx: list[int]) -> list[float]:
    
    base_model = estimator.get_mlx_model_name()
    adapter_path = str(estimator.get_adapter_path())
    if dataset.data_dir is not None:
        records = load_pair_idx(dataset.data_dir, record_idx)
    else:
        raise ValueError("Data path not found in BaselineDataCard")
    
    loaded = load(
                    base_model,
                    adapter_path=adapter_path,
                    lazy=True,
                )
    
    model = loaded[0]
    tokenizer = loaded[1]

    estimated_loss = [score_pair(model, tokenizer, r)[0] for r in records]

    return estimated_loss

