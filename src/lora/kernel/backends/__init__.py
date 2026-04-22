from __future__ import annotations

from typing import Any

from lora.kernel.backends.frozen_pair import FrozenPairFeatureBackend
from lora.kernel.backends.lora_ntk import LoRANTKFeatureBackend
from lora.kernel.config import KernelRunConfig


def create_feature_backend(config: KernelRunConfig) -> Any:
    backend_name = config.backend.lower()
    if backend_name == "frozen_pair":
        return FrozenPairFeatureBackend(
            base_model=config.base_model,
            pooling=str(config.backend_args.get("pooling", "completion_mean")),
        )
    if backend_name == "lora_ntk":
        return LoRANTKFeatureBackend(
            base_model=config.base_model,
            adapter_path=config.adapter_path,
            leaf_filter=str(config.backend_args.get("leaf_filter", "lora_b_only")),
        )
    raise ValueError(f"Unknown kernel backend: {config.backend}")
