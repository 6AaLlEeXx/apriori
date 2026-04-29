from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import json
import numpy as np
from numpy.typing import NDArray

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from kernel.data import PairRecord
from kernel.scoring import (
    build_inputs_and_targets,
    score_from_batch,
    tokenize_pair,
)
from kernel.config import KernelRunConfig


def create_feature_backend(config: KernelRunConfig) -> Any:
    backend_name = config.backend.lower()
    if backend_name == "lora_ntk":
        return LoRANTKFeatureBackend(
            base_model=config.base_model,
            adapter_path=config.adapter_path,
            leaf_filter=str(config.backend_args.get("leaf_filter", "lora_b_only")),
        )
    raise ValueError(f"Unknown kernel backend: {config.backend}")


class LoRANTKFeatureBackend:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        leaf_filter: str = "lora_b_only",
        adapter_config: dict[str, Any] | None = None,
    ) -> None:
        if adapter_config is None:
            if adapter_path is None:
                raise ValueError(
                    "LoRA NTK features require either `adapter_path` or "
                    "`adapter_config`."
                )
            adapter_config = self._load_adapter_config(adapter_path)
        adapter_config = dict(adapter_config)
        fine_tune_type = adapter_config.get("fine_tune_type", "lora")
        if fine_tune_type not in {"lora", "dora"}:
            raise ValueError(
                "LoRA NTK backend only supports LoRA/DoRA adapter configs."
            )

        loaded = load(base_model, lazy=True)
        self.model: Any = loaded[0]
        self.tokenizer: Any = loaded[1]
        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            int(adapter_config["num_layers"]),
            dict(adapter_config["lora_parameters"]),
            use_dora=(fine_tune_type == "dora"),
        )
        self.model.eval()
        self.leaf_filter = leaf_filter
        self._value_and_grad = nn.value_and_grad(self.model, score_from_batch)
        self._selected_leaf_names = self._build_leaf_name_filter()

    @staticmethod
    def _load_adapter_config(adapter_path: str | Path) -> dict[str, Any]:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing adapter config: {config_path}")
        payload = json.loads(config_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Adapter config must be a JSON object: {config_path}")
        return payload

    @classmethod
    def from_mlx_args(
        cls,
        *,
        base_model: str,
        mlx_args: dict[str, Any],
        leaf_filter: str = "lora_b_only",
    ) -> LoRANTKFeatureBackend:
        if "num_layers" not in mlx_args:
            raise ValueError(
                "LoRA NTK selector requires `mlx_args.num_layers` in the run config."
            )
        raw_lora_parameters = mlx_args.get("lora_parameters")
        if not isinstance(raw_lora_parameters, dict):
            raise ValueError(
                "LoRA NTK selector requires `mlx_args.lora_parameters` "
                "in the run config."
            )
        adapter_config = {
            "fine_tune_type": str(mlx_args.get("fine_tune_type", "lora")),
            "num_layers": int(mlx_args["num_layers"]),
            "lora_parameters": _normalize_lora_parameters(raw_lora_parameters),
        }
        return cls(
            base_model=base_model,
            leaf_filter=leaf_filter,
            adapter_config=adapter_config,
        )

    def _build_leaf_name_filter(self) -> set[str]:
        leaves = [name for name, _ in tree_flatten(self.model.trainable_parameters())]
        if self.leaf_filter == "all":
            return set(leaves)
        if self.leaf_filter == "lora_b_only":
            return {name for name in leaves if name.endswith("lora_b")}
        raise ValueError(
            f"Unsupported leaf filter: {self.leaf_filter}. Use `all` or `lora_b_only`."
        )

    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]:
        pair_tokens = tokenize_pair(self.tokenizer, record.prompt, record.completion)
        inputs, targets = build_inputs_and_targets(pair_tokens)
        _, grad = self._value_and_grad(
            self.model,
            inputs,
            targets,
            pair_tokens.prompt_offset,
        )
        flattened: list[NDArray[np.float32]] = []
        for name, value in tree_flatten(grad):
            if name not in self._selected_leaf_names:
                continue
            value_array = cast(Any, value)
            flattened.append(
                np.array(value_array.astype(mx.float32), copy=False).reshape(-1)
            )
        if not flattened:
            raise ValueError("No gradient leaves selected for LoRA NTK features.")
        return np.concatenate(flattened, axis=0)


def _normalize_lora_parameters(raw_parameters: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_parameters)
    rank = normalized.get("rank")
    rank_value = int(rank) if rank is not None else None
    scale = normalized.get("scale")
    alpha = normalized.pop("alpha", None)

    if scale is None:
        if alpha is not None:
            alpha_value = float(alpha)
            normalized["scale"] = (
                alpha_value / rank_value
                if rank_value is not None and rank_value != 0
                else alpha_value
            )
        else:
            normalized["scale"] = 20.0

    if normalized.get("dropout") is None:
        normalized["dropout"] = 0.0
    return normalized
