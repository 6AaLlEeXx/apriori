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

from lora.kernel.data import PairRecord
from lora.kernel.scoring import (
    build_inputs_and_targets,
    score_from_batch,
    tokenize_pair,
)


class LoRANTKFeatureBackend:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        leaf_filter: str = "lora_b_only",
    ) -> None:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing adapter config: {config_path}")
        adapter_config = json.loads(config_path.read_text())
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
