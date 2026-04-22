from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
import mlx.core as mx
from mlx_lm import load

from lora.kernel.data import PairRecord
from lora.kernel.scoring import build_inputs_and_targets, tokenize_pair


class FrozenPairFeatureBackend:
    def __init__(
        self,
        base_model: str,
        pooling: str = "completion_mean",
    ) -> None:
        if pooling != "completion_mean":
            raise ValueError(
                f"Unsupported frozen-pair pooling mode: {pooling}. "
                "Only `completion_mean` is currently implemented."
            )
        self.pooling = pooling
        loaded = load(base_model, lazy=True)
        self.model: Any = loaded[0]
        self.tokenizer: Any = loaded[1]
        self.model.eval()

    def extract_feature(self, record: PairRecord) -> NDArray[np.float32]:
        pair_tokens = tokenize_pair(self.tokenizer, record.prompt, record.completion)
        inputs, _ = build_inputs_and_targets(pair_tokens)
        hidden = self.model.model(inputs)
        hidden_np = np.array(hidden.astype(mx.float32), copy=False)[0]
        start = max(pair_tokens.prompt_offset - 1, 0)
        if start >= hidden_np.shape[0]:
            raise ValueError(
                f"No completion positions available for pooled feature: {record.pair_id}"
            )
        pooled = hidden_np[start:, :].mean(axis=0)
        return pooled.astype(np.float32, copy=False)
