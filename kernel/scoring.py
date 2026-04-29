from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from kernel.data import PairRecord


@dataclass(frozen=True)
class PairTokens:
    token_ids: NDArray[np.int32]
    prompt_offset: int
    sequence_length: int
    completion_targets: int


def tokenize_pair(
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> PairTokens:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Kernel score-delta runs require a tokenizer with "
            "`apply_chat_template`. Use a chat-template-compatible model for v1."
        )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    token_ids = tokenizer.apply_chat_template(messages, return_dict=False)
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        return_dict=False,
    )
    token_array = np.asarray(token_ids, dtype=np.int32)
    prompt_offset = int(len(prompt_only))
    completion_targets = max(0, len(token_array) - prompt_offset)
    return PairTokens(
        token_ids=token_array,
        prompt_offset=prompt_offset,
        sequence_length=int(len(token_array)),
        completion_targets=completion_targets,
    )


def build_inputs_and_targets(pair_tokens: PairTokens) -> tuple[mx.array, mx.array]:
    inputs = mx.array(pair_tokens.token_ids[:-1][None, :])
    targets = mx.array(pair_tokens.token_ids[1:][None, :])
    return inputs, targets


def score_from_batch(
    model: Any,
    inputs: mx.array,
    targets: mx.array,
    prompt_offset: int,
) -> mx.array:
    logits = model(inputs)
    losses = nn.losses.cross_entropy(logits, targets)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = steps >= prompt_offset
    ntoks = mask.sum()
    if int(ntoks.item()) <= 0:
        raise ValueError(
            "This pair has no supervised completion tokens after truncation."
        )
    return -(losses.astype(mx.float32) * mask).sum() / ntoks


def score_pair(
    model: Any,
    tokenizer: Any,
    record: PairRecord,
) -> tuple[float, PairTokens]:
    pair_tokens = tokenize_pair(tokenizer, record.prompt, record.completion)
    inputs, targets = build_inputs_and_targets(pair_tokens)
    score = score_from_batch(
        model=model,
        inputs=inputs,
        targets=targets,
        prompt_offset=pair_tokens.prompt_offset,
    )
    return float(score.item()), pair_tokens


class ModelScorer:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        lazy: bool = True,
    ) -> None:
        loaded = load(
            base_model,
            adapter_path=adapter_path,
            lazy=lazy,
        )
        self.model: Any = loaded[0]
        self.tokenizer: Any = loaded[1]
        self.model.eval()

    def score_record(self, record: PairRecord) -> tuple[float, PairTokens]:
        return score_pair(
            model=self.model,
            tokenizer=self.tokenizer,
            record=record,
        )
