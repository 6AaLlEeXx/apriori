# LoRA Kernel Experiments

This pipeline lives entirely inside this standalone `lora/` project and is separate from the older `src/` NTK benchmark.

The current experiment predicts held-out adapter scores for prompt/completion pairs:

- score each pair with the base model
- score the same pair with a trained LoRA adapter
- fit KRR on `score_delta = adapter_score - base_score`
- compare predicted vs true held-out score deltas

## Commands

Preview a run name:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_frozen_pair.yaml --dry-run
```

Run a tiny end-to-end smoke test:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_frozen_pair_smoke.yaml
```

Run a tiny LoRA-NTK smoke test:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_lora_ntk_smoke.yaml
```

Run the frozen-feature baseline:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_frozen_pair.yaml
```

Run the experimental LoRA-tangent backend:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_lora_ntk.yaml
```

Run the same experiment on GSM8K:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/gsm8k_frozen_pair.yaml
uv run mlx-lora-run-kernel --config configs/kernel/gsm8k_lora_ntk.yaml
```

Run the same experiment on SQL Create Context:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/sql_create_context_frozen_pair.yaml
uv run mlx-lora-run-kernel --config configs/kernel/sql_create_context_lora_ntk.yaml
```

Override the config's model for one kernel run:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/dolly_frozen_pair.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Build the aggregate markdown report:

```bash
uv run mlx-lora-make-kernel-report
```

Build the cross-dataset comparison table:

```bash
uv run mlx-lora-make-kernel-comparison
```

## Config Notes

- If `adapter_path` is empty, the kernel runner automatically picks the latest completed LoRA adapter for the same dataset and base model from `results/runs/*/summary.json`.
- `target` is currently fixed to `score_delta`.
- `backend=frozen_pair` uses mean pooled final hidden states over the completion span.
- `backend=lora_ntk` computes score Jacobians with respect to the inserted LoRA parameters.
- `leaf_filter=lora_b_only` is the practical default for the NTK backend because MLX LoRA initializes `lora_b` to zero, so `lora_a` gradients are zero at initialization.
- The comparison report chooses one primary run per dataset/backend pair by taking the largest train split first, then breaking ties by higher test delta Pearson.

## Outputs

Each run writes to:

```text
results/kernel/runs/<run_name>/
  metadata.json
  resolved_config.yaml
  summary.json
  eval.json
  report.md
  scores/{train,valid,test}.jsonl
  features/<backend>_{train,valid,test}.npy
  predictions/{train,valid,test}.jsonl
```

The key metrics are:

- Pearson and Spearman correlation on score deltas
- RMSE and MAE on score deltas
- sign accuracy on score deltas
- the same correlation/error metrics on reconstructed adapter scores

## Practical Limits

- `frozen_pair` is the cheap baseline and can handle larger subsets.
- `lora_ntk` is expensive. Keep it small at first, such as `64-128` training examples.
- The current implementation materializes explicit feature arrays, so `lora_ntk` is still a small-subset experiment rather than a large-scale one.
