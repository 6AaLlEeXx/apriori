# LoRA Kernel Experiments

This pipeline predicts held-out adapter score deltas for prepared prompt/completion pairs:

- score each pair with the base model
- score the same pair with a trained LoRA adapter
- fit KRR on `score_delta = adapter_score - base_score`
- compare predicted vs. true held-out score deltas

The kernel runner is dataset-agnostic at the prepared-data level: any dataset under `data/<name>/` can be used if it has non-empty `train.jsonl`, `valid.jsonl`, and `test.jsonl` files with `prompt` and `completion` fields.

## Generate Kernel Configs

Generate one LoRA-NTK config from a data recipe:

```bash
uv run mlx-lora-generate-kernel-configs \
  --data-config configs/data/instruction/alpaca.yaml
```

Generate configs for every data recipe:

```bash
uv run mlx-lora-generate-kernel-configs --all-data-configs
```

Generate configs with an explicit model:

```bash
uv run mlx-lora-generate-kernel-configs \
  --data-config configs/data/gsm8k.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Generate a config with an explicit backend:

```bash
uv run mlx-lora-generate-kernel-configs \
  --data-config configs/data/sql/spider.yaml \
  --backends lora_ntk
```

Preview without writing files:

```bash
uv run mlx-lora-generate-kernel-configs \
  --all-data-configs \
  --dry-run
```

Generated files are written to `configs/kernel/generated/` by default. Existing hand-written configs under `configs/kernel/` remain valid.

## Run Experiments

Preview a run name:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/generated/alpaca_lora_ntk.yaml \
  --dry-run
```

Run an experiment:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/generated/alpaca_lora_ntk.yaml
```

Override the config's model for one run:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/generated/alpaca_lora_ntk.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

If `adapter_path` is empty, the kernel runner picks the latest completed LoRA adapter for the same dataset and exact base model from `results/runs/*/summary.json`.

## Reports

Build the aggregate markdown report:

```bash
uv run mlx-lora-make-kernel-report
```

Build a comparison table from discovered completed kernel runs:

```bash
uv run mlx-lora-make-kernel-comparison
```

Filter the comparison table:

```bash
uv run mlx-lora-make-kernel-comparison \
  --datasets dolly gsm8k spider \
  --backends lora_ntk \
  --base-models mlx-community/SmolLM2-1.7B-Instruct
```

The comparison report groups runs by base model, dataset, and backend. For each group it chooses the run with the largest train split, then breaks ties by higher test delta Pearson and newer run name.

## V1 Constraints

- `target` is currently fixed to `score_delta`.
- Prepared data must contain non-empty `prompt` and `completion` values.
- The tokenizer must expose `apply_chat_template`; plain tokenizer fallback is not implemented yet.
- `adapter_path` must resolve to a LoRA/DoRA adapter directory with `adapter_config.json`.
- `backend=lora_ntk` computes score Jacobians with respect to inserted LoRA parameters and should be kept to small subsets.

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

The key metrics are Pearson/Spearman correlation, RMSE, MAE, and sign accuracy
on score deltas, plus correlation/error metrics on reconstructed adapter
scores. Reports also compare KRR against a train-mean score-delta baseline;
positive RMSE gain means KRR has lower held-out error than that baseline.
