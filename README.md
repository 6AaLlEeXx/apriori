# MLX LoRA Workbench

This directory is a standalone UV project for local MLX-LM LoRA work:

- prepare datasets into `data/`
- train MLX-LM LoRA adapters into `results/adapters/`
- track run metadata, logs, and metrics in `results/runs/`
- run kernel score-delta experiments into `results/kernel/`
- write markdown reports into `reports/`

All relative paths in configs and CLIs are resolved against this project directory, not against the parent repository.

## Setup

```bash
cd lora
uv sync --group dev
```

The MLX dependencies are installed only on macOS. Dataset preparation uses Hugging Face `datasets`. The checked-in configs default to `mlx-community/SmolLM2-1.7B-Instruct`, but the model is just a parameter.

## Prepare Data

```bash
uv run mlx-lora-prepare-data --config configs/data/dolly.yaml
uv run mlx-lora-prepare-data --config configs/data/gsm8k.yaml
uv run mlx-lora-prepare-data --config configs/data/samsum.yaml
uv run mlx-lora-prepare-data --config configs/data/sql_create_context.yaml
uv run mlx-lora-prepare-data --config configs/data/conala.yaml
uv run mlx-lora-prepare-data --config configs/data/conala_mined.yaml
```

Each config writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `metadata.json` under `data/<dataset>/`. The data prep runner supports Hugging Face datasets plus local JSONL, JSON, and CSV sources. Prompt/completion mapping, computed fields, split strategy, source append, length filters, and optional token-supervision filtering live in YAML.

For Dolly, pass the same model used for training so truncation checks use the right chat template:

```bash
uv run mlx-lora-prepare-data \
  --config configs/data/dolly.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

To add a normal supervised dataset, add one YAML file under `configs/data/` with `source`, `split`, `mapping`, and optional `filters`. You should only need Python for cases that cannot be represented as row templates or source composition.

**50 ready-to-use dataset recipes** (instruction, math, commonsense, QA, code, SQL, summarization, classification, medical, multilingual, translation, tools) ship under `configs/data/<category>/`. See [docs/datasets.md](docs/datasets.md) for the full catalog, schema reference, and mapping cookbook.

## Train Adapters

Preview a run:

```bash
uv run mlx-lora-run --config configs/dolly.yaml --dry-run
```

Train:

```bash
uv run mlx-lora-run --config configs/dolly.yaml
```

Use another MLX-LM-compatible model without editing the config:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Run-time subsampling (keep full `data/<dataset>/train.jsonl`, choose subset per run):

```bash
uv run mlx-lora-run \
  --run-name some-name \
  --config configs/dolly.yaml \
  --base-model mlx-community/Qwen3.5-9B-Instruct-4bit \
  --sample-selector selectors/default_selector.py \
  --max-examples 2000
```

`--sample-selector` must point to a Python module defining:

```python
def select_samples(rows, max_example=None):
    ...
```

- `rows` is the full parsed `train.jsonl` list of JSON objects.
- return value must be an iterable of selected row objects.
- `--max-examples` is optional and passed through as `max_example`.
- if omitted, training uses the full selector output.

The default selector (`selectors/default_selector.py`) returns rows unchanged.
When a selector is used, the run writes a materialized sampled split under
`results/runs/<run_name>/data/train.jsonl` and copies `valid.jsonl`/`test.jsonl`
for the same run.

The run writes:

```text
results/
  adapters/<run_name>/
  runs/<run_name>/
    command.txt
    metadata.json
    mlx_config.yaml
    resolved_config.yaml
    summary.json
    metrics.jsonl
    logs/
      train.log
      test.log
```

## Evaluate Prediction Files

```bash
uv run mlx-lora-eval-preds \
  --config configs/gsm8k.yaml \
  --predictions results/runs/<run_name>/predictions.jsonl \
  --run-dir results/runs/<run_name>
```

The evaluator reads metric names from the training config's `evaluation` block. Built-in metrics include `exact_match`, `token_f1`, `gsm8k_answer_accuracy`, `sql_exact_match`, `rouge_l_f1`, and `conala_exact_match`.

## Kernel Experiments

Preview:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_frozen_pair.yaml --dry-run
```

Run:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_frozen_pair.yaml
uv run mlx-lora-run-kernel --config configs/kernel/dolly_lora_ntk.yaml
```

Use the same model override for kernel runs:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/dolly_frozen_pair.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

If a kernel config leaves `adapter_path` empty, the runner chooses the latest completed adapter for the same dataset and base model from `results/runs/*/summary.json`.

Kernel outputs are written under:

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

## Reports

```bash
uv run mlx-lora-make-report
uv run mlx-lora-make-kernel-report
uv run mlx-lora-make-kernel-comparison
```

Reports are written to `reports/`.

## Tests

```bash
uv run pytest
```
