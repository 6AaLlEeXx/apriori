# MLX LoRA Tracking

This standalone UV project keeps LoRA training lightweight:

- `cli/run.py` prepares a stable run directory, launches `mlx_lm.lora`, and captures MLX logs to local files plus `metrics.jsonl`.
- `cli/eval_predictions.py` scores a prediction JSONL file and stores `eval.json` inside a run.
- `cli/make_report.py` builds a markdown table across all runs.

## Layout

```text
configs/
  base.yaml
  base_smollm2.yaml
  data/
    dolly.yaml
    gsm8k.yaml
    samsum.yaml
    sql_create_context.yaml
    conala.yaml
    conala_mined.yaml
  dolly.yaml
  gsm8k.yaml
  samsum.yaml
  sql_create_context.yaml
  conala.yaml
docs/
  mlops.md
src/
  lora/
    cli/
    kernel/
tests/
  test_mlops.py
data/
  <dataset>/
    train.jsonl
    valid.jsonl
    test.jsonl
results/
  adapters/<run_name>/
  runs/<run_name>/
    command.txt
    metadata.json
    mlx_config.yaml
    resolved_config.yaml
    summary.json
    metrics.jsonl
    eval.json
    logs/
      train.log
      test.log
reports/
  lora_runs.md
```

## Run Naming

Runs are named as:

```text
YYYYMMDD-HHMMSS__<model>__<dataset>__r<rank>__l<layers>__seq<max_seq_length>__s<seed>
```

Example with the default model:

```text
20260311-213000__smollm2-1-7b-instruct__gsm8k__r8__l12__seq640__s42
```

This keeps runs sortable while still exposing the main knobs that matter when you compare adapters later.

## Commands

Preview a run without starting training:

```bash
uv run mlx-lora-run --config configs/dolly.yaml --dry-run
```

Train a tracked adapter:

```bash
uv run mlx-lora-run --config configs/dolly.yaml
```

Override the config's model for one run:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --base-model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Train without the post-train test pass:

```bash
uv run mlx-lora-run --config configs/dolly.yaml --skip-test
```

Build the markdown comparison report:

```bash
uv run mlx-lora-make-report
```

## Preparing Datasets

Install dependencies once:

```bash
uv sync --group dev
```

Then prepare each dataset into the local `data/<dataset>/` layout expected by `mlx_lm.lora`:

```bash
uv run mlx-lora-prepare-data --config configs/data/dolly.yaml
uv run mlx-lora-prepare-data --config configs/data/gsm8k.yaml
uv run mlx-lora-prepare-data --config configs/data/samsum.yaml
uv run mlx-lora-prepare-data --config configs/data/sql_create_context.yaml
uv run mlx-lora-prepare-data --config configs/data/conala.yaml
uv run mlx-lora-prepare-data --config configs/data/conala_mined.yaml
```

Each config writes:

```text
data/<dataset>/
  train.jsonl
  valid.jsonl
  test.jsonl
  metadata.json
```

The generic prep runner supports Hugging Face datasets plus local JSONL, JSON, and CSV sources. A data config declares:

- `source`: where rows come from, including published splits, local split files, and optional appended sources.
- `split`: `existing`, `ratios`, `train_valid`, or `train_valid_existing_test`.
- `mapping`: prompt/completion templates with optional computed fields such as `first_non_empty`.
- `filters`: optional character limits and token-supervision filtering.

Useful options:

- `--output-dir` overrides where JSONL files are written.
- `--seed` overrides split and subset sampling.
- `--base-model` or `--tokenizer-model` overrides the tokenizer used by token-supervision filtering.
- `configs/data/sql_create_context.yaml` keeps the SQL dataset to 25,000 examples by default.
- `configs/data/dolly.yaml` drops pathological long rows and rows that would leave fewer than one supervised completion token after truncation.
- `configs/data/conala_mined.yaml` appends a sampled mined subset to the curated CoNaLa train split.

The runner converts source rows into prompt/completion JSONL records for `mlx_lm.lora`. This is an inference from the current MLX-LM LoRA docs and the current Hugging Face dataset cards for Dolly, GSM8K, SAMSum, `sql-create-context`, and CoNaLa.

Troubleshooting:

- If `datasets` fails with `ModuleNotFoundError: No module named '_lzma'`, your current Python build was compiled without lzma support. Rebuild the pyenv interpreter with Homebrew `xz`, then recreate `.venv`.

## Metrics You Keep

The training wrapper parses current MLX log lines into `metrics.jsonl` and surfaces them in `summary.json`:

- train loss by step
- validation loss by step
- learning rate
- iterations per second
- tokens per second
- trained tokens
- peak memory
- test loss
- test perplexity

## Evaluating Model Outputs

The local evaluator expects JSONL records shaped like:

```json
{"prediction": "SELECT name FROM student", "reference": "SELECT name FROM student"}
```

Then run:

```bash
uv run mlx-lora-eval-preds \
  --config configs/sql_create_context.yaml \
  --predictions results/runs/<run_name>/predictions.jsonl \
  --run-dir results/runs/<run_name>
```

Evaluation uses metric names from the training config's `evaluation` block. You can override them directly:

```bash
uv run mlx-lora-eval-preds \
  --metric exact_match \
  --metric token_f1 \
  --primary-metric token_f1 \
  --predictions results/runs/<run_name>/predictions.jsonl
```

Built-in metrics are `exact_match`, `token_f1`, `gsm8k_answer_accuracy`, `sql_exact_match`, `rouge_l_f1`, and `conala_exact_match`.

## Notes

- Training configs live in `configs/`; data prep configs live in `configs/data/`. Both use local `data/<dataset>` paths by default.
- `base_model` is a normal config field. You can change it in YAML for reproducibility or override it once with `--base-model`.
- `mlx_args` maps to `mlx_lm.lora` arguments. Supported CLI args are passed directly, while config-only keys such as `lora_parameters` are written into `results/runs/<run_name>/mlx_config.yaml` and passed with `-c`.
- MLX-LM uses `lora_parameters.scale`, not `alpha`. The wrapper accepts legacy `alpha` and converts it to `scale = alpha / rank` for compatibility.
- The wrapper normalizes `gradient_accumulation_steps` to MLX-LM's current `--grad-accumulation-steps` flag.
- If you see `nan` losses in Dolly training with `mask_prompt: true`, discard that run and rebuild `data/dolly/` with the current prep script before starting a new run. The prep step now removes rows that would produce zero supervised completion tokens after truncation.
- The LoRA scaffold is local-only. It does not configure W&B or any external experiment tracker.
