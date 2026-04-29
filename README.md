# MLX LoRA Workbench

This directory is a standalone UV project for local MLX-LM LoRA work:

- prepare datasets into `data/`
- train MLX-LM LoRA adapters into `results/adapters/`
- track run metadata, logs, and metrics in `results/runs/`
- run kernel score-delta experiments into `results/kernel/`
- write markdown reports into `reports/`

All relative paths in configs and CLIs are resolved against this project directory, not against the parent repository.

## Codebase Layout

The codebase is intentionally flat. Top-level modules handle shared workflows, while small packages keep CLI, kernel, and selection concerns separate:

- `cli/` contains command-line entrypoints.
- `kernel/` contains LoRA-NTK feature extraction, scoring, KRR experiments, adapter comparison, and reports.
- `selection/` contains training-set selectors, including LoRA-NTK k-means.
- `data_prep.py` prepares raw datasets into prompt/completion JSONL splits.
- `mlops.py` manages LoRA run configs, commands, logs, metadata, and summaries.
- `eval.py` contains task-level prediction metrics.
- `paths.py` centralizes project-relative path handling.

## Setup

```bash
cd lora
uv sync --group dev
```

The MLX dependencies are installed only on macOS. Dataset preparation uses Hugging Face `datasets`. The checked-in configs default to `mlx-community/SmolLM2-1.7B-Instruct`, but the model is just a parameter.

## Prepare Data

```bash
uv run mlx-lora-prepare-data --config <data-config.yaml>
```

Flags:

- `--config` - required data recipe, for example `configs/data/dolly.yaml`.
- `--output-dir` - override the config's `output_dir`.
- `--seed` - override the split/subsample seed.
- `--tokenizer-model` - tokenizer used by token-supervision filters.
- `--base-model` - alias for `--tokenizer-model`, useful when matching training.

The data config is the source of truth for the dataset source, split strategy, prompt/completion mapping, computed fields, source append, length filters, token-supervision filters, and output location. Each run writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `metadata.json` under the configured output directory, usually `data/<dataset>/`.

To add a normal supervised dataset, add one YAML file under `configs/data/` with `source`, `split`, `mapping`, and optional `filters`. You should only need Python for cases that cannot be represented as row templates or source composition.

**50 ready-to-use dataset recipes** (instruction, math, commonsense, QA, code, SQL, summarization, classification, medical, multilingual, translation, tools) ship under `configs/data/<category>/`. See [docs/datasets.md](docs/datasets.md) for the full catalog, schema reference, and mapping cookbook.

## Train Adapters

Train:

```bash
uv run mlx-lora-run --config configs/dolly.yaml
```

Flags:

- `--config` - required LoRA run config.
- `--run-name` - explicit run name; defaults to a timestamped generated name.
- `--base-model` - override the config's `base_model`.
- `--dry-run` - create metadata/commands and print paths without training.
- `--skip-test` - skip the post-training `mlx_lm.lora --test` pass.
- `--sample-selector` - Python file defining `select_samples(...)`.
- `--max-examples` - value passed through to the selector as `max_example`.

Run-time subsampling keeps the full prepared dataset untouched and materializes
the selected train split under the run directory:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --sample-selector selection/lora_ntk_kmeans.py \
  --max-examples 2000
```

`--sample-selector` must point to a Python module defining:

```python
def select_samples(rows, max_example=None, context=None):
    ...
```

- `rows` is the full parsed `train.jsonl` list of JSON objects.
- return value must be an iterable of selected row objects.
- `--max-examples` is optional and passed through as `max_example`.
- `context`, when accepted by the selector, includes `source_data_dir`,
  `run_dir`, `sampled_data_dir`, `base_model`, `mlx_args`, and `seed`.
- if `--max-examples` is omitted, training uses the full selector output.

The default selector (`selection/default.py`) returns rows unchanged.
When a selector is used, the run writes a materialized sampled split under
`results/runs/<run_name>/data/train.jsonl` and copies `valid.jsonl`/`test.jsonl`
for the same run.

The LoRA-NTK k-means selector extracts gradient features from the configured base model and LoRA settings, clusters the train split into `--max-examples` clusters, and keeps the nearest real row to each center.

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
  --predictions <predictions.jsonl> \
  --config <training-config.yaml>
```

Flags:

- `--predictions` - required JSONL with `prediction` and `reference` fields.
- `--config` - optional training config whose `evaluation` block is used.
- `--metric` - metric to compute; can be passed multiple times.
- `--primary-metric` - metric reported as `metric_name`/`metric_value`.
- `--run-dir` - writes to `<run-dir>/eval.json` unless `--output` is set.
- `--output` - explicit evaluation JSON path.

The evaluator reads metric names from the training config's `evaluation` block. Built-in metrics include `exact_match`, `token_f1`, `gsm8k_answer_accuracy`, `sql_exact_match`, `rouge_l_f1`, and `conala_exact_match`.

## Compare Full vs Subset Adapters

Compare an adapter trained on full data with one trained on a selected subset:

```bash
uv run mlx-lora-compare-adapters \
  --config <training-config.yaml> \
  --full-adapter <full-adapter-dir> \
  --subset-adapter <subset-adapter-dir>
```

Flags:

- `--config` - required training config defining `data_dir` and `base_model`.
- `--full-adapter` - required adapter trained on full data.
- `--subset-adapter` - required adapter trained on selected data.
- `--split` - `train`, `valid`, or `test`; defaults to `test`.
- `--limit` - max scored examples; use `0` for all selected split records.
- `--seed` - seed used when subsampling the split.
- `--base-model` - optional base model override.
- `--output-dir` - explicit output directory.

This scores the same held-out examples with the base model, full adapter, and subset adapter, then compares `full_score_delta` against `subset_score_delta`. Outputs are written under `results/comparisons/` by default.

## Kernel Experiments

Run:

```bash
uv run mlx-lora-run-kernel --config configs/kernel/dolly_lora_ntk.yaml
```

Flags:

- `--config` - required kernel run config.
- `--run-name` - explicit run name; defaults to a timestamped generated name.
- `--base-model` - override the config's `base_model`.
- `--dry-run` - resolve config and print the generated run name.

Generate kernel configs from data/training configs:

```bash
uv run mlx-lora-generate-kernel-configs --data-config <data-config.yaml>
```

Generator flags:

- `--data-config` - data recipe to convert; can be passed multiple times.
- `--all-data-configs` - generate configs for every recipe under `configs/data/`.
- `--training-config` - optional LoRA config used when its dataset matches.
- `--base-model` - override generated `base_model`.
- `--backends` - backend names; currently defaults to `lora_ntk`.
- `--output-dir` - generated config directory.
- `--force` - overwrite existing generated configs.
- `--dry-run` - print planned paths without writing.

Use any config under `configs/kernel/`. The same model override pattern works for kernel runs.

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

Reports are written to `reports/` by default.

### LoRA Run Report

```bash
uv run mlx-lora-make-report
```

Flags:

- `--input-root` - root containing tracked LoRA runs.
- `--output` - markdown output path.

### Kernel Run Report

```bash
uv run mlx-lora-make-kernel-report
```

Flags:

- `--output-root` - kernel results root.
- `--output` - markdown output path.

### Kernel Comparison Report

```bash
uv run mlx-lora-make-kernel-comparison
```

Flags:

- `--output-root` - kernel results root.
- `--output` - markdown output path.
- `--datasets` - dataset names to include.
- `--backends` - backend names to include.
- `--base-models` - base models to include.

## Tests

```bash
uv run pytest
```

Useful pytest selectors/flags:

- `tests/<file>.py` - run one test file.
- `-k <pattern>` - run matching tests.
- `-q` - quieter output.
