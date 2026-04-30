# MLX LoRA Workbench

This directory is a standalone UV project for local MLX-LM LoRA work:

- prepare datasets into `data/`
- train MLX-LM LoRA adapters into `results/adapters/`
- track run metadata, logs, and metrics in `results/runs/`
- run kernel score-delta experiments into `results/kernel/`
- write markdown reports into `reports/`

All relative paths in configs and CLIs are resolved against this project directory, not against the parent repository.

## Codebase Layout

The codebase is intentionally flat. Top-level modules handle shared workflows,
while small directories keep CLI, kernel, selector, transformation, and
projection concerns separate:

- `cli/` contains command-line entrypoints.
- `kernel/` contains LoRA-NTK feature extraction, scoring, KRR experiments, adapter comparison, and reports.
- `selectors/` contains training-set selector files, including LoRA-NTK k-means.
- `transformations/` contains feature transformations such as `identity`, `sign`, and `thresholded_sign`.
- `projection/` contains feature projections such as `identity` and `sparse_random`.
- `feature_pipeline.py` composes feature extraction, transformations, and projection.
- `selector_algorithms.py` contains feature-matrix selection algorithms such as k-means.
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

## First Empirical Validation

The first validation asks six practical questions:

- is LoRA-NTK k-means better than a random subset at the same subset size?
- does `sign(features)` alone preserve quality relative to raw k-means?
- can sparse random projection reduce feature dimensionality while preserving
  most of the k-means result quality?
- does adding `sign(features)` preserve quality relative to projected features?
- does each subset adapter stay close to the full-data adapter?
- do LoRA-NTK features predict held-out adapter score deltas?

Subset methods:

| Method | Run suffix | Selector | Transformation | Projection |
| --- | --- | --- | --- | --- |
| `random` | `random` | `selectors/random.py` | - | - |
| `kmeans` | `kmeans` | `selectors/lora_ntk_kmeans.py` | `identity` | `identity` |
| `kmeans+sign` | `kmeans-sign` | `selectors/lora_ntk_kmeans.py` | `sign` | `identity` |
| `kmeans+thresholded_sign` | `kmeans-thresholded-sign` | `selectors/lora_ntk_kmeans.py` | `thresholded_sign` | `identity` |
| `kmeans+sparse_random` | `kmeans-srp` | `selectors/lora_ntk_kmeans.py` | `identity` | `sparse_random` |
| `kmeans+sparse_random+sign` | `kmeans-srp-sign` | `selectors/lora_ntk_kmeans.py` | `sign` | `sparse_random` |
| `kmeans+sparse_random+thresholded_sign` | `kmeans-srp-thresholded-sign` | `selectors/lora_ntk_kmeans.py` | `thresholded_sign` | `sparse_random` |

Run the workflow on one dataset/model pair first:

```bash
uv run mlx-lora-prepare-data \
  --config configs/data/<dataset>.yaml \
  --base-model <model>
```

Train the full-data adapter:

```bash
uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-full \
  --base-model <model>
```

Train the five selected-subset adapters:

```bash
uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-random-<n> \
  --base-model <model> \
  --sample-selector selectors/random.py \
  --max-examples <n>

uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-kmeans-<n> \
  --base-model <model> \
  --sample-selector selectors/lora_ntk_kmeans.py \
  --max-examples <n>

uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-kmeans-sign-<n> \
  --base-model <model> \
  --sample-selector selectors/lora_ntk_kmeans.py \
  --max-examples <n> \
  --selector-transformation sign

uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-kmeans-srp-<n> \
  --base-model <model> \
  --sample-selector selectors/lora_ntk_kmeans.py \
  --max-examples <n> \
  --selector-projection sparse_random \
  --selector-projection-components 1024

uv run mlx-lora-run \
  --config configs/<dataset>.yaml \
  --run-name <dataset>-kmeans-srp-sign-<n> \
  --base-model <model> \
  --sample-selector selectors/lora_ntk_kmeans.py \
  --max-examples <n> \
  --selector-transformation sign \
  --selector-projection sparse_random \
  --selector-projection-components 1024
```

Compare each subset adapter against the full adapter on the same held-out split:

```bash
uv run mlx-lora-compare-adapters \
  --config configs/<dataset>.yaml \
  --full-adapter results/adapters/<dataset>-full \
  --subset-adapter results/adapters/<dataset>-<run-suffix>-<n> \
  --split test \
  --limit 0 \
  --method <method>
```

Run the NTK prediction experiment:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/<dataset>_lora_ntk.yaml \
  --base-model <model>
```

If the kernel config leaves `adapter_path` empty, run this after the subset
adapter so the runner picks that latest completed adapter. Otherwise, set
`adapter_path` in the kernel config to the adapter being explained.

Then generate the reports:

```bash
uv run mlx-lora-make-report
uv run mlx-lora-make-adapter-comparison
uv run mlx-lora-make-kernel-report
uv run mlx-lora-make-kernel-comparison
```

The adapter comparison report is the main cross-method table. Use it to compare:

- `kmeans` vs. `random`: higher delta Pearson/sign accuracy and lower delta RMSE.
- `kmeans+sign` vs. `kmeans`: little or no degradation from sign-only features.
- `kmeans+sparse_random` vs. `kmeans`: similar quality with lower projection dim.
- `kmeans+sparse_random+sign` vs. `kmeans+sparse_random`: little or no degradation.
- every subset method vs. full fine-tuning: high adapter-score Pearson, low
  adapter-score RMSE, and small mean delta gap.

Its main method plots are grouped by subset size, so `method_delta_pearson.svg`,
`method_delta_rmse.svg`, `method_adapter_rmse.svg`, and
`method_sign_accuracy.svg` compare methods at each `<n>`. Collapsed method
averages are written separately as `average_method_*.svg`.

The kernel reports separately test whether LoRA-NTK features predict held-out
adapter score deltas. After the first run works, repeat across multiple `<n>`
values, datasets, and seeds.

The same workflow can be launched with:

```bash
./orchestrate.sh
```

Useful overrides:

- `SMOKE_RUN=1` - use the tiny Dolly smoke configs, one subset size, small
  comparison/kernel limits, and short training runs.
- `DEBUG=0` - suppress extra orchestration/selector debug chatter.
- `PLAN_ONLY=1` - print the commands without executing them.
- `PREPARE_DATA=0` - reuse existing prepared data instead of rebuilding it.
- `PREPARED_DATA_DIR=data/dolly` - directory checked when `PREPARE_DATA=0`.
- `REUSE_EXISTING_ADAPTERS=1` - skip subset adapter training when the exact
  generated run name already has a completed adapter.
- `REUSE_EXISTING_COMPARISONS=1` - skip adapter comparisons when the exact
  comparison output already exists.
- `METHODS="random kmeans-srp-sign"` - restrict selector methods for a faster
  check.
- `METHODS="kmeans-thresholded-sign kmeans-srp-thresholded-sign"` - run the
  thresholded sign selector variants.
- `N_VALUES="128 512 2000"` - choose the three subset sizes.
- `BASE_MODEL=<model>` - use a different MLX-LM-compatible model.
- `TRAIN_FULL=0 FULL_RUN_NAME=<run-name>` - reuse an existing full-data adapter
  instead of retraining it.
- `PROJECTION_COMPONENTS=1024` - set sparse random projection dimension.
- `THRESHOLDED_SIGN_THRESHOLD=0.01` - set the thresholded sign dead-zone.
- `SELECTOR_PROJECTION_CHUNK_SIZE=16` - rows per sparse projection chunk.
- `SELECTOR_MAX_KMEANS_FEATURE_GB=4` - unprojected k-means memory guard.
- `RUN_KERNEL=0` - skip kernel prediction runs.
- `KERNEL_ADAPTER_RUNS="<run-name> ..."` - restrict kernel prediction to a
  subset of adapters. By default, every adapter trained by the orchestration is
  tested.

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

The data config is the source of truth for the dataset source, split strategy,
prompt/completion mapping, computed fields, source append, length filters,
token-supervision filters, and output location. Each run writes `train.jsonl`,
`valid.jsonl`, `test.jsonl`, and `metadata.json` under the configured output
directory, usually `data/<dataset>/`.

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
- `--selector-transformation` - feature transformation before selector clustering; repeatable, defaults to `identity`.
- `--selector-thresholded-sign-threshold` - absolute-value dead-zone threshold for `thresholded_sign`; defaults to `0.01`.
- `--selector-projection` - feature projection before selector clustering; defaults to `identity`.
- `--selector-projection-components` - output dimension for projections that need one.
- `--selector-projection-chunk-size` - rows per chunk for memory-conscious selector projections; defaults to `16`.
- `--selector-max-kmeans-feature-gb` - maximum unprojected k-means feature matrix size before failing with a projection hint; defaults to `4`.
- `--selector-debug` - print selector feature/cache progress while preparing sampled data.

Run-time subsampling keeps the full prepared dataset untouched and materializes
the selected train split under the run directory:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --sample-selector selectors/lora_ntk_kmeans.py \
  --max-examples 2000 \
  --selector-transformation thresholded_sign \
  --selector-thresholded-sign-threshold 0.01 \
  --selector-projection sparse_random \
  --selector-projection-components 1024
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
  `run_dir`, `sampled_data_dir`, `base_model`, `mlx_args`, `seed`,
  `selector_transformations`, `selector_transformation_params`,
  `selector_projection`, and `selector_projection_components`.
- if `--max-examples` is omitted, training uses the full selector output.

The identity selector (`selectors/identity.py`) returns rows unchanged.
When a selector is used, the run writes a materialized sampled split under
`results/runs/<run_name>/data/train.jsonl` and copies `valid.jsonl`/`test.jsonl`
for the same run.

The LoRA-NTK k-means selector extracts gradient features from the configured
base model and LoRA settings, applies the requested feature transformations and
projection, clusters the train split into `--max-examples` clusters, and keeps
the nearest real row to each center.

Raw LoRA-NTK selector features are cached under
`results/selector_feature_cache/` by default. The cache key includes the base
model, seed, LoRA feature settings, and train-row content, so k-means variants
such as `kmeans`, `kmeans+sign`, `kmeans+sparse_random`, and
`kmeans+sparse_random+thresholded_sign` reuse the same expensive raw feature
matrix while still applying their own transformation/projection steps.

For large train splits, unprojected LoRA-NTK k-means is usually not practical:
the raw feature matrix can be tens of GiB. The runner fails early when the
unprojected matrix exceeds `--selector-max-kmeans-feature-gb`; use sparse random
projection for full-size runs.

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
- `--method` - optional subset method label used by aggregate reports.

This scores the same held-out examples with the base model, full adapter, and
subset adapter, then compares both `full_score_delta` vs. `subset_score_delta`
and full adapter scores vs. subset adapter scores. Outputs are written under
`results/comparisons/` by default.

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
- `--assets-dir` - directory for generated visualization assets.
- `--no-plots` - skip visualization generation.

### Adapter Comparison Report

```bash
uv run mlx-lora-make-adapter-comparison
```

Flags:

- `--output-root` - results root containing `comparisons/` and `runs/`.
- `--output` - markdown output path.
- `--datasets` - dataset names to include.
- `--methods` - subset method labels to include.
- `--base-models` - base models to include.
- `--assets-dir` - directory for generated visualization assets.
- `--no-plots` - skip visualization generation.

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
