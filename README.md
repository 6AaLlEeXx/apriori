# Score-Induced Tangent Kernels

This repository studies whether local tangent information from a language model can predict how fine-tuning on a small subset of examples changes held-out example scores.

For an input-output example `z = (x, y)`, the implemented score is the supervised completion log score. The code calls the resulting restricted tangent feature a score-gradient feature:

```text
psi_I(z) = grad_{theta_I} s_z(f_theta(x)) | theta = theta_0
```

where `I` is the selected trainable parameter block. In this codebase that block is the LoRA adapter block, and the active backend is named `score_gradient`.

For a trained adapter `a`, the kernel target is the held-out score delta:

```text
Delta_a(z) = s_z(f_{theta_0,a}(x)) - s_z(f_{theta_0}(x))
```

Kernel ridge regression learns this delta from a kernel fit split and is evaluated against a train-mean delta baseline.

## Current Scope

The supported workflow is `./orchestrate.sh`. Standalone report dashboards, standalone prediction evaluation, and generated config tooling have been removed so the repository stays centered on the paper experiment path. Dataset, training, and kernel configs are retained because orchestration is expected to run across datasets via environment overrides.

The remaining workflow performs:

1. dataset preparation, unless `PREPARE_DATA=0`
2. full-data LoRA adapter training, unless `TRAIN_FULL_ADAPTER=0`
3. subset adapter training for `SUBSET_SELECTION_METHODS x SUBSET_TRAIN_SIZES`
4. optional full-vs-subset adapter scoring summaries
5. kernel score-delta prediction experiments
6. kernel PDF figures

## Layout

```text
.
|-- orchestrate.sh                 # end-to-end experiment runner
|-- cli/                           # orchestrated command entrypoints
|-- scripts/
|   `-- orchestrate/               # shell modules sourced by orchestrate.sh
|-- configs/
|   |-- data/                      # raw dataset -> prepared JSONL split recipes
|   |-- train/                     # LoRA fine-tuning configs over prepared data
|   `-- kernel/                    # score-delta KRR configs for trained adapters
|-- kernel/                        # score deltas, tangent features, KRR
|-- selectors/                     # random subset selector
|-- transformations/               # identity, sign, thresholded_sign
|-- reporting/                     # PDF figure generation
|   |-- paper/                     # exact paper-compatible renderer
|   `-- standard/                  # conventional matplotlib renderer
|-- tests/                         # tests for the active workflow
|-- data/                          # prepared JSONL splits, normally generated
|-- results/                       # adapters, runs, caches, comparisons
`-- reports/                       # orchestration manifests and kernel figures
```

Important modules:

- `data_prep.py` prepares configured datasets into `train.jsonl`, `valid.jsonl`, and `test.jsonl`.
- `mlops.py` resolves MLX-LM configs, run directories, metadata, and adapter summaries.
- `kernel/run.py` scores base/adapter models, caches scores/features, fits KRR, and writes predictions.
- `scripts/orchestrate/` splits orchestration into context/defaults, shared helpers, manifest writing, and one workflow step per shell module.
- `reporting/plots.py` selects between the exact paper renderer and the conventional standard renderer.
- `reporting/paper/plots.py` preserves the paper-compatible plot layout used for reproducibility.
- `reporting/standard/` contains smaller matplotlib modules for conventional PDF figures.

Config roles:

- `configs/data/...` is consumed by `mlx-lora-prepare-data` and writes `data/<dataset>/{train,valid,test}.jsonl`.
- `configs/train/...` is consumed by `mlx-lora-run` and writes adapter runs under `results/`.
- `configs/kernel/...` is consumed by `mlx-lora-run-kernel` and evaluates score-delta prediction for trained adapters.

## Configuration Model

YAML and JSON configs support `extends`. The parent path is resolved relative to the child config, and nested mappings are merged recursively. This is why dataset-specific training configs can extend `configs/train/base/base.yaml`, and kernel configs can extend `configs/kernel/base/score_delta.yaml`.

Data configs define the prepared JSONL splits:

| Field | Meaning |
| --- | --- |
| `dataset_name` | Logical dataset key used in output paths and run metadata. |
| `upstream_dataset`, `source` | Input dataset identity and source. Supported source types are `hf`, `jsonl`, `json`, and `csv`. |
| `split` | Split strategy and seed. Supported strategies are `ratios`, `train_valid_test`, `train_valid`, `existing`, and `train_valid_existing_test`. |
| `prompt_completion_mapping` | Converts raw rows into `prompt` and `completion` text via templates, prompt parts, or computed fields. |
| `filters` | Length filters, prompt-hash exclusions, and optional token-supervision filtering. |
| `metadata` | Free-form notes retained with prepared data. |

Training configs describe MLX-LM LoRA runs:

| Field | Meaning |
| --- | --- |
| `dataset_name`, `task`, `base_model`, `prepared_data_dir` | Dataset/model identity and prepared data location. |
| `results_root` | Root for `results/runs/` and `results/adapters/`. |
| `mlx_command` | Training command, normally `mlx_lm.lora`. |
| `test_after_train` | Whether `mlx_lm.lora --test` runs after training. |
| `mlx_lora_args` | MLX-LM CLI arguments such as `iters`, `batch_size`, `learning_rate`, and `lora_parameters`. |
| `subset_training` | Optional effective-iteration scaling for sampled subset runs. |
| `extra_args` | Additional CLI arguments appended to the MLX-LM command. |
| `evaluation`, `run_tags`, `notes` | Metadata written into run summaries. |

When subset training uses a selector, `subset_training.iteration_scaling_policy` can be `none`, `fixed`, `match_full_exposure`, `match_full_passes`, or `per_example`. The active policy scales `mlx_lora_args.iters` by `selected_train_examples / original_train_examples`, with optional `min_iters` and `cap_at_full_iters`.

Kernel configs describe score-delta prediction runs:

| Field | Meaning |
| --- | --- |
| `dataset_name`, `task`, `base_model`, `prepared_data_dir` | Dataset/model identity and prepared data location. |
| `adapter_path` | Adapter to evaluate. If empty, the latest completed matching adapter is resolved from `results/runs/`. |
| `kernel_results_root` | Root for kernel runs and kernel caches. |
| `feature_backend` | Feature backend. The active supported backend is `score_gradient`. |
| `prediction_target` | Prediction target. The active implemented target is `score_delta`. |
| `krr_fit_examples`, `validation_examples`, `test_examples` | Number of prepared examples used for KRR fit, validation, and test. |
| `krr` | KRR method and hyperparameters. |
| `feature_backend_args` | Feature extraction and transformation arguments. |
| `provenance_configs`, `run_tags`, `notes` | Metadata written into kernel summaries. |

## Setup

This is a UV project. Training, adapter scoring, and score-gradient extraction use MLX-LM, so real experiments are intended for macOS on Apple Silicon.

```bash
uv sync --group dev
```

Useful entrypoints:

```bash
uv run mlx-lora-prepare-data --help
uv run mlx-lora-run --help
uv run mlx-lora-run-kernel --help
uv run mlx-lora-compare-adapters --help
uv run mlx-lora-make-kernel-figures --help
```

Run the local test suite:

```bash
uv run pytest
```

## Orchestration

Always inspect the plan before launching an expensive run:

```bash
PLAN_ONLY=1 ./orchestrate.sh
```

Run a small end-to-end smoke check:

```bash
SMOKE_RUN=1 ./orchestrate.sh
```

Run the default Dolly experiment:

```bash
./orchestrate.sh
```

Each run writes:

```text
results/orchestrations/<run-id>/manifest.md
reports/orchestrations/<run-id>/kernel_figures.md
reports/orchestrations/<run-id>/assets/kernel_figures/
```

## Outputs and Caches

Training writes tracked run metadata and adapters:

```text
results/
  adapters/<run-name>/
  runs/<run-name>/
    command.txt
    metadata.json
    metrics.jsonl
    mlx_config.yaml
    resolved_config.yaml
    summary.json
    logs/{train,test}.log
```

Adapter comparisons, when enabled, write scores and a metric summary:

```text
results/orchestrations/<run-id>/comparisons/<subset-run>__vs__<full-run>/
  scores.jsonl
  summary.json
```

Kernel runs write resolved configs, scores, predictions, and evaluation payloads:

```text
results/orchestrations/<run-id>/kernel/
  runs/<kernel-run-name>/
    metadata.json
    resolved_config.yaml
    summary.json
    eval.json
    scores/{train,valid,test}.jsonl
    features/
    predictions/{train,valid,test}.jsonl
  cache/
    scores/
    features/
```

The score cache stores base/adapter supervised completion scores for selected records. The feature cache stores raw score-gradient matrices; cheap feature transforms such as `sign` and `thresholded_sign` are applied after loading the raw feature cache.

## Figure Rendering

Kernel plots are PDF-only. There are two renderers:

- `paper` is the default renderer. It preserves the exact paper-compatible layout used to reproduce existing plots, including fixed figure geometry and manually positioned legends.
- `standard` is the conventional renderer for new analysis. It uses ordinary matplotlib/seaborn axes, grids, legends, and PDF export, with the plotting code split into smaller modules under `reporting/standard/`.

Use the default `paper` renderer when regenerating figures that need to match the paper. Use `standard` when you want simpler, more conventional plots for inspection or future reports.

Through orchestration:

```bash
KERNEL_FIGURE_STYLE=standard ./orchestrate.sh
```

Or directly from collected kernel run summaries:

```bash
uv run mlx-lora-make-kernel-figures \
  --kernel-results "dolly=results/orchestrations/<run-id>/kernel" \
  --output-dir "reports/orchestrations/<run-id>/assets/kernel_standard" \
  --output "reports/orchestrations/<run-id>/kernel_standard_figures.md" \
  --plot-style standard
```

Combined PDFs are named `paper_kernel_figures.pdf` for the exact renderer and `standard_kernel_figures.pdf` for the conventional renderer. With `--individual-pdfs`, the standard renderer prefixes filenames with `standard_` so they can live next to paper-compatible figures without collisions.

## Switching Datasets

The workflow is not Dolly-specific. To orchestrate another dataset, override the dataset label plus the matching data, training, and kernel configs:

```bash
DATASET_NAME=samsum \
DATA_PREP_CONFIG=configs/data/summarization/samsum.yaml \
ADAPTER_TRAIN_CONFIG=configs/train/summarization/samsum.yaml \
KERNEL_EXPERIMENT_CONFIGS="configs/kernel/summarization/samsum_score_gradient.yaml configs/kernel/summarization/samsum_score_gradient_thresholded_sign.yaml" \
./orchestrate.sh
```

Use the same pattern for configs under `classification`, `code`, `commonsense`, `instruction`, `math`, `medical`, `multilingual`, `qa`, `reasoning`, `sql`, `summarization`, `tools`, and `translation`.

## Kernel Semantics

The active prediction target is `score_delta`, the difference between adapter and base supervised completion log score. The active feature backend is `score_gradient`, which extracts score-gradient features from the LoRA parameter block; the default `feature_backend_args.leaf_filter: lora_b_only` keeps only LoRA `B` matrices.

Feature transforms are configured in `feature_backend_args`:

| Transform | Meaning |
| --- | --- |
| `identity` or no transform | Use raw score-gradient features. |
| `sign` | Replace each feature with its sign. |
| `thresholded_sign` | Keep only signs whose absolute raw feature value exceeds `threshold`. |

KRR supports `krr.method: nystrom` and `krr.method: dual`. The default paper-oriented configs use Nystrom KRR with `ridge_lambda`, `rank`, and `num_landmarks`; `dual` uses the full train-train kernel and `ridge_lambda`. The baseline used in summaries and plots is the train-mean score delta.

## Adapter Comparisons

Adapter comparison is an optional orchestration step controlled by `RUN_ADAPTER_COMPARISONS`. It scores a full-data adapter and each subset adapter on the same prepared split, then reports delta prediction metrics such as Pearson correlation, sign accuracy, and RMSE. Use `ADAPTER_COMPARISON_SPLIT` to choose the split, `ADAPTER_COMPARISON_EXAMPLE_LIMIT=0` to score the full split, and `REUSE_EXISTING_ADAPTER_COMPARISONS=1` to avoid recomputing completed comparison directories.

## Selectors

The active orchestrated selector is `random`, implemented by `selectors/random.py`. Custom selectors can still be passed directly to `mlx-lora-run --sample-selector`; a selector module must define:

```python
def select_samples(rows, subset_size=None, context=None):
    ...
```

`rows` are prepared training JSON objects, `subset_size` comes from `--subset-train-size`, and `context` includes paths plus model, `mlx_lora_args`, seed, and debug settings. The selector must return the selected training rows.

## Paper-Style Template

This template trains or reuses random subset adapters for `n in {16, 256, 512}`, then evaluates raw and thresholded-sign features with matching kernel fit sizes.

```bash
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"

RUN_ID="dolly-paper-${RUN_STAMP}" \
ADAPTER_RUN_PREFIX="dolly-paper-${RUN_STAMP}" \
DATASET_NAME=dolly \
DATA_PREP_CONFIG=configs/data/instruction/dolly.yaml \
ADAPTER_TRAIN_CONFIG=configs/train/instruction/dolly.yaml \
BASE_MODEL=mlx-community/SmolLM2-1.7B-Instruct \
SUBSET_SELECTION_METHODS="random" \
SUBSET_TRAIN_SIZES="16 256 512" \
KERNEL_ADAPTER_RUNS="dolly-paper-${RUN_STAMP}-random-subset512" \
KERNEL_EXPERIMENT_CONFIGS="configs/kernel/instruction/dolly_score_gradient.yaml configs/kernel/instruction/dolly_score_gradient_thresholded_sign.yaml" \
KERNEL_FIT_SIZES="16 256 512" \
KERNEL_VALID_EXAMPLES=32 \
KERNEL_TEST_EXAMPLES=256 \
KERNEL_FIGURE_ADAPTER_NAME_CONTAINS="random-subset512" \
KERNEL_FIGURE_INDIVIDUAL_PDFS=1 \
./orchestrate.sh
```

`SUBSET_TRAIN_SIZES` controls which subset adapters are trained. `KERNEL_FIT_SIZES` controls how many examples KRR uses to fit the score-delta predictor.

## Main Controls

| Variable | Meaning |
| --- | --- |
| `SMOKE_RUN=1` | Use tiny configs and limits for a fast check. |
| `PLAN_ONLY=1` | Print commands without running them. |
| `DATASET_NAME` | Dataset/run label, default `dolly`. |
| `DATA_PREP_CONFIG` | Data prep config, default `configs/data/instruction/dolly.yaml`. |
| `ADAPTER_TRAIN_CONFIG` | LoRA training config, default `configs/train/instruction/dolly.yaml`. |
| `BASE_MODEL` | MLX-LM-compatible base model. |
| `SUBSET_SELECTION_METHODS` | Subset selection methods. The active workflow uses `random`. |
| `SUBSET_TRAIN_SIZES` | Subset sizes for adapter training. |
| `RUN_ADAPTER_COMPARISONS=0` | Skip full-vs-subset adapter scoring summaries. |
| `RUN_KERNEL_EXPERIMENTS=0` | Skip kernel prediction and kernel figures. |
| `KERNEL_EXPERIMENT_CONFIG` | Single default kernel config used when `KERNEL_EXPERIMENT_CONFIGS` is unset. |
| `KERNEL_EXPERIMENT_CONFIGS` | One or more kernel configs. |
| `KERNEL_FIT_SIZES` | Optional fit-set sizes for KRR sweeps. |
| `KERNEL_FIGURE_ADAPTER_NAME_CONTAINS` | Filter kernel figures to matching adapter runs. |
| `KERNEL_FIGURE_INDIVIDUAL_PDFS=1` | Write one PDF per panel instead of one multi-page PDF. |
| `KERNEL_FIGURE_STYLE=standard` | Use the conventional matplotlib renderer instead of the exact paper renderer. |

Advanced controls:

| Variable | Meaning |
| --- | --- |
| `RUN_ID` | Explicit orchestration id. Defaults to a timestamp. |
| `ADAPTER_RUN_PREFIX` | Prefix for generated adapter run names. Defaults to `<DATASET_NAME>-<RUN_ID>`. |
| `ORCHESTRATION_DIR` | Override orchestration results directory. |
| `REPORT_OUTPUT_DIR` | Override report directory. |
| `PREPARE_DATA=0` | Reuse prepared data instead of running data prep. |
| `PREPARED_DATA_DIR` | Prepared data directory to validate/use when data prep is skipped. |
| `TRAIN_FULL_ADAPTER=0` | Reuse an existing full-data adapter instead of training one. |
| `FULL_ADAPTER_RUN_NAME` | Full-data adapter run name to train or reuse. |
| `REUSE_EXISTING_SUBSET_ADAPTERS=1` | Skip subset adapter training when completed adapters already exist. |
| `REQUIRE_EXISTING_SUBSET_ADAPTERS=1` | Fail unless the expected subset adapters already exist. |
| `REUSE_EXISTING_ADAPTER_COMPARISONS=1` | Reuse completed full-vs-subset comparison outputs. |
| `ADAPTER_COMPARISON_SPLIT` | Split used for adapter comparisons, default `test`. |
| `ADAPTER_COMPARISON_EXAMPLE_LIMIT` | Adapter comparison example limit. Use `0` for all examples. |
| `SKIP_LORA_TEST=1` | Skip the post-training `mlx_lm.lora --test` pass. |
| `SELECTOR_DEBUG=0` | Disable selector debug output passed through orchestration. |
| `KERNEL_ADAPTER_RUNS` | Space-separated adapter run names to evaluate with KRR. Defaults to all trained/reused adapters. |
| `KERNEL_VALID_EXAMPLES` | Override validation examples for generated kernel configs. |
| `KERNEL_TEST_EXAMPLES` | Override test examples for generated kernel configs and plots. |
| `KERNEL_FIGURE_SPLIT` | Prediction split used for predicted-vs-true plot panels, default `test`. |

## Validation

Fast checks:

```bash
uv run pytest
PLAN_ONLY=1 SMOKE_RUN=1 ./orchestrate.sh
```
