# Score-Induced Tangent Kernels

This repository studies whether local tangent information from a large language model can predict how fine-tuning on a small subset of examples changes held-out example scores.

The current implementation uses MLX-LM adapters, but the core object is more general than LoRA. Given a model with parameters $\theta \in \mathbb{R}^p$, choose a subset of parameters $I \subseteq \{1,\ldots,p\}$. The selected parameters may be LoRA weights, one layer, attention weights, MLP weights, or any other parameter block.

For an input-output example $z=(x,y)$, let the model output be

$$
f_\theta(x) \in \mathbb{R}^d
$$

and let

$$
s_z : \mathbb{R}^d \rightarrow \mathbb{R}
$$

be a scalar score functional. In this codebase, the score is the supervised completion log score, implemented as the negative token-level cross entropy on the completion tokens.

The score-induced tangent feature for example $z$, restricted to parameter subset $I$, is

$$
\psi_I(z) =
\nabla_{\theta_I} s_z(f_\theta(x)) \big|_{\theta=\theta_0}.
$$

The associated score-induced tangent kernel is

$$
K_I(z,z') =
\langle \psi_I(z), \psi_I(z') \rangle.
$$

By the chain rule,

$$
\psi_I(z) =
J_I f_{\theta_0}(x)^\top
\nabla_f s_z(f_{\theta_0}(x)),
$$

so this is not simply the raw output NTK. It is the output tangent kernel contracted through the gradient of a task-specific scalar score. When $s_z$ is nonlinear, as it is for cross entropy, the feature depends on the model prediction and the target example.

For a trained adapter $a$, the target predicted by kernel ridge regression is the held-out score delta

$$
\Delta_a(z) =
s_z(f_{\theta_0,a}(x)) - s_z(f_{\theta_0}(x)).
$$

Given a kernel fit set $S_k$, KRR learns $\widehat{\Delta}_a$ from $\{(z_i,\Delta_a(z_i)): z_i \in S_k\}$ and is evaluated by test RMSE:

$$
\mathrm{RMSE} =
\sqrt{
\frac{1}{|\mathcal{T}|}
\sum_{z \in \mathcal{T}}
\left(\widehat{\Delta}_a(z)-\Delta_a(z)\right)^2
}.
$$

The baseline predicts the train-set mean score delta for every held-out example:

$$
\bar{\Delta}_a =
\frac{1}{|S_k|}
\sum_{z_i \in S_k}
\Delta_a(z_i).
$$

Historically, the code calls this backend `lora_ntk`, because the implemented parameter subset is the LoRA adapter block. In the paper text, a clearer name is **score-induced tangent kernel**, or **parameter-restricted score tangent kernel** when emphasizing the selected parameter block.

## What This Work Tests

The empirical question is:

> Can score-gradient features, restricted to a small trainable parameter block, explain and predict the held-out effect of subset fine-tuning?

The main workflow trains adapters on selected subsets, scores the resulting adapter on held-out examples, and asks whether KRR over score-induced tangent features can predict those score deltas.

The important plots are:

- predicted score delta vs. true score delta for several kernel fit sizes
- absolute test RMSE for KRR features against the train-mean baseline
- optional RMSE gain plots, where positive values mean KRR has lower RMSE than the baseline
- adapter comparison plots, when comparing selected-subset adapters against a full-data adapter

## Codebase Layout

The project is intentionally flat. Top-level modules own the shared workflow; subdirectories contain focused implementation pieces.

```text
.
|-- orchestrate.sh                 # end-to-end experiment runner
|-- cli/                           # command-line entrypoints
|-- configs/                       # training, data, and kernel configs
|   |-- data/                      # dataset preparation recipes
|   `-- kernel/                    # KRR / tangent-feature configs
|-- kernel/                        # score deltas, feature extraction, KRR, reports
|-- selectors/                     # random and tangent-feature k-means selectors
|-- transformations/               # identity, sign, thresholded_sign
|-- projection/                    # identity and sparse_random projection
|-- reporting/                     # SVG/PDF plot generation
|-- docs/                          # dataset catalog and extra notes
|-- tests/                         # unit tests
|-- data/                          # prepared JSONL splits, normally generated
|-- results/                       # adapters, runs, caches, comparisons
`-- reports/                       # generated Markdown and paper figures
```

Important modules:

- `data_prep.py` prepares raw Hugging Face datasets into `train.jsonl`, `valid.jsonl`, and `test.jsonl`.
- `mlops.py` resolves MLX-LM training configs, run directories, metadata, and adapter summaries.
- `feature_pipeline.py` composes feature extraction, transformations, and projections for selectors.
- `selector_algorithms.py` contains matrix-based selection algorithms such as k-means.
- `kernel/features.py` extracts score-gradient features. The current backend is LoRA-restricted and still named `lora_ntk`.
- `kernel/run.py` scores base/adapter models, caches scores and features, fits KRR, evaluates RMSE/correlation, and writes kernel predictions.
- `reporting/plots.py` generates the report and paper SVG/PDF figures.

## Setup

This is a UV project. Reporting and most tests are ordinary Python. Training, adapter scoring, and score-gradient extraction use MLX-LM, so real experiments are intended for macOS on Apple Silicon.

```bash
cd lora
uv sync --group dev
```

Check that the command entrypoints are available:

```bash
uv run mlx-lora-prepare-data --help
uv run mlx-lora-run --help
uv run mlx-lora-run-kernel --help
```

Run the fast local test suite:

```bash
uv run pytest
```

Before launching a real experiment, ask the orchestrator to print the plan:

```bash
PLAN_ONLY=1 SMOKE_RUN=1 ./orchestrate.sh
```

Then run the tiny smoke workflow:

```bash
SMOKE_RUN=1 ./orchestrate.sh
```

The smoke run is the useful first check because it exercises data preparation, adapter training, subset selection, adapter comparison, kernel prediction, cache creation, and report generation with small limits.

## Orchestration First

Most experiments should be launched through:

```bash
./orchestrate.sh
```

The orchestrator is the source of truth for an end-to-end run. It performs:

1. dataset preparation, unless `PREPARE_DATA=0`
2. full-data adapter training, unless `TRAIN_FULL=0`
3. subset adapter training for every `METHODS` x `N_VALUES` pair
4. optional full-vs-subset adapter comparisons
5. optional KRR experiments for one or more kernel configs
6. report and plot generation

Every orchestration writes a manifest:

```text
results/orchestrations/<run-id>/manifest.md
```

and reports:

```text
reports/orchestrations/<run-id>/
```

## Default Run

Without overrides, the orchestrator runs Dolly with SmolLM2:

```bash
./orchestrate.sh
```

The default run is intentionally expensive. It prepares Dolly, trains a full-data adapter, trains subset adapters for multiple subset sizes and methods, compares them against the full adapter, runs kernel prediction experiments, and generates reports.

Use `PLAN_ONLY=1` before any new configuration:

```bash
PLAN_ONLY=1 ./orchestrate.sh
```

## Paper-Style Run Template

A typical paper-style run now focuses on random subset adapters and tests KRR at fixed kernel fit sizes. This example trains or reuses adapters for $n \in \{16,256,512\}$, then evaluates KRR with raw and thresholded-sign features using kernel fit sizes $k \in \{16,256,512\}$.

```bash
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"

RUN_ID="dolly-paper-${RUN_STAMP}" \
RUN_PREFIX="dolly-paper-${RUN_STAMP}" \
DATASET=dolly \
DATA_CONFIG=configs/data/dolly.yaml \
TRAIN_CONFIG=configs/dolly.yaml \
BASE_MODEL=mlx-community/SmolLM2-1.7B-Instruct \
METHODS="random" \
N_VALUES="16 256 512" \
KERNEL_ADAPTER_RUNS="dolly-paper-${RUN_STAMP}-random-512" \
KERNEL_CONFIGS="configs/kernel/dolly_lora_ntk.yaml configs/kernel/dolly_lora_ntk_thresholded_sign.yaml" \
KERNEL_TRAIN_LIMITS="16 256 512" \
KERNEL_VALID_LIMIT=32 \
KERNEL_TEST_LIMIT=256 \
./orchestrate.sh
```

The key idea is that `N_VALUES` controls which subset adapters are trained, while `KERNEL_TRAIN_LIMITS` controls how many examples KRR uses to fit the score-delta predictor. These are intentionally separate axes.

## Orchestration Variables

Common controls:

| Variable | Meaning |
| --- | --- |
| `SMOKE_RUN=1` | use tiny configs and small limits for a fast end-to-end check |
| `PLAN_ONLY=1` | print commands without executing them |
| `DATASET` | dataset name used for run naming and default paths |
| `DATA_CONFIG` | YAML recipe for preparing data |
| `TRAIN_CONFIG` | MLX-LM adapter training config |
| `BASE_MODEL` | MLX-LM-compatible base model |
| `RUN_ID` | orchestration id, used in output paths |
| `RUN_PREFIX` | prefix for generated adapter run names |
| `PREPARE_DATA=0` | reuse existing prepared data |
| `TRAIN_FULL=0` | reuse an existing full adapter |
| `FULL_RUN_NAME` | full adapter run name to train or reuse |
| `SKIP_LORA_TEST=1` | skip the MLX-LM post-training test pass |

Subset adapter controls:

| Variable | Meaning |
| --- | --- |
| `N_VALUES="16 256 512"` | subset sizes for adapter training |
| `METHODS="random"` | subset selectors to run |
| `REUSE_EXISTING_ADAPTERS=1` | skip subset training when a completed adapter exists |
| `REQUIRE_EXISTING_ADAPTERS=1` | fail unless the expected subset adapters already exist |
| `PROJECTION_COMPONENTS=1024` | sparse random projection dimension |
| `THRESHOLDED_SIGN_THRESHOLD=0.01` | selector threshold for thresholded-sign subset methods |
| `SELECTOR_MAX_KMEANS_FEATURE_GB=4` | memory guard for unprojected k-means |
| `SELECTOR_PROJECTION_CHUNK_SIZE=16` | rows per projection chunk |

Comparison controls:

| Variable | Meaning |
| --- | --- |
| `RUN_COMPARISONS=0` | skip full-vs-subset adapter comparisons |
| `COMPARE_SPLIT=test` | split used by adapter comparison |
| `COMPARE_LIMIT=0` | number of examples scored, with `0` meaning all |
| `REUSE_EXISTING_COMPARISONS=1` | reuse completed comparison outputs |

Kernel controls:

| Variable | Meaning |
| --- | --- |
| `RUN_KERNEL=0` | skip KRR experiments |
| `KERNEL_CONFIGS="<a> <b>"` | one or more kernel configs, for example raw and thresholded-sign |
| `KERNEL_ADAPTER_RUNS="<run> ..."` | restrict KRR to selected adapters |
| `KERNEL_TRAIN_LIMITS="16 256 512"` | kernel fit-set sizes |
| `KERNEL_VALID_LIMIT=32` | validation examples used by KRR run |
| `KERNEL_TEST_LIMIT=256` | test examples used by KRR run and scatter plots |

If `KERNEL_ADAPTER_RUNS` is unset, the orchestrator runs KRR for every adapter it trained or reused in that orchestration. For paper-style runs, set `KERNEL_ADAPTER_RUNS` explicitly to avoid unnecessary kernel work.

Kernel feature thresholds are controlled by the kernel YAML itself, for example `backend_args.threshold` in `configs/kernel/*thresholded_sign*.yaml`. `THRESHOLDED_SIGN_THRESHOLD` only affects selector methods such as `kmeans-thresholded-sign`.

## Subset Methods

`METHODS` is a space-separated list of method suffixes:

| Method suffix | Selector | Transformation | Projection |
| --- | --- | --- | --- |
| `random` | `selectors/random.py` | none | none |
| `kmeans` | `selectors/lora_ntk_kmeans.py` | identity | identity |
| `kmeans-sign` | `selectors/lora_ntk_kmeans.py` | `sign` | identity |
| `kmeans-thresholded-sign` | `selectors/lora_ntk_kmeans.py` | `thresholded_sign` | identity |
| `kmeans-srp` | `selectors/lora_ntk_kmeans.py` | identity | `sparse_random` |
| `kmeans-srp-sign` | `selectors/lora_ntk_kmeans.py` | `sign` | `sparse_random` |
| `kmeans-srp-thresholded-sign` | `selectors/lora_ntk_kmeans.py` | `thresholded_sign` | `sparse_random` |

The k-means selector uses score-gradient features from the configured base model and adapter settings, then optionally applies a feature transformation and projection before clustering.

## Kernel Feature Variants

Kernel configs live under `configs/kernel/`. The raw Dolly config is:

```text
configs/kernel/dolly_lora_ntk.yaml
```

and the thresholded-sign variant is:

```text
configs/kernel/dolly_lora_ntk_thresholded_sign.yaml
```

The current feature backend extracts gradients with respect to LoRA leaves. By default, `backend_args.leaf_filter: lora_b_only` keeps only LoRA $B$ matrices. Use `leaf_filter: all` in a kernel config to include all trainable LoRA leaves.

Feature transforms are applied after raw feature extraction:

$$
\mathrm{sign}(\psi)_j = \mathrm{sign}(\psi_j)
$$

and

$$
\mathrm{thresholded\_sign}_\tau(\psi)_j =
\mathrm{sign}(\psi_j)\mathbf{1}\{|\psi_j| \ge \tau\}.
$$

The raw feature cache deliberately ignores transform-only backend arguments. That means thresholded-sign kernel runs reuse the expensive raw feature matrix and apply the threshold transformation on load.

## Naming and Outputs

The orchestrator generates run names deterministically from `RUN_PREFIX`, methods, and subset sizes:

```text
<RUN_PREFIX>-full
<RUN_PREFIX>-random-16
<RUN_PREFIX>-random-256
<RUN_PREFIX>-random-512
```

Kernel run names add the kernel config variant and the fit/valid/test limits:

```text
<adapter-run>-<kernel-config-name>-kernel-n16-v32-t256
<adapter-run>-<kernel-config-name>-kernel-n256-v32-t256
<adapter-run>-<kernel-config-name>-kernel-n512-v32-t256
```

Adapter outputs:

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
```

Orchestrated outputs:

```text
results/orchestrations/<run-id>/
  manifest.md
  comparisons/
  kernel/
    runs/
    cache/
      scores/
      features/
  kernel_configs/

reports/orchestrations/<run-id>/
  lora_runs.md
  adapter_comparisons.md
  kernel_runs.md
  kernel_comparison.md
  assets/
```

Kernel run outputs:

```text
results/orchestrations/<run-id>/kernel/runs/<kernel-run-name>/
  metadata.json
  resolved_config.yaml
  summary.json
  eval.json
  report.md
  scores/{train,valid,test}.jsonl
  predictions/{train,valid,test}.jsonl
```

## Caching

The kernel runner caches two expensive artifacts:

- base and adapter scores under `<kernel-output-root>/cache/scores/`
- raw score-gradient feature matrices under `<kernel-output-root>/cache/features/`

The score cache key includes the base model, adapter identity, selected records, and split. The feature cache key includes the base model, backend, adapter configuration, selected records, and raw extraction arguments.

Transform-only arguments such as `threshold`, `feature_transform`, and `feature_transformations` are excluded from the raw feature cache key. This is intentional: raw features are computed once, and transforms such as `thresholded_sign` are reapplied cheaply when the matrix is loaded.

Selector feature caches are separate and live under:

```text
results/selector_feature_cache/
```

They are used by tangent-feature k-means selection, not by the KRR kernel runs.

## Paper Figures

After kernel runs are complete, generate one SVG/PDF per paper panel:

```bash
uv run mlx-lora-make-kernel-paper-figures \
  --experiment Dolly=results/orchestrations/<dolly-run>/kernel \
  --experiment SAMSum=results/orchestrations/<samsum-run>/kernel \
  --experiment OpenMath=results/orchestrations/<openmath-run>/kernel \
  --adapter-contains random-512 \
  --train-sizes 16 256 512 \
  --features raw thresholded_sign \
  --individual \
  --pdf \
  --output-dir reports/paper/assets/paper \
  --output reports/paper/kernel_individual_figures.md
```

This writes predicted-vs-true plots, absolute RMSE plots, and RMSE-gain plots under:

```text
reports/paper/assets/paper/
```

## Standalone Commands

The orchestrator is preferred, but each stage can be run by hand.

Prepare data:

```bash
uv run mlx-lora-prepare-data \
  --config configs/data/dolly.yaml \
  --base-model mlx-community/SmolLM2-1.7B-Instruct
```

Train one adapter:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --run-name dolly-full \
  --base-model mlx-community/SmolLM2-1.7B-Instruct
```

Train one subset adapter:

```bash
uv run mlx-lora-run \
  --config configs/dolly.yaml \
  --run-name dolly-random-512 \
  --base-model mlx-community/SmolLM2-1.7B-Instruct \
  --sample-selector selectors/random.py \
  --max-examples 512
```

Compare a subset adapter against a full adapter:

```bash
uv run mlx-lora-compare-adapters \
  --config configs/dolly.yaml \
  --base-model mlx-community/SmolLM2-1.7B-Instruct \
  --full-adapter results/adapters/dolly-full \
  --subset-adapter results/adapters/dolly-random-512 \
  --split test \
  --limit 0 \
  --method random
```

Run one KRR experiment:

```bash
uv run mlx-lora-run-kernel \
  --config configs/kernel/dolly_lora_ntk.yaml \
  --run-name dolly-random-512-kernel-n64 \
  --base-model mlx-community/SmolLM2-1.7B-Instruct
```

Generate aggregate reports:

```bash
uv run mlx-lora-make-report
uv run mlx-lora-make-adapter-comparison
uv run mlx-lora-make-kernel-report
uv run mlx-lora-make-kernel-comparison
```

## Tests

Run everything:

```bash
uv run pytest
```

Run a focused file:

```bash
uv run pytest tests/test_reporting_plots.py
```

Static checks used during development:

```bash
uv run ruff check .
uv run basedpyright .
```
