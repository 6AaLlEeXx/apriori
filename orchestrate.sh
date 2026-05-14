#!/usr/bin/env bash
set -euo pipefail

# End-to-end empirical validation orchestrator.
#
# Default experiment:
# - dataset: Dolly
# - subset sizes: 128, 512, 2000
# - method: random subset selection
#
# This is intentionally expensive: with the defaults it runs one full-data
# adapter, three subset adapters, and kernel prediction for all trained adapters.
# Use PLAN_ONLY=1 to print the commands first. Use SMOKE_RUN=1 for a small
# end-to-end check intended to finish quickly on a laptop.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"

SMOKE_RUN="${SMOKE_RUN:-0}"
if [[ "$SMOKE_RUN" == "1" ]]; then
  DEFAULT_DATASET_NAME="dolly_smoke"
  DEFAULT_DATA_PREP_CONFIG="configs/data/instruction/dolly_smoke.yaml"
  DEFAULT_ADAPTER_TRAIN_CONFIG="configs/train/instruction/dolly_smoke.yaml"
  DEFAULT_KERNEL_EXPERIMENT_CONFIG="configs/kernel/instruction/dolly_score_gradient_smoke.yaml"
  DEFAULT_SUBSET_TRAIN_SIZES="8"
  DEFAULT_SUBSET_SELECTION_METHODS="random"
  DEFAULT_ADAPTER_COMPARISON_EXAMPLE_LIMIT="8"
else
  DEFAULT_DATASET_NAME="dolly"
  DEFAULT_DATA_PREP_CONFIG="configs/data/instruction/dolly.yaml"
  DEFAULT_ADAPTER_TRAIN_CONFIG="configs/train/instruction/dolly.yaml"
  DEFAULT_KERNEL_EXPERIMENT_CONFIG="configs/kernel/instruction/dolly_score_gradient.yaml"
  DEFAULT_SUBSET_TRAIN_SIZES="128 512 2000"
  DEFAULT_SUBSET_SELECTION_METHODS="random"
  DEFAULT_ADAPTER_COMPARISON_EXAMPLE_LIMIT="0"
fi

DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET_NAME}"
DATA_PREP_CONFIG="${DATA_PREP_CONFIG:-$DEFAULT_DATA_PREP_CONFIG}"
ADAPTER_TRAIN_CONFIG="${ADAPTER_TRAIN_CONFIG:-$DEFAULT_ADAPTER_TRAIN_CONFIG}"
KERNEL_EXPERIMENT_CONFIG="${KERNEL_EXPERIMENT_CONFIG:-$DEFAULT_KERNEL_EXPERIMENT_CONFIG}"
KERNEL_EXPERIMENT_CONFIGS="${KERNEL_EXPERIMENT_CONFIGS:-$KERNEL_EXPERIMENT_CONFIG}"
BASE_MODEL="${BASE_MODEL:-mlx-community/SmolLM2-1.7B-Instruct}"
SUBSET_TRAIN_SIZES="${SUBSET_TRAIN_SIZES:-$DEFAULT_SUBSET_TRAIN_SIZES}"
SUBSET_SELECTION_METHODS="${SUBSET_SELECTION_METHODS:-$DEFAULT_SUBSET_SELECTION_METHODS}"
ADAPTER_COMPARISON_SPLIT="${ADAPTER_COMPARISON_SPLIT:-test}"
ADAPTER_COMPARISON_EXAMPLE_LIMIT="${ADAPTER_COMPARISON_EXAMPLE_LIMIT:-$DEFAULT_ADAPTER_COMPARISON_EXAMPLE_LIMIT}"
PREPARE_DATA="${PREPARE_DATA:-1}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-data/${DATASET_NAME}}"
REUSE_EXISTING_SUBSET_ADAPTERS="${REUSE_EXISTING_SUBSET_ADAPTERS:-0}"
REQUIRE_EXISTING_SUBSET_ADAPTERS="${REQUIRE_EXISTING_SUBSET_ADAPTERS:-0}"
REUSE_EXISTING_ADAPTER_COMPARISONS="${REUSE_EXISTING_ADAPTER_COMPARISONS:-0}"
RUN_ADAPTER_COMPARISONS="${RUN_ADAPTER_COMPARISONS:-1}"
RUN_KERNEL_EXPERIMENTS="${RUN_KERNEL_EXPERIMENTS:-1}"
SKIP_LORA_TEST="${SKIP_LORA_TEST:-0}"
KERNEL_FIT_SIZES="${KERNEL_FIT_SIZES:-}"
KERNEL_VALID_EXAMPLES="${KERNEL_VALID_EXAMPLES:-}"
KERNEL_TEST_EXAMPLES="${KERNEL_TEST_EXAMPLES:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SELECTOR_DEBUG="${SELECTOR_DEBUG:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
ADAPTER_RUN_PREFIX="${ADAPTER_RUN_PREFIX:-${DATASET_NAME}-${RUN_ID}}"
FULL_ADAPTER_RUN="${FULL_ADAPTER_RUN_NAME:-${ADAPTER_RUN_PREFIX}-full}"
TRAIN_FULL_ADAPTER="${TRAIN_FULL_ADAPTER:-1}"
ORCHESTRATION_DIR="${ORCHESTRATION_DIR:-results/orchestrations/${RUN_ID}}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-reports/orchestrations/${RUN_ID}}"
KERNEL_RESULTS_ROOT="${ORCHESTRATION_DIR}/kernel"
GENERATED_KERNEL_CONFIG_DIR="${ORCHESTRATION_DIR}/kernel_configs"
KERNEL_FIGURE_SPLIT="${KERNEL_FIGURE_SPLIT:-test}"
KERNEL_FIGURE_ADAPTER_NAME_CONTAINS="${KERNEL_FIGURE_ADAPTER_NAME_CONTAINS:-}"
KERNEL_FIGURE_INDIVIDUAL_PDFS="${KERNEL_FIGURE_INDIVIDUAL_PDFS:-0}"
KERNEL_FIGURE_STYLE="${KERNEL_FIGURE_STYLE:-paper}"

read -r -a SUBSET_SIZE_ARRAY <<< "$SUBSET_TRAIN_SIZES"
if [[ "${#SUBSET_SIZE_ARRAY[@]}" -eq 0 ]]; then
  echo "SUBSET_TRAIN_SIZES must contain at least one subset size." >&2
  exit 2
fi
read -r -a SUBSET_METHOD_ARRAY <<< "$SUBSET_SELECTION_METHODS"
if [[ "${#SUBSET_METHOD_ARRAY[@]}" -eq 0 ]]; then
  echo "SUBSET_SELECTION_METHODS must contain at least one method suffix." >&2
  exit 2
fi
read -r -a KERNEL_EXPERIMENT_CONFIG_ARRAY <<< "$KERNEL_EXPERIMENT_CONFIGS"
if [[ "${#KERNEL_EXPERIMENT_CONFIG_ARRAY[@]}" -eq 0 ]]; then
  echo "KERNEL_EXPERIMENT_CONFIGS must contain at least one kernel config." >&2
  exit 2
fi
KERNEL_EXPERIMENT_CONFIG_COUNT="${#KERNEL_EXPERIMENT_CONFIG_ARRAY[@]}"
if [[ -n "$KERNEL_FIT_SIZES" ]]; then
  read -r -a KERNEL_FIT_SIZE_ARRAY <<< "$KERNEL_FIT_SIZES"
else
  KERNEL_FIT_SIZE_ARRAY=("")
fi

prepared_data_exists() {
  local prepared_data_dir="$1"
  [[ -f "${prepared_data_dir}/train.jsonl" && -f "${prepared_data_dir}/valid.jsonl" && -f "${prepared_data_dir}/test.jsonl" ]]
}

adapter_run_complete() {
  local run_name="$1"
  local summary_path="results/runs/${run_name}/summary.json"
  [[ -f "results/adapters/${run_name}/adapters.safetensors" ]] || return 1
  [[ -f "$summary_path" ]] || return 1
  grep -q '"status": "completed"' "$summary_path"
}

comparison_complete() {
  local subset_run="$1"
  local full_run="$2"
  local comparison_dir="${ORCHESTRATION_DIR}/comparisons/${subset_run}__vs__${full_run}"
  [[ -f "${comparison_dir}/summary.json" && -f "${comparison_dir}/scores.jsonl" ]]
}

SUBSET_RUN_COUNT=$((${#SUBSET_SIZE_ARRAY[@]} * ${#SUBSET_METHOD_ARRAY[@]}))
PLANNED_ADAPTER_COUNT=$((1 + SUBSET_RUN_COUNT))
DATA_PREP_COMMAND_COUNT=1
if [[ "$PREPARE_DATA" != "1" ]]; then
  DATA_PREP_COMMAND_COUNT=0
fi
TRAIN_FULL_ADAPTER_COMMAND_COUNT=1
if [[ "$TRAIN_FULL_ADAPTER" != "1" ]]; then
  TRAIN_FULL_ADAPTER_COMMAND_COUNT=0
fi
REUSED_SUBSET_COUNT=0
REUSED_COMPARISON_COUNT=0
if [[ "$REUSE_EXISTING_SUBSET_ADAPTERS" == "1" ]]; then
  for subset_size in "${SUBSET_SIZE_ARRAY[@]}"; do
    for suffix in "${SUBSET_METHOD_ARRAY[@]}"; do
      subset_run="${ADAPTER_RUN_PREFIX}-${suffix}-subset${subset_size}"
      if adapter_run_complete "$subset_run"; then
        REUSED_SUBSET_COUNT=$((REUSED_SUBSET_COUNT + 1))
      fi
      if [[ "$REUSE_EXISTING_ADAPTER_COMPARISONS" == "1" ]] && comparison_complete "$subset_run" "$FULL_ADAPTER_RUN"; then
        REUSED_COMPARISON_COUNT=$((REUSED_COMPARISON_COUNT + 1))
      fi
    done
  done
elif [[ "$REUSE_EXISTING_ADAPTER_COMPARISONS" == "1" ]]; then
  for subset_size in "${SUBSET_SIZE_ARRAY[@]}"; do
    for suffix in "${SUBSET_METHOD_ARRAY[@]}"; do
      subset_run="${ADAPTER_RUN_PREFIX}-${suffix}-subset${subset_size}"
      if comparison_complete "$subset_run" "$FULL_ADAPTER_RUN"; then
        REUSED_COMPARISON_COUNT=$((REUSED_COMPARISON_COUNT + 1))
      fi
    done
  done
fi
SUBSET_TRAIN_COMMAND_COUNT=$((SUBSET_RUN_COUNT - REUSED_SUBSET_COUNT))
if [[ "$REQUIRE_EXISTING_SUBSET_ADAPTERS" == "1" ]]; then
  SUBSET_TRAIN_COMMAND_COUNT=0
fi
COMPARISON_COMMAND_COUNT=$((SUBSET_RUN_COUNT - REUSED_COMPARISON_COUNT))
if [[ "$RUN_ADAPTER_COMPARISONS" != "1" ]]; then
  COMPARISON_COMMAND_COUNT=0
fi
KERNEL_COMMAND_COUNT=0
if [[ "$RUN_KERNEL_EXPERIMENTS" == "1" ]]; then
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a PLANNED_KERNEL_ARRAY <<< "$KERNEL_ADAPTER_RUNS"
    KERNEL_COMMAND_COUNT=$((${#PLANNED_KERNEL_ARRAY[@]} * ${#KERNEL_FIT_SIZE_ARRAY[@]} * KERNEL_EXPERIMENT_CONFIG_COUNT))
  else
    KERNEL_COMMAND_COUNT=$((PLANNED_ADAPTER_COUNT * ${#KERNEL_FIT_SIZE_ARRAY[@]} * KERNEL_EXPERIMENT_CONFIG_COUNT))
  fi
fi
REPORT_COMMAND_COUNT=0
if [[ "$RUN_KERNEL_EXPERIMENTS" == "1" ]]; then
  REPORT_COMMAND_COUNT=1
fi
TOTAL_COMMANDS=$((DATA_PREP_COMMAND_COUNT + TRAIN_FULL_ADAPTER_COMMAND_COUNT + SUBSET_TRAIN_COMMAND_COUNT + COMPARISON_COMMAND_COUNT + KERNEL_COMMAND_COUNT + REPORT_COMMAND_COUNT))
COMMAND_INDEX=0

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

section() {
  printf '\n[%s] == %s ==\n' "$(timestamp)" "$*"
}

format_command() {
  printf '%q ' "$@"
}

run_cmd() {
  COMMAND_INDEX=$((COMMAND_INDEX + 1))
  local start="$SECONDS"
  log "[$COMMAND_INDEX/$TOTAL_COMMANDS] START $(format_command "$@")"
  if [[ "$PLAN_ONLY" != "1" ]]; then
    local status=0
    if "$@"; then
      status=0
    else
      status=$?
    fi
    local elapsed=$((SECONDS - start))
    if [[ "$status" -eq 0 ]]; then
      log "[$COMMAND_INDEX/$TOTAL_COMMANDS] DONE in ${elapsed}s"
    else
      log "[$COMMAND_INDEX/$TOTAL_COMMANDS] FAILED after ${elapsed}s with exit ${status}"
    fi
    return "$status"
  else
    log "[$COMMAND_INDEX/$TOTAL_COMMANDS] PLAN only"
  fi
}

method_label() {
  case "$1" in
    random) echo "random" ;;
    *)
      echo "Unknown method suffix: $1" >&2
      return 2
      ;;
  esac
}

selector_path() {
  case "$1" in
    random) echo "selectors/random.py" ;;
    *)
      echo "Unknown method suffix: $1" >&2
      return 2
      ;;
  esac
}

write_manifest() {
  mkdir -p "$ORCHESTRATION_DIR" "$REPORT_OUTPUT_DIR"
  cat > "${ORCHESTRATION_DIR}/manifest.md" <<EOF
# Orchestration ${RUN_ID}

- Dataset: \`${DATASET_NAME}\`
- Smoke run: \`${SMOKE_RUN}\`
- Selector debug: \`${SELECTOR_DEBUG}\`
- Data prep config: \`${DATA_PREP_CONFIG}\`
- Prepare data: \`${PREPARE_DATA}\`
- Prepared data dir: \`${PREPARED_DATA_DIR}\`
- Adapter train config: \`${ADAPTER_TRAIN_CONFIG}\`
- Skip LoRA post-train test: \`${SKIP_LORA_TEST}\`
- Kernel experiment config: \`${KERNEL_EXPERIMENT_CONFIG}\`
- Kernel experiment configs: \`${KERNEL_EXPERIMENT_CONFIGS}\`
- Base model: \`${BASE_MODEL}\`
- Subset sizes: \`${SUBSET_TRAIN_SIZES}\`
- Subset selection methods: \`${SUBSET_SELECTION_METHODS}\`
- Adapter comparison split: \`${ADAPTER_COMPARISON_SPLIT}\`
- Adapter comparison example limit: \`${ADAPTER_COMPARISON_EXAMPLE_LIMIT}\`
- Run adapter comparisons: \`${RUN_ADAPTER_COMPARISONS}\`
- Adapter run prefix: \`${ADAPTER_RUN_PREFIX}\`
- Full adapter run: \`${FULL_ADAPTER_RUN}\`
- Train full adapter: \`${TRAIN_FULL_ADAPTER}\`
- Reuse existing subset adapters: \`${REUSE_EXISTING_SUBSET_ADAPTERS}\`
- Require existing subset adapters: \`${REQUIRE_EXISTING_SUBSET_ADAPTERS}\`
- Reuse existing comparisons: \`${REUSE_EXISTING_ADAPTER_COMPARISONS}\`
- Kernel adapter runs: \`all trained adapters unless KERNEL_ADAPTER_RUNS is set\`
- Kernel fit sizes: \`${KERNEL_FIT_SIZES:-config default}\`
- Kernel validation examples override: \`${KERNEL_VALID_EXAMPLES:-config default}\`
- Kernel test examples override: \`${KERNEL_TEST_EXAMPLES:-config default}\`
- Kernel figure style: \`${KERNEL_FIGURE_STYLE}\`
- Kernel figures: \`${REPORT_OUTPUT_DIR}\`

This orchestration trains or reuses one full adapter and one subset adapter for
each method/subset-size pair, optionally compares every subset adapter against
the full adapter, runs kernel experiments for the configured adapter set, and
generates paper-oriented kernel figures.
EOF
}

kernel_run_suffix() {
  local krr_fit_size="$1"
  local suffix="kernel"
  if [[ -n "$krr_fit_size" ]]; then
    suffix="${suffix}-fit${krr_fit_size}"
  fi
  if [[ -n "$KERNEL_VALID_EXAMPLES" ]]; then
    suffix="${suffix}-valid${KERNEL_VALID_EXAMPLES}"
  fi
  if [[ -n "$KERNEL_TEST_EXAMPLES" ]]; then
    suffix="${suffix}-test${KERNEL_TEST_EXAMPLES}"
  fi
  echo "$suffix"
}

kernel_config_variant() {
  local config_path="$1"
  local base
  base="$(basename "$config_path")"
  base="${base%.*}"
  echo "$base"
}

write_kernel_config() {
  local adapter_run="$1"
  local krr_fit_size="${2:-}"
  local kernel_base_config="${3:-$KERNEL_EXPERIMENT_CONFIG}"
  local variant="${4:-}"
  local suffix
  suffix="$(kernel_run_suffix "$krr_fit_size")"
  local config_component=""
  if [[ -n "$variant" ]]; then
    config_component="-${variant}"
  fi
  local config_path="${GENERATED_KERNEL_CONFIG_DIR}/${adapter_run}${config_component}-${suffix}.yaml"
  mkdir -p "$GENERATED_KERNEL_CONFIG_DIR"
  cat > "$config_path" <<EOF
extends: "${ROOT_DIR}/${kernel_base_config}"
base_model: "${BASE_MODEL}"
adapter_path: "${ROOT_DIR}/results/adapters/${adapter_run}"
kernel_results_root: "${ROOT_DIR}/${KERNEL_RESULTS_ROOT}"
EOF
  if [[ -n "$krr_fit_size" ]]; then
    printf 'krr_fit_examples: %s\n' "$krr_fit_size" >> "$config_path"
  fi
  if [[ -n "$KERNEL_VALID_EXAMPLES" ]]; then
    printf 'validation_examples: %s\n' "$KERNEL_VALID_EXAMPLES" >> "$config_path"
  fi
  if [[ -n "$KERNEL_TEST_EXAMPLES" ]]; then
    printf 'test_examples: %s\n' "$KERNEL_TEST_EXAMPLES" >> "$config_path"
  fi
  cat >> "$config_path" <<EOF
notes: "Orchestrated kernel run for ${adapter_run}${krr_fit_size:+ with krr_fit_examples=${krr_fit_size}}."
EOF
  echo "$config_path"
}

log "Orchestration run id: ${RUN_ID}"
log "Output directory: ${ORCHESTRATION_DIR}"
log "Report directory: ${REPORT_OUTPUT_DIR}"
log "Smoke run: ${SMOKE_RUN}; debug: ${SELECTOR_DEBUG}; plan only: ${PLAN_ONLY}"
log "Dataset: ${DATASET_NAME}; subset train sizes: ${SUBSET_TRAIN_SIZES}; subset selection methods: ${SUBSET_SELECTION_METHODS}"
log "Kernel experiment configs: ${KERNEL_EXPERIMENT_CONFIGS}"
log "Full run: ${FULL_ADAPTER_RUN}; train full adapter: ${TRAIN_FULL_ADAPTER}"
log "Skip LoRA post-train test: ${SKIP_LORA_TEST}"
log "Prepare data: ${PREPARE_DATA}; prepared data dir: ${PREPARED_DATA_DIR}"
log "Run adapter comparisons: ${RUN_ADAPTER_COMPARISONS}"
log "Reuse existing subset adapters: ${REUSE_EXISTING_SUBSET_ADAPTERS}; reusable subset adapters found: ${REUSED_SUBSET_COUNT}"
log "Require existing subset adapters: ${REQUIRE_EXISTING_SUBSET_ADAPTERS}"
log "Reuse existing comparisons: ${REUSE_EXISTING_ADAPTER_COMPARISONS}; reusable comparisons found: ${REUSED_COMPARISON_COUNT}"
log "Kernel fit sizes: ${KERNEL_FIT_SIZES:-config default}; validation override: ${KERNEL_VALID_EXAMPLES:-config default}; test override: ${KERNEL_TEST_EXAMPLES:-config default}"
log "Planned work: ${PLANNED_ADAPTER_COUNT} adapters, ${SUBSET_RUN_COUNT} comparisons, ${KERNEL_COMMAND_COUNT} kernel runs"
if [[ "$PLAN_ONLY" == "1" ]]; then
  log "PLAN_ONLY=1: commands will be printed but not executed."
fi

write_manifest

if [[ "$PREPARE_DATA" == "1" ]]; then
  section "Prepare data"
  run_cmd uv run mlx-lora-prepare-data \
    --config "$DATA_PREP_CONFIG" \
    --base-model "$BASE_MODEL"
else
  section "Reuse prepared data"
  log "Skipping data prep; using ${PREPARED_DATA_DIR}"
  if [[ "$PLAN_ONLY" != "1" ]] && ! prepared_data_exists "$PREPARED_DATA_DIR"; then
    echo "Missing prepared data files under: ${PREPARED_DATA_DIR}" >&2
    exit 2
  fi
fi

TRAINED_ADAPTER_RUNS=("$FULL_ADAPTER_RUN")

if [[ "$TRAIN_FULL_ADAPTER" == "1" ]]; then
  section "Train full-data adapter"
  FULL_CMD=(
    uv run mlx-lora-run
    --config "$ADAPTER_TRAIN_CONFIG" \
    --run-name "$FULL_ADAPTER_RUN" \
    --base-model "$BASE_MODEL"
  )
  if [[ "$SKIP_LORA_TEST" == "1" ]]; then
    FULL_CMD+=(--skip-test)
  fi
  run_cmd "${FULL_CMD[@]}"
else
  section "Reuse full-data adapter"
  log "Skipping full adapter training; using ${FULL_ADAPTER_RUN}"
  if [[ "$PLAN_ONLY" != "1" ]]; then
    if ! adapter_run_complete "$FULL_ADAPTER_RUN"; then
      echo "Missing completed full adapter run: ${FULL_ADAPTER_RUN}" >&2
      exit 2
    fi
  fi
fi

section "Train subset adapters and compare against full adapter"
SUBSET_INDEX=0
for subset_size in "${SUBSET_SIZE_ARRAY[@]}"; do
  for suffix in "${SUBSET_METHOD_ARRAY[@]}"; do
    SUBSET_INDEX=$((SUBSET_INDEX + 1))
    method="$(method_label "$suffix")"
    selector="$(selector_path "$suffix")"
    subset_run="${ADAPTER_RUN_PREFIX}-${suffix}-subset${subset_size}"

    section "Subset ${SUBSET_INDEX}/${SUBSET_RUN_COUNT}: ${method}, subset size=${subset_size}"
    CMD=(
      uv run mlx-lora-run
      --config "$ADAPTER_TRAIN_CONFIG"
      --run-name "$subset_run"
      --base-model "$BASE_MODEL"
      --sample-selector "$selector"
      --subset-train-size "$subset_size"
    )
    if [[ "$SELECTOR_DEBUG" == "1" ]]; then
      CMD+=(--selector-debug)
    fi
    if [[ "$SKIP_LORA_TEST" == "1" ]]; then
      CMD+=(--skip-test)
    fi
    if [[ "$REUSE_EXISTING_SUBSET_ADAPTERS" == "1" ]] && adapter_run_complete "$subset_run"; then
      log "Skipping subset adapter training; using existing completed run ${subset_run}"
    elif [[ "$REQUIRE_EXISTING_SUBSET_ADAPTERS" == "1" ]]; then
      log "Requiring existing completed subset adapter run ${subset_run}"
      if [[ "$PLAN_ONLY" != "1" ]]; then
        echo "Missing completed subset adapter run: ${subset_run}" >&2
        exit 2
      fi
    else
      run_cmd "${CMD[@]}"
    fi
    TRAINED_ADAPTER_RUNS+=("$subset_run")

    if [[ "$RUN_ADAPTER_COMPARISONS" != "1" ]]; then
      log "Skipping adapter comparison; RUN_ADAPTER_COMPARISONS=${RUN_ADAPTER_COMPARISONS}"
    else
      comparison_dir="${ORCHESTRATION_DIR}/comparisons/${subset_run}__vs__${FULL_ADAPTER_RUN}"
      if [[ "$REUSE_EXISTING_ADAPTER_COMPARISONS" == "1" ]] && comparison_complete "$subset_run" "$FULL_ADAPTER_RUN"; then
        log "Skipping adapter comparison; using existing ${comparison_dir}"
      else
        run_cmd uv run mlx-lora-compare-adapters \
          --config "$ADAPTER_TRAIN_CONFIG" \
          --base-model "$BASE_MODEL" \
          --full-adapter "results/adapters/${FULL_ADAPTER_RUN}" \
          --subset-adapter "results/adapters/${subset_run}" \
          --split "$ADAPTER_COMPARISON_SPLIT" \
          --example-limit "$ADAPTER_COMPARISON_EXAMPLE_LIMIT" \
          --subset-method-label "$method" \
          --output-dir "$comparison_dir"
      fi
    fi
  done
done

if [[ "$RUN_KERNEL_EXPERIMENTS" == "1" ]]; then
  section "Run kernel prediction experiments"
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a KERNEL_RUN_ARRAY <<< "$KERNEL_ADAPTER_RUNS"
  else
    KERNEL_RUN_ARRAY=("${TRAINED_ADAPTER_RUNS[@]}")
  fi
  KERNEL_INDEX=0
  for adapter_run in "${KERNEL_RUN_ARRAY[@]}"; do
    for kernel_base_config in "${KERNEL_EXPERIMENT_CONFIG_ARRAY[@]}"; do
      kernel_variant=""
      if [[ "$KERNEL_EXPERIMENT_CONFIG_COUNT" -gt 1 ]]; then
        kernel_variant="$(kernel_config_variant "$kernel_base_config")"
      fi
      for krr_fit_size in "${KERNEL_FIT_SIZE_ARRAY[@]}"; do
        KERNEL_INDEX=$((KERNEL_INDEX + 1))
        kernel_suffix="$(kernel_run_suffix "$krr_fit_size")"
        if [[ -n "$kernel_variant" ]]; then
          kernel_run_name="${adapter_run}-${kernel_variant}-${kernel_suffix}"
        else
          kernel_run_name="${adapter_run}-${kernel_suffix}"
        fi
        log "Kernel ${KERNEL_INDEX}/${KERNEL_COMMAND_COUNT} for adapter ${adapter_run}, config=${kernel_base_config}, krr_fit_examples=${krr_fit_size:-config default}"
        kernel_config="$(write_kernel_config "$adapter_run" "$krr_fit_size" "$kernel_base_config" "$kernel_variant")"
        run_cmd uv run mlx-lora-run-kernel \
          --config "$kernel_config" \
          --run-name "$kernel_run_name" \
          --base-model "$BASE_MODEL"
      done
    done
  done
else
  section "Skip kernel prediction experiments"
  log "RUN_KERNEL_EXPERIMENTS=${RUN_KERNEL_EXPERIMENTS}"
fi

section "Generate kernel figures"
if [[ "$RUN_KERNEL_EXPERIMENTS" == "1" ]]; then
  KERNEL_FIGURE_CMD=(
    uv run mlx-lora-make-kernel-figures
    --kernel-results "${DATASET_NAME}=${KERNEL_RESULTS_ROOT}"
    --output-dir "${REPORT_OUTPUT_DIR}/assets/kernel_figures"
    --output "${REPORT_OUTPUT_DIR}/kernel_figures.md"
    --split "$KERNEL_FIGURE_SPLIT"
    --plot-style "$KERNEL_FIGURE_STYLE"
  )
  if [[ -n "$KERNEL_FIGURE_ADAPTER_NAME_CONTAINS" ]]; then
    KERNEL_FIGURE_CMD+=(--adapter-name-contains "$KERNEL_FIGURE_ADAPTER_NAME_CONTAINS")
  fi
  if [[ "$KERNEL_FIGURE_INDIVIDUAL_PDFS" == "1" ]]; then
    KERNEL_FIGURE_CMD+=(--individual-pdfs)
  fi
  run_cmd "${KERNEL_FIGURE_CMD[@]}"
else
  log "Skipping kernel figures; RUN_KERNEL_EXPERIMENTS=${RUN_KERNEL_EXPERIMENTS}"
fi

section "Done"
log "Manifest: ${ORCHESTRATION_DIR}/manifest.md"
log "Kernel figures: ${REPORT_OUTPUT_DIR}"
