#!/usr/bin/env bash
set -euo pipefail

# End-to-end empirical validation orchestrator.
#
# Default experiment:
# - dataset: Dolly
# - subset sizes: 128, 512, 2000
# - methods: random, kmeans, kmeans+sign, kmeans+sparse_random,
#   kmeans+sparse_random+sign
#
# This is intentionally expensive: with the defaults it runs one full-data
# adapter, 15 subset adapters, and kernel prediction for all trained adapters.
# Use PLAN_ONLY=1 to print the commands first. Use SMOKE_RUN=1 for a small
# end-to-end check intended to finish quickly on a laptop.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"

SMOKE_RUN="${SMOKE_RUN:-0}"
if [[ "$SMOKE_RUN" == "1" ]]; then
  DEFAULT_DATASET="dolly_smoke"
  DEFAULT_DATA_CONFIG="configs/data/dolly_smoke.yaml"
  DEFAULT_TRAIN_CONFIG="configs/dolly_smoke.yaml"
  DEFAULT_KERNEL_CONFIG="configs/kernel/dolly_lora_ntk_smoke.yaml"
  DEFAULT_N_VALUES="8"
  DEFAULT_METHODS="random kmeans kmeans-sign kmeans-srp kmeans-srp-sign"
  DEFAULT_PROJECTION_COMPONENTS="32"
  DEFAULT_COMPARE_LIMIT="8"
else
  DEFAULT_DATASET="dolly"
  DEFAULT_DATA_CONFIG="configs/data/dolly.yaml"
  DEFAULT_TRAIN_CONFIG="configs/dolly.yaml"
  DEFAULT_KERNEL_CONFIG="configs/kernel/dolly_lora_ntk.yaml"
  DEFAULT_N_VALUES="128 512 2000"
  DEFAULT_METHODS="random kmeans kmeans-sign kmeans-srp kmeans-srp-sign"
  DEFAULT_PROJECTION_COMPONENTS="1024"
  DEFAULT_COMPARE_LIMIT="0"
fi

DATASET="${DATASET:-$DEFAULT_DATASET}"
DATA_CONFIG="${DATA_CONFIG:-$DEFAULT_DATA_CONFIG}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$DEFAULT_TRAIN_CONFIG}"
KERNEL_CONFIG="${KERNEL_CONFIG:-$DEFAULT_KERNEL_CONFIG}"
BASE_MODEL="${BASE_MODEL:-mlx-community/SmolLM2-1.7B-Instruct}"
N_VALUES="${N_VALUES:-$DEFAULT_N_VALUES}"
METHODS="${METHODS:-$DEFAULT_METHODS}"
PROJECTION_COMPONENTS="${PROJECTION_COMPONENTS:-$DEFAULT_PROJECTION_COMPONENTS}"
SELECTOR_PROJECTION_CHUNK_SIZE="${SELECTOR_PROJECTION_CHUNK_SIZE:-16}"
SELECTOR_MAX_KMEANS_FEATURE_GB="${SELECTOR_MAX_KMEANS_FEATURE_GB:-4}"
THRESHOLDED_SIGN_THRESHOLD="${THRESHOLDED_SIGN_THRESHOLD:-0.01}"
COMPARE_SPLIT="${COMPARE_SPLIT:-test}"
COMPARE_LIMIT="${COMPARE_LIMIT:-$DEFAULT_COMPARE_LIMIT}"
PREPARE_DATA="${PREPARE_DATA:-1}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-data/${DATASET}}"
REUSE_EXISTING_ADAPTERS="${REUSE_EXISTING_ADAPTERS:-0}"
REUSE_EXISTING_COMPARISONS="${REUSE_EXISTING_COMPARISONS:-0}"
RUN_KERNEL="${RUN_KERNEL:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
DEBUG="${DEBUG:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-${DATASET}-${RUN_ID}}"
FULL_RUN="${FULL_RUN_NAME:-${RUN_PREFIX}-full}"
TRAIN_FULL="${TRAIN_FULL:-1}"
ORCH_DIR="${ORCH_DIR:-results/orchestrations/${RUN_ID}}"
REPORT_DIR="${REPORT_DIR:-reports/orchestrations/${RUN_ID}}"
KERNEL_OUTPUT_ROOT="${ORCH_DIR}/kernel"
KERNEL_CONFIG_DIR="${ORCH_DIR}/kernel_configs"

read -r -a N_ARRAY <<< "$N_VALUES"
if [[ "${#N_ARRAY[@]}" -eq 0 ]]; then
  echo "N_VALUES must contain at least one subset size." >&2
  exit 2
fi
read -r -a METHOD_ARRAY <<< "$METHODS"
if [[ "${#METHOD_ARRAY[@]}" -eq 0 ]]; then
  echo "METHODS must contain at least one method suffix." >&2
  exit 2
fi

prepared_data_exists() {
  local data_dir="$1"
  [[ -f "${data_dir}/train.jsonl" && -f "${data_dir}/valid.jsonl" && -f "${data_dir}/test.jsonl" ]]
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
  local comparison_dir="${ORCH_DIR}/comparisons/${subset_run}__vs__${full_run}"
  [[ -f "${comparison_dir}/summary.json" && -f "${comparison_dir}/scores.jsonl" ]]
}

SUBSET_RUN_COUNT=$((${#N_ARRAY[@]} * ${#METHOD_ARRAY[@]}))
PLANNED_ADAPTER_COUNT=$((1 + SUBSET_RUN_COUNT))
DATA_PREP_COMMAND_COUNT=1
if [[ "$PREPARE_DATA" != "1" ]]; then
  DATA_PREP_COMMAND_COUNT=0
fi
TRAIN_FULL_COMMAND_COUNT=1
if [[ "$TRAIN_FULL" != "1" ]]; then
  TRAIN_FULL_COMMAND_COUNT=0
fi
REUSED_SUBSET_COUNT=0
REUSED_COMPARISON_COUNT=0
if [[ "$REUSE_EXISTING_ADAPTERS" == "1" ]]; then
  for n in "${N_ARRAY[@]}"; do
    for suffix in "${METHOD_ARRAY[@]}"; do
      subset_run="${RUN_PREFIX}-${suffix}-${n}"
      if adapter_run_complete "$subset_run"; then
        REUSED_SUBSET_COUNT=$((REUSED_SUBSET_COUNT + 1))
      fi
      if [[ "$REUSE_EXISTING_COMPARISONS" == "1" ]] && comparison_complete "$subset_run" "$FULL_RUN"; then
        REUSED_COMPARISON_COUNT=$((REUSED_COMPARISON_COUNT + 1))
      fi
    done
  done
elif [[ "$REUSE_EXISTING_COMPARISONS" == "1" ]]; then
  for n in "${N_ARRAY[@]}"; do
    for suffix in "${METHOD_ARRAY[@]}"; do
      subset_run="${RUN_PREFIX}-${suffix}-${n}"
      if comparison_complete "$subset_run" "$FULL_RUN"; then
        REUSED_COMPARISON_COUNT=$((REUSED_COMPARISON_COUNT + 1))
      fi
    done
  done
fi
SUBSET_TRAIN_COMMAND_COUNT=$((SUBSET_RUN_COUNT - REUSED_SUBSET_COUNT))
COMPARISON_COMMAND_COUNT=$((SUBSET_RUN_COUNT - REUSED_COMPARISON_COUNT))
KERNEL_COMMAND_COUNT=0
if [[ "$RUN_KERNEL" == "1" ]]; then
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a PLANNED_KERNEL_ARRAY <<< "$KERNEL_ADAPTER_RUNS"
    KERNEL_COMMAND_COUNT="${#PLANNED_KERNEL_ARRAY[@]}"
  else
    KERNEL_COMMAND_COUNT="$PLANNED_ADAPTER_COUNT"
  fi
fi
REPORT_COMMAND_COUNT=2
if [[ "$RUN_KERNEL" == "1" ]]; then
  REPORT_COMMAND_COUNT=4
fi
TOTAL_COMMANDS=$((DATA_PREP_COMMAND_COUNT + TRAIN_FULL_COMMAND_COUNT + SUBSET_TRAIN_COMMAND_COUNT + COMPARISON_COMMAND_COUNT + KERNEL_COMMAND_COUNT + REPORT_COMMAND_COUNT))
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
    kmeans) echo "kmeans" ;;
    kmeans-sign) echo "kmeans+sign" ;;
    kmeans-thresholded-sign) echo "kmeans+thresholded_sign" ;;
    kmeans-srp) echo "kmeans+sparse_random" ;;
    kmeans-srp-sign) echo "kmeans+sparse_random+sign" ;;
    kmeans-srp-thresholded-sign) echo "kmeans+sparse_random+thresholded_sign" ;;
    *)
      echo "Unknown method suffix: $1" >&2
      return 2
      ;;
  esac
}

selector_path() {
  case "$1" in
    random) echo "selectors/random.py" ;;
    kmeans | kmeans-sign | kmeans-thresholded-sign | kmeans-srp | kmeans-srp-sign | kmeans-srp-thresholded-sign)
      echo "selectors/lora_ntk_kmeans.py"
      ;;
    *)
      echo "Unknown method suffix: $1" >&2
      return 2
      ;;
  esac
}

append_selector_args() {
  local suffix="$1"
  case "$suffix" in
    random | kmeans)
      ;;
    kmeans-sign)
      CMD+=(--selector-transformation sign)
      ;;
    kmeans-thresholded-sign)
      CMD+=(
        --selector-transformation thresholded_sign
        --selector-thresholded-sign-threshold "$THRESHOLDED_SIGN_THRESHOLD"
      )
      ;;
    kmeans-srp)
      CMD+=(
        --selector-projection sparse_random
        --selector-projection-components "$PROJECTION_COMPONENTS"
      )
      ;;
    kmeans-srp-sign)
      CMD+=(
        --selector-transformation sign
        --selector-projection sparse_random
        --selector-projection-components "$PROJECTION_COMPONENTS"
      )
      ;;
    kmeans-srp-thresholded-sign)
      CMD+=(
        --selector-transformation thresholded_sign
        --selector-thresholded-sign-threshold "$THRESHOLDED_SIGN_THRESHOLD"
        --selector-projection sparse_random
        --selector-projection-components "$PROJECTION_COMPONENTS"
      )
      ;;
    *)
      echo "Unknown method suffix: $suffix" >&2
      return 2
      ;;
  esac
}

write_manifest() {
  mkdir -p "$ORCH_DIR" "$REPORT_DIR"
  cat > "${ORCH_DIR}/manifest.md" <<EOF
# Orchestration ${RUN_ID}

- Dataset: \`${DATASET}\`
- Smoke run: \`${SMOKE_RUN}\`
- Debug: \`${DEBUG}\`
- Data config: \`${DATA_CONFIG}\`
- Prepare data: \`${PREPARE_DATA}\`
- Prepared data dir: \`${PREPARED_DATA_DIR}\`
- Training config: \`${TRAIN_CONFIG}\`
- Kernel config: \`${KERNEL_CONFIG}\`
- Base model: \`${BASE_MODEL}\`
- Subset sizes: \`${N_VALUES}\`
- Methods: \`${METHODS}\`
- Projection components: \`${PROJECTION_COMPONENTS}\`
- Selector projection chunk size: \`${SELECTOR_PROJECTION_CHUNK_SIZE}\`
- Selector max k-means feature GiB: \`${SELECTOR_MAX_KMEANS_FEATURE_GB}\`
- Thresholded sign threshold: \`${THRESHOLDED_SIGN_THRESHOLD}\`
- Compare split: \`${COMPARE_SPLIT}\`
- Compare limit: \`${COMPARE_LIMIT}\`
- Run prefix: \`${RUN_PREFIX}\`
- Full run: \`${FULL_RUN}\`
- Train full adapter: \`${TRAIN_FULL}\`
- Reuse existing subset adapters: \`${REUSE_EXISTING_ADAPTERS}\`
- Reuse existing comparisons: \`${REUSE_EXISTING_COMPARISONS}\`
- Kernel adapter runs: \`all trained adapters unless KERNEL_ADAPTER_RUNS is set\`
- Reports: \`${REPORT_DIR}\`

This orchestration trains one full adapter and one subset adapter for each
method/subset-size pair, compares every subset adapter against the full adapter,
runs kernel experiments for the configured adapter set, and generates Markdown
reports with SVG visualizations.
EOF
}

write_kernel_config() {
  local adapter_run="$1"
  local config_path="${KERNEL_CONFIG_DIR}/${adapter_run}.yaml"
  mkdir -p "$KERNEL_CONFIG_DIR"
  cat > "$config_path" <<EOF
extends: "${ROOT_DIR}/${KERNEL_CONFIG}"
base_model: "${BASE_MODEL}"
adapter_path: "${ROOT_DIR}/results/adapters/${adapter_run}"
output_root: "${ROOT_DIR}/${KERNEL_OUTPUT_ROOT}"
notes: "Orchestrated kernel run for ${adapter_run}."
EOF
  echo "$config_path"
}

log "Orchestration run id: ${RUN_ID}"
log "Output directory: ${ORCH_DIR}"
log "Report directory: ${REPORT_DIR}"
log "Smoke run: ${SMOKE_RUN}; debug: ${DEBUG}; plan only: ${PLAN_ONLY}"
log "Dataset: ${DATASET}; n values: ${N_VALUES}; methods: ${METHODS}"
log "Full run: ${FULL_RUN}; train full adapter: ${TRAIN_FULL}"
log "Prepare data: ${PREPARE_DATA}; prepared data dir: ${PREPARED_DATA_DIR}"
log "Reuse existing subset adapters: ${REUSE_EXISTING_ADAPTERS}; reusable subset adapters found: ${REUSED_SUBSET_COUNT}"
log "Reuse existing comparisons: ${REUSE_EXISTING_COMPARISONS}; reusable comparisons found: ${REUSED_COMPARISON_COUNT}"
log "Planned work: ${PLANNED_ADAPTER_COUNT} adapters, ${SUBSET_RUN_COUNT} comparisons, ${KERNEL_COMMAND_COUNT} kernel runs"
if [[ "$PLAN_ONLY" == "1" ]]; then
  log "PLAN_ONLY=1: commands will be printed but not executed."
fi

write_manifest

if [[ "$PREPARE_DATA" == "1" ]]; then
  section "Prepare data"
  run_cmd uv run mlx-lora-prepare-data \
    --config "$DATA_CONFIG" \
    --base-model "$BASE_MODEL"
else
  section "Reuse prepared data"
  log "Skipping data prep; using ${PREPARED_DATA_DIR}"
  if [[ "$PLAN_ONLY" != "1" ]] && ! prepared_data_exists "$PREPARED_DATA_DIR"; then
    echo "Missing prepared data files under: ${PREPARED_DATA_DIR}" >&2
    exit 2
  fi
fi

TRAINED_ADAPTER_RUNS=("$FULL_RUN")

if [[ "$TRAIN_FULL" == "1" ]]; then
  section "Train full-data adapter"
  run_cmd uv run mlx-lora-run \
    --config "$TRAIN_CONFIG" \
    --run-name "$FULL_RUN" \
    --base-model "$BASE_MODEL"
else
  section "Reuse full-data adapter"
  log "Skipping full adapter training; using ${FULL_RUN}"
  if [[ "$PLAN_ONLY" != "1" ]]; then
    if ! adapter_run_complete "$FULL_RUN"; then
      echo "Missing completed full adapter run: ${FULL_RUN}" >&2
      exit 2
    fi
  fi
fi

section "Train subset adapters and compare against full adapter"
SUBSET_INDEX=0
for n in "${N_ARRAY[@]}"; do
  for suffix in "${METHOD_ARRAY[@]}"; do
    SUBSET_INDEX=$((SUBSET_INDEX + 1))
    method="$(method_label "$suffix")"
    selector="$(selector_path "$suffix")"
    subset_run="${RUN_PREFIX}-${suffix}-${n}"

    section "Subset ${SUBSET_INDEX}/${SUBSET_RUN_COUNT}: ${method}, n=${n}"
    CMD=(
      uv run mlx-lora-run
      --config "$TRAIN_CONFIG"
      --run-name "$subset_run"
      --base-model "$BASE_MODEL"
      --sample-selector "$selector"
      --max-examples "$n"
      --selector-projection-chunk-size "$SELECTOR_PROJECTION_CHUNK_SIZE"
      --selector-max-kmeans-feature-gb "$SELECTOR_MAX_KMEANS_FEATURE_GB"
    )
    if [[ "$DEBUG" == "1" ]]; then
      CMD+=(--selector-debug)
    fi
    append_selector_args "$suffix"
    if [[ "$REUSE_EXISTING_ADAPTERS" == "1" ]] && adapter_run_complete "$subset_run"; then
      log "Skipping subset adapter training; using existing completed run ${subset_run}"
    else
      run_cmd "${CMD[@]}"
    fi
    TRAINED_ADAPTER_RUNS+=("$subset_run")

    comparison_dir="${ORCH_DIR}/comparisons/${subset_run}__vs__${FULL_RUN}"
    if [[ "$REUSE_EXISTING_COMPARISONS" == "1" ]] && comparison_complete "$subset_run" "$FULL_RUN"; then
      log "Skipping adapter comparison; using existing ${comparison_dir}"
    else
      run_cmd uv run mlx-lora-compare-adapters \
        --config "$TRAIN_CONFIG" \
        --base-model "$BASE_MODEL" \
        --full-adapter "results/adapters/${FULL_RUN}" \
        --subset-adapter "results/adapters/${subset_run}" \
        --split "$COMPARE_SPLIT" \
        --limit "$COMPARE_LIMIT" \
        --method "$method" \
        --output-dir "$comparison_dir"
    fi
  done
done

if [[ "$RUN_KERNEL" == "1" ]]; then
  section "Run kernel prediction experiments"
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a KERNEL_RUN_ARRAY <<< "$KERNEL_ADAPTER_RUNS"
  else
    KERNEL_RUN_ARRAY=("${TRAINED_ADAPTER_RUNS[@]}")
  fi
  KERNEL_INDEX=0
  for adapter_run in "${KERNEL_RUN_ARRAY[@]}"; do
    KERNEL_INDEX=$((KERNEL_INDEX + 1))
    log "Kernel ${KERNEL_INDEX}/${#KERNEL_RUN_ARRAY[@]} for adapter ${adapter_run}"
    kernel_config="$(write_kernel_config "$adapter_run")"
    kernel_run_name="${adapter_run}-kernel"
    run_cmd uv run mlx-lora-run-kernel \
      --config "$kernel_config" \
      --run-name "$kernel_run_name" \
      --base-model "$BASE_MODEL"
  done
else
  section "Skip kernel prediction experiments"
  log "RUN_KERNEL=${RUN_KERNEL}"
fi

section "Generate reports and visuals"
run_cmd uv run mlx-lora-make-report \
  --output "${REPORT_DIR}/lora_runs.md"

run_cmd uv run mlx-lora-make-adapter-comparison \
  --output-root "$ORCH_DIR" \
  --output "${REPORT_DIR}/adapter_comparisons.md" \
  --assets-dir "${REPORT_DIR}/assets/adapter_comparisons"

if [[ "$RUN_KERNEL" == "1" ]]; then
  run_cmd uv run mlx-lora-make-kernel-report \
    --output-root "$KERNEL_OUTPUT_ROOT" \
    --output "${REPORT_DIR}/kernel_runs.md" \
    --assets-dir "${REPORT_DIR}/assets/kernel"

  run_cmd uv run mlx-lora-make-kernel-comparison \
    --output-root "$KERNEL_OUTPUT_ROOT" \
    --output "${REPORT_DIR}/kernel_comparison.md"
fi

section "Done"
log "Manifest: ${ORCH_DIR}/manifest.md"
log "Reports: ${REPORT_DIR}"
