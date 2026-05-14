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
  DEFAULT_DATASET="dolly_smoke"
  DEFAULT_DATA_CONFIG="configs/data/instruction/dolly_smoke.yaml"
  DEFAULT_TRAIN_CONFIG="configs/train/instruction/dolly_smoke.yaml"
  DEFAULT_KERNEL_CONFIG="configs/kernel/instruction/dolly_lora_ntk_smoke.yaml"
  DEFAULT_N_VALUES="8"
  DEFAULT_METHODS="random"
  DEFAULT_COMPARE_LIMIT="8"
else
  DEFAULT_DATASET="dolly"
  DEFAULT_DATA_CONFIG="configs/data/instruction/dolly.yaml"
  DEFAULT_TRAIN_CONFIG="configs/train/instruction/dolly.yaml"
  DEFAULT_KERNEL_CONFIG="configs/kernel/instruction/dolly_lora_ntk.yaml"
  DEFAULT_N_VALUES="128 512 2000"
  DEFAULT_METHODS="random"
  DEFAULT_COMPARE_LIMIT="0"
fi

DATASET="${DATASET:-$DEFAULT_DATASET}"
DATA_CONFIG="${DATA_CONFIG:-$DEFAULT_DATA_CONFIG}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$DEFAULT_TRAIN_CONFIG}"
KERNEL_CONFIG="${KERNEL_CONFIG:-$DEFAULT_KERNEL_CONFIG}"
KERNEL_CONFIGS="${KERNEL_CONFIGS:-$KERNEL_CONFIG}"
BASE_MODEL="${BASE_MODEL:-mlx-community/SmolLM2-1.7B-Instruct}"
N_VALUES="${N_VALUES:-$DEFAULT_N_VALUES}"
METHODS="${METHODS:-$DEFAULT_METHODS}"
COMPARE_SPLIT="${COMPARE_SPLIT:-test}"
COMPARE_LIMIT="${COMPARE_LIMIT:-$DEFAULT_COMPARE_LIMIT}"
PREPARE_DATA="${PREPARE_DATA:-1}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-data/${DATASET}}"
REUSE_EXISTING_ADAPTERS="${REUSE_EXISTING_ADAPTERS:-0}"
REQUIRE_EXISTING_ADAPTERS="${REQUIRE_EXISTING_ADAPTERS:-0}"
REUSE_EXISTING_COMPARISONS="${REUSE_EXISTING_COMPARISONS:-0}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
RUN_KERNEL="${RUN_KERNEL:-1}"
SKIP_LORA_TEST="${SKIP_LORA_TEST:-0}"
KERNEL_TRAIN_LIMITS="${KERNEL_TRAIN_LIMITS:-}"
KERNEL_VALID_LIMIT="${KERNEL_VALID_LIMIT:-}"
KERNEL_TEST_LIMIT="${KERNEL_TEST_LIMIT:-}"
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
PAPER_FIGURE_SPLIT="${PAPER_FIGURE_SPLIT:-test}"
PAPER_FIGURE_ADAPTER_CONTAINS="${PAPER_FIGURE_ADAPTER_CONTAINS:-}"
PAPER_FIGURE_INDIVIDUAL="${PAPER_FIGURE_INDIVIDUAL:-0}"
PAPER_FIGURE_STYLE="${PAPER_FIGURE_STYLE:-paper}"

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
read -r -a KERNEL_CONFIG_ARRAY <<< "$KERNEL_CONFIGS"
if [[ "${#KERNEL_CONFIG_ARRAY[@]}" -eq 0 ]]; then
  echo "KERNEL_CONFIGS must contain at least one kernel config." >&2
  exit 2
fi
KERNEL_CONFIG_COUNT="${#KERNEL_CONFIG_ARRAY[@]}"
if [[ -n "$KERNEL_TRAIN_LIMITS" ]]; then
  read -r -a KERNEL_TRAIN_LIMIT_ARRAY <<< "$KERNEL_TRAIN_LIMITS"
else
  KERNEL_TRAIN_LIMIT_ARRAY=("")
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
if [[ "$REQUIRE_EXISTING_ADAPTERS" == "1" ]]; then
  SUBSET_TRAIN_COMMAND_COUNT=0
fi
COMPARISON_COMMAND_COUNT=$((SUBSET_RUN_COUNT - REUSED_COMPARISON_COUNT))
if [[ "$RUN_COMPARISONS" != "1" ]]; then
  COMPARISON_COMMAND_COUNT=0
fi
KERNEL_COMMAND_COUNT=0
if [[ "$RUN_KERNEL" == "1" ]]; then
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a PLANNED_KERNEL_ARRAY <<< "$KERNEL_ADAPTER_RUNS"
    KERNEL_COMMAND_COUNT=$((${#PLANNED_KERNEL_ARRAY[@]} * ${#KERNEL_TRAIN_LIMIT_ARRAY[@]} * KERNEL_CONFIG_COUNT))
  else
    KERNEL_COMMAND_COUNT=$((PLANNED_ADAPTER_COUNT * ${#KERNEL_TRAIN_LIMIT_ARRAY[@]} * KERNEL_CONFIG_COUNT))
  fi
fi
REPORT_COMMAND_COUNT=0
if [[ "$RUN_KERNEL" == "1" ]]; then
  REPORT_COMMAND_COUNT=1
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
- Skip LoRA post-train test: \`${SKIP_LORA_TEST}\`
- Kernel config: \`${KERNEL_CONFIG}\`
- Kernel configs: \`${KERNEL_CONFIGS}\`
- Base model: \`${BASE_MODEL}\`
- Subset sizes: \`${N_VALUES}\`
- Methods: \`${METHODS}\`
- Compare split: \`${COMPARE_SPLIT}\`
- Compare limit: \`${COMPARE_LIMIT}\`
- Run comparisons: \`${RUN_COMPARISONS}\`
- Run prefix: \`${RUN_PREFIX}\`
- Full run: \`${FULL_RUN}\`
- Train full adapter: \`${TRAIN_FULL}\`
- Reuse existing subset adapters: \`${REUSE_EXISTING_ADAPTERS}\`
- Require existing subset adapters: \`${REQUIRE_EXISTING_ADAPTERS}\`
- Reuse existing comparisons: \`${REUSE_EXISTING_COMPARISONS}\`
- Kernel adapter runs: \`all trained adapters unless KERNEL_ADAPTER_RUNS is set\`
- Kernel train limits: \`${KERNEL_TRAIN_LIMITS:-config default}\`
- Kernel valid limit override: \`${KERNEL_VALID_LIMIT:-config default}\`
- Kernel test limit override: \`${KERNEL_TEST_LIMIT:-config default}\`
- Paper figure style: \`${PAPER_FIGURE_STYLE}\`
- Paper figures: \`${REPORT_DIR}\`

This orchestration trains or reuses one full adapter and one subset adapter for
each method/subset-size pair, optionally compares every subset adapter against
the full adapter, runs kernel experiments for the configured adapter set, and
generates paper-oriented kernel figures.
EOF
}

kernel_run_suffix() {
  local train_limit="$1"
  local suffix="kernel"
  if [[ -n "$train_limit" ]]; then
    suffix="${suffix}-n${train_limit}"
  fi
  if [[ -n "$KERNEL_VALID_LIMIT" ]]; then
    suffix="${suffix}-v${KERNEL_VALID_LIMIT}"
  fi
  if [[ -n "$KERNEL_TEST_LIMIT" ]]; then
    suffix="${suffix}-t${KERNEL_TEST_LIMIT}"
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
  local train_limit="${2:-}"
  local kernel_base_config="${3:-$KERNEL_CONFIG}"
  local variant="${4:-}"
  local suffix
  suffix="$(kernel_run_suffix "$train_limit")"
  local config_component=""
  if [[ -n "$variant" ]]; then
    config_component="-${variant}"
  fi
  local config_path="${KERNEL_CONFIG_DIR}/${adapter_run}${config_component}-${suffix}.yaml"
  mkdir -p "$KERNEL_CONFIG_DIR"
  cat > "$config_path" <<EOF
extends: "${ROOT_DIR}/${kernel_base_config}"
base_model: "${BASE_MODEL}"
adapter_path: "${ROOT_DIR}/results/adapters/${adapter_run}"
output_root: "${ROOT_DIR}/${KERNEL_OUTPUT_ROOT}"
EOF
  if [[ -n "$train_limit" ]]; then
    printf 'train_limit: %s\n' "$train_limit" >> "$config_path"
  fi
  if [[ -n "$KERNEL_VALID_LIMIT" ]]; then
    printf 'valid_limit: %s\n' "$KERNEL_VALID_LIMIT" >> "$config_path"
  fi
  if [[ -n "$KERNEL_TEST_LIMIT" ]]; then
    printf 'test_limit: %s\n' "$KERNEL_TEST_LIMIT" >> "$config_path"
  fi
  cat >> "$config_path" <<EOF
notes: "Orchestrated kernel run for ${adapter_run}${train_limit:+ with train_limit=${train_limit}}."
EOF
  echo "$config_path"
}

log "Orchestration run id: ${RUN_ID}"
log "Output directory: ${ORCH_DIR}"
log "Report directory: ${REPORT_DIR}"
log "Smoke run: ${SMOKE_RUN}; debug: ${DEBUG}; plan only: ${PLAN_ONLY}"
log "Dataset: ${DATASET}; n values: ${N_VALUES}; methods: ${METHODS}"
log "Kernel configs: ${KERNEL_CONFIGS}"
log "Full run: ${FULL_RUN}; train full adapter: ${TRAIN_FULL}"
log "Skip LoRA post-train test: ${SKIP_LORA_TEST}"
log "Prepare data: ${PREPARE_DATA}; prepared data dir: ${PREPARED_DATA_DIR}"
log "Run comparisons: ${RUN_COMPARISONS}"
log "Reuse existing subset adapters: ${REUSE_EXISTING_ADAPTERS}; reusable subset adapters found: ${REUSED_SUBSET_COUNT}"
log "Require existing subset adapters: ${REQUIRE_EXISTING_ADAPTERS}"
log "Reuse existing comparisons: ${REUSE_EXISTING_COMPARISONS}; reusable comparisons found: ${REUSED_COMPARISON_COUNT}"
log "Kernel train limits: ${KERNEL_TRAIN_LIMITS:-config default}; valid override: ${KERNEL_VALID_LIMIT:-config default}; test override: ${KERNEL_TEST_LIMIT:-config default}"
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
  FULL_CMD=(
    uv run mlx-lora-run
    --config "$TRAIN_CONFIG" \
    --run-name "$FULL_RUN" \
    --base-model "$BASE_MODEL"
  )
  if [[ "$SKIP_LORA_TEST" == "1" ]]; then
    FULL_CMD+=(--skip-test)
  fi
  run_cmd "${FULL_CMD[@]}"
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
    )
    if [[ "$DEBUG" == "1" ]]; then
      CMD+=(--selector-debug)
    fi
    if [[ "$SKIP_LORA_TEST" == "1" ]]; then
      CMD+=(--skip-test)
    fi
    if [[ "$REUSE_EXISTING_ADAPTERS" == "1" ]] && adapter_run_complete "$subset_run"; then
      log "Skipping subset adapter training; using existing completed run ${subset_run}"
    elif [[ "$REQUIRE_EXISTING_ADAPTERS" == "1" ]]; then
      log "Requiring existing completed subset adapter run ${subset_run}"
      if [[ "$PLAN_ONLY" != "1" ]]; then
        echo "Missing completed subset adapter run: ${subset_run}" >&2
        exit 2
      fi
    else
      run_cmd "${CMD[@]}"
    fi
    TRAINED_ADAPTER_RUNS+=("$subset_run")

    if [[ "$RUN_COMPARISONS" != "1" ]]; then
      log "Skipping adapter comparison; RUN_COMPARISONS=${RUN_COMPARISONS}"
    else
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
    for kernel_base_config in "${KERNEL_CONFIG_ARRAY[@]}"; do
      kernel_variant=""
      if [[ "$KERNEL_CONFIG_COUNT" -gt 1 ]]; then
        kernel_variant="$(kernel_config_variant "$kernel_base_config")"
      fi
      for train_limit in "${KERNEL_TRAIN_LIMIT_ARRAY[@]}"; do
        KERNEL_INDEX=$((KERNEL_INDEX + 1))
        kernel_suffix="$(kernel_run_suffix "$train_limit")"
        if [[ -n "$kernel_variant" ]]; then
          kernel_run_name="${adapter_run}-${kernel_variant}-${kernel_suffix}"
        else
          kernel_run_name="${adapter_run}-${kernel_suffix}"
        fi
        log "Kernel ${KERNEL_INDEX}/${KERNEL_COMMAND_COUNT} for adapter ${adapter_run}, config=${kernel_base_config}, train_limit=${train_limit:-config default}"
        kernel_config="$(write_kernel_config "$adapter_run" "$train_limit" "$kernel_base_config" "$kernel_variant")"
        run_cmd uv run mlx-lora-run-kernel \
          --config "$kernel_config" \
          --run-name "$kernel_run_name" \
          --base-model "$BASE_MODEL"
      done
    done
  done
else
  section "Skip kernel prediction experiments"
  log "RUN_KERNEL=${RUN_KERNEL}"
fi

section "Generate paper figures"
if [[ "$RUN_KERNEL" == "1" ]]; then
  PAPER_CMD=(
    uv run mlx-lora-make-kernel-paper-figures
    --experiment "${DATASET}=${KERNEL_OUTPUT_ROOT}"
    --output-dir "${REPORT_DIR}/assets/kernel_paper"
    --output "${REPORT_DIR}/kernel_paper_figures.md"
    --split "$PAPER_FIGURE_SPLIT"
    --plot-style "$PAPER_FIGURE_STYLE"
  )
  if [[ -n "$PAPER_FIGURE_ADAPTER_CONTAINS" ]]; then
    PAPER_CMD+=(--adapter-contains "$PAPER_FIGURE_ADAPTER_CONTAINS")
  fi
  if [[ "$PAPER_FIGURE_INDIVIDUAL" == "1" ]]; then
    PAPER_CMD+=(--individual)
  fi
  run_cmd "${PAPER_CMD[@]}"
else
  log "Skipping paper figures; RUN_KERNEL=${RUN_KERNEL}"
fi

section "Done"
log "Manifest: ${ORCH_DIR}/manifest.md"
log "Paper figures: ${REPORT_DIR}"
