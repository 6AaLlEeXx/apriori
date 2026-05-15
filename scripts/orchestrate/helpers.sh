#!/usr/bin/env bash

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
  fi
  log "[$COMMAND_INDEX/$TOTAL_COMMANDS] PLAN only"
}

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

subset_run_name() {
  local suffix="$1"
  local subset_size="$2"
  echo "${ADAPTER_RUN_PREFIX}-${suffix}-subset${subset_size}"
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
