#!/usr/bin/env bash

initialize_orchestration_context() {
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

  count_reusable_outputs

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
}

count_reusable_outputs() {
  REUSED_SUBSET_COUNT=0
  REUSED_COMPARISON_COUNT=0
  if [[ "$REUSE_EXISTING_SUBSET_ADAPTERS" == "1" ]]; then
    for subset_size in "${SUBSET_SIZE_ARRAY[@]}"; do
      for suffix in "${SUBSET_METHOD_ARRAY[@]}"; do
        subset_run="$(subset_run_name "$suffix" "$subset_size")"
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
        subset_run="$(subset_run_name "$suffix" "$subset_size")"
        if comparison_complete "$subset_run" "$FULL_ADAPTER_RUN"; then
          REUSED_COMPARISON_COUNT=$((REUSED_COMPARISON_COUNT + 1))
        fi
      done
    done
  fi
}

print_orchestration_plan() {
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
}

finish_orchestration() {
  section "Done"
  log "Manifest: ${ORCHESTRATION_DIR}/manifest.md"
  log "Kernel figures: ${REPORT_OUTPUT_DIR}"
}
