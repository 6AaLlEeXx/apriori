#!/usr/bin/env bash

run_subset_adapters_and_comparisons_step() {
  section "Train subset adapters and compare against full adapter"
  local subset_index=0
  local subset_size
  local suffix

  for subset_size in "${SUBSET_SIZE_ARRAY[@]}"; do
    for suffix in "${SUBSET_METHOD_ARRAY[@]}"; do
      subset_index=$((subset_index + 1))
      local method
      local selector
      local subset_run
      method="$(method_label "$suffix")"
      selector="$(selector_path "$suffix")"
      subset_run="$(subset_run_name "$suffix" "$subset_size")"

      section "Subset ${subset_index}/${SUBSET_RUN_COUNT}: ${method}, subset size=${subset_size}"
      local train_cmd=(
        uv run mlx-lora-run
        --config "$ADAPTER_TRAIN_CONFIG"
        --run-name "$subset_run"
        --base-model "$BASE_MODEL"
        --sample-selector "$selector"
        --subset-train-size "$subset_size"
      )
      if [[ "$SELECTOR_DEBUG" == "1" ]]; then
        train_cmd+=(--selector-debug)
      fi
      if [[ "$SKIP_LORA_TEST" == "1" ]]; then
        train_cmd+=(--skip-test)
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
        run_cmd "${train_cmd[@]}"
      fi
      TRAINED_ADAPTER_RUNS+=("$subset_run")

      run_adapter_comparison_step "$subset_run" "$method"
    done
  done
}

run_adapter_comparison_step() {
  local subset_run="$1"
  local method="$2"

  if [[ "$RUN_ADAPTER_COMPARISONS" != "1" ]]; then
    log "Skipping adapter comparison; RUN_ADAPTER_COMPARISONS=${RUN_ADAPTER_COMPARISONS}"
    return
  fi

  local comparison_dir="${ORCHESTRATION_DIR}/comparisons/${subset_run}__vs__${FULL_ADAPTER_RUN}"
  if [[ "$REUSE_EXISTING_ADAPTER_COMPARISONS" == "1" ]] && comparison_complete "$subset_run" "$FULL_ADAPTER_RUN"; then
    log "Skipping adapter comparison; using existing ${comparison_dir}"
    return
  fi

  run_cmd uv run mlx-lora-compare-adapters \
    --config "$ADAPTER_TRAIN_CONFIG" \
    --base-model "$BASE_MODEL" \
    --full-adapter "results/adapters/${FULL_ADAPTER_RUN}" \
    --subset-adapter "results/adapters/${subset_run}" \
    --split "$ADAPTER_COMPARISON_SPLIT" \
    --example-limit "$ADAPTER_COMPARISON_EXAMPLE_LIMIT" \
    --subset-method-label "$method" \
    --output-dir "$comparison_dir"
}
