#!/usr/bin/env bash

run_full_adapter_step() {
  TRAINED_ADAPTER_RUNS=("$FULL_ADAPTER_RUN")

  if [[ "$TRAIN_FULL_ADAPTER" == "1" ]]; then
    section "Train full-data adapter"
    local full_cmd=(
      uv run mlx-lora-run
      --config "$ADAPTER_TRAIN_CONFIG"
      --run-name "$FULL_ADAPTER_RUN"
      --base-model "$BASE_MODEL"
    )
    if [[ "$SKIP_LORA_TEST" == "1" ]]; then
      full_cmd+=(--skip-test)
    fi
    run_cmd "${full_cmd[@]}"
    return
  fi

  section "Reuse full-data adapter"
  log "Skipping full adapter training; using ${FULL_ADAPTER_RUN}"
  if [[ "$PLAN_ONLY" != "1" ]] && ! adapter_run_complete "$FULL_ADAPTER_RUN"; then
    echo "Missing completed full adapter run: ${FULL_ADAPTER_RUN}" >&2
    exit 2
  fi
}
