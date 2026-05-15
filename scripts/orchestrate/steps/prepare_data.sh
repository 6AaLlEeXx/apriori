#!/usr/bin/env bash

run_prepare_data_step() {
  if [[ "$PREPARE_DATA" == "1" ]]; then
    section "Prepare data"
    run_cmd uv run mlx-lora-prepare-data \
      --config "$DATA_PREP_CONFIG" \
      --base-model "$BASE_MODEL"
    return
  fi

  section "Reuse prepared data"
  log "Skipping data prep; using ${PREPARED_DATA_DIR}"
  if [[ "$PLAN_ONLY" != "1" ]] && ! prepared_data_exists "$PREPARED_DATA_DIR"; then
    echo "Missing prepared data files under: ${PREPARED_DATA_DIR}" >&2
    exit 2
  fi
}
