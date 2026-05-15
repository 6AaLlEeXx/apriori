#!/usr/bin/env bash

run_kernel_experiments_step() {
  if [[ "$RUN_KERNEL_EXPERIMENTS" != "1" ]]; then
    section "Skip kernel prediction experiments"
    log "RUN_KERNEL_EXPERIMENTS=${RUN_KERNEL_EXPERIMENTS}"
    return
  fi

  section "Run kernel prediction experiments"
  local kernel_run_array
  if [[ -n "${KERNEL_ADAPTER_RUNS:-}" ]]; then
    read -r -a kernel_run_array <<< "$KERNEL_ADAPTER_RUNS"
  else
    kernel_run_array=("${TRAINED_ADAPTER_RUNS[@]}")
  fi

  local kernel_index=0
  local adapter_run
  local kernel_base_config
  local krr_fit_size
  for adapter_run in "${kernel_run_array[@]}"; do
    for kernel_base_config in "${KERNEL_EXPERIMENT_CONFIG_ARRAY[@]}"; do
      local kernel_variant=""
      if [[ "$KERNEL_EXPERIMENT_CONFIG_COUNT" -gt 1 ]]; then
        kernel_variant="$(kernel_config_variant "$kernel_base_config")"
      fi
      for krr_fit_size in "${KERNEL_FIT_SIZE_ARRAY[@]}"; do
        kernel_index=$((kernel_index + 1))
        local kernel_suffix
        local kernel_run_name
        local kernel_config
        kernel_suffix="$(kernel_run_suffix "$krr_fit_size")"
        if [[ -n "$kernel_variant" ]]; then
          kernel_run_name="${adapter_run}-${kernel_variant}-${kernel_suffix}"
        else
          kernel_run_name="${adapter_run}-${kernel_suffix}"
        fi
        log "Kernel ${kernel_index}/${KERNEL_COMMAND_COUNT} for adapter ${adapter_run}, config=${kernel_base_config}, krr_fit_examples=${krr_fit_size:-config default}"
        kernel_config="$(write_kernel_config "$adapter_run" "$krr_fit_size" "$kernel_base_config" "$kernel_variant")"
        run_cmd uv run mlx-lora-run-kernel \
          --config "$kernel_config" \
          --run-name "$kernel_run_name" \
          --base-model "$BASE_MODEL"
      done
    done
  done
}
