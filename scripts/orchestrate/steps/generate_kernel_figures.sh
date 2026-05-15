#!/usr/bin/env bash

run_kernel_figures_step() {
  section "Generate kernel figures"
  if [[ "$RUN_KERNEL_EXPERIMENTS" != "1" ]]; then
    log "Skipping kernel figures; RUN_KERNEL_EXPERIMENTS=${RUN_KERNEL_EXPERIMENTS}"
    return
  fi

  local kernel_figure_cmd=(
    uv run mlx-lora-make-kernel-figures
    --kernel-results "${DATASET_NAME}=${KERNEL_RESULTS_ROOT}"
    --output-dir "${REPORT_OUTPUT_DIR}/assets/kernel_figures"
    --output "${REPORT_OUTPUT_DIR}/kernel_figures.md"
    --split "$KERNEL_FIGURE_SPLIT"
    --plot-style "$KERNEL_FIGURE_STYLE"
  )
  if [[ -n "$KERNEL_FIGURE_ADAPTER_NAME_CONTAINS" ]]; then
    kernel_figure_cmd+=(--adapter-name-contains "$KERNEL_FIGURE_ADAPTER_NAME_CONTAINS")
  fi
  if [[ "$KERNEL_FIGURE_INDIVIDUAL_PDFS" == "1" ]]; then
    kernel_figure_cmd+=(--individual-pdfs)
  fi
  run_cmd "${kernel_figure_cmd[@]}"
}
