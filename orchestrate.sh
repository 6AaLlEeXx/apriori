#!/usr/bin/env bash
set -euo pipefail

# End-to-end empirical validation orchestrator. The workflow is split into
# scripts/orchestrate so this entrypoint only wires the semantic steps together.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"

ORCHESTRATE_SCRIPT_DIR="${ROOT_DIR}/scripts/orchestrate"

source "${ORCHESTRATE_SCRIPT_DIR}/helpers.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/context.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/manifest.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/steps/prepare_data.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/steps/train_full_adapter.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/steps/train_subsets_and_compare.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/steps/run_kernel_experiments.sh"
source "${ORCHESTRATE_SCRIPT_DIR}/steps/generate_kernel_figures.sh"

initialize_orchestration_context
print_orchestration_plan
write_manifest
run_prepare_data_step
run_full_adapter_step
run_subset_adapters_and_comparisons_step
run_kernel_experiments_step
run_kernel_figures_step
finish_orchestration
