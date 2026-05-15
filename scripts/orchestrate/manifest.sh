#!/usr/bin/env bash

write_manifest() {
  mkdir -p "$ORCHESTRATION_DIR" "$REPORT_OUTPUT_DIR"
  cat > "${ORCHESTRATION_DIR}/manifest.md" <<EOF
# Orchestration ${RUN_ID}

- Dataset: \`${DATASET_NAME}\`
- Smoke run: \`${SMOKE_RUN}\`
- Selector debug: \`${SELECTOR_DEBUG}\`
- Data prep config: \`${DATA_PREP_CONFIG}\`
- Prepare data: \`${PREPARE_DATA}\`
- Prepared data dir: \`${PREPARED_DATA_DIR}\`
- Adapter train config: \`${ADAPTER_TRAIN_CONFIG}\`
- Skip LoRA post-train test: \`${SKIP_LORA_TEST}\`
- Kernel experiment config: \`${KERNEL_EXPERIMENT_CONFIG}\`
- Kernel experiment configs: \`${KERNEL_EXPERIMENT_CONFIGS}\`
- Base model: \`${BASE_MODEL}\`
- Subset sizes: \`${SUBSET_TRAIN_SIZES}\`
- Subset selection methods: \`${SUBSET_SELECTION_METHODS}\`
- Adapter comparison split: \`${ADAPTER_COMPARISON_SPLIT}\`
- Adapter comparison example limit: \`${ADAPTER_COMPARISON_EXAMPLE_LIMIT}\`
- Run adapter comparisons: \`${RUN_ADAPTER_COMPARISONS}\`
- Adapter run prefix: \`${ADAPTER_RUN_PREFIX}\`
- Full adapter run: \`${FULL_ADAPTER_RUN}\`
- Train full adapter: \`${TRAIN_FULL_ADAPTER}\`
- Reuse existing subset adapters: \`${REUSE_EXISTING_SUBSET_ADAPTERS}\`
- Require existing subset adapters: \`${REQUIRE_EXISTING_SUBSET_ADAPTERS}\`
- Reuse existing comparisons: \`${REUSE_EXISTING_ADAPTER_COMPARISONS}\`
- Kernel adapter runs: \`all trained adapters unless KERNEL_ADAPTER_RUNS is set\`
- Kernel fit sizes: \`${KERNEL_FIT_SIZES:-config default}\`
- Kernel validation examples override: \`${KERNEL_VALID_EXAMPLES:-config default}\`
- Kernel test examples override: \`${KERNEL_TEST_EXAMPLES:-config default}\`
- Kernel figure style: \`${KERNEL_FIGURE_STYLE}\`
- Kernel figures: \`${REPORT_OUTPUT_DIR}\`

This orchestration trains or reuses one full adapter and one subset adapter for each method/subset-size pair, optionally compares every subset adapter against the full adapter, runs kernel experiments for the configured adapter set, and generates paper-oriented kernel figures.
EOF
}
