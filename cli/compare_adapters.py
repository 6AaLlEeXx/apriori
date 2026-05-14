from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from kernel.adapter_compare import run_adapter_comparison
from mlops import load_lora_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a full-data adapter against a subset-trained adapter on "
            "the same prepared split."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the LoRA training config that defines prepared_data_dir/base_model.",
    )
    parser.add_argument(
        "--full-adapter",
        required=True,
        help="Adapter trained on the full training set.",
    )
    parser.add_argument(
        "--subset-adapter",
        required=True,
        help="Adapter trained on the selected subset.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Prepared data split to score. Defaults to test.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=512,
        help="Maximum examples to score from the selected split. Use 0 for all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when subsampling the split.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to results/comparisons/<run>.",
    )
    parser.add_argument(
        "--subset-method-label",
        default=None,
        help="Optional subset selection method label for reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_lora_run_config(args.config)
    if args.base_model is not None:
        config = replace(config, base_model=args.base_model)

    paths, summary = run_adapter_comparison(
        base_model=config.base_model,
        prepared_data_dir=config.prepared_data_dir,
        full_adapter_path=args.full_adapter,
        subset_adapter_path=args.subset_adapter,
        dataset_name=config.dataset_name,
        task=config.task,
        subset_method_label=args.subset_method_label,
        split=args.split,
        example_limit=args.example_limit,
        seed=args.seed,
        output_dir=args.output_dir,
        results_root=config.results_root,
    )
    print(f"Comparison directory: {paths.output_dir}")
    print(f"Scores: {paths.scores_path}")
    print(f"Summary: {paths.summary_path}")
    print(
        "Delta Pearson: "
        f"{summary['metrics']['delta']['pearson']:.4f}; "
        "Sign accuracy: "
        f"{summary['metrics']['delta']['sign_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
