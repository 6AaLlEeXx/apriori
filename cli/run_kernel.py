from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from kernel.config import build_kernel_run_name, load_kernel_run_config
from kernel.run import run_kernel_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tracked kernel-score experiment for a LoRA adapter."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the kernel run config YAML/JSON file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name. Defaults to a generated timestamped name.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Optional model id or local model path. Overrides `base_model` "
            "from the config for this kernel run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the config and print the generated run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kernel_run_config(args.config)
    if args.base_model is not None:
        config = replace(config, base_model=args.base_model)
    run_name = args.run_name or build_kernel_run_name(config)
    if args.dry_run:
        print(f"Kernel run name: {run_name}")
        return
    paths = run_kernel_experiment(
        config=config,
        config_source=args.config,
        run_name=run_name,
    )
    print(f"Kernel run directory: {paths.run_dir}")
    print(f"Kernel summary: {paths.summary_path}")
    print(f"Kernel report: {paths.report_path}")


if __name__ == "__main__":
    main()
