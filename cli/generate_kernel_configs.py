from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from kernel.generate_configs import (
    DEFAULT_GENERATED_KERNEL_CONFIG_DIR,
    DEFAULT_KERNEL_BACKENDS,
    plan_generated_kernel_configs,
    write_generated_kernel_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate kernel score-delta configs from data/training configs."
    )
    parser.add_argument(
        "--data-config",
        action="append",
        default=[],
        help="Data prep config to convert. May be passed more than once.",
    )
    parser.add_argument(
        "--all-data-configs",
        action="store_true",
        help="Generate configs for every YAML file under configs/data.",
    )
    parser.add_argument(
        "--training-config",
        action="append",
        default=[],
        help="Optional LoRA training config used when its dataset matches.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the generated base_model for every generated config.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_KERNEL_BACKENDS,
        help="Kernel backends to generate. Defaults to lora_ntk.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_GENERATED_KERNEL_CONFIG_DIR,
        help="Directory for generated kernel configs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated configs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned config paths without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planned = plan_generated_kernel_configs(
        data_configs=args.data_config,
        all_data_configs=args.all_data_configs,
        training_configs=args.training_config,
        base_model=args.base_model,
        backends=args.backends,
        output_dir=args.output_dir,
    )
    paths = write_generated_kernel_configs(
        planned,
        force=args.force,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    for path in paths:
        print(f"{action} kernel config: {path}")


if __name__ == "__main__":
    main()
