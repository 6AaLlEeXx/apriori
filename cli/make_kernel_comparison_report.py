from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from kernel.report import make_kernel_comparison_report
from paths import DEFAULT_KERNEL_RESULTS_ROOT, DEFAULT_REPORTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison table for the primary kernel runs per dataset/backend."
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_KERNEL_RESULTS_ROOT,
        help="Kernel run output root. Defaults to results/kernel.",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_REPORTS_ROOT}/lora_kernel_comparison.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional datasets to include. Defaults to discovered completed runs.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Optional backends to include. Defaults to discovered completed runs.",
    )
    parser.add_argument(
        "--base-models",
        nargs="*",
        default=None,
        help="Optional base models to include. Defaults to discovered completed runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_kernel_comparison_report(
        output_root=args.output_root,
        output_path=args.output,
        datasets=list(args.datasets) if args.datasets else None,
        backends=list(args.backends) if args.backends else None,
        base_models=list(args.base_models) if args.base_models else None,
    )
    print(f"Wrote kernel comparison report to {output_path}")


if __name__ == "__main__":
    main()
