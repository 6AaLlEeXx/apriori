from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from lora.kernel.report import make_kernel_report
from lora.paths import DEFAULT_KERNEL_RESULTS_ROOT, DEFAULT_REPORTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate tracked kernel runs into a markdown report."
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_KERNEL_RESULTS_ROOT,
        help="Kernel run output root. Defaults to results/kernel.",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_REPORTS_ROOT}/lora_kernel_runs.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_kernel_report(
        output_root=args.output_root,
        output_path=args.output,
    )
    print(f"Wrote kernel report to {output_path}")


if __name__ == "__main__":
    main()
