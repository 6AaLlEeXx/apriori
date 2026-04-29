from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from kernel.adapter_compare import make_adapter_comparison_report
from paths import DEFAULT_REPORTS_ROOT, DEFAULT_RESULTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate full-vs-subset adapter comparisons into a report."
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Results root containing comparisons and runs.",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_REPORTS_ROOT}/adapter_comparisons.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method labels to include.",
    )
    parser.add_argument(
        "--base-models",
        nargs="*",
        default=None,
        help="Optional base models to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_adapter_comparison_report(
        output_root=args.output_root,
        output_path=args.output,
        datasets=list(args.datasets) if args.datasets else None,
        methods=list(args.methods) if args.methods else None,
        base_models=list(args.base_models) if args.base_models else None,
    )
    print(f"Wrote adapter comparison report to {output_path}")


if __name__ == "__main__":
    main()
