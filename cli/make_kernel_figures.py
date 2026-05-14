from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from kernel.report import collect_kernel_run_summaries
from reporting import generate_kernel_plots, markdown_plot_section


def _parse_kernel_results(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Kernel results must be formatted as LABEL=KERNEL_RESULTS_ROOT."
        )
    label, kernel_results_root = value.split("=", 1)
    label = label.strip()
    kernel_results_root = kernel_results_root.strip()
    if not label or not kernel_results_root:
        raise argparse.ArgumentTypeError(
            "Kernel results must include both a label and a kernel results root."
        )
    return label, kernel_results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready kernel prediction and RMSE-gain figures."
    )
    parser.add_argument(
        "--kernel-results",
        action="append",
        type=_parse_kernel_results,
        required=True,
        help=(
            "Experiment label and kernel results root, formatted as "
            "LABEL=results/orchestrations/<run>/kernel. Repeat for each panel."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/paper/kernel",
        help="Directory for generated PDF figures.",
    )
    parser.add_argument(
        "--output",
        default="reports/paper/kernel_figures.md",
        help="Markdown file linking to the generated PDF figures.",
    )
    parser.add_argument(
        "--kernel-fit-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Kernel fit-set sizes to include. Defaults to all completed runs.",
    )
    parser.add_argument(
        "--feature-transforms",
        nargs="+",
        default=None,
        help="Feature transforms to include. Defaults to all completed runs.",
    )
    parser.add_argument(
        "--adapter-name-contains",
        default=None,
        help="Only include kernel runs whose run name or adapter path contains this text.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Prediction split for predicted-vs-true panels.",
    )
    parser.add_argument(
        "--individual-pdfs",
        action="store_true",
        help="Write one PDF per panel instead of a combined multi-page PDF.",
    )
    parser.add_argument(
        "--plot-style",
        choices=["paper", "standard"],
        default="paper",
        help=(
            "Plot renderer to use. 'paper' preserves the exact paper-compatible "
            "layout; 'standard' uses conventional matplotlib PDF figures."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [
        (label, collect_kernel_run_summaries(kernel_results_root))
        for label, kernel_results_root in args.kernel_results
    ]
    output_dir = Path(args.output_dir)
    output_path = Path(args.output)
    artifacts = generate_kernel_plots(
        experiments,
        output_dir,
        train_sizes=(
            list(args.kernel_fit_sizes) if args.kernel_fit_sizes else None
        ),
        features=(
            list(args.feature_transforms) if args.feature_transforms else None
        ),
        adapter_contains=args.adapter_name_contains,
        split=args.split,
        individual=args.individual_pdfs,
        plot_style=args.plot_style,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# Kernel {args.plot_style.title()} Figures\n"
        + markdown_plot_section(
            artifacts,
            report_path=output_path,
            heading="Figures",
        )
    )
    print(f"Wrote {args.plot_style} kernel figures to {output_path}")


if __name__ == "__main__":
    main()
