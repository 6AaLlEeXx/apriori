from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from kernel.report import collect_kernel_run_summaries
from reporting import generate_kernel_paper_plots, markdown_plot_section


def _parse_experiment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Experiments must be formatted as LABEL=KERNEL_OUTPUT_ROOT."
        )
    label, output_root = value.split("=", 1)
    label = label.strip()
    output_root = output_root.strip()
    if not label or not output_root:
        raise argparse.ArgumentTypeError(
            "Experiments must include both a label and a kernel output root."
        )
    return label, output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready kernel prediction and RMSE-gain figures."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=_parse_experiment,
        required=True,
        help=(
            "Experiment label and kernel output root, formatted as "
            "LABEL=results/orchestrations/<run>/kernel. Repeat for each panel."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/paper/kernel",
        help="Directory for generated SVG figures.",
    )
    parser.add_argument(
        "--output",
        default="reports/paper/kernel_figures.md",
        help="Markdown file embedding the generated figures.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[16, 256, 512],
        help="Kernel fit-set sizes to include.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["raw", "thresholded_sign"],
        help="Feature transforms to include, e.g. raw thresholded_sign.",
    )
    parser.add_argument(
        "--adapter-contains",
        default=None,
        help="Only include kernel runs whose run name or adapter path contains this text.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Prediction split for predicted-vs-true panels.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write one vector graphic per panel instead of combined multi-panel figures.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write matching PDF files next to the generated SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [
        (label, collect_kernel_run_summaries(output_root))
        for label, output_root in args.experiment
    ]
    output_dir = Path(args.output_dir)
    output_path = Path(args.output)
    artifacts = generate_kernel_paper_plots(
        experiments,
        output_dir,
        train_sizes=list(args.train_sizes),
        features=list(args.features),
        adapter_contains=args.adapter_contains,
        split=args.split,
        individual=args.individual,
        pdf=args.pdf,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Kernel Paper Figures\n"
        + markdown_plot_section(
            artifacts,
            report_path=output_path,
            heading="Figures",
        )
    )
    print(f"Wrote paper kernel figures to {output_path}")


if __name__ == "__main__":
    main()
