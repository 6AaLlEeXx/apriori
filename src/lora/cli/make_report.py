from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from lora.mlops import collect_run_summaries, render_markdown_report
from lora.paths import DEFAULT_REPORTS_ROOT, DEFAULT_RESULTS_ROOT, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a markdown report for tracked LoRA runs."
    )
    parser.add_argument(
        "--input-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory that contains runs and adapters.",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_REPORTS_ROOT}/lora_runs.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = collect_run_summaries(args.input_root)
    report = render_markdown_report(summaries)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Wrote LoRA report to {output_path}")


if __name__ == "__main__":
    main()
