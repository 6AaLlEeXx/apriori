from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from eval import evaluate_predictions, load_predictions
from mlops import load_lora_run_config
from paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction JSONL files for a tracked LoRA run."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a JSONL file with `prediction` and `reference` fields.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training config whose evaluation block should be used.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Metric to compute. Can be passed multiple times.",
    )
    parser.add_argument(
        "--primary-metric",
        default=None,
        help="Metric to report as metric_name/metric_value.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. If set, output defaults to <run-dir>/eval.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for the evaluation JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_predictions(args.predictions)
    evaluation = {}
    if args.config is not None:
        evaluation = load_lora_run_config(args.config).evaluation
    if args.metrics is not None:
        evaluation = dict(evaluation)
        evaluation["metrics"] = args.metrics
    if args.primary_metric is not None:
        evaluation = dict(evaluation)
        evaluation["primary_metric"] = args.primary_metric
    payload = evaluate_predictions(records, evaluation or None)

    if args.output is not None:
        output_path = resolve_project_path(args.output)
    elif args.run_dir is not None:
        output_path = resolve_project_path(args.run_dir) / "eval.json"
    else:
        output_path = resolve_project_path(args.predictions).with_suffix(".eval.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote evaluation to {output_path}")


if __name__ == "__main__":
    main()
