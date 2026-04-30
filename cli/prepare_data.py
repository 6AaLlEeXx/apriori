from __future__ import annotations

import argparse

from data_prep import load_data_prep_config, prepare_dataset_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare prompt/completion JSONL splits from a data YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a data prep config, for example configs/data/dolly.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config output_dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config split seed.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Override filters.token_supervision.tokenizer_model when enabled.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Alias for --tokenizer-model when token supervision is enabled; "
            "ignored otherwise for consistency with training CLIs."
        ),
    )
    return parser.parse_args()


def token_supervision_enabled(config: dict) -> bool:
    filters = config.get("filters")
    if not isinstance(filters, dict):
        return False
    token_filter = filters.get("token_supervision")
    return isinstance(token_filter, dict) and bool(token_filter.get("enabled"))


def main() -> None:
    args = parse_args()
    config = load_data_prep_config(args.config)
    tokenizer_model = args.tokenizer_model
    if args.base_model is not None and token_supervision_enabled(config):
        tokenizer_model = args.base_model
    counts = prepare_dataset_from_config(
        args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        tokenizer_model=tokenizer_model,
    )
    dataset_name = config.get("dataset_name", "dataset")
    output_dir = args.output_dir or config.get("output_dir") or f"data/{dataset_name}"
    print(f"Wrote {dataset_name} splits to {output_dir}: {counts}")


if __name__ == "__main__":
    main()
