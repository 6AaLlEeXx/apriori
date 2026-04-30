from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from mlops import (
    build_mlx_config_payload,
    build_run_metadata,
    build_run_name,
    build_summary,
    build_test_command,
    build_train_command,
    load_lora_run_config,
    prepare_sampled_data_dir,
    prepare_run,
    save_lora_run_config,
    run_command_with_logging,
    write_command,
    write_metadata,
    write_mlx_runtime_config,
    write_summary,
)
from paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tracked MLX-LM LoRA fine-tuning job."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the LoRA run config YAML/JSON file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name. Defaults to a timestamped generated name.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Optional model id or local model path. Overrides `base_model` "
            "from the config for this run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the run directory and print the command without executing it.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the post-train `mlx_lm.lora --test` pass.",
    )
    parser.add_argument(
        "--sample-selector",
        default=None,
        help=(
            "Optional path to a Python selector module defining "
            "`select_samples(rows, max_example=None, context=None)`."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional max-examples value passed to `select_samples`.",
    )
    parser.add_argument(
        "--selector-transformation",
        action="append",
        default=None,
        help=(
            "Feature transformation applied before selector clustering. "
            "Can be passed multiple times. Defaults to identity."
        ),
    )
    parser.add_argument(
        "--selector-projection",
        default="identity",
        help=(
            "Feature projection applied before selector clustering. "
            "Use sparse_random for SparseRandomProjection. Defaults to identity."
        ),
    )
    parser.add_argument(
        "--selector-projection-components",
        type=int,
        default=None,
        help="Output dimension for selector projections that need one.",
    )
    parser.add_argument(
        "--selector-projection-chunk-size",
        type=int,
        default=16,
        help=(
            "Rows per chunk for memory-conscious selector projections. "
            "Defaults to 16."
        ),
    )
    parser.add_argument(
        "--selector-max-kmeans-feature-gb",
        type=float,
        default=4.0,
        help=(
            "Maximum unprojected k-means feature matrix size in GiB before "
            "failing with a projection hint. Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--selector-debug",
        action="store_true",
        help="Print selector feature/cache progress while preparing sampled data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_lora_run_config(args.config)
    if args.base_model is not None:
        config = replace(config, base_model=args.base_model)
    run_name = args.run_name or build_run_name(config)
    paths = prepare_run(config, run_name)

    save_lora_run_config(config, paths.resolved_config_path)
    metadata = build_run_metadata(
        config=config,
        paths=paths,
        config_source=args.config,
        status="prepared",
    )
    write_metadata(paths.metadata_path, metadata)
    write_mlx_runtime_config(
        paths.mlx_config_path,
        build_mlx_config_payload(config),
    )

    train_data_dir = config.data_dir
    if args.sample_selector:
        train_data_dir, sampling_meta = prepare_sampled_data_dir(
            source_data_dir=config.data_dir,
            run_dir=paths.run_dir,
            sample_selector=args.sample_selector,
            max_example=args.max_examples,
            selector_context={
                "base_model": config.base_model,
                "mlx_args": config.mlx_args,
                "seed": config.mlx_args.get("seed", 42),
                "selector_transformations": (
                    args.selector_transformation or ["identity"]
                ),
                "selector_projection": args.selector_projection,
                "selector_projection_components": (
                    args.selector_projection_components
                ),
                "selector_projection_chunk_size": (
                    args.selector_projection_chunk_size
                ),
                "selector_max_kmeans_feature_bytes": int(
                    args.selector_max_kmeans_feature_gb * 1024**3
                ),
                "shared_feature_cache_root": str(
                    resolve_project_path(config.output_root)
                    / "selector_feature_cache"
                ),
                "selector_debug": args.selector_debug,
            },
        )
        if args.selector_debug:
            print(
                "[selector] selected "
                f"{sampling_meta['selected_train_examples']}/"
                f"{sampling_meta['original_train_examples']} training rows; "
                f"sampled_data_dir={sampling_meta['sampled_data_dir']}",
                flush=True,
            )
        metadata["sampling"] = sampling_meta
    metadata["train_data_dir"] = str(train_data_dir)
    write_metadata(paths.metadata_path, metadata)

    train_command = build_train_command(config, paths, data_dir=train_data_dir)
    write_command(paths.command_path, train_command)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status="prepared",
        ),
    )

    print(f"Run directory: {paths.run_dir}")
    print(f"Adapter directory: {paths.adapter_dir}")
    print(f"Command: {paths.command_path}")

    if args.dry_run:
        return

    metadata["status"] = "running"
    write_metadata(paths.metadata_path, metadata)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status="running",
        ),
    )

    train_exit_code = run_command_with_logging(
        command=train_command,
        log_path=paths.train_log,
        metrics_path=paths.metrics_path,
        source_phase="train",
    )

    status = "completed" if train_exit_code == 0 else "failed"
    test_exit_code: int | None = None

    if train_exit_code == 0 and config.test_after_train and not args.skip_test:
        test_command = build_test_command(config, paths, data_dir=train_data_dir)
        test_exit_code = run_command_with_logging(
            command=test_command,
            log_path=paths.test_log,
            metrics_path=paths.metrics_path,
            source_phase="test",
        )
        if test_exit_code != 0:
            status = "failed"

    metadata["status"] = status
    metadata["train_exit_code"] = train_exit_code
    metadata["test_exit_code"] = test_exit_code
    write_metadata(paths.metadata_path, metadata)
    write_summary(
        paths.summary_path,
        build_summary(
            config=config,
            paths=paths,
            metadata=metadata,
            status=status,
            train_exit_code=train_exit_code,
            test_exit_code=test_exit_code,
        ),
    )

    if train_exit_code != 0:
        raise SystemExit(train_exit_code)
    if test_exit_code not in {None, 0}:
        raise SystemExit(test_exit_code)


if __name__ == "__main__":
    main()
