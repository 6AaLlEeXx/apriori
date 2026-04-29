from __future__ import annotations

from pathlib import Path
import json

import pytest

from kernel.config import (
    KernelRunConfig,
    build_kernel_run_name,
    load_kernel_run_config,
    resolve_adapter_path,
)


def test_load_kernel_run_config_supports_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "dataset_name: dolly",
                "adapter_path: results/adapters/example",
                "backend: lora_ntk",
                "kernel:",
                "  method: nystrom",
                "  ridge_lambda: 0.01",
                "  rank: 16",
                "  num_landmarks: 16",
            ]
        )
    )
    child.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "backend: lora_ntk",
                "train_limit: 64",
                "backend_args:",
                "  leaf_filter: lora_b_only",
            ]
        )
    )

    config = load_kernel_run_config(child)
    assert config.dataset_name == "dolly"
    assert config.backend == "lora_ntk"
    assert config.train_limit == 64
    assert config.kernel.rank == 16
    assert config.backend_args["leaf_filter"] == "lora_b_only"


def test_load_kernel_run_config_rejects_unsupported_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "kernel.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: dolly",
                "backend: unsupported_backend",
            ]
        )
    )

    with pytest.raises(ValueError, match="Unsupported kernel backend"):
        load_kernel_run_config(config_path)


def test_build_kernel_run_name_includes_backend_and_limits() -> None:
    config = load_kernel_run_config("configs/kernel/dolly_lora_ntk.yaml")
    run_name = build_kernel_run_name(config)
    assert "dolly" in run_name
    assert "smollm2-1-7b-instruct" in run_name
    assert "lora-ntk" in run_name
    assert "n64" in run_name


def test_resolve_adapter_path_matches_dataset_aliases(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapters" / "sql"
    adapter_dir.mkdir(parents=True)
    adapter_dir.joinpath("adapter_config.json").write_text("{}")
    summary_dir = tmp_path / "runs" / "example"
    summary_dir.mkdir(parents=True)
    summary_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run_name": "example",
                "status": "completed",
                "dataset_name": "sql-create-context",
                "base_model": "mlx-community/SmolLM2-1.7B-Instruct",
                "adapter_dir": str(adapter_dir),
                "updated_at": "2026-03-12T05:00:00+03:00",
            }
        )
    )

    config = KernelRunConfig(
        dataset_name="sql_create_context",
        base_model="mlx-community/SmolLM2-1.7B-Instruct",
        adapter_path="",
    )

    resolved = resolve_adapter_path(config, output_root=tmp_path)
    assert resolved == str(adapter_dir)
