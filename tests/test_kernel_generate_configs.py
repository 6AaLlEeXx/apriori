from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kernel.generate_configs import (
    plan_generated_kernel_configs,
    write_generated_kernel_configs,
)


def _write_data_config(path: Path, dataset_name: str = "toy") -> None:
    path.write_text(
        "\n".join(
            [
                f"dataset_name: {dataset_name}",
                "source_dataset: local/toy",
                f"output_dir: data/{dataset_name}",
                "source:",
                "  type: jsonl",
                "  path: source.jsonl",
                "split:",
                "  strategy: ratios",
                "mapping:",
                '  prompt_template: "{question}"',
                '  completion_template: "{answer}"',
                "metadata:",
                "  task: exact_match",
            ]
        )
    )


def test_generate_kernel_config_from_data_config(tmp_path: Path) -> None:
    data_config = tmp_path / "toy.yaml"
    output_dir = tmp_path / "generated"
    _write_data_config(data_config)

    planned = plan_generated_kernel_configs(
        data_configs=[data_config],
        base_model="models/test",
        output_dir=output_dir,
    )
    paths = write_generated_kernel_configs(planned)

    assert paths == [output_dir / "toy_lora_ntk.yaml"]
    payload = yaml.safe_load(paths[0].read_text())
    assert payload["dataset_name"] == "toy"
    assert payload["task"] == "exact_match"
    assert payload["base_model"] == "models/test"
    assert payload["data_dir"] == "data/toy"
    assert payload["backend"] == "lora_ntk"
    assert payload["target"] == "score_delta"
    assert payload["train_limit"] == 128
    assert payload["valid_limit"] == 128
    assert payload["test_limit"] == 128
    assert payload["backend_args"] == {"leaf_filter": "lora_b_only"}
    assert "data" in payload["source_config"]


def test_generate_kernel_config_uses_cli_model_before_training_config(
    tmp_path: Path,
) -> None:
    data_config = tmp_path / "toy.yaml"
    training_config = tmp_path / "train.yaml"
    _write_data_config(data_config)
    training_config.write_text(
        "\n".join(
            [
                "dataset_name: toy",
                "task: train-task",
                "base_model: models/from-training",
                "data_dir: custom/toy",
            ]
        )
    )

    planned = plan_generated_kernel_configs(
        data_configs=[data_config],
        training_configs=[training_config],
        base_model="models/from-cli",
        backends=["lora_ntk"],
        output_dir=tmp_path / "generated",
    )

    payload = planned[0].payload
    assert payload["base_model"] == "models/from-cli"
    assert payload["task"] == "train-task"
    assert payload["data_dir"] == "custom/toy"
    assert payload["backend_args"] == {"leaf_filter": "lora_b_only"}
    assert payload["source_config"]["training"].endswith("train.yaml")


def test_generate_kernel_config_accepts_split_limit_overrides(
    tmp_path: Path,
) -> None:
    data_config = tmp_path / "toy.yaml"
    _write_data_config(data_config)

    planned = plan_generated_kernel_configs(
        data_configs=[data_config],
        base_model="models/test",
        train_limit=64,
        valid_limit=0,
        test_limit=2048,
        output_dir=tmp_path / "generated",
    )

    payload = planned[0].payload
    assert payload["train_limit"] == 64
    assert payload["valid_limit"] == 0
    assert payload["test_limit"] == 2048


def test_generate_kernel_config_rejects_negative_split_limits(
    tmp_path: Path,
) -> None:
    data_config = tmp_path / "toy.yaml"
    _write_data_config(data_config)

    with pytest.raises(ValueError, match="test_limit"):
        plan_generated_kernel_configs(
            data_configs=[data_config],
            base_model="models/test",
            test_limit=-1,
            output_dir=tmp_path / "generated",
        )


def test_generate_kernel_config_rejects_unsupported_backend(tmp_path: Path) -> None:
    data_config = tmp_path / "toy.yaml"
    _write_data_config(data_config)

    with pytest.raises(ValueError, match="Unsupported kernel backend"):
        plan_generated_kernel_configs(
            data_configs=[data_config],
            base_model="models/test",
            backends=["unsupported_backend"],
            output_dir=tmp_path / "generated",
        )


def test_generate_kernel_config_refuses_overwrite_without_force(
    tmp_path: Path,
) -> None:
    data_config = tmp_path / "toy.yaml"
    output_dir = tmp_path / "generated"
    _write_data_config(data_config)
    planned = plan_generated_kernel_configs(
        data_configs=[data_config],
        base_model="models/test",
        output_dir=output_dir,
    )

    write_generated_kernel_configs(planned)
    with pytest.raises(FileExistsError, match="already exists"):
        write_generated_kernel_configs(planned)

    write_generated_kernel_configs(planned, force=True)


def test_generate_kernel_config_dry_run_does_not_write(tmp_path: Path) -> None:
    data_config = tmp_path / "toy.yaml"
    output_dir = tmp_path / "generated"
    _write_data_config(data_config)
    planned = plan_generated_kernel_configs(
        data_configs=[data_config],
        base_model="models/test",
        output_dir=output_dir,
    )

    paths = write_generated_kernel_configs(planned, dry_run=True)

    assert paths == [output_dir / "toy_lora_ntk.yaml"]
    assert not paths[0].exists()


def test_generate_kernel_config_uses_data_dir_name_for_name_collisions(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    output_dir = tmp_path / "generated"
    _write_data_config(left, dataset_name="toy")
    _write_data_config(right, dataset_name="toy")
    right.write_text(
        right.read_text().replace("output_dir: data/toy", "output_dir: data/toy_extra")
    )

    planned = plan_generated_kernel_configs(
        data_configs=[left, right],
        base_model="models/test",
        output_dir=output_dir,
    )

    assert [item.output_path.name for item in planned] == [
        "toy_lora_ntk.yaml",
        "toy_extra_lora_ntk.yaml",
    ]
