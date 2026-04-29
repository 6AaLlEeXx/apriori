from __future__ import annotations

from pathlib import Path
import json
import sys

from cli.run import main as run_cli_main
from eval import evaluate_predictions
from mlops import (
    LoraRunConfig,
    RunPaths,
    build_mlx_config_payload,
    build_run_name,
    build_train_command,
    build_test_command,
    load_lora_run_config,
    prepare_sampled_data_dir,
    parse_mlx_log_line,
    render_markdown_report,
    summarize_metric_events,
)


def test_load_lora_run_config_supports_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "dataset_name: dolly",
                "task: generic",
                "base_model: mlx-community/SmolLM2-1.7B-Instruct",
                "data_dir: data/dolly",
                "mlx_args:",
                "  batch_size: 2",
                "  num_layers: 8",
            ]
        )
    )
    child.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "dataset_name: gsm8k",
                "task: gsm8k",
                "mlx_args:",
                "  num_layers: 12",
            ]
        )
    )

    config = load_lora_run_config(child)
    assert config.dataset_name == "gsm8k"
    assert config.task == "gsm8k"
    assert config.mlx_args["batch_size"] == 2
    assert config.mlx_args["num_layers"] == 12


def test_build_run_name_includes_key_knobs() -> None:
    config = LoraRunConfig(
        dataset_name="gsm8k",
        base_model="mlx-community/SmolLM2-1.7B-Instruct",
        mlx_args={
            "num_layers": 12,
            "max_seq_length": 640,
            "seed": 42,
            "lora_parameters": {"rank": 8},
        },
    )

    run_name = build_run_name(config)
    assert "smollm2-1-7b-instruct" in run_name
    assert "gsm8k" in run_name
    assert "__r8__" in run_name
    assert "__l12__" in run_name
    assert run_name.endswith("__s42")


def test_build_train_command_maps_mlx_args_to_cli_flags() -> None:
    config = LoraRunConfig(
        dataset_name="dolly",
        data_dir="data/dolly",
        mlx_args={
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "mask_prompt": True,
            "max_seq_length": 512,
            "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0},
        },
    )
    paths = RunPaths(
        run_name="run-1",
        run_dir=Path("results/runs/run-1"),
        adapter_dir=Path("results/adapters/run-1"),
        logs_dir=Path("results/runs/run-1/logs"),
        train_log=Path("results/runs/run-1/logs/train.log"),
        test_log=Path("results/runs/run-1/logs/test.log"),
        metrics_path=Path("results/runs/run-1/metrics.jsonl"),
        metadata_path=Path("results/runs/run-1/metadata.json"),
        resolved_config_path=Path("results/runs/run-1/config.resolved.yaml"),
        summary_path=Path("results/runs/run-1/summary.md"),
        eval_path=Path("results/runs/run-1/eval.json"),
        command_path=Path("results/runs/run-1/command.txt"),
        mlx_config_path=Path("results/runs/run-1/mlx_config.yaml"),
    )

    command = build_train_command(config, paths)
    assert "--batch-size" in command
    assert "--grad-accumulation-steps" in command
    assert "--mask-prompt" in command
    assert "--max-seq-length" in command
    assert "--gradient-accumulation-steps" not in command
    assert "--lora-parameters" not in command
    assert "-c" in command
    payload = build_mlx_config_payload(config)
    assert payload["lora_parameters"]["rank"] == 8
    assert payload["lora_parameters"]["scale"] == 2.0
    assert "alpha" not in payload["lora_parameters"]


def test_build_train_and_test_command_support_data_dir_override() -> None:
    config = LoraRunConfig(dataset_name="dolly", data_dir="data/dolly")
    paths = RunPaths(
        run_name="run-1",
        run_dir=Path("results/runs/run-1"),
        adapter_dir=Path("results/adapters/run-1"),
        logs_dir=Path("results/runs/run-1/logs"),
        train_log=Path("results/runs/run-1/logs/train.log"),
        test_log=Path("results/runs/run-1/logs/test.log"),
        metrics_path=Path("results/runs/run-1/metrics.jsonl"),
        metadata_path=Path("results/runs/run-1/metadata.json"),
        resolved_config_path=Path("results/runs/run-1/config.resolved.yaml"),
        summary_path=Path("results/runs/run-1/summary.md"),
        eval_path=Path("results/runs/run-1/eval.json"),
        command_path=Path("results/runs/run-1/command.txt"),
        mlx_config_path=Path("results/runs/run-1/mlx_config.yaml"),
    )

    train_command = build_train_command(config, paths, data_dir="data/dolly_subset")
    test_command = build_test_command(config, paths, data_dir="data/dolly_subset")
    assert str(Path("data/dolly_subset").resolve()) in train_command
    assert str(Path("data/dolly_subset").resolve()) in test_command


def test_prepare_sampled_data_dir_applies_selector(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True)
    train_rows = [{"id": i, "prompt": f"p{i}", "completion": f"c{i}"} for i in range(6)]
    source_dir.joinpath("train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in train_rows) + "\n"
    )
    source_dir.joinpath("valid.jsonl").write_text(
        json.dumps({"prompt": "v", "completion": "v"}) + "\n"
    )
    source_dir.joinpath("test.jsonl").write_text(
        json.dumps({"prompt": "t", "completion": "t"}) + "\n"
    )
    selector_path = tmp_path / "selector.py"
    selector_path.write_text(
        "\n".join(
            [
                "def select_samples(rows, max_example=None):",
                "    selected = [row for row in rows if row['id'] % 2 == 0]",
                "    if max_example is None:",
                "        return selected",
                "    return selected[:max_example]",
            ]
        )
        + "\n"
    )

    sampled_dir, metadata = prepare_sampled_data_dir(
        source_data_dir=source_dir,
        run_dir=tmp_path / "run",
        sample_selector=selector_path,
        max_example=2,
    )

    sampled_rows = [
        json.loads(line) for line in sampled_dir.joinpath("train.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in sampled_rows] == [0, 2]
    assert sampled_dir.joinpath("valid.jsonl").exists()
    assert sampled_dir.joinpath("test.jsonl").exists()
    assert metadata["original_train_examples"] == 6
    assert metadata["selected_train_examples"] == 2
    assert metadata["max_example"] == 2


def test_prepare_sampled_data_dir_default_selector_returns_all(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True)
    rows = [{"prompt": "a", "completion": "b"}, {"prompt": "c", "completion": "d"}]
    source_dir.joinpath("train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    sampled_dir, metadata = prepare_sampled_data_dir(
        source_data_dir=source_dir,
        run_dir=tmp_path / "run",
        sample_selector="selectors/identity.py",
        max_example=1,
    )
    sampled_rows = [
        json.loads(line) for line in sampled_dir.joinpath("train.jsonl").read_text().splitlines()
    ]
    assert sampled_rows == rows
    assert metadata["selected_train_examples"] == 2


def test_prepare_sampled_data_dir_passes_selector_context(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True)
    rows = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(3)]
    source_dir.joinpath("train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    selector_path = tmp_path / "selector.py"
    selector_path.write_text(
        "\n".join(
            [
                "def select_samples(rows, max_example=None, context=None):",
                "    assert context['purpose'] == 'test'",
                "    return rows[:context['limit']]",
            ]
        )
        + "\n"
    )

    sampled_dir, metadata = prepare_sampled_data_dir(
        source_data_dir=source_dir,
        run_dir=tmp_path / "run",
        sample_selector=selector_path,
        selector_context={"purpose": "test", "limit": 2},
    )

    sampled_rows = [
        json.loads(line) for line in sampled_dir.joinpath("train.jsonl").read_text().splitlines()
    ]
    assert [row["prompt"] for row in sampled_rows] == ["p0", "p1"]
    assert metadata["selector_context"]["purpose"] == "test"


def test_run_cli_dry_run_applies_selector_max_examples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True)
    rows = [{"id": i, "prompt": f"p{i}", "completion": f"c{i}"} for i in range(5)]
    source_dir.joinpath("train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    source_dir.joinpath("valid.jsonl").write_text(
        json.dumps({"prompt": "v", "completion": "v"}) + "\n"
    )
    source_dir.joinpath("test.jsonl").write_text(
        json.dumps({"prompt": "t", "completion": "t"}) + "\n"
    )
    selector_path = tmp_path / "selector.py"
    selector_path.write_text(
        "\n".join(
            [
                "def select_samples(rows, max_example=None):",
                "    return rows[:max_example]",
            ]
        )
        + "\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: toy",
                "task: generic",
                f"data_dir: {source_dir}",
                f"output_root: {tmp_path / 'results'}",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-lora-run",
            "--config",
            str(config_path),
            "--run-name",
            "selector-dry-run",
            "--sample-selector",
            str(selector_path),
            "--max-examples",
            "2",
            "--dry-run",
        ],
    )

    run_cli_main()

    run_dir = tmp_path / "results" / "runs" / "selector-dry-run"
    sampled_rows = [
        json.loads(line)
        for line in run_dir.joinpath("data", "train.jsonl").read_text().splitlines()
    ]
    metadata = json.loads(run_dir.joinpath("metadata.json").read_text())
    summary = json.loads(run_dir.joinpath("summary.json").read_text())
    assert [row["id"] for row in sampled_rows] == [0, 1]
    assert metadata["sampling"]["max_example"] == 2
    assert metadata["sampling"]["selected_train_examples"] == 2
    assert metadata["sampling"]["selector_context"]["selector_projection"] == "identity"
    assert metadata["sampling"]["selector_context"]["selector_transformations"] == [
        "identity"
    ]
    assert summary["data_dir"] == metadata["train_data_dir"]
    assert summary["sampling"] == metadata["sampling"]


def test_parse_mlx_log_line_handles_train_val_and_test() -> None:
    train = parse_mlx_log_line(
        "Iter 10: Train loss 2.586, Learning Rate 1.000e-05, "
        "It/sec 1.426, Tokens/sec 251.622, Trained Tokens 1764, Peak mem 31.753 GB"
    )
    val = parse_mlx_log_line("Iter 50: Val loss 1.487, Val took 14.056s")
    test = parse_mlx_log_line("Test loss 1.234, Test ppl 3.436.")

    assert train == {
        "event": "train",
        "step": 10,
        "train_loss": 2.586,
        "learning_rate": 1.0e-05,
        "it_per_sec": 1.426,
        "tokens_per_sec": 251.622,
        "trained_tokens": 1764,
        "peak_mem_gb": 31.753,
    }
    assert val == {
        "event": "val",
        "step": 50,
        "val_loss": 1.487,
        "val_seconds": 14.056,
    }
    assert test == {
        "event": "test",
        "test_loss": 1.234,
        "test_ppl": 3.436,
    }


def test_summarize_metric_events_reports_best_val_and_peak_mem() -> None:
    events = [
        {"event": "val", "step": 1, "val_loss": 2.0},
        {"event": "train", "step": 10, "train_loss": 1.7, "peak_mem_gb": 18.0},
        {"event": "val", "step": 20, "val_loss": 1.4},
        {"event": "train", "step": 20, "train_loss": 1.3, "peak_mem_gb": 18.5},
        {"event": "test", "test_loss": 1.1, "test_ppl": 3.0},
    ]

    summary = summarize_metric_events(events)
    assert summary["last_train_step"] == 20
    assert summary["last_train_loss"] == 1.3
    assert summary["best_val_loss"] == 1.4
    assert summary["best_val_step"] == 20
    assert summary["peak_mem_gb"] == 18.5
    assert summary["test_ppl"] == 3.0


def test_evaluate_predictions_supports_gsm8k_and_samsum() -> None:
    gsm8k = evaluate_predictions(
        [
            {"prediction": "The answer is 42", "reference": "42"},
            {"prediction": "17", "reference": "18"},
        ],
        {
            "name": "gsm8k",
            "metrics": ["token_f1", "gsm8k_answer_accuracy"],
            "primary_metric": "gsm8k_answer_accuracy",
        },
    )
    samsum = evaluate_predictions(
        [{"prediction": "alice is late", "reference": "alice is late"}],
        {
            "name": "samsum",
            "metrics": ["token_f1", "rouge_l_f1"],
            "primary_metric": "rouge_l_f1",
        },
    )

    assert gsm8k["metric_name"] == "gsm8k_answer_accuracy"
    assert gsm8k["metric_value"] == 0.5
    assert samsum["metric_name"] == "rouge_l_f1"
    assert samsum["metric_value"] == 1.0


def test_render_markdown_report_includes_task_metric() -> None:
    report = render_markdown_report(
        [
            {
                "run_name": "run-1",
                "dataset_name": "gsm8k",
                "task": "gsm8k",
                "status": "completed",
                "adapter_dir": "results/adapters/run-1",
                "metrics": {
                    "last_train_loss": 1.2,
                    "best_val_loss": 1.1,
                    "test_ppl": 2.9,
                    "peak_mem_gb": 18.2,
                },
                "task_eval": {
                    "metric_name": "gsm8k_answer_accuracy",
                    "metric_value": 0.62,
                },
            }
        ]
    )

    assert "gsm8k_answer_accuracy=0.620" in report
    assert "run-1" in report
