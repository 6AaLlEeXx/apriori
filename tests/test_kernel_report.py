from __future__ import annotations

from pathlib import Path
import json

from kernel.report import make_kernel_comparison_report, make_kernel_report


def _write_run(
    root: Path,
    run_name: str,
    dataset_name: str,
    backend: str,
    train_n: int,
    test_delta_pearson: float,
    base_model: str = "mlx-community/SmolLM2-1.7B-Instruct",
    feature_label: str = "raw",
) -> None:
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True)
    run_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "status": "completed",
                "base_model": base_model,
                "dataset_name": dataset_name,
                "backend": backend,
                "feature_transform_label": feature_label,
                "feature_transform": {
                    "names": (
                        ["identity"] if feature_label == "raw" else [feature_label]
                    ),
                    "params": {},
                    "label": feature_label,
                },
                "feature_dim": 2048,
                "split_sizes": {"train": train_n, "valid": 8, "test": 8},
                "test_delta_pearson": test_delta_pearson,
                "valid_delta_pearson": test_delta_pearson - 0.1,
            }
        )
    )
    run_dir.joinpath("eval.json").write_text(
        json.dumps(
            {
                "test": {
                    "delta": {
                        "pearson": test_delta_pearson,
                        "spearman": 0.5,
                        "rmse": 0.25,
                        "sign_accuracy": 0.9,
                    },
                    "adapter_score": {
                        "pearson": 0.95,
                    },
                },
                "feature_transform": {
                    "names": (
                        ["identity"] if feature_label == "raw" else [feature_label]
                    ),
                    "params": {},
                    "label": feature_label,
                },
            }
        )
    )


def _write_kernel_predictions(root: Path, run_name: str) -> None:
    predictions_dir = root / "runs" / run_name / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {
        "train": [
            {"score_delta": -0.5, "predicted_score_delta": -0.4},
            {"score_delta": 0.5, "predicted_score_delta": 0.4},
        ],
        "test": [
            {"score_delta": -1.0, "predicted_score_delta": -0.8},
            {"score_delta": 0.0, "predicted_score_delta": 0.1},
            {"score_delta": 1.0, "predicted_score_delta": 0.9},
        ],
    }
    for split, rows in rows_by_split.items():
        predictions_dir.joinpath(f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )


def test_make_kernel_comparison_report_selects_largest_train_split(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="small-ntk",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.9,
    )
    _write_run(
        output_root,
        run_name="large-ntk",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=256,
        test_delta_pearson=0.7,
    )
    _write_run(
        output_root,
        run_name="gsm8k-ntk-run",
        dataset_name="gsm8k",
        backend="lora_ntk",
        train_n=64,
        test_delta_pearson=0.94,
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(
        output_root=output_root,
        output_path=output_path,
        datasets=["dolly", "gsm8k"],
        backends=["lora_ntk"],
    )

    report = output_path.read_text()
    assert "`large-ntk`" in report
    assert "`small-ntk`" not in report
    assert "`gsm8k-ntk-run`" in report


def test_kernel_comparison_report_auto_discovers_completed_runs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="dolly-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.8,
    )
    _write_run(
        output_root,
        run_name="gsm8k-run",
        dataset_name="gsm8k",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.7,
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "`dolly-run`" in report
    assert "`gsm8k-run`" in report
    assert "missing" not in report


def test_kernel_comparison_report_keeps_base_models_separate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="model-a-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.8,
        base_model="org/model-a",
    )
    _write_run(
        output_root,
        run_name="model-b-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.9,
        base_model="org/model-b",
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "`model-a-run`" in report
    assert "`model-b-run`" in report
    assert "| model-a | dolly | lora_ntk |" in report
    assert "| model-b | dolly | lora_ntk |" in report


def test_kernel_comparison_report_filters_dataset_backend_and_model(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="model-a-dolly",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.8,
        base_model="org/model-a",
    )
    _write_run(
        output_root,
        run_name="model-b-dolly",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.9,
        base_model="org/model-b",
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(
        output_root=output_root,
        output_path=output_path,
        datasets=["dolly"],
        backends=["lora_ntk"],
        base_models=["org/model-a"],
    )

    report = output_path.read_text()
    assert "`model-a-dolly`" in report
    assert "`model-b-dolly`" not in report


def test_make_kernel_report_writes_prediction_scatter_plot(tmp_path: Path) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="dolly-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.8,
    )
    _write_kernel_predictions(output_root, "dolly-run")

    output_path = tmp_path / "kernel.md"
    make_kernel_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "Visualizations" in report
    assert "Baseline RMSE" in report
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "ntk_predicted_vs_true_00_dolly-run.svg"
    ).exists()
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "kernel_train_size_test_delta_rmse_vs_baseline.svg"
    ).exists()
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "kernel_average_test_delta_rmse_vs_baseline.svg"
    ).exists()
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "kernel_train_size_test_delta_rmse_gain_by_feature.svg"
    ).exists()


def test_kernel_comparison_report_keeps_feature_transforms_separate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="raw-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.8,
        feature_label="raw",
    )
    _write_run(
        output_root,
        run_name="thresholded-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=16,
        test_delta_pearson=0.7,
        feature_label="thresholded_sign(threshold=0.1)",
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "`raw-run`" in report
    assert "`thresholded-run`" in report
    assert "thresholded_sign(threshold=0.1)" in report


def test_make_kernel_report_writes_train_size_sweep_plots(tmp_path: Path) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="adapter-kernel-n32",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=32,
        test_delta_pearson=0.6,
    )
    _write_run(
        output_root,
        run_name="adapter-kernel-n64",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=64,
        test_delta_pearson=0.8,
    )

    output_path = tmp_path / "kernel.md"
    make_kernel_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "Kernel Train Size: Test Delta Pearson" in report
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "kernel_train_size_test_delta_pearson.svg"
    ).exists()
    assert (
        tmp_path
        / "assets"
        / "kernel"
        / "kernel_train_size_test_delta_rmse.svg"
    ).exists()
