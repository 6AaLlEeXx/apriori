from __future__ import annotations

from pathlib import Path
import json

from lora.kernel.report import make_kernel_comparison_report


def _write_run(
    root: Path,
    run_name: str,
    dataset_name: str,
    backend: str,
    train_n: int,
    test_delta_pearson: float,
    base_model: str = "mlx-community/SmolLM2-1.7B-Instruct",
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
                }
            }
        )
    )


def test_make_kernel_comparison_report_selects_largest_train_split(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="small-frozen",
        dataset_name="dolly",
        backend="frozen_pair",
        train_n=16,
        test_delta_pearson=0.9,
    )
    _write_run(
        output_root,
        run_name="large-frozen",
        dataset_name="dolly",
        backend="frozen_pair",
        train_n=256,
        test_delta_pearson=0.7,
    )
    _write_run(
        output_root,
        run_name="ntk-run",
        dataset_name="dolly",
        backend="lora_ntk",
        train_n=64,
        test_delta_pearson=0.94,
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(
        output_root=output_root,
        output_path=output_path,
        datasets=["dolly", "gsm8k"],
        backends=["frozen_pair", "lora_ntk"],
    )

    report = output_path.read_text()
    assert "`large-frozen`" in report
    assert "`small-frozen`" not in report
    assert "`ntk-run`" in report
    assert (
        "| SmolLM2-1.7B-Instruct | gsm8k | frozen_pair | - | - | - | - | - | - | - | missing |"
        in report
    )


def test_kernel_comparison_report_auto_discovers_completed_runs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="dolly-run",
        dataset_name="dolly",
        backend="frozen_pair",
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
        backend="frozen_pair",
        train_n=16,
        test_delta_pearson=0.8,
        base_model="org/model-a",
    )
    _write_run(
        output_root,
        run_name="model-b-run",
        dataset_name="dolly",
        backend="frozen_pair",
        train_n=16,
        test_delta_pearson=0.9,
        base_model="org/model-b",
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(output_root=output_root, output_path=output_path)

    report = output_path.read_text()
    assert "`model-a-run`" in report
    assert "`model-b-run`" in report
    assert "| model-a | dolly | frozen_pair |" in report
    assert "| model-b | dolly | frozen_pair |" in report


def test_kernel_comparison_report_filters_dataset_backend_and_model(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "kernel"
    _write_run(
        output_root,
        run_name="model-a-dolly",
        dataset_name="dolly",
        backend="frozen_pair",
        train_n=16,
        test_delta_pearson=0.8,
        base_model="org/model-a",
    )
    _write_run(
        output_root,
        run_name="model-b-dolly",
        dataset_name="dolly",
        backend="frozen_pair",
        train_n=16,
        test_delta_pearson=0.9,
        base_model="org/model-b",
    )

    output_path = tmp_path / "comparison.md"
    make_kernel_comparison_report(
        output_root=output_root,
        output_path=output_path,
        datasets=["dolly"],
        backends=["frozen_pair"],
        base_models=["org/model-a"],
    )

    report = output_path.read_text()
    assert "`model-a-dolly`" in report
    assert "`model-b-dolly`" not in report
