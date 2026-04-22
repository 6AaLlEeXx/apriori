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
) -> None:
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True)
    run_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "status": "completed",
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
    assert "| gsm8k | frozen_pair | - | - | - | - | - | - | - | missing |" in report
