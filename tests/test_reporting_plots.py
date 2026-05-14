from __future__ import annotations

from pathlib import Path
import json

from reporting.plots import generate_kernel_paper_plots, generate_kernel_plots


def test_kernel_paper_plots_auto_select_observed_sizes_and_features(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "kernel" / "toy-random-512-kernel-n16"
    predictions_path = run_dir / "predictions" / "test.jsonl"
    predictions_path.parent.mkdir(parents=True)
    predictions_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "score_delta": value,
                    "predicted_score_delta": value + 0.01,
                }
            )
            for value in (0.1, 0.2, 0.3)
        )
        + "\n"
    )

    artifacts = generate_kernel_paper_plots(
        [
            (
                "Toy",
                [
                    {
                        "status": "completed",
                        "run_name": "toy-random-512-kernel-n16",
                        "run_dir": str(run_dir),
                        "adapter_path": "results/adapters/toy-random-512",
                        "split_sizes": {"train": 16},
                        "feature_transform_label": "raw",
                        "test_delta_rmse_gain": 0.02,
                        "test_baseline_delta_rmse": 0.2,
                        "eval": {"test": {"delta": {"rmse": 0.18}}},
                    }
                ],
            )
        ],
        tmp_path / "paper",
        adapter_contains="random-512",
        individual=True,
    )

    names = {artifact.path.name for artifact in artifacts}
    assert "toy_predicted_vs_true_k16.pdf" in names
    assert "toy_rmse_gain.pdf" in names
    assert "toy_rmse.pdf" in names
    pdf_path = tmp_path / "paper" / "toy_predicted_vs_true_k16.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")

    combined_artifacts = generate_kernel_paper_plots(
        [
            (
                "Toy",
                [
                    {
                        "status": "completed",
                        "run_name": "toy-random-512-kernel-n16",
                        "run_dir": str(run_dir),
                        "adapter_path": "results/adapters/toy-random-512",
                        "split_sizes": {"train": 16},
                        "feature_transform_label": "raw",
                        "test_delta_rmse_gain": 0.02,
                        "test_baseline_delta_rmse": 0.2,
                        "eval": {"test": {"delta": {"rmse": 0.18}}},
                    }
                ],
            )
        ],
        tmp_path / "combined",
        adapter_contains="random-512",
    )

    assert [artifact.path.name for artifact in combined_artifacts] == [
        "paper_kernel_figures.pdf"
    ]
    combined_path = tmp_path / "combined" / "paper_kernel_figures.pdf"
    assert combined_path.exists()
    assert combined_path.read_bytes().startswith(b"%PDF")

    standard_artifacts = generate_kernel_plots(
        [
            (
                "Toy",
                [
                    {
                        "status": "completed",
                        "run_name": "toy-random-512-kernel-n16",
                        "run_dir": str(run_dir),
                        "adapter_path": "results/adapters/toy-random-512",
                        "split_sizes": {"train": 16},
                        "feature_transform_label": "raw",
                        "test_delta_rmse_gain": 0.02,
                        "test_baseline_delta_rmse": 0.2,
                        "eval": {"test": {"delta": {"rmse": 0.18}}},
                    }
                ],
            )
        ],
        tmp_path / "standard",
        adapter_contains="random-512",
        individual=True,
        plot_style="standard",
    )

    standard_names = {artifact.path.name for artifact in standard_artifacts}
    assert "toy_standard_predicted_vs_true_k16.pdf" in standard_names
    assert "toy_standard_rmse_gain.pdf" in standard_names
    assert "toy_standard_rmse.pdf" in standard_names
    standard_pdf_path = tmp_path / "standard" / "toy_standard_rmse.pdf"
    assert standard_pdf_path.exists()
    assert standard_pdf_path.read_bytes().startswith(b"%PDF")
