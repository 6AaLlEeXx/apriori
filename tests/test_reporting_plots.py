from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET

from reporting.plots import (
    generate_adapter_comparison_plots,
    generate_kernel_prediction_plots,
)


def _svg_width(path: Path) -> int:
    root = ET.parse(path).getroot()
    return int(root.attrib["width"])


def test_kernel_plots_expand_for_long_titles_and_legends(tmp_path: Path) -> None:
    run_name = (
        "samsum-samsum-llama32-1b-20260430-052414-"
        "kmeans-srp-thresholded-sign-1024-kernel-n256-v64-t2048"
    )
    run_dir = tmp_path / "kernel" / run_name
    predictions_path = run_dir / "predictions" / "test.jsonl"
    predictions_path.parent.mkdir(parents=True)
    predictions_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "score_delta": value,
                    "predicted_score_delta": value + 0.1,
                }
            )
            for value in (0.1, 0.2, 0.3)
        )
        + "\n"
    )

    generate_kernel_prediction_plots(
        [
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "split_sizes": {"train": 256},
                "eval": {
                    "test": {
                        "delta": {
                            "pearson": 0.9,
                            "rmse": 0.1,
                        }
                    }
                },
            }
        ],
        tmp_path / "plots",
    )

    scatter_path = tmp_path / "plots" / f"ntk_predicted_vs_true_00_{run_name}.svg"
    line_path = tmp_path / "plots" / "kernel_train_size_test_delta_pearson.svg"

    assert _svg_width(scatter_path) > 620
    assert run_name in scatter_path.read_text()
    assert _svg_width(line_path) > 820


def test_adapter_grouped_bar_expands_for_long_method_legend(tmp_path: Path) -> None:
    method = "kmeans+sparse_random+thresholded_sign"

    generate_adapter_comparison_plots(
        [
            {
                "method": method,
                "selector_details": {"max_examples": 512},
                "metrics": {
                    "delta": {
                        "pearson": 0.8,
                        "sign_accuracy": 0.75,
                        "rmse": 0.2,
                    },
                    "adapter_score": {"rmse": 0.3},
                },
            }
        ],
        tmp_path / "plots",
    )

    grouped_path = tmp_path / "plots" / "method_delta_pearson.svg"
    assert _svg_width(grouped_path) > 860
    assert method in grouped_path.read_text()
