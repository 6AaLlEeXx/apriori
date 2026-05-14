from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from reporting.artifacts import PlotArtifact
from reporting.kernel_data import KernelPlotData, prepare_kernel_plot_data, slug
from reporting.standard.lines import rmse_figure, rmse_gain_figure
from reporting.standard.scatter import prediction_figure, prediction_grid_figure
from reporting.standard.style import apply_standard_style, save_pdf


def _save_individual_prediction_figures(
    data: KernelPlotData,
    output_dir: Path,
) -> list[PlotArtifact]:
    artifacts: list[PlotArtifact] = []
    for experiment in data.experiment_names:
        experiment_slug = slug(experiment)
        for train_size in sorted(data.train_sizes):
            path = save_pdf(
                output_dir
                / f"{experiment_slug}_standard_predicted_vs_true_k{train_size}.pdf",
                prediction_figure(
                    data,
                    experiment=experiment,
                    train_size=train_size,
                ),
            )
            artifacts.append(
                PlotArtifact(
                    f"{experiment}: Standard Predicted vs True, k={train_size}",
                    path,
                    "Conventionally styled predicted adapter score deltas against true score deltas.",
                )
            )
    return artifacts


def _save_individual_metric_figures(
    data: KernelPlotData,
    output_dir: Path,
) -> list[PlotArtifact]:
    artifacts: list[PlotArtifact] = []
    for experiment in data.experiment_names:
        experiment_slug = slug(experiment)
        gain_path = save_pdf(
            output_dir / f"{experiment_slug}_standard_rmse_gain.pdf",
            rmse_gain_figure(data, experiment),
        )
        artifacts.append(
            PlotArtifact(
                f"{experiment}: Standard Baseline RMSE Improvement",
                gain_path,
                "Conventionally styled baseline test score-delta RMSE minus KRR test score-delta RMSE.",
            )
        )
        rmse_path = save_pdf(
            output_dir / f"{experiment_slug}_standard_rmse.pdf",
            rmse_figure(data, experiment),
        )
        artifacts.append(
            PlotArtifact(
                f"{experiment}: Standard Test RMSE",
                rmse_path,
                "Conventionally styled absolute test score-delta RMSE for KRR and train-mean baseline.",
            )
        )
    return artifacts


def _save_combined_pdf(data: KernelPlotData, output_dir: Path) -> list[PlotArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "standard_kernel_figures.pdf"
    with PdfPages(path) as pdf:
        prediction_grid = prediction_grid_figure(data)
        pdf.savefig(prediction_grid, bbox_inches="tight")
        plt.close(prediction_grid)
        for experiment in data.experiment_names:
            gain_fig = rmse_gain_figure(data, experiment)
            pdf.savefig(gain_fig, bbox_inches="tight")
            plt.close(gain_fig)
            rmse_fig = rmse_figure(data, experiment)
            pdf.savefig(rmse_fig, bbox_inches="tight")
            plt.close(rmse_fig)
    return [
        PlotArtifact(
            "Standard Kernel Figures",
            path,
            "Multi-page PDF with conventionally styled prediction-vs-true panels, RMSE improvement, and absolute RMSE figures.",
        )
    ]


def generate_kernel_standard_plots(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    features: list[str] | None = None,
    adapter_contains: str | None = None,
    split: str = "test",
    individual: bool = False,
) -> list[PlotArtifact]:
    apply_standard_style()
    data = prepare_kernel_plot_data(
        experiments,
        train_sizes=train_sizes,
        features=features,
        adapter_contains=adapter_contains,
        split=split,
    )
    output_dir = Path(output_dir)
    if not individual:
        return _save_combined_pdf(data, output_dir)
    return [
        *_save_individual_prediction_figures(data, output_dir),
        *_save_individual_metric_figures(data, output_dir),
    ]
