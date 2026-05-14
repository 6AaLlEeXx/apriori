from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from reporting.artifacts import PlotArtifact, markdown_plot_section
from reporting.paper.plots import generate_kernel_paper_plots
from reporting.standard.plots import generate_kernel_standard_plots


PlotStyle = Literal["paper", "standard"]


def generate_kernel_plots(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    features: list[str] | None = None,
    adapter_contains: str | None = None,
    split: str = "test",
    individual: bool = False,
    plot_style: PlotStyle = "paper",
) -> list[PlotArtifact]:
    if plot_style == "paper":
        return generate_kernel_paper_plots(
            experiments,
            output_dir,
            train_sizes=train_sizes,
            features=features,
            adapter_contains=adapter_contains,
            split=split,
            individual=individual,
        )
    if plot_style == "standard":
        return generate_kernel_standard_plots(
            experiments,
            output_dir,
            train_sizes=train_sizes,
            features=features,
            adapter_contains=adapter_contains,
            split=split,
            individual=individual,
        )
    raise ValueError(f"Unknown plot style: {plot_style}")


__all__ = [
    "PlotArtifact",
    "PlotStyle",
    "generate_kernel_paper_plots",
    "generate_kernel_plots",
    "generate_kernel_standard_plots",
    "markdown_plot_section",
]
