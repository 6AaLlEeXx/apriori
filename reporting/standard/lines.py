from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from reporting.kernel_data import (
    KernelPlotData,
    baseline_series,
    feature_label,
    krr_rmse,
    metric_series,
    rmse_gain,
)
from reporting.standard.style import (
    BASELINE_DASH,
    feature_marker,
    feature_palette,
    mark_missing,
)


def _plot_feature_series(
    ax: Any,
    data: KernelPlotData,
    series: dict[str, list[tuple[int, float]]],
) -> None:
    palette = feature_palette(data.features)
    for feature in data.features:
        points = series.get(feature, [])
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(
            xs,
            ys,
            color=palette[feature],
            linewidth=1.7,
            marker=feature_marker(feature),
            markersize=5.2,
            markerfacecolor="#ffffff",
            markeredgecolor=palette[feature],
            markeredgewidth=1.2,
            label=feature_label(feature),
        )


def _line_figure(
    data: KernelPlotData,
    *,
    experiment: str,
    title: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    include_zero: bool = False,
    baseline: list[tuple[int, float]] | None = None,
) -> Any:
    fig, ax = plt.subplots(figsize=(5.1, 3.55), constrained_layout=True)
    if baseline:
        xs = [point[0] for point in baseline]
        ys = [point[1] for point in baseline]
        ax.plot(
            xs,
            ys,
            color="#555555",
            linestyle=BASELINE_DASH,
            linewidth=1.5,
            label="Train-mean baseline",
        )
    _plot_feature_series(ax, data, series)
    if include_zero:
        ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
    if not any(series.values()) and not baseline:
        mark_missing(ax)

    ax.set_title(title)
    ax.set_xlabel("kernel fit examples (k)")
    ax.set_ylabel(y_label)
    ax.set_xticks(sorted(data.train_sizes))
    ax.grid(True, axis="both")
    ax.legend(loc="best", frameon=True)
    sns.despine(ax=ax)
    return fig


def rmse_gain_figure(data: KernelPlotData, experiment: str) -> Any:
    return _line_figure(
        data,
        experiment=experiment,
        title=f"{experiment}: baseline improvement",
        y_label="baseline RMSE - KRR RMSE",
        series=metric_series(data, experiment, rmse_gain),
        include_zero=True,
    )


def rmse_figure(data: KernelPlotData, experiment: str) -> Any:
    return _line_figure(
        data,
        experiment=experiment,
        title=f"{experiment}: test RMSE",
        y_label="test score-delta RMSE",
        series=metric_series(data, experiment, krr_rmse),
        baseline=baseline_series(data, experiment),
    )
