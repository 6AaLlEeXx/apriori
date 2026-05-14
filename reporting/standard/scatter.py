from __future__ import annotations

from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

from reporting.kernel_data import KernelPlotData, feature_label, prediction_points
from reporting.standard.style import (
    IDEAL_DASH,
    feature_marker,
    feature_palette,
    mark_missing,
    padded_bounds,
)


def _legend_handles(features: list[str], palette: dict[str, Any]) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color="#242424",
            linestyle=IDEAL_DASH,
            linewidth=1.2,
            label="ideal",
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=feature_marker(feature),
            color="none",
            markerfacecolor=palette[feature],
            markeredgecolor="#ffffff",
            markeredgewidth=0.5,
            alpha=0.78,
            markersize=6,
            label=feature_label(feature),
        )
        for feature in features
    )
    return handles


def plot_prediction_panel(
    ax: Any,
    data: KernelPlotData,
    *,
    experiment: str,
    train_size: int,
    show_labels: bool = True,
) -> None:
    palette = feature_palette(data.features)
    points_by_feature = {
        feature: prediction_points(
            data.selected.get((experiment, train_size, feature)),
            split=data.split,
        )
        for feature in data.features
    }
    values = [
        value
        for points in points_by_feature.values()
        for point in points
        for value in point
    ]
    low, high = padded_bounds(values)
    ax.plot(
        [low, high],
        [low, high],
        color="#242424",
        linestyle=IDEAL_DASH,
        linewidth=1.2,
        label="ideal",
        zorder=1,
    )
    for feature in data.features:
        points = points_by_feature.get(feature, [])
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(
            xs,
            ys,
            color=palette[feature],
            marker=feature_marker(feature),
            s=28,
            alpha=0.72,
            linewidth=0.4,
            edgecolors="#ffffff",
            label=feature_label(feature),
            zorder=2,
        )
    if not values:
        mark_missing(ax)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(f"{experiment}, k = {train_size}")
    ax.set_xlabel("true score delta" if show_labels else "")
    ax.set_ylabel("predicted score delta" if show_labels else "")
    sns.despine(ax=ax)


def prediction_figure(
    data: KernelPlotData,
    *,
    experiment: str,
    train_size: int,
) -> Any:
    palette = feature_palette(data.features)
    fig, ax = plt.subplots(figsize=(4.8, 4.1), constrained_layout=True)
    plot_prediction_panel(
        ax,
        data,
        experiment=experiment,
        train_size=train_size,
    )
    ax.legend(
        handles=_legend_handles(data.features, palette),
        loc="best",
        frameon=True,
    )
    return fig


def prediction_grid_figure(data: KernelPlotData) -> Any:
    rows = max(len(data.train_sizes), 1)
    columns = max(len(data.experiment_names), 1)
    palette = feature_palette(data.features)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    display_train_sizes = sorted(data.train_sizes, reverse=True)
    for row, train_size in enumerate(display_train_sizes):
        for col, experiment in enumerate(data.experiment_names):
            ax = axes[row][col]
            plot_prediction_panel(
                ax,
                data,
                experiment=experiment,
                train_size=train_size,
                show_labels=False,
            )
            if row == rows - 1:
                ax.set_xlabel("true score delta")
            if col == 0:
                ax.set_ylabel("predicted score delta")
    fig.suptitle("Predicted vs true score delta", x=0.02, ha="left", fontsize=13)
    fig.legend(
        handles=_legend_handles(data.features, palette),
        loc="lower center",
        ncol=min(len(data.features) + 1, 4),
        bbox_to_anchor=(0.5, -0.02),
    )
    return fig
