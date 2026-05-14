from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import math
import re

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns

from reporting.artifacts import PlotArtifact


FIGURE_WIDTH = 420 / 72
FIGURE_HEIGHT = 330 / 72
PREDICTION_AXES = (58 / 420, 50 / 330, 344 / 420, 242 / 330)
LINE_AXES = (60 / 420, 50 / 330, 342 / 420, 242 / 330)
PREDICTION_LABEL_CENTER_X = 230 / 420
LINE_LABEL_CENTER_X = 231 / 420
TITLE_Y = 1 - 22 / 330
X_LABEL_Y = 1 - 320 / 330
Y_LABEL_X = 14 / 420
Y_LABEL_Y = 1 - 159 / 330
IDEAL_DASH = (0, (5.5, 2.4))


def _slug(value: Any) -> str:
    text = str(value or "plot").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "plot"


def _nested(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _axis_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = 0.08 * (high - low)
    return low - padding, high + padding


def _ticks(low: float, high: float, count: int = 6) -> list[float]:
    if count <= 1 or low == high:
        return [low]
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def _scatter_tick_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _line_tick_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _style_plots() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        rc={
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 100,
            "font.family": "Arial",
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _kernel_feature_label(summary: dict[str, Any]) -> str:
    direct = summary.get("feature_transform_label")
    if direct:
        return str(direct)
    transform = summary.get("feature_transform")
    if isinstance(transform, dict) and transform.get("label"):
        return str(transform["label"])
    eval_payload = summary.get("eval")
    if isinstance(eval_payload, dict):
        eval_transform = eval_payload.get("feature_transform")
        if isinstance(eval_transform, dict) and eval_transform.get("label"):
            return str(eval_transform["label"])
    return "raw"


def _kernel_feature_key(summary: dict[str, Any]) -> str:
    label = _kernel_feature_label(summary)
    normalized = label.lower().strip()
    if normalized in {"", "identity", "none", "raw"}:
        return "raw"
    if normalized.startswith("thresholded_sign"):
        return "thresholded_sign"
    if normalized.startswith("sign"):
        return "sign"
    return normalized


def _kernel_train_size(summary: dict[str, Any]) -> int | None:
    split_sizes = summary.get("split_sizes") or {}
    if isinstance(split_sizes, dict):
        value = _number(split_sizes.get("train"))
        if value is not None:
            return int(value)
    value = _number(summary.get("train_limit"))
    return int(value) if value is not None else None


def _prediction_points(
    summary: dict[str, Any] | None,
    *,
    split: str,
) -> list[tuple[float, float]]:
    if summary is None:
        return []
    run_dir = summary.get("run_dir")
    if not run_dir:
        return []
    try:
        rows = _read_jsonl(Path(str(run_dir)) / "predictions" / f"{split}.jsonl")
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    points: list[tuple[float, float]] = []
    for row in rows:
        actual = _number(row.get("score_delta"))
        predicted = _number(row.get("predicted_score_delta"))
        if actual is not None and predicted is not None:
            points.append((actual, predicted))
    return points


def _feature_label(feature: str) -> str:
    labels = {
        "raw": "LoRA-NTK",
        "sign": "Sign LoRA-NTK",
        "thresholded_sign": "Thresholded-sign LoRA-NTK",
    }
    return labels.get(feature, feature.replace("_", " "))


def _feature_palette(features: list[str]) -> dict[str, Any]:
    fixed = {
        "raw": "#111111",
        "thresholded_sign": "#d62728",
        "sign": "#1f77b4",
    }
    fallback = sns.color_palette("tab10", n_colors=max(len(features), 1)).as_hex()
    palette: dict[str, Any] = {}
    fallback_index = 0
    for feature in features:
        if feature in fixed:
            palette[feature] = fixed[feature]
        else:
            palette[feature] = fallback[fallback_index % len(fallback)]
            fallback_index += 1
    return palette


def _rmse_gain(summary: dict[str, Any]) -> float | None:
    value = _number(summary.get("test_delta_rmse_gain"))
    if value is not None:
        return value
    krr = _number(_nested(summary, "eval.test.delta.rmse"))
    baseline = _number(_nested(summary, "baseline.test.delta.rmse"))
    if baseline is None:
        baseline = _number(summary.get("test_baseline_delta_rmse"))
    if krr is None or baseline is None:
        return None
    return baseline - krr


def _baseline_rmse(summary: dict[str, Any]) -> float | None:
    value = _number(summary.get("test_baseline_delta_rmse"))
    if value is not None:
        return value
    return _number(_nested(summary, "baseline.test.delta.rmse"))


def _krr_rmse(summary: dict[str, Any]) -> float | None:
    return _number(_nested(summary, "eval.test.delta.rmse"))


def _matches_adapter(summary: dict[str, Any], adapter_contains: str | None) -> bool:
    if not adapter_contains:
        return True
    needle = adapter_contains.lower()
    haystacks = [
        str(summary.get("run_name") or ""),
        str(summary.get("adapter_path") or ""),
    ]
    return any(needle in value.lower() for value in haystacks)


def _available_train_sizes(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    adapter_contains: str | None,
) -> list[int]:
    values: set[int] = set()
    for _, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not _matches_adapter(summary, adapter_contains):
                continue
            train_size = _kernel_train_size(summary)
            if train_size is not None:
                values.add(train_size)
    return sorted(values)


def _available_features(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    adapter_contains: str | None,
) -> list[str]:
    values: set[str] = set()
    for _, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not _matches_adapter(summary, adapter_contains):
                continue
            values.add(_kernel_feature_key(summary))
    preferred = ["raw", "thresholded_sign", "sign"]
    ordered = [feature for feature in preferred if feature in values]
    ordered.extend(sorted(values - set(ordered)))
    return ordered


def _select_summaries(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    *,
    train_sizes: list[int],
    features: list[str],
    adapter_contains: str | None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    wanted_train_sizes = set(train_sizes)
    wanted_features = set(features)
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for experiment, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not _matches_adapter(summary, adapter_contains):
                continue
            train_size = _kernel_train_size(summary)
            feature = _kernel_feature_key(summary)
            if train_size not in wanted_train_sizes or feature not in wanted_features:
                continue
            key = (experiment, train_size, feature)
            previous = selected.get(key)
            run_name = str(summary.get("run_name") or "")
            if previous is None or run_name > str(previous.get("run_name") or ""):
                selected[key] = summary
    return selected


def _paper_figure(axes_rect: tuple[float, float, float, float]) -> tuple[Any, Any]:
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes(axes_rect)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.8,
        color="#111111",
        labelcolor="#111111",
        labelsize=10,
        pad=4,
    )
    return fig, ax


def _add_figure_labels(
    fig: Any,
    *,
    title: str,
    x_label: str,
    y_label: str,
    center_x: float,
) -> None:
    fig.text(
        center_x,
        TITLE_Y,
        title,
        ha="center",
        va="center",
        fontsize=12,
        color="#111111",
    )
    fig.text(
        center_x,
        X_LABEL_Y,
        x_label,
        ha="center",
        va="center",
        fontsize=10,
        color="#111111",
    )
    fig.text(
        Y_LABEL_X,
        Y_LABEL_Y,
        y_label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
        color="#111111",
    )


def _style_legend(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("#ffffff")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.8)
    frame.set_alpha(1.0)


def _plot_x(value: float, *, left: float, width: float) -> float:
    return (value - left) / width


def _plot_y(value: float, *, top: float, height: float) -> float:
    return (top + height - value) / height


def _draw_grid(ax: Any, *, x_ticks: list[float], y_ticks: list[float]) -> None:
    for y_value in y_ticks:
        ax.axhline(y_value, color="#d9d9d9", linewidth=0.6, zorder=0)
    for x_value in x_ticks:
        ax.axvline(x_value, color="#e6e6e6", linewidth=0.6, zorder=0)


def _draw_prediction_legend(
    ax: Any,
    features: list[str],
    palette: dict[str, Any],
) -> None:
    left = 58
    top = 38
    width = 344
    height = 242
    ax.add_patch(
        Rectangle(
            (
                _plot_x(66, left=left, width=width),
                _plot_y(106, top=top, height=height),
            ),
            200 / width,
            58 / height,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            linewidth=0.8,
            zorder=5,
        )
    )
    ideal_y = _plot_y(60, top=top, height=height)
    ax.plot(
        [_plot_x(76, left=left, width=width), _plot_x(100, left=left, width=width)],
        [ideal_y, ideal_y],
        color="#111111",
        linestyle=IDEAL_DASH,
        linewidth=1.2,
        transform=ax.transAxes,
        zorder=6,
    )
    ax.text(
        _plot_x(108, left=left, width=width),
        ideal_y,
        "ideal",
        transform=ax.transAxes,
        va="center",
        fontsize=10,
        color="#111111",
        zorder=6,
    )
    for index, feature in enumerate(features[:2]):
        y_svg = 75 + 18 * index
        y = _plot_y(y_svg, top=top, height=height)
        ax.scatter(
            [_plot_x(88, left=left, width=width)],
            [y],
            color=palette[feature],
            s=26,
            alpha=0.8,
            linewidth=0,
            transform=ax.transAxes,
            zorder=6,
        )
        ax.text(
            _plot_x(108, left=left, width=width),
            y,
            _feature_label(feature),
            transform=ax.transAxes,
            va="center",
            fontsize=10,
            color="#111111",
            zorder=6,
        )


def _prediction_panel(
    ax: Any,
    points_by_feature: dict[str, list[tuple[float, float]]],
    features: list[str],
    palette: dict[str, Any],
) -> None:
    values = [
        value
        for points in points_by_feature.values()
        for point in points
        for value in point
    ]
    low, high = _axis_bounds(values)
    ticks = _ticks(low, high)
    _draw_grid(ax, x_ticks=ticks, y_ticks=ticks)
    ax.plot(
        [low, high],
        [low, high],
        color="#111111",
        linestyle=IDEAL_DASH,
        linewidth=1.2,
        zorder=2,
    )
    for feature in features:
        points = points_by_feature.get(feature, [])
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(
            xs,
            ys,
            color=palette[feature],
            s=17,
            alpha=0.62,
            linewidth=0,
            zorder=3,
        )
    if not values:
        ax.text(
            0.5,
            0.5,
            "missing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([_scatter_tick_label(value) for value in ticks])
    ax.set_yticklabels([_scatter_tick_label(value) for value in ticks])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _legend_handles(features: list[str], palette: dict[str, Any]) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color="#111111",
            linestyle=IDEAL_DASH,
            linewidth=1.2,
            label="ideal",
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[feature],
            markeredgecolor="none",
            alpha=0.8,
            markersize=6,
            label=_feature_label(feature),
        )
        for feature in features
    )
    return handles


def _prediction_grid_figure(
    experiments: list[str],
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    *,
    split: str,
) -> Any:
    rows = max(len(train_sizes), 1)
    columns = max(len(experiments), 1)
    palette = _feature_palette(features)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns + 1.0, 2.8 * rows + 0.9),
        squeeze=False,
    )
    display_train_sizes = sorted(train_sizes, reverse=True)
    for row, train_size in enumerate(display_train_sizes):
        for col, experiment in enumerate(experiments):
            ax = axes[row][col]
            points_by_feature = {
                feature: _prediction_points(
                    selected.get((experiment, train_size, feature)),
                    split=split,
                )
                for feature in features
            }
            _prediction_panel(
                ax,
                points_by_feature,
                features,
                palette,
            )
            ax.set_title(f"{experiment}, k = {train_size}")
    fig.suptitle("Predicted vs True Score Delta", x=0.02, ha="left", fontsize=12)
    fig.legend(
        handles=_legend_handles(features, palette),
        loc="lower center",
        ncol=min(len(features) + 1, 4),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def _series_for_experiment(
    experiment: str,
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    value_getter: Callable[[dict[str, Any]], float | None],
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for feature in features:
        values: list[tuple[int, float]] = []
        for train_size in sorted(train_sizes):
            summary = selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = value_getter(summary)
            if value is not None:
                values.append((train_size, value))
        if values:
            series[feature] = values
    return series


def _baseline_series_for_experiment(
    experiment: str,
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for train_size in sorted(train_sizes):
        for feature in features:
            summary = selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = _baseline_rmse(summary)
            if value is not None:
                values.append((train_size, value))
                break
    return values


def _draw_line_legend(
    ax: Any,
    features: list[str],
    series: dict[str, list[tuple[int, float]]],
    palette: dict[str, Any],
    *,
    baseline_series: list[tuple[int, float]] | None,
) -> None:
    left = 60
    top = 38
    width = 342
    height = 242
    present_features = [feature for feature in features if series.get(feature)]
    entries = len(present_features) + (1 if baseline_series else 0)
    if entries == 0:
        return

    if baseline_series:
        rect_x = 184
        rect_y = 81.56
        rect_w = 210
        rect_h = 68 if entries == 3 else 9 + 20 * entries
        row_y = 95.56
    else:
        rect_x = 194
        rect_w = 200
        rect_h = 49 if entries == 2 else 9 + 20 * entries
        rect_y = 223 if entries == 2 else 280 - 8 - rect_h
        row_y = 238

    ax.add_patch(
        Rectangle(
            (
                _plot_x(rect_x, left=left, width=width),
                _plot_y(rect_y + rect_h, top=top, height=height),
            ),
            rect_w / width,
            rect_h / height,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            linewidth=0.8,
            zorder=5,
        )
    )

    current_row = 0
    if baseline_series:
        y = _plot_y(row_y, top=top, height=height)
        ax.plot(
            [
                _plot_x(rect_x + 10, left=left, width=width),
                _plot_x(rect_x + 34, left=left, width=width),
            ],
            [y, y],
            color="#555555",
            linestyle=IDEAL_DASH,
            linewidth=1.4,
            transform=ax.transAxes,
            zorder=6,
        )
        ax.text(
            _plot_x(rect_x + 42, left=left, width=width),
            y,
            "Train-mean baseline",
            transform=ax.transAxes,
            va="center",
            fontsize=10,
            color="#111111",
            zorder=6,
        )
        current_row += 1

    for feature in present_features:
        y = _plot_y(row_y + 20 * current_row, top=top, height=height)
        color = palette[feature]
        ax.plot(
            [
                _plot_x(rect_x + 10, left=left, width=width),
                _plot_x(rect_x + 34, left=left, width=width),
            ],
            [y, y],
            color=color,
            linewidth=1.5,
            transform=ax.transAxes,
            zorder=6,
        )
        ax.scatter(
            [_plot_x(rect_x + 22, left=left, width=width)],
            [y],
            facecolors="#ffffff",
            edgecolors=color,
            s=26,
            linewidth=1.3,
            transform=ax.transAxes,
            zorder=7,
        )
        ax.text(
            _plot_x(rect_x + 42, left=left, width=width),
            y,
            _feature_label(feature),
            transform=ax.transAxes,
            va="center",
            fontsize=10,
            color="#111111",
            zorder=6,
        )
        current_row += 1


def _line_figure(
    title: str,
    y_label: str,
    train_sizes: list[int],
    series: dict[str, list[tuple[int, float]]],
    features: list[str],
    *,
    include_zero: bool = False,
    baseline_series: list[tuple[int, float]] | None = None,
) -> Any:
    palette = _feature_palette(features)
    fig, ax = _paper_figure(LINE_AXES)
    _add_figure_labels(
        fig,
        title=title,
        x_label="kernel fit examples (k)",
        y_label=y_label,
        center_x=LINE_LABEL_CENTER_X,
    )
    x_values = [float(value) for value in train_sizes]
    xmin, xmax = _axis_bounds(x_values)
    y_values = [value for values in series.values() for _, value in values]
    if baseline_series:
        y_values.extend(value for _, value in baseline_series)
    y_values.append(0.0)
    raw_ymin = min(y_values)
    ymin, ymax = _axis_bounds(y_values)
    ymin = 0.0 if raw_ymin >= 0.0 else min(0.0, ymin)
    if include_zero:
        ymax = max(0.0, ymax)
        if 0.0 < ymax < 1.0:
            rounded_ymax = round(ymax / 0.025) * 0.025
            ymax = rounded_ymax if rounded_ymax >= max(y_values) else rounded_ymax + 0.025
        if ymin == ymax:
            ymax = 1.0
    y_ticks = _ticks(ymin, ymax)
    x_ticks = [float(value) for value in sorted(train_sizes)]
    _draw_grid(ax, x_ticks=x_ticks, y_ticks=y_ticks)
    if baseline_series:
        xs = [point[0] for point in baseline_series]
        ys = [point[1] for point in baseline_series]
        ax.plot(
            xs,
            ys,
            color="#555555",
            linestyle=IDEAL_DASH,
            linewidth=1.4,
            zorder=2,
        )
    for feature in features:
        values = series.get(feature, [])
        if not values:
            continue
        xs = [point[0] for point in values]
        ys = [point[1] for point in values]
        ax.plot(
            xs,
            ys,
            color=palette[feature],
            linewidth=1.5,
            marker="o",
            markersize=4.8,
            markerfacecolor="#ffffff",
            markeredgecolor=palette[feature],
            markeredgewidth=1.5,
            zorder=3,
        )
    if include_zero:
        ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    if not any(series.values()) and not baseline_series:
        ax.text(
            0.5,
            0.5,
            "missing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(value) for value in sorted(train_sizes)])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_line_tick_label(value) for value in y_ticks])
    ax.set_xlabel("")
    ax.set_ylabel("")
    _draw_line_legend(
        ax,
        features,
        series,
        palette,
        baseline_series=baseline_series,
    )
    return fig


def _save_pdf(path: Path, fig: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def _generate_individual_plots(
    experiments: list[str],
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    output_dir: Path,
    *,
    split: str,
) -> list[PlotArtifact]:
    artifacts: list[PlotArtifact] = []
    for experiment in experiments:
        experiment_slug = _slug(experiment)
        palette = _feature_palette(features)
        for train_size in sorted(train_sizes):
            points_by_feature = {
                feature: _prediction_points(
                    selected.get((experiment, train_size, feature)),
                    split=split,
                )
                for feature in features
            }
            fig, ax = _paper_figure(PREDICTION_AXES)
            _add_figure_labels(
                fig,
                title=f"{experiment}, k = {train_size}",
                x_label="true score delta",
                y_label="predicted score delta",
                center_x=PREDICTION_LABEL_CENTER_X,
            )
            _prediction_panel(
                ax,
                points_by_feature,
                features,
                palette,
            )
            _draw_prediction_legend(ax, features, palette)
            path = _save_pdf(
                output_dir / f"{experiment_slug}_predicted_vs_true_k{train_size}.pdf",
                fig,
            )
            artifacts.append(
                PlotArtifact(
                    f"{experiment}: Predicted vs True, k={train_size}",
                    path,
                    "Predicted adapter score deltas against true score deltas.",
                )
            )

        gain_series = _series_for_experiment(
            experiment,
            train_sizes,
            features,
            selected,
            _rmse_gain,
        )
        gain_path = _save_pdf(
            output_dir / f"{experiment_slug}_rmse_gain.pdf",
            _line_figure(
                f"{experiment}: baseline improvement",
                "baseline RMSE - KRR RMSE",
                train_sizes,
                gain_series,
                features,
                include_zero=True,
            ),
        )
        artifacts.append(
            PlotArtifact(
                f"{experiment}: Baseline RMSE Improvement",
                gain_path,
                "Baseline test score-delta RMSE minus KRR test score-delta RMSE.",
            )
        )

        rmse_series = _series_for_experiment(
            experiment,
            train_sizes,
            features,
            selected,
            _krr_rmse,
        )
        rmse_path = _save_pdf(
            output_dir / f"{experiment_slug}_rmse.pdf",
            _line_figure(
                f"{experiment}: test RMSE",
                "test score-delta RMSE",
                train_sizes,
                rmse_series,
                features,
                baseline_series=_baseline_series_for_experiment(
                    experiment,
                    train_sizes,
                    features,
                    selected,
                ),
            ),
        )
        artifacts.append(
            PlotArtifact(
                f"{experiment}: Test RMSE",
                rmse_path,
                "Absolute test score-delta RMSE for KRR and train-mean baseline.",
            )
        )
    return artifacts


def generate_kernel_paper_plots(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    features: list[str] | None = None,
    adapter_contains: str | None = None,
    split: str = "test",
    individual: bool = False,
) -> list[PlotArtifact]:
    _style_plots()
    output_dir = Path(output_dir)
    train_sizes = train_sizes or _available_train_sizes(experiments, adapter_contains)
    features = features or _available_features(experiments, adapter_contains)
    if not train_sizes:
        train_sizes = [0]
    if not features:
        features = ["raw"]
    selected = _select_summaries(
        experiments,
        train_sizes=train_sizes,
        features=features,
        adapter_contains=adapter_contains,
    )
    experiment_names = [label for label, _ in experiments]

    if individual:
        return _generate_individual_plots(
            experiment_names,
            train_sizes,
            features,
            selected,
            output_dir,
            split=split,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "paper_kernel_figures.pdf"
    with PdfPages(path) as pdf:
        prediction_grid = _prediction_grid_figure(
            experiment_names,
            train_sizes,
            features,
            selected,
            split=split,
        )
        pdf.savefig(prediction_grid)
        plt.close(prediction_grid)

        for experiment in experiment_names:
            gain_series = _series_for_experiment(
                experiment,
                train_sizes,
                features,
                selected,
                _rmse_gain,
            )
            gain_fig = _line_figure(
                f"{experiment}: baseline improvement",
                "baseline RMSE - KRR RMSE",
                train_sizes,
                gain_series,
                features,
                include_zero=True,
            )
            pdf.savefig(gain_fig)
            plt.close(gain_fig)

            rmse_series = _series_for_experiment(
                experiment,
                train_sizes,
                features,
                selected,
                _krr_rmse,
            )
            rmse_fig = _line_figure(
                f"{experiment}: test RMSE",
                "test score-delta RMSE",
                train_sizes,
                rmse_series,
                features,
                baseline_series=_baseline_series_for_experiment(
                    experiment,
                    train_sizes,
                    features,
                    selected,
                ),
            )
            pdf.savefig(rmse_fig)
            plt.close(rmse_fig)

    return [
        PlotArtifact(
            "Kernel Paper Figures",
            path,
            "Multi-page PDF with prediction-vs-true panels, RMSE improvement, and absolute RMSE figures.",
        )
    ]

