from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import re


_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
]
_METHOD_ORDER = [
    "random",
    "kmeans",
    "kmeans+sign",
    "kmeans+thresholded_sign",
    "kmeans+sparse_random",
    "kmeans+sparse_random+sign",
    "kmeans+sparse_random+thresholded_sign",
]


@dataclass(frozen=True)
class PlotArtifact:
    title: str
    path: Path
    description: str


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _write(path: Path, svg: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)
    return path


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
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = 0.08 * (high - low)
    return low - padding, high + padding


def _positive_max(values: list[float]) -> float:
    high = max(values) if values else 1.0
    return high if high > 0 else 1.0


def _bar_chart(
    rows: list[tuple[str, float]],
    *,
    title: str,
    y_label: str,
    higher_is_better: bool = True,
) -> str:
    width = max(760, 88 * max(len(rows), 1) + 160)
    height = 420
    left = 76
    right = 24
    top = 54
    bottom = 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = _positive_max([value for _, value in rows])
    if higher_is_better and ymax <= 1.0:
        ymax = 1.0
    bar_w = plot_w / max(len(rows), 1) * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" font-family="Arial" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = top + plot_h - (value / ymax) * plot_h
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{value:.2f}</text>'
        )
    for index, (label, value) in enumerate(rows):
        center = left + (index + 0.5) * plot_w / max(len(rows), 1)
        bar_h = (value / ymax) * plot_h if ymax else 0
        x = center - bar_w / 2
        y = top + plot_h - bar_h
        color = _COLORS[index % len(_COLORS)]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111827">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 18}" text-anchor="end" transform="rotate(-35 {center:.1f} {top + plot_h + 18})" font-family="Arial" font-size="11" fill="#374151">{_escape(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _grouped_bar_chart(
    groups: list[float],
    methods: list[str],
    values: dict[tuple[float, str], float],
    *,
    title: str,
    y_label: str,
    higher_is_better: bool = True,
) -> str:
    method_count = max(len(methods), 1)
    group_count = max(len(groups), 1)
    width = max(860, group_count * max(108, method_count * 24) + 260)
    height = 460
    left = 78
    right = 230
    top = 54
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    numeric_values = list(values.values())
    ymax = _positive_max(numeric_values)
    if higher_is_better and ymax <= 1.0:
        ymax = 1.0
    ymin = min(0.0, min(numeric_values) if numeric_values else 0.0)
    if ymin == ymax:
        padding = abs(ymax) * 0.1 or 1.0
        ymin -= padding
        ymax += padding

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    zero_y = sy(0.0)
    group_w = plot_w / group_count
    inner_w = group_w * 0.74
    gap = 3
    bar_w = max(4.0, (inner_w - gap * (method_count - 1)) / method_count)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" font-family="Arial" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = sy(value)
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{value:.2f}</text>'
        )
    for group_index, group in enumerate(groups):
        group_center = left + (group_index + 0.5) * group_w
        start_x = group_center - inner_w / 2
        parts.append(
            f'<text x="{group_center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{group:.0f}</text>'
        )
        for method_index, method in enumerate(methods):
            value = values.get((group, method))
            if value is None:
                continue
            x = start_x + method_index * (bar_w + gap)
            y = sy(value)
            rect_y = min(y, zero_y)
            rect_h = abs(zero_y - y)
            color = _COLORS[method_index % len(_COLORS)]
            parts.append(
                f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" rx="2" fill="{color}"/>'
            )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">Selected training examples</text>'
    )
    for index, method in enumerate(methods):
        color = _COLORS[index % len(_COLORS)]
        legend_y = top + 18 + index * 22
        parts.append(
            f'<rect x="{left + plot_w + 30}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + plot_w + 48}" y="{legend_y}" font-family="Arial" font-size="12" fill="#374151">{_escape(method)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _line_chart(
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    width = 820
    height = 460
    left = 78
    right = 180
    top = 54
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [point for values in series.values() for point in values]
    if not points:
        return _empty_svg(title)
    xmin, xmax = _axis_bounds([x for x, _ in points])
    ymin, ymax = _axis_bounds([y for _, y in points])

    def sx(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{_escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" font-family="Arial" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for tick in range(5):
        x_value = xmin + (xmax - xmin) * tick / 4
        y_value = ymin + (ymax - ymin) * tick / 4
        x = sx(x_value)
        y = sy(y_value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{x_value:.0f}</text>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{y_value:.2f}</text>')
    for index, (name, values) in enumerate(series.items()):
        values = sorted(values)
        color = _COLORS[index % len(_COLORS)]
        line_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in values:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + index * 22
        parts.append(f'<rect x="{left + plot_w + 28}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{left + plot_w + 46}" y="{legend_y}" font-family="Arial" font-size="12" fill="#374151">{_escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _scatter_chart(
    points: list[tuple[float, float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool = True,
) -> str:
    width = 620
    height = 540
    left = 78
    right = 36
    top = 54
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not points:
        return _empty_svg(title)
    if len(points) > 1500:
        step = max(1, len(points) // 1500)
        points = points[::step]
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    xmin, xmax = _axis_bounds(x_values + (y_values if diagonal else []))
    ymin, ymax = _axis_bounds(y_values + (x_values if diagonal else []))

    def sx(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{_escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" font-family="Arial" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for tick in range(5):
        x_value = xmin + (xmax - xmin) * tick / 4
        y_value = ymin + (ymax - ymin) * tick / 4
        x = sx(x_value)
        y = sy(y_value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{x_value:.2f}</text>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{y_value:.2f}</text>')
    if diagonal:
        low = max(xmin, ymin)
        high = min(xmax, ymax)
        parts.append(
            f'<line x1="{sx(low):.1f}" y1="{sy(low):.1f}" x2="{sx(high):.1f}" y2="{sy(high):.1f}" stroke="#111827" stroke-width="1.5" stroke-dasharray="5 5"/>'
        )
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.7" fill="#2563eb" opacity="0.62"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _empty_svg(title: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="220" viewBox="0 0 620 220">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="32" y="42" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
            '<text x="32" y="92" font-family="Arial" font-size="13" fill="#6b7280">No data available for this plot.</text>',
            "</svg>",
        ]
    )


def _aggregate_by_method(
    summaries: list[dict[str, Any]],
    metric_path: str,
) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        value = _number(_nested(summary, metric_path))
        method = str(summary.get("method") or "-")
        if value is not None and method != "-":
            grouped[method].append(value)
    ordered = [method for method in _METHOD_ORDER if method in grouped]
    ordered.extend(sorted(method for method in grouped if method not in ordered))
    return [
        (method, mean)
        for method in ordered
        for mean in [_mean(grouped[method])]
        if mean is not None
    ]


def _series_by_method(
    summaries: list[dict[str, Any]],
    metric_path: str,
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for summary in summaries:
        details = summary.get("selector_details") or {}
        x = _number(details.get("max_examples"))
        y = _number(_nested(summary, metric_path))
        method = str(summary.get("method") or "-")
        if x is not None and y is not None and method != "-":
            grouped[(method, x)].append(y)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (method, x), values in grouped.items():
        mean = _mean(values)
        if mean is not None:
            out[method].append((x, mean))
    return dict(out)


def _grouped_values_by_subset_size(
    summaries: list[dict[str, Any]],
    metric_path: str,
) -> tuple[list[float], list[str], dict[tuple[float, str], float]]:
    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    methods_seen: set[str] = set()
    groups_seen: set[float] = set()
    for summary in summaries:
        details = summary.get("selector_details") or {}
        subset_size = _number(details.get("max_examples"))
        value = _number(_nested(summary, metric_path))
        method = str(summary.get("method") or "-")
        if subset_size is None or value is None or method == "-":
            continue
        grouped[(subset_size, method)].append(value)
        groups_seen.add(subset_size)
        methods_seen.add(method)
    methods = [method for method in _METHOD_ORDER if method in methods_seen]
    methods.extend(sorted(method for method in methods_seen if method not in methods))
    values = {
        key: mean
        for key, method_values in grouped.items()
        for mean in [_mean(method_values)]
        if mean is not None
    }
    return sorted(groups_seen), methods, values


def generate_adapter_comparison_plots(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_scatter_plots: int = 16,
) -> list[PlotArtifact]:
    output_dir = Path(output_dir)
    artifacts: list[PlotArtifact] = []

    grouped_bar_specs = [
        (
            "method_delta_pearson.svg",
            "Method Comparison: Delta Pearson",
            "Delta Pearson",
            "metrics.delta.pearson",
            True,
            "Full-vs-subset score-delta correlation by method and subset size.",
        ),
        (
            "method_sign_accuracy.svg",
            "Method Comparison: Sign Accuracy",
            "Sign Accuracy",
            "metrics.delta.sign_accuracy",
            True,
            "Agreement on whether fine-tuning helps or hurts each example by method and subset size.",
        ),
        (
            "method_delta_rmse.svg",
            "Method Error: Delta RMSE",
            "Delta RMSE",
            "metrics.delta.rmse",
            False,
            "Error between subset and full score deltas by method and subset size; lower is better.",
        ),
        (
            "method_adapter_rmse.svg",
            "Method Error: Adapter Score RMSE",
            "Adapter Score RMSE",
            "metrics.adapter_score.rmse",
            False,
            "Direct adapter-score error relative to full fine-tuning by method and subset size; lower is better.",
        ),
    ]
    for filename, title, ylabel, metric_path, higher, description in grouped_bar_specs:
        groups, methods, values = _grouped_values_by_subset_size(
            summaries,
            metric_path,
        )
        if not groups or not methods:
            continue
        path = output_dir / filename
        _write(
            path,
            _grouped_bar_chart(
                groups,
                methods,
                values,
                title=title,
                y_label=ylabel,
                higher_is_better=higher,
            ),
        )
        artifacts.append(PlotArtifact(title, path, description))

    average_bar_specs = [
        (
            "average_method_delta_pearson.svg",
            "Average Method Comparison: Delta Pearson",
            "Delta Pearson",
            "metrics.delta.pearson",
            True,
            "Average full-vs-subset score-delta correlation by method across all subset sizes.",
        ),
        (
            "average_method_sign_accuracy.svg",
            "Average Method Comparison: Sign Accuracy",
            "Sign Accuracy",
            "metrics.delta.sign_accuracy",
            True,
            "Average agreement on whether fine-tuning helps or hurts each example across all subset sizes.",
        ),
        (
            "average_method_delta_rmse.svg",
            "Average Method Error: Delta RMSE",
            "Delta RMSE",
            "metrics.delta.rmse",
            False,
            "Average error between subset and full score deltas by method across all subset sizes; lower is better.",
        ),
        (
            "average_method_adapter_rmse.svg",
            "Average Method Error: Adapter Score RMSE",
            "Adapter Score RMSE",
            "metrics.adapter_score.rmse",
            False,
            "Average direct adapter-score error relative to full fine-tuning by method across all subset sizes; lower is better.",
        ),
    ]
    for filename, title, ylabel, metric_path, higher, description in average_bar_specs:
        rows = _aggregate_by_method(summaries, metric_path)
        if not rows:
            continue
        path = output_dir / filename
        _write(path, _bar_chart(rows, title=title, y_label=ylabel, higher_is_better=higher))
        artifacts.append(PlotArtifact(title, path, description))

    curves = _series_by_method(summaries, "metrics.delta.pearson")
    if curves:
        path = output_dir / "subset_size_delta_pearson.svg"
        _write(
            path,
            _line_chart(
                curves,
                title="Subset Size Curve: Delta Pearson",
                x_label="Selected training examples",
                y_label="Delta Pearson",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Subset Size Curve: Delta Pearson",
                path,
                "How quickly each subset method approaches full-adapter behavior as n grows.",
            )
        )

    adapter_curves = _series_by_method(summaries, "metrics.adapter_score.rmse")
    if adapter_curves:
        path = output_dir / "subset_size_adapter_rmse.svg"
        _write(
            path,
            _line_chart(
                adapter_curves,
                title="Subset Size Curve: Adapter RMSE",
                x_label="Selected training examples",
                y_label="Adapter Score RMSE",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Subset Size Curve: Adapter RMSE",
                path,
                "Direct degradation relative to the full adapter as n changes; lower is better.",
            )
        )

    projection_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for summary in summaries:
        details = summary.get("selector_details") or {}
        projection_dim = _number(details.get("projection_dim"))
        value = _number(_nested(summary, "metrics.delta.pearson"))
        method = str(summary.get("method") or "-")
        if projection_dim is not None and value is not None:
            projection_series[method].append((projection_dim, value))
    if projection_series:
        path = output_dir / "projection_tradeoff_delta_pearson.svg"
        _write(
            path,
            _line_chart(
                dict(projection_series),
                title="Projection Tradeoff",
                x_label="Projected feature dimension",
                y_label="Delta Pearson",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Projection Tradeoff",
                path,
                "Quality as projected feature dimension changes.",
            )
        )

    timing_rows = []
    dim_rows = []
    for summary in summaries:
        method = str(summary.get("method") or "-")
        details = summary.get("selector_details") or {}
        seconds = _number(details.get("selector_total_seconds"))
        projected_dim = _number(details.get("projection_dim"))
        if seconds is not None and method != "-":
            timing_rows.append((method, seconds))
        if projected_dim is not None and method != "-":
            dim_rows.append((method, projected_dim))
    if timing_rows:
        path = output_dir / "selector_runtime_seconds.svg"
        _write(
            path,
            _bar_chart(
                _average_duplicate_labels(timing_rows),
                title="Selector Runtime",
                y_label="Seconds",
                higher_is_better=False,
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Selector Runtime",
                path,
                "Average selector pipeline runtime by method; lower is better.",
            )
        )
    if dim_rows:
        path = output_dir / "selector_projection_dim.svg"
        _write(
            path,
            _bar_chart(
                _average_duplicate_labels(dim_rows),
                title="Projected Feature Dimension",
                y_label="Dimensions",
                higher_is_better=False,
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Projected Feature Dimension",
                path,
                "Feature dimensionality after transformation/projection.",
            )
        )

    for index, summary in enumerate(summaries[:max_scatter_plots]):
        scores_path = summary.get("scores_path")
        if not scores_path:
            continue
        try:
            score_rows = _read_jsonl(scores_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        points = [
            (full, subset)
            for row in score_rows
            for full, subset in [
                (
                    _number(row.get("full_score_delta")),
                    _number(row.get("subset_score_delta")),
                )
            ]
            if full is not None and subset is not None
        ]
        if not points:
            continue
        method = summary.get("method") or "method"
        name = summary.get("subset_run_name") or summary.get("comparison_dir") or index
        path = output_dir / f"full_vs_subset_{index:02d}_{_slug(name)}.svg"
        _write(
            path,
            _scatter_chart(
                points,
                title=f"Full vs Subset Delta: {method}",
                x_label="Full adapter score delta",
                y_label="Subset adapter score delta",
            ),
        )
        artifacts.append(
            PlotArtifact(
                f"Full vs Subset Delta: {method}",
                path,
                "Per-example subset score deltas against full-adapter score deltas.",
            )
        )

    return artifacts


def _average_duplicate_labels(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for label, value in rows:
        grouped[label].append(value)
    ordered = [method for method in _METHOD_ORDER if method in grouped]
    ordered.extend(sorted(label for label in grouped if label not in ordered))
    return [
        (label, mean)
        for label in ordered
        for mean in [_mean(grouped[label])]
        if mean is not None
    ]


def generate_kernel_prediction_plots(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    split: str = "test",
    max_scatter_plots: int = 16,
) -> list[PlotArtifact]:
    output_dir = Path(output_dir)
    artifacts: list[PlotArtifact] = []
    for index, summary in enumerate(summaries[:max_scatter_plots]):
        run_dir = summary.get("run_dir")
        if not run_dir:
            continue
        predictions_path = Path(str(run_dir)) / "predictions" / f"{split}.jsonl"
        try:
            rows = _read_jsonl(predictions_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        points = [
            (actual, predicted)
            for row in rows
            for actual, predicted in [
                (
                    _number(row.get("score_delta")),
                    _number(row.get("predicted_score_delta")),
                )
            ]
            if actual is not None and predicted is not None
        ]
        if not points:
            continue
        run_name = summary.get("run_name") or f"kernel-{index}"
        path = output_dir / f"ntk_predicted_vs_true_{index:02d}_{_slug(run_name)}.svg"
        _write(
            path,
            _scatter_chart(
                points,
                title=f"NTK Predicted vs True Delta: {run_name}",
                x_label="True score delta",
                y_label="Predicted score delta",
            ),
        )
        artifacts.append(
            PlotArtifact(
                f"NTK Predicted vs True Delta: {run_name}",
                path,
                "Held-out NTK-predicted score deltas against true adapter score deltas.",
            )
        )
    return artifacts


def markdown_plot_section(
    artifacts: list[PlotArtifact],
    *,
    report_path: str | Path,
    heading: str = "Visualizations",
) -> str:
    if not artifacts:
        return ""
    report_dir = Path(report_path).parent
    lines = ["", f"## {heading}", ""]
    for artifact in artifacts:
        rel_path = os.path.relpath(artifact.path, report_dir)
        lines.extend(
            [
                f"### {artifact.title}",
                "",
                artifact.description,
                "",
                f"![{artifact.title}]({rel_path})",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
