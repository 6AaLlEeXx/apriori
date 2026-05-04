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


def _estimated_text_width(value: Any, font_size: float) -> float:
    """Approximate Arial text width for sizing plain SVG canvases."""
    width = 0.0
    for char in str(value):
        if char in "ilI1.,:;|! ":
            width += 0.28 * font_size
        elif char in "mwMW@#%&":
            width += 0.86 * font_size
        elif char in "-_/":
            width += 0.34 * font_size
        else:
            width += 0.55 * font_size
    return width * 1.08


def _max_text_width(values: list[Any], font_size: float) -> float:
    return max((_estimated_text_width(value, font_size) for value in values), default=0.0)


def _ceil_px(value: float) -> int:
    return int(math.ceil(value))


def _bar_chart(
    rows: list[tuple[str, float]],
    *,
    title: str,
    y_label: str,
    higher_is_better: bool = True,
) -> str:
    left = 76
    right = 24
    title_width = _estimated_text_width(title, 18)
    width = max(760, 88 * max(len(rows), 1) + 160, _ceil_px(left + title_width + 24))
    height = 420
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
    left = 78
    base_right = 230
    base_width = max(860, group_count * max(108, method_count * 24) + 260)
    plot_w = base_width - left - base_right
    legend_width = _max_text_width(list(methods), 12)
    right = max(base_right, _ceil_px(legend_width + 72))
    title_width = _estimated_text_width(title, 18)
    width = max(left + plot_w + right, _ceil_px(left + title_width + 24))
    height = 460
    top = 54
    bottom = 78
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
    left = 78
    base_width = 820
    base_right = 180
    plot_w = base_width - left - base_right
    legend_width = _max_text_width(list(series), 12)
    right = max(base_right, _ceil_px(legend_width + 70))
    title_width = _estimated_text_width(title, 18)
    width = max(left + plot_w + right, _ceil_px(left + title_width + 24))
    height = 460
    top = 54
    bottom = 70
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
    left = 78
    right = 36
    title_width = _estimated_text_width(title, 18)
    width = max(620, _ceil_px(left + title_width + 24))
    height = 540
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
    width = max(620, _ceil_px(32 + _estimated_text_width(title, 18) + 24))
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220" viewBox="0 0 {width} 220">',
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


def _kernel_adapter_label(summary: dict[str, Any]) -> str:
    run_name = str(summary.get("run_name") or "kernel")
    marker = "-kernel"
    if marker in run_name:
        run_name = run_name.split(marker, 1)[0]
    feature = _kernel_feature_label(summary)
    if feature == "raw":
        return run_name
    return f"{run_name} / {feature}"


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
    label = _kernel_feature_label(summary).strip().lower()
    if not label or label in {"raw", "identity"}:
        return "raw"
    if "thresholded_sign" in label or "thresholded-sign" in label:
        return "thresholded_sign"
    if label == "sign" or label.endswith("/ sign"):
        return "sign"
    return label


def _kernel_train_size(summary: dict[str, Any]) -> int | None:
    split_sizes = summary.get("split_sizes") or {}
    if not isinstance(split_sizes, dict):
        return None
    train_size = _number(split_sizes.get("train"))
    return int(train_size) if train_size is not None else None


def _kernel_prediction_points(
    summary: dict[str, Any],
    *,
    split: str = "test",
    max_points: int = 500,
) -> list[tuple[float, float]]:
    run_dir = summary.get("run_dir")
    if not run_dir:
        return []
    predictions_path = Path(str(run_dir)) / "predictions" / f"{split}.jsonl"
    try:
        rows = _read_jsonl(predictions_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
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
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return points


def _paper_feature_label(feature: str) -> str:
    if feature == "raw":
        return "LoRA-NTK"
    if feature == "thresholded_sign":
        return "Signed LoRA-NTK"
    if feature == "sign":
        return "Sign LoRA-NTK"
    return feature.replace("_", " ")


def _paper_summary_matches_adapter(
    summary: dict[str, Any],
    adapter_contains: str | None,
) -> bool:
    if not adapter_contains:
        return True
    needle = adapter_contains.lower()
    haystacks = [
        str(summary.get("run_name") or ""),
        str(summary.get("adapter_path") or ""),
    ]
    return any(needle in value.lower() for value in haystacks)


def _select_paper_kernel_summaries(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    *,
    train_sizes: list[int],
    features: list[str],
    adapter_contains: str | None = None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    wanted_train_sizes = set(train_sizes)
    wanted_features = set(features)
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for experiment, summaries in experiments:
        for summary in summaries:
            if str(summary.get("status", "completed")).lower() != "completed":
                continue
            if not _paper_summary_matches_adapter(summary, adapter_contains):
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


def _paper_prediction_grid_svg(
    experiments: list[str],
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    *,
    split: str,
    title: str,
) -> str:
    display_train_sizes = sorted(train_sizes, reverse=True)
    panel_w = 260
    panel_h = 220
    gap_x = 34
    gap_y = 42
    left = 94
    right = 28
    top = 78
    bottom = 76
    width = left + len(experiments) * panel_w + max(len(experiments) - 1, 0) * gap_x + right
    height = top + len(display_train_sizes) * panel_h + max(len(display_train_sizes) - 1, 0) * gap_y + bottom
    colors = {
        "raw": "#2563eb",
        "thresholded_sign": "#dc2626",
        "sign": "#9333ea",
    }

    points_by_cell: dict[tuple[str, int, str], list[tuple[float, float]]] = {}
    bounds_by_experiment: dict[str, tuple[float, float]] = {}
    for experiment in experiments:
        values: list[float] = []
        for train_size in display_train_sizes:
            for feature in features:
                summary = selected.get((experiment, train_size, feature))
                points = (
                    _kernel_prediction_points(summary, split=split)
                    if summary is not None
                    else []
                )
                points_by_cell[(experiment, train_size, feature)] = points
                values.extend([value for point in points for value in point])
        bounds_by_experiment[experiment] = _axis_bounds(values)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="19" font-weight="700">{_escape(title)}</text>',
    ]

    for column, experiment in enumerate(experiments):
        x0 = left + column * (panel_w + gap_x)
        parts.append(
            f'<text x="{x0 + panel_w / 2:.1f}" y="58" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{_escape(experiment)}</text>'
        )

    for row, train_size in enumerate(display_train_sizes):
        y0 = top + row * (panel_h + gap_y)
        parts.append(
            f'<text x="{left - 24}" y="{y0 + panel_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left - 24} {y0 + panel_h / 2:.1f})" font-family="Arial" font-size="13" font-weight="700" fill="#111827">k = {train_size}</text>'
        )
        for column, experiment in enumerate(experiments):
            x0 = left + column * (panel_w + gap_x)
            xmin, xmax = bounds_by_experiment[experiment]
            ymin, ymax = xmin, xmax
            plot_pad = 34
            plot_w = panel_w - plot_pad - 12
            plot_h = panel_h - plot_pad - 18
            px0 = x0 + plot_pad
            py0 = y0 + 12

            def sx(value: float) -> float:
                return px0 + ((value - xmin) / (xmax - xmin)) * plot_w

            def sy(value: float) -> float:
                return py0 + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>'
            )
            for tick in range(3):
                value = xmin + (xmax - xmin) * tick / 2
                x = sx(value)
                y = sy(value)
                parts.append(f'<line x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py0 + plot_h}" stroke="#f3f4f6"/>')
                parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0 + plot_w}" y2="{y:.1f}" stroke="#f3f4f6"/>')
                parts.append(f'<text x="{x:.1f}" y="{py0 + plot_h + 14}" text-anchor="middle" font-family="Arial" font-size="9" fill="#6b7280">{value:.2f}</text>')
                parts.append(f'<text x="{px0 - 5}" y="{y + 3:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="#6b7280">{value:.2f}</text>')
            parts.append(f'<line x1="{px0}" y1="{py0 + plot_h}" x2="{px0 + plot_w}" y2="{py0 + plot_h}" stroke="#9ca3af"/>')
            parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + plot_h}" stroke="#9ca3af"/>')
            parts.append(
                f'<line x1="{sx(xmin):.1f}" y1="{sy(xmin):.1f}" x2="{sx(xmax):.1f}" y2="{sy(xmax):.1f}" stroke="#111827" stroke-width="1.1" stroke-dasharray="4 4"/>'
            )
            has_points = False
            for feature in features:
                points = points_by_cell.get((experiment, train_size, feature), [])
                color = colors.get(feature, _COLORS[len(feature) % len(_COLORS)])
                if points:
                    has_points = True
                for x, y in points:
                    parts.append(
                        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.2" fill="{color}" opacity="0.58"/>'
                    )
            if not has_points:
                parts.append(
                    f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + panel_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#9ca3af">missing</text>'
                )
            if row == len(display_train_sizes) - 1:
                parts.append(
                    f'<text x="{px0 + plot_w / 2:.1f}" y="{y0 + panel_h - 4}" text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">true delta</text>'
                )
            if column == 0:
                parts.append(
                    f'<text x="{x0 + 9}" y="{py0 + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 {x0 + 9} {py0 + plot_h / 2:.1f})" font-family="Arial" font-size="10" fill="#374151">predicted</text>'
                )

    legend_x = left
    legend_y = height - 34
    for index, feature in enumerate(features):
        color = colors.get(feature, _COLORS[index % len(_COLORS)])
        x = legend_x + index * 170
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="4" fill="{color}" opacity="0.72"/>')
        parts.append(
            f'<text x="{x + 12}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#374151">{_escape(_paper_feature_label(feature))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _paper_rmse_gain_svg(
    experiments: list[str],
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    *,
    title: str,
) -> str:
    panel_w = 300
    panel_h = 270
    gap_x = 34
    left = 86
    right = 30
    top = 72
    bottom = 78
    width = left + len(experiments) * panel_w + max(len(experiments) - 1, 0) * gap_x + right
    height = top + panel_h + bottom
    colors = {
        "raw": "#2563eb",
        "thresholded_sign": "#dc2626",
        "sign": "#9333ea",
    }
    values: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    all_y: list[float] = []
    for experiment in experiments:
        for feature in features:
            for train_size in sorted(train_sizes):
                summary = selected.get((experiment, train_size, feature))
                if summary is None:
                    continue
                value = _number(summary.get("test_delta_rmse_gain"))
                if value is None:
                    krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
                    baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
                    if baseline_rmse is not None and krr_rmse is not None:
                        value = baseline_rmse - krr_rmse
                if value is None:
                    continue
                values[(experiment, feature)].append((train_size, value))
                all_y.append(value)

    ymin = min(0.0, min(all_y) if all_y else 0.0)
    ymax = max(0.0, max(all_y) if all_y else 1.0)
    if ymin == ymax:
        padding = abs(ymax) * 0.1 or 1.0
        ymin -= padding
        ymax += padding
    else:
        padding = (ymax - ymin) * 0.12
        ymin -= padding
        ymax += padding
    xmin, xmax = _axis_bounds([float(value) for value in train_sizes])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="19" font-weight="700">{_escape(title)}</text>',
        f'<text x="22" y="{top + panel_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + panel_h / 2:.1f})" font-family="Arial" font-size="12" fill="#374151">Baseline RMSE - KRR RMSE</text>',
    ]
    for column, experiment in enumerate(experiments):
        x0 = left + column * (panel_w + gap_x)
        px0 = x0 + 42
        py0 = top + 22
        plot_w = panel_w - 58
        plot_h = panel_h - 58

        def sx(value: float) -> float:
            return px0 + ((value - xmin) / (xmax - xmin)) * plot_w

        def sy(value: float) -> float:
            return py0 + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

        zero_y = sy(0.0)
        parts.append(
            f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>'
        )
        parts.append(
            f'<text x="{x0 + panel_w / 2:.1f}" y="{top + 18}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{_escape(experiment)}</text>'
        )
        for tick in range(5):
            y_value = ymin + (ymax - ymin) * tick / 4
            y = sy(y_value)
            parts.append(f'<line x1="{px0 - 4}" y1="{y:.1f}" x2="{px0 + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{px0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#6b7280">{y_value:.2f}</text>')
        for train_size in sorted(train_sizes):
            x = sx(train_size)
            parts.append(f'<line x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py0 + plot_h}" stroke="#f3f4f6"/>')
            parts.append(f'<text x="{x:.1f}" y="{py0 + plot_h + 16}" text-anchor="middle" font-family="Arial" font-size="10" fill="#6b7280">{train_size}</text>')
        parts.append(f'<line x1="{px0}" y1="{zero_y:.1f}" x2="{px0 + plot_w}" y2="{zero_y:.1f}" stroke="#9ca3af"/>')
        parts.append(f'<line x1="{px0}" y1="{py0 + plot_h}" x2="{px0 + plot_w}" y2="{py0 + plot_h}" stroke="#9ca3af"/>')
        parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + plot_h}" stroke="#9ca3af"/>')
        for index, feature in enumerate(features):
            points = values.get((experiment, feature), [])
            if not points:
                continue
            color = colors.get(feature, _COLORS[index % len(_COLORS)])
            line_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            parts.append(f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{px0 + plot_w / 2:.1f}" y="{top + panel_h - 10}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">kernel fit examples (k)</text>'
        )

    legend_x = left
    legend_y = height - 32
    for index, feature in enumerate(features):
        color = colors.get(feature, _COLORS[index % len(_COLORS)])
        x = legend_x + index * 170
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 22}" y="{legend_y + 1}" font-family="Arial" font-size="12" fill="#374151">{_escape(_paper_feature_label(feature))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def generate_kernel_paper_plots(
    experiments: list[tuple[str, list[dict[str, Any]]]],
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    features: list[str] | None = None,
    adapter_contains: str | None = None,
    split: str = "test",
) -> list[PlotArtifact]:
    output_dir = Path(output_dir)
    train_sizes = train_sizes or [16, 256, 512]
    features = features or ["raw", "thresholded_sign"]
    selected = _select_paper_kernel_summaries(
        experiments,
        train_sizes=train_sizes,
        features=features,
        adapter_contains=adapter_contains,
    )
    experiment_names = [label for label, _ in experiments]
    artifacts: list[PlotArtifact] = []

    prediction_path = output_dir / "paper_kernel_predicted_vs_true.svg"
    _write(
        prediction_path,
        _paper_prediction_grid_svg(
            experiment_names,
            train_sizes,
            features,
            selected,
            split=split,
            title="Predicted vs True Score Delta",
        ),
    )
    artifacts.append(
        PlotArtifact(
            "Predicted vs True Score Delta",
            prediction_path,
            "Prediction-vs-true score-delta panels by experiment and kernel fit size.",
        )
    )

    gain_path = output_dir / "paper_kernel_rmse_gain.svg"
    _write(
        gain_path,
        _paper_rmse_gain_svg(
            experiment_names,
            train_sizes,
            features,
            selected,
            title="KRR Improvement over Train-Mean Baseline",
        ),
    )
    artifacts.append(
        PlotArtifact(
            "KRR Improvement over Train-Mean Baseline",
            gain_path,
            "Baseline test score-delta RMSE minus KRR test score-delta RMSE by experiment and kernel fit size.",
        )
    )
    return artifacts


def _kernel_series_by_train_size(
    summaries: list[dict[str, Any]],
    metric_path: str,
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for summary in summaries:
        split_sizes = summary.get("split_sizes") or {}
        if not isinstance(split_sizes, dict):
            continue
        train_size = _number(split_sizes.get("train"))
        value = _number(_nested(summary, metric_path))
        if train_size is None or value is None:
            continue
        series[_kernel_adapter_label(summary)].append((train_size, value))
    return {
        name: values
        for name, values in series.items()
        if values
    }


def _kernel_baseline_rmse_gain_series(
    summaries: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for summary in summaries:
        split_sizes = summary.get("split_sizes") or {}
        if not isinstance(split_sizes, dict):
            continue
        train_size = _number(split_sizes.get("train"))
        krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
        baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
        if baseline_rmse is None:
            baseline_rmse = _number(summary.get("test_baseline_delta_rmse"))
        if train_size is None or krr_rmse is None or baseline_rmse is None:
            continue
        series[_kernel_adapter_label(summary)].append(
            (train_size, baseline_rmse - krr_rmse)
        )
    return {
        name: values
        for name, values in series.items()
        if values
    }


def _kernel_average_rmse_by_train_size_series(
    summaries: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for summary in summaries:
        split_sizes = summary.get("split_sizes") or {}
        if not isinstance(split_sizes, dict):
            continue
        train_size = _number(split_sizes.get("train"))
        krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
        baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
        if baseline_rmse is None:
            baseline_rmse = _number(summary.get("test_baseline_delta_rmse"))
        if train_size is None:
            continue
        if krr_rmse is not None:
            grouped[(f"KRR {_kernel_feature_label(summary)}", train_size)].append(
                krr_rmse
            )
        if baseline_rmse is not None:
            grouped[("Train-mean baseline", train_size)].append(baseline_rmse)

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (name, train_size), values in grouped.items():
        mean = _mean(values)
        if mean is not None:
            series[name].append((train_size, mean))
    return {
        name: values
        for name, values in series.items()
        if values
    }


def _kernel_average_rmse_gain_by_feature_series(
    summaries: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for summary in summaries:
        split_sizes = summary.get("split_sizes") or {}
        if not isinstance(split_sizes, dict):
            continue
        train_size = _number(split_sizes.get("train"))
        krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
        baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
        if baseline_rmse is None:
            baseline_rmse = _number(summary.get("test_baseline_delta_rmse"))
        if train_size is None or krr_rmse is None or baseline_rmse is None:
            continue
        grouped[(_kernel_feature_label(summary), train_size)].append(
            baseline_rmse - krr_rmse
        )

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (feature, train_size), values in grouped.items():
        mean = _mean(values)
        if mean is not None:
            series[feature].append((train_size, mean))
    return {
        name: values
        for name, values in series.items()
        if values
    }


def _kernel_average_rmse_rows(summaries: list[dict[str, Any]]) -> list[tuple[str, float]]:
    krr_values: list[float] = []
    baseline_values: list[float] = []
    for summary in summaries:
        krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
        baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
        if baseline_rmse is None:
            baseline_rmse = _number(summary.get("test_baseline_delta_rmse"))
        if krr_rmse is not None:
            krr_values.append(krr_rmse)
        if baseline_rmse is not None:
            baseline_values.append(baseline_rmse)
    rows: list[tuple[str, float]] = []
    krr_mean = _mean(krr_values)
    baseline_mean = _mean(baseline_values)
    if krr_mean is not None and baseline_mean is not None:
        rows.extend(
            [
                ("KRR", krr_mean),
                ("Train-mean baseline", baseline_mean),
            ]
        )
    return rows


def generate_kernel_prediction_plots(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    split: str = "test",
    max_scatter_plots: int = 16,
) -> list[PlotArtifact]:
    output_dir = Path(output_dir)
    artifacts: list[PlotArtifact] = []
    train_delta_series = _kernel_series_by_train_size(
        summaries,
        "eval.test.delta.pearson",
    )
    if train_delta_series:
        path = output_dir / "kernel_train_size_test_delta_pearson.svg"
        _write(
            path,
            _line_chart(
                train_delta_series,
                title="Kernel Train Size: Test Delta Pearson",
                x_label="Kernel fit training examples",
                y_label="Test Delta Pearson",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Kernel Train Size: Test Delta Pearson",
                path,
                "Held-out score-delta correlation as the kernel fit set grows.",
            )
        )

    train_rmse_series = _kernel_series_by_train_size(
        summaries,
        "eval.test.delta.rmse",
    )
    if train_rmse_series:
        path = output_dir / "kernel_train_size_test_delta_rmse.svg"
        _write(
            path,
            _line_chart(
                train_rmse_series,
                title="Kernel Train Size: Test Delta RMSE",
                x_label="Kernel fit training examples",
                y_label="Test Delta RMSE",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Kernel Train Size: Test Delta RMSE",
                path,
                "Held-out score-delta error as the kernel fit set grows; lower is better.",
            )
        )

    baseline_gain_series = _kernel_baseline_rmse_gain_series(summaries)
    if baseline_gain_series:
        path = output_dir / "kernel_train_size_test_delta_rmse_gain.svg"
        _write(
            path,
            _line_chart(
                baseline_gain_series,
                title="Kernel vs Baseline: Test Delta RMSE Gain",
                x_label="Kernel fit training examples",
                y_label="Baseline RMSE - KRR RMSE",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Kernel vs Baseline: Test Delta RMSE Gain",
                path,
                "Positive values mean KRR has lower test score-delta RMSE than the train-mean baseline.",
            )
        )

    average_gain_by_feature = _kernel_average_rmse_gain_by_feature_series(summaries)
    if average_gain_by_feature:
        path = output_dir / "kernel_train_size_test_delta_rmse_gain_by_feature.svg"
        _write(
            path,
            _line_chart(
                average_gain_by_feature,
                title="Kernel Feature Transform: Test Delta RMSE Gain",
                x_label="Kernel fit training examples",
                y_label="Baseline RMSE - KRR RMSE",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Kernel Feature Transform: Test Delta RMSE Gain",
                path,
                "Average KRR improvement over the train-mean baseline by kernel feature transform and fit-set size.",
            )
        )

    average_rmse_by_train_size = _kernel_average_rmse_by_train_size_series(summaries)
    if average_rmse_by_train_size:
        path = output_dir / "kernel_train_size_test_delta_rmse_vs_baseline.svg"
        _write(
            path,
            _line_chart(
                average_rmse_by_train_size,
                title="Kernel Train Size: Test Delta RMSE vs Baseline",
                x_label="Kernel fit training examples",
                y_label="Test Delta RMSE",
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Kernel Train Size: Test Delta RMSE vs Baseline",
                path,
                "Average test score-delta RMSE for KRR and the train-mean baseline at each kernel fit-set size.",
            )
        )

    average_rmse_rows = _kernel_average_rmse_rows(summaries)
    if average_rmse_rows:
        path = output_dir / "kernel_average_test_delta_rmse_vs_baseline.svg"
        _write(
            path,
            _bar_chart(
                average_rmse_rows,
                title="Average Test Delta RMSE: KRR vs Baseline",
                y_label="Test Delta RMSE",
                higher_is_better=False,
            ),
        )
        artifacts.append(
            PlotArtifact(
                "Average Test Delta RMSE: KRR vs Baseline",
                path,
                "Average test score-delta RMSE for KRR and the train-mean baseline across kernel runs.",
            )
        )

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
        feature = _kernel_feature_label(summary)
        title_suffix = f"{run_name} [{feature}]" if feature != "raw" else str(run_name)
        path = output_dir / f"ntk_predicted_vs_true_{index:02d}_{_slug(run_name)}.svg"
        _write(
            path,
            _scatter_chart(
                points,
                title=f"NTK Predicted vs True Delta: {title_suffix}",
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
