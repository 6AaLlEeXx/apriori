from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


IDEAL_DASH = (0, (5.5, 2.4))
BASELINE_DASH = (0, (4.0, 2.2))


def apply_standard_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        rc={
            "axes.edgecolor": "#242424",
            "axes.labelcolor": "#242424",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "font.family": "Arial",
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.color": "#242424",
            "ytick.color": "#242424",
        },
    )


def feature_palette(features: list[str]) -> dict[str, Any]:
    fixed = {
        "raw": "#1f2937",
        "thresholded_sign": "#d95f02",
        "sign": "#1b9e77",
    }
    fallback = sns.color_palette("colorblind", n_colors=max(len(features), 1)).as_hex()
    palette: dict[str, Any] = {}
    fallback_index = 0
    for feature in features:
        if feature in fixed:
            palette[feature] = fixed[feature]
        else:
            palette[feature] = fallback[fallback_index % len(fallback)]
            fallback_index += 1
    return palette


def feature_marker(feature: str) -> str:
    markers = {
        "raw": "o",
        "thresholded_sign": "s",
        "sign": "^",
    }
    return markers.get(feature, "D")


def padded_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = 0.08 * (high - low)
    return low - padding, high + padding


def save_pdf(path: Path, fig: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def mark_missing(ax: Any) -> None:
    ax.text(
        0.5,
        0.5,
        "missing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#777777",
    )
