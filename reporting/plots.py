from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import re
import xml.etree.ElementTree as ET


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


def _pdf_number(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _svg_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text.endswith("%"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _svg_color(value: Any) -> tuple[float, float, float] | None:
    text = str(value or "").strip()
    if not text or text == "none":
        return None
    if text.startswith("#") and len(text) == 7:
        return (
            int(text[1:3], 16) / 255,
            int(text[3:5], 16) / 255,
            int(text[5:7], 16) / 255,
        )
    if text == "white":
        return (1.0, 1.0, 1.0)
    if text == "black":
        return (0.0, 0.0, 0.0)
    return None


def _pdf_color_command(value: Any, *, stroke: bool) -> str:
    color = _svg_color(value)
    if color is None:
        return ""
    operator = "RG" if stroke else "rg"
    return (
        f"{_pdf_number(color[0])} {_pdf_number(color[1])} "
        f"{_pdf_number(color[2])} {operator}"
    )


def _pdf_escape_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


_HELVETICA_WIDTHS = {
    " ": 278,
    "!": 278,
    '"': 355,
    "#": 556,
    "$": 556,
    "%": 889,
    "&": 667,
    "'": 191,
    "(": 333,
    ")": 333,
    "*": 389,
    "+": 584,
    ",": 278,
    "-": 333,
    ".": 278,
    "/": 278,
    ":": 278,
    ";": 278,
    "<": 584,
    "=": 584,
    ">": 584,
    "?": 556,
    "@": 1015,
    "[": 278,
    "\\": 278,
    "]": 278,
    "^": 469,
    "_": 556,
    "`": 333,
    "{": 334,
    "|": 260,
    "}": 334,
    "~": 584,
}
_HELVETICA_WIDTHS.update({str(number): 556 for number in range(10)})
_HELVETICA_WIDTHS.update(
    {
        "A": 667,
        "B": 667,
        "C": 722,
        "D": 722,
        "E": 667,
        "F": 611,
        "G": 778,
        "H": 722,
        "I": 278,
        "J": 500,
        "K": 667,
        "L": 556,
        "M": 833,
        "N": 722,
        "O": 778,
        "P": 667,
        "Q": 778,
        "R": 722,
        "S": 667,
        "T": 611,
        "U": 722,
        "V": 667,
        "W": 944,
        "X": 667,
        "Y": 667,
        "Z": 611,
        "a": 556,
        "b": 556,
        "c": 500,
        "d": 556,
        "e": 556,
        "f": 278,
        "g": 556,
        "h": 556,
        "i": 222,
        "j": 222,
        "k": 500,
        "l": 222,
        "m": 833,
        "n": 556,
        "o": 556,
        "p": 556,
        "q": 556,
        "r": 333,
        "s": 500,
        "t": 278,
        "u": 556,
        "v": 500,
        "w": 722,
        "x": 500,
        "y": 500,
        "z": 500,
    }
)


def _pdf_text_width(value: str, font_size: float) -> float:
    units = sum(_HELVETICA_WIDTHS.get(character, 556) for character in value)
    return units * font_size / 1000


def _pdf_text_offset(value: str, font_size: float, anchor: str) -> float:
    if anchor == "middle":
        return -_pdf_text_width(value, font_size) / 2
    if anchor == "end":
        return -_pdf_text_width(value, font_size)
    return 0.0


def _pdf_alpha_name(value: float) -> str:
    clipped = max(0.0, min(1.0, value))
    return f"GS{int(round(clipped * 1000)):03d}"


def _pdf_alpha_command(
    attrs: dict[str, str],
    alpha_values: dict[str, float],
) -> str:
    raw = attrs.get("opacity") or attrs.get("fill-opacity") or attrs.get("stroke-opacity")
    if raw is None:
        return ""
    alpha = _svg_float(raw, 1.0)
    if alpha >= 1:
        return ""
    # SVG opacity that looks reasonable in-browser can render too washed out in
    # paper PDFs. Keep the vector transparency but enforce a print-friendly floor.
    alpha = max(alpha, 0.86)
    name = _pdf_alpha_name(alpha)
    alpha_values[name] = alpha
    return f"/{name} gs"


def _pdf_y(height: float, value: float) -> float:
    return height - value


def _svg_tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _pdf_rect_commands(
    attrs: dict[str, str],
    *,
    page_width: float,
    page_height: float,
) -> list[str]:
    x = _svg_float(attrs.get("x"))
    y = _svg_float(attrs.get("y"))
    width = page_width if attrs.get("width") == "100%" else _svg_float(attrs.get("width"))
    height = page_height if attrs.get("height") == "100%" else _svg_float(attrs.get("height"))
    fill = _svg_color(attrs.get("fill"))
    stroke = _svg_color(attrs.get("stroke"))
    commands: list[str] = []
    fill_command = _pdf_color_command(attrs.get("fill"), stroke=False)
    stroke_command = _pdf_color_command(attrs.get("stroke"), stroke=True)
    if fill_command:
        commands.append(fill_command)
    if stroke_command:
        commands.append(stroke_command)
    stroke_width = _svg_float(attrs.get("stroke-width"), 1.0)
    if stroke is not None:
        commands.append(f"{_pdf_number(stroke_width)} w")
    commands.append(
        f"{_pdf_number(x)} {_pdf_number(_pdf_y(page_height, y + height))} "
        f"{_pdf_number(width)} {_pdf_number(height)} re"
    )
    if fill is not None and stroke is not None:
        commands.append("B")
    elif fill is not None:
        commands.append("f")
    elif stroke is not None:
        commands.append("S")
    return commands


def _pdf_line_commands(attrs: dict[str, str], *, page_height: float) -> list[str]:
    stroke = _svg_color(attrs.get("stroke"))
    if stroke is None:
        return []
    x1 = _svg_float(attrs.get("x1"))
    y1 = _svg_float(attrs.get("y1"))
    x2 = _svg_float(attrs.get("x2"))
    y2 = _svg_float(attrs.get("y2"))
    commands = [
        _pdf_color_command(attrs.get("stroke"), stroke=True),
        f"{_pdf_number(_svg_float(attrs.get('stroke-width'), 1.0))} w",
    ]
    dash = attrs.get("stroke-dasharray")
    if dash:
        values = " ".join(_pdf_number(_svg_float(value)) for value in dash.split())
        commands.append(f"[{values}] 0 d")
    commands.append(
        f"{_pdf_number(x1)} {_pdf_number(_pdf_y(page_height, y1))} m "
        f"{_pdf_number(x2)} {_pdf_number(_pdf_y(page_height, y2))} l S"
    )
    return [command for command in commands if command]


def _pdf_circle_path(cx: float, cy: float, radius: float, page_height: float) -> str:
    kappa = 0.5522847498
    c = radius * kappa
    y = _pdf_y(page_height, cy)
    return " ".join(
        [
            f"{_pdf_number(cx + radius)} {_pdf_number(y)} m",
            f"{_pdf_number(cx + radius)} {_pdf_number(y + c)} "
            f"{_pdf_number(cx + c)} {_pdf_number(y + radius)} "
            f"{_pdf_number(cx)} {_pdf_number(y + radius)} c",
            f"{_pdf_number(cx - c)} {_pdf_number(y + radius)} "
            f"{_pdf_number(cx - radius)} {_pdf_number(y + c)} "
            f"{_pdf_number(cx - radius)} {_pdf_number(y)} c",
            f"{_pdf_number(cx - radius)} {_pdf_number(y - c)} "
            f"{_pdf_number(cx - c)} {_pdf_number(y - radius)} "
            f"{_pdf_number(cx)} {_pdf_number(y - radius)} c",
            f"{_pdf_number(cx + c)} {_pdf_number(y - radius)} "
            f"{_pdf_number(cx + radius)} {_pdf_number(y - c)} "
            f"{_pdf_number(cx + radius)} {_pdf_number(y)} c",
            "h",
        ]
    )


def _pdf_circle_commands(attrs: dict[str, str], *, page_height: float) -> list[str]:
    fill = _svg_color(attrs.get("fill"))
    stroke = _svg_color(attrs.get("stroke"))
    if fill is None and stroke is None:
        return []
    cx = _svg_float(attrs.get("cx"))
    cy = _svg_float(attrs.get("cy"))
    radius = _svg_float(attrs.get("r"))
    commands: list[str] = []
    fill_command = _pdf_color_command(attrs.get("fill"), stroke=False)
    stroke_command = _pdf_color_command(attrs.get("stroke"), stroke=True)
    if fill_command:
        commands.append(fill_command)
    if stroke_command:
        commands.append(stroke_command)
    if stroke is not None:
        commands.append(f"{_pdf_number(_svg_float(attrs.get('stroke-width'), 1.0))} w")
    commands.append(_pdf_circle_path(cx, cy, radius, page_height))
    if fill is not None and stroke is not None:
        commands.append("B")
    elif fill is not None:
        commands.append("f")
    else:
        commands.append("S")
    return commands


def _pdf_polyline_commands(attrs: dict[str, str], *, page_height: float) -> list[str]:
    stroke = _svg_color(attrs.get("stroke"))
    points_raw = attrs.get("points", "")
    if stroke is None or not points_raw.strip():
        return []
    points: list[tuple[float, float]] = []
    for raw_pair in points_raw.split():
        if "," not in raw_pair:
            continue
        x_raw, y_raw = raw_pair.split(",", 1)
        points.append((_svg_float(x_raw), _svg_float(y_raw)))
    if not points:
        return []
    path = [
        f"{_pdf_number(points[0][0])} {_pdf_number(_pdf_y(page_height, points[0][1]))} m"
    ]
    path.extend(
        f"{_pdf_number(x)} {_pdf_number(_pdf_y(page_height, y))} l"
        for x, y in points[1:]
    )
    commands = [
        _pdf_color_command(attrs.get("stroke"), stroke=True),
        f"{_pdf_number(_svg_float(attrs.get('stroke-width'), 1.0))} w",
        "1 J 1 j",
    ]
    dash = attrs.get("stroke-dasharray")
    if dash:
        values = " ".join(_pdf_number(_svg_float(value)) for value in dash.split())
        commands.append(f"[{values}] 0 d")
    commands.extend([" ".join(path), "S"])
    return commands


def _pdf_text_commands(attrs: dict[str, str], value: str, *, page_height: float) -> list[str]:
    if not value:
        return []
    fill_command = _pdf_color_command(attrs.get("fill", "#111111"), stroke=False)
    font_size = _svg_float(attrs.get("font-size"), 10.0)
    anchor = attrs.get("text-anchor", "start")
    x = _svg_float(attrs.get("x"))
    y = _svg_float(attrs.get("y"))
    offset = _pdf_text_offset(value, font_size, anchor)
    escaped = _pdf_escape_text(value)
    commands = [fill_command, f"BT /F1 {_pdf_number(font_size)} Tf"]
    transform = attrs.get("transform", "")
    rotate_match = re.search(r"rotate\(([-0-9.]+)(?:\s+[-0-9.]+\s+[-0-9.]+)?\)", transform)
    if rotate_match:
        # SVG's y-axis points down; PDF's points up, so the visual angle flips.
        angle = -math.radians(float(rotate_match.group(1)))
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        commands.extend(
            [
                f"1 0 0 1 {_pdf_number(x)} {_pdf_number(_pdf_y(page_height, y))} cm",
                f"{_pdf_number(cos_value)} {_pdf_number(sin_value)} "
                f"{_pdf_number(-sin_value)} {_pdf_number(cos_value)} 0 0 cm",
                f"1 0 0 1 {_pdf_number(offset)} 0 Tm",
            ]
        )
    else:
        commands.append(
            f"1 0 0 1 {_pdf_number(x + offset)} {_pdf_number(_pdf_y(page_height, y))} Tm"
        )
    commands.append(f"({escaped}) Tj ET")
    return [command for command in commands if command]


def _pdf_stream_from_svg(svg: str) -> tuple[float, float, str, dict[str, float]]:
    root = ET.fromstring(svg)
    width = _svg_float(root.attrib.get("width"), 420.0)
    height = _svg_float(root.attrib.get("height"), 330.0)
    alpha_values: dict[str, float] = {}
    commands: list[str] = []
    for element in root:
        tag = _svg_tag_name(element)
        attrs = {str(key): str(value) for key, value in element.attrib.items()}
        body: list[str]
        if tag == "rect":
            body = _pdf_rect_commands(attrs, page_width=width, page_height=height)
        elif tag == "line":
            body = _pdf_line_commands(attrs, page_height=height)
        elif tag == "circle":
            body = _pdf_circle_commands(attrs, page_height=height)
        elif tag == "polyline":
            body = _pdf_polyline_commands(attrs, page_height=height)
        elif tag == "text":
            body = _pdf_text_commands(attrs, element.text or "", page_height=height)
        else:
            continue
        if not body:
            continue
        commands.append("q")
        alpha_command = _pdf_alpha_command(attrs, alpha_values)
        if alpha_command:
            commands.append(alpha_command)
        commands.extend(body)
        commands.append("Q")
    return width, height, "\n".join(commands), alpha_values


def _write_pdf_from_svg(path: Path, svg: str) -> Path:
    width, height, stream, alpha_values = _pdf_stream_from_svg(svg)
    stream_bytes = (stream + "\n").encode("utf-8")
    alpha_names = sorted(alpha_values)
    font_id = 3
    alpha_object_ids = {name: index + 4 for index, name in enumerate(alpha_names)}
    contents_id = 4 + len(alpha_names)
    page_id = contents_id + 1
    max_object_id = page_id

    ext_gstate = ""
    if alpha_object_ids:
        states = " ".join(
            f"/{name} {object_id} 0 R"
            for name, object_id in alpha_object_ids.items()
        )
        ext_gstate = f" /ExtGState << {states} >>"

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode(),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        contents_id: (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode()
            + stream_bytes
            + b"endstream"
        ),
        page_id: (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {_pdf_number(width)} {_pdf_number(height)}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >>{ext_gstate} >> "
            f"/Contents {contents_id} 0 R >>"
        ).encode(),
    }
    for name, object_id in alpha_object_ids.items():
        alpha = _pdf_number(alpha_values[name])
        objects[object_id] = (
            f"<< /Type /ExtGState /ca {alpha} /CA {alpha} >>"
        ).encode()

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (max_object_id + 1)
    for object_id in range(1, max_object_id + 1):
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {max_object_id + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for object_id in range(1, max_object_id + 1):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {max_object_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))
    return path


def _write_paper_vector(path: Path, svg: str, *, pdf: bool) -> Path:
    _write(path, svg)
    if pdf:
        _write_pdf_from_svg(path.with_suffix(".pdf"), svg)
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
        return "Thresholded-sign LoRA-NTK"
    if feature == "sign":
        return "Sign LoRA-NTK"
    return feature.replace("_", " ")


def _paper_feature_color(feature: str) -> str:
    colors = {
        "raw": "#111111",
        "thresholded_sign": "#d62728",
        "sign": "#9467bd",
    }
    return colors.get(feature, _COLORS[len(feature) % len(_COLORS)])


def _paper_tick_label(value: float, *, compact: bool = False) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    if compact and abs_value < 0.1 and value != 0:
        return f"{value:.3f}"
    if abs_value >= 0.01 or value == 0:
        return f"{value:.2f}" if compact else f"{value:.3f}"
    return f"{value:.4f}"


def _paper_ticks(low: float, high: float, count: int = 6) -> list[float]:
    if count <= 1 or low == high:
        return [low]
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def _paper_rmse_gain_value(summary: dict[str, Any]) -> float | None:
    value = _number(summary.get("test_delta_rmse_gain"))
    if value is not None:
        return value
    krr_rmse = _number(_nested(summary, "eval.test.delta.rmse"))
    baseline_rmse = _number(_nested(summary, "baseline.test.delta.rmse"))
    if baseline_rmse is None:
        baseline_rmse = _number(summary.get("test_baseline_delta_rmse"))
    if baseline_rmse is None or krr_rmse is None:
        return None
    return baseline_rmse - krr_rmse


def _paper_baseline_rmse_value(summary: dict[str, Any]) -> float | None:
    value = _number(summary.get("test_baseline_delta_rmse"))
    if value is not None:
        return value
    value = _number(_nested(summary, "baseline.test.delta.rmse"))
    if value is not None:
        return value
    return _number(_nested(summary, "eval.test.baseline.delta.rmse"))


def _paper_krr_rmse_value(summary: dict[str, Any]) -> float | None:
    value = _number(_nested(summary, "eval.test.delta.rmse"))
    if value is not None:
        return value
    baseline_rmse = _paper_baseline_rmse_value(summary)
    gain = _paper_rmse_gain_value(summary)
    if baseline_rmse is None or gain is None:
        return None
    return baseline_rmse - gain


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


def _paper_prediction_single_svg(
    experiment: str,
    train_size: int,
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    *,
    split: str,
) -> str:
    width = 420
    height = 330
    left = 58
    right = 18
    top = 38
    bottom = 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    title = f"{experiment}, k = {train_size}"
    points_by_feature: dict[str, list[tuple[float, float]]] = {}
    values: list[float] = []
    for feature in features:
        summary = selected.get((experiment, train_size, feature))
        points = (
            _kernel_prediction_points(summary, split=split)
            if summary is not None
            else []
        )
        points_by_feature[feature] = points
        values.extend([value for point in points for value in point])
    xmin, xmax = _axis_bounds(values)
    ymin, ymax = xmin, xmax

    def sx(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="22" text-anchor="middle" font-family="Arial" font-size="12" fill="#111111">{_escape(title)}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">true score delta</text>',
        f'<text x="14" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2:.1f})" font-family="Arial" font-size="10" fill="#111111">predicted score delta</text>',
    ]
    for value in _paper_ticks(xmin, xmax, 6):
        x = sx(value)
        y = sy(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#d9d9d9" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 3.5}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<line x1="{left - 3.5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 16}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">{_paper_tick_label(value, compact=True)}</text>'
        )
        parts.append(
            f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#111111">{_paper_tick_label(value, compact=True)}</text>'
        )
    parts.append(
        f'<line x1="{sx(xmin):.1f}" y1="{sy(xmin):.1f}" x2="{sx(xmax):.1f}" y2="{sy(xmax):.1f}" stroke="#111111" stroke-width="1.2" stroke-dasharray="5.5 2.4"/>'
    )
    for feature in features:
        color = _paper_feature_color(feature)
        for x_value, y_value in points_by_feature.get(feature, []):
            parts.append(
                f'<circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" r="2.3" fill="{color}" opacity="0.62"/>'
            )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111111" stroke-width="0.8"/>'
    )
    legend_w = 200
    legend_h = 40 + 18 * max(len(features) - 1, 0)
    legend_x = left + 8
    legend_y = top + 10
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="2" fill="#ffffff" stroke="#cccccc" stroke-width="0.8"/>'
    )
    parts.append(
        f'<line x1="{legend_x + 10}" y1="{legend_y + 12}" x2="{legend_x + 34}" y2="{legend_y + 12}" stroke="#111111" stroke-width="1.2" stroke-dasharray="5.5 2.4"/>'
    )
    parts.append(
        f'<text x="{legend_x + 42}" y="{legend_y + 15}" font-family="Arial" font-size="10" fill="#111111">ideal</text>'
    )
    for index, feature in enumerate(features):
        y = legend_y + 30 + 18 * index
        color = _paper_feature_color(feature)
        parts.append(f'<circle cx="{legend_x + 22}" cy="{y - 3}" r="3" fill="{color}" opacity="0.8"/>')
        parts.append(
            f'<text x="{legend_x + 42}" y="{y}" font-family="Arial" font-size="10" fill="#111111">{_escape(_paper_feature_label(feature))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _paper_rmse_gain_single_svg(
    experiment: str,
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
) -> str:
    width = 420
    height = 330
    left = 60
    right = 18
    top = 38
    bottom = 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    values_by_feature: dict[str, list[tuple[int, float]]] = {}
    y_values: list[float] = []
    for feature in features:
        values: list[tuple[int, float]] = []
        for train_size in sorted(train_sizes):
            summary = selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = _paper_rmse_gain_value(summary)
            if value is None:
                continue
            values.append((train_size, value))
            y_values.append(value)
        values_by_feature[feature] = values
    raw_ymin = min(y_values) if y_values else 0.0
    raw_ymax = max(y_values) if y_values else 1.0
    has_negative_gain = raw_ymin < 0
    if has_negative_gain:
        padding = (raw_ymax - raw_ymin) * 0.08 or 1.0
        ymin = raw_ymin - padding
        ymax = raw_ymax + padding
    else:
        ymin = 0.0
        ymax = raw_ymax + (abs(raw_ymax) * 0.08 or 1.0)
    xmin, xmax = _axis_bounds([float(value) for value in train_sizes])

    def sx(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    title = f"{experiment}: baseline improvement"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="22" text-anchor="middle" font-family="Arial" font-size="12" fill="#111111">{_escape(title)}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">kernel fit examples (k)</text>',
        f'<text x="14" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2:.1f})" font-family="Arial" font-size="10" fill="#111111">baseline RMSE - KRR RMSE</text>',
    ]
    for y_value in _paper_ticks(ymin, ymax, 6):
        y = sy(y_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{left - 3.5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#111111">{_paper_tick_label(y_value)}</text>'
        )
    for train_size in sorted(train_sizes):
        x = sx(train_size)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6e6e6" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 3.5}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 16}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">{train_size}</text>'
        )
    for feature in features:
        points = values_by_feature.get(feature, [])
        if not points:
            continue
        color = _paper_feature_color(feature)
        line_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x_value, y_value in points:
            parts.append(
                f'<circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" r="3.4" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>'
            )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111111" stroke-width="0.8"/>'
    )
    legend_w = 200
    legend_h = 29 + 20 * max(len(features) - 1, 0)
    legend_x = left + plot_w - legend_w - 8
    if has_negative_gain:
        legend_y = top + (plot_h - legend_h) / 2
    else:
        legend_y = top + plot_h - legend_h - 8
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="2" fill="#ffffff" stroke="#cccccc" stroke-width="0.8"/>'
    )
    for index, feature in enumerate(features):
        y = legend_y + 19 + 20 * index
        color = _paper_feature_color(feature)
        parts.append(
            f'<line x1="{legend_x + 10}" y1="{y - 4}" x2="{legend_x + 34}" y2="{y - 4}" stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle cx="{legend_x + 22}" cy="{y - 4}" r="3" fill="#ffffff" stroke="{color}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 42}" y="{y - 1}" font-family="Arial" font-size="10" fill="#111111">{_escape(_paper_feature_label(feature))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _paper_rmse_single_svg(
    experiment: str,
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
) -> str:
    width = 420
    height = 330
    left = 60
    right = 18
    top = 38
    bottom = 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    values_by_feature: dict[str, list[tuple[int, float]]] = {}
    baseline_by_train_size: list[tuple[int, float]] = []
    y_values: list[float] = []

    for feature in features:
        values: list[tuple[int, float]] = []
        for train_size in sorted(train_sizes):
            summary = selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            value = _paper_krr_rmse_value(summary)
            if value is None:
                continue
            values.append((train_size, value))
            y_values.append(value)
        values_by_feature[feature] = values

    for train_size in sorted(train_sizes):
        baseline_value: float | None = None
        for feature in features:
            summary = selected.get((experiment, train_size, feature))
            if summary is None:
                continue
            baseline_value = _paper_baseline_rmse_value(summary)
            if baseline_value is not None:
                break
        if baseline_value is None:
            continue
        baseline_by_train_size.append((train_size, baseline_value))
        y_values.append(baseline_value)

    ymin = 0.0
    ymax = max(y_values) if y_values else 1.0
    ymax += abs(ymax - ymin) * 0.08 or 1.0
    xmin, xmax = _axis_bounds([float(value) for value in train_sizes])

    def sx(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    title = f"{experiment}: test RMSE"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="22" text-anchor="middle" font-family="Arial" font-size="12" fill="#111111">{_escape(title)}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">kernel fit examples (k)</text>',
        f'<text x="14" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2:.1f})" font-family="Arial" font-size="10" fill="#111111">test score-delta RMSE</text>',
    ]
    for y_value in _paper_ticks(ymin, ymax, 6):
        y = sy(y_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{left - 3.5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#111111">{_paper_tick_label(y_value)}</text>'
        )
    for train_size in sorted(train_sizes):
        x = sx(train_size)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6e6e6" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 3.5}" stroke="#111111" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 16}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111111">{train_size}</text>'
        )

    if baseline_by_train_size:
        baseline_points = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}" for x, y in baseline_by_train_size
        )
        parts.append(
            f'<polyline points="{baseline_points}" fill="none" stroke="#555555" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="5.5 2.4"/>'
        )
    for feature in features:
        points = values_by_feature.get(feature, [])
        if not points:
            continue
        color = _paper_feature_color(feature)
        line_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x_value, y_value in points:
            parts.append(
                f'<circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" r="3.4" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>'
            )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111111" stroke-width="0.8"/>'
    )
    legend_w = 210
    legend_h = 68
    if experiment.strip().lower() == "dolly":
        legend_x = left + plot_w - legend_w - 8
        legend_y = top + plot_h * 0.18
    else:
        legend_x = left + 8
        legend_y = top + plot_h - legend_h - 8
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="2" fill="#ffffff" stroke="#cccccc" stroke-width="0.8"/>'
    )
    baseline_y = legend_y + 18
    parts.append(
        f'<line x1="{legend_x + 10}" y1="{baseline_y - 4}" x2="{legend_x + 34}" y2="{baseline_y - 4}" stroke="#555555" stroke-width="1.4" stroke-dasharray="5.5 2.4"/>'
    )
    parts.append(
        f'<text x="{legend_x + 42}" y="{baseline_y - 1}" font-family="Arial" font-size="10" fill="#111111">Train-mean baseline</text>'
    )
    for index, feature in enumerate(features):
        y = legend_y + 38 + 20 * index
        color = _paper_feature_color(feature)
        parts.append(
            f'<line x1="{legend_x + 10}" y1="{y - 4}" x2="{legend_x + 34}" y2="{y - 4}" stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle cx="{legend_x + 22}" cy="{y - 4}" r="3" fill="#ffffff" stroke="{color}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 42}" y="{y - 1}" font-family="Arial" font-size="10" fill="#111111">{_escape(_paper_feature_label(feature))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _generate_individual_kernel_paper_plots(
    experiments: list[str],
    train_sizes: list[int],
    features: list[str],
    selected: dict[tuple[str, int, str], dict[str, Any]],
    output_dir: Path,
    *,
    split: str,
    pdf: bool,
) -> list[PlotArtifact]:
    artifacts: list[PlotArtifact] = []
    for experiment in experiments:
        experiment_slug = _slug(experiment)
        for train_size in sorted(train_sizes):
            path = output_dir / f"{experiment_slug}_predicted_vs_true_k{train_size}.svg"
            _write_paper_vector(
                path,
                _paper_prediction_single_svg(
                    experiment,
                    train_size,
                    features,
                    selected,
                    split=split,
                ),
                pdf=pdf,
            )
            artifacts.append(
                PlotArtifact(
                    f"{experiment}: Predicted vs True, k={train_size}",
                    path,
                    "Predicted adapter score deltas against true score deltas.",
                )
            )
        gain_path = output_dir / f"{experiment_slug}_rmse_gain.svg"
        _write_paper_vector(
            gain_path,
            _paper_rmse_gain_single_svg(
                experiment,
                train_sizes,
                features,
                selected,
            ),
            pdf=pdf,
        )
        artifacts.append(
            PlotArtifact(
                f"{experiment}: Baseline RMSE Improvement",
                gain_path,
                "Baseline test score-delta RMSE minus KRR test score-delta RMSE.",
            )
        )
        rmse_path = output_dir / f"{experiment_slug}_rmse.svg"
        _write_paper_vector(
            rmse_path,
            _paper_rmse_single_svg(
                experiment,
                train_sizes,
                features,
                selected,
            ),
            pdf=pdf,
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
    pdf: bool = False,
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

    if individual:
        return _generate_individual_kernel_paper_plots(
            experiment_names,
            train_sizes,
            features,
            selected,
            output_dir,
            split=split,
            pdf=pdf,
        )

    prediction_path = output_dir / "paper_kernel_predicted_vs_true.svg"
    _write_paper_vector(
        prediction_path,
        _paper_prediction_grid_svg(
            experiment_names,
            train_sizes,
            features,
            selected,
            split=split,
            title="Predicted vs True Score Delta",
        ),
        pdf=pdf,
    )
    artifacts.append(
        PlotArtifact(
            "Predicted vs True Score Delta",
            prediction_path,
            "Prediction-vs-true score-delta panels by experiment and kernel fit size.",
        )
    )

    gain_path = output_dir / "paper_kernel_rmse_gain.svg"
    _write_paper_vector(
        gain_path,
        _paper_rmse_gain_svg(
            experiment_names,
            train_sizes,
            features,
            selected,
            title="KRR Improvement over Train-Mean Baseline",
        ),
        pdf=pdf,
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
