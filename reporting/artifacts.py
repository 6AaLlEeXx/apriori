from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class PlotArtifact:
    title: str
    path: Path
    description: str


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
                f"[Open PDF]({rel_path})",
                "",
            ]
        )
    return "\n".join(lines)
