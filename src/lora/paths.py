from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = "data"
DEFAULT_RESULTS_ROOT = "results"
DEFAULT_KERNEL_RESULTS_ROOT = "results/kernel"
DEFAULT_REPORTS_ROOT = "reports"


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return PROJECT_ROOT / expanded


def resolve_existing_project_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.exists():
        return expanded
    return resolve_project_path(expanded)
