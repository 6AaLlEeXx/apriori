from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re

from paths import (
    DEFAULT_KERNEL_RESULTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    resolve_existing_project_path,
    resolve_project_path,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


SUPPORTED_KERNEL_BACKENDS = {"lora_ntk"}


@dataclass
class KernelMethodConfig:
    method: str = "nystrom"
    ridge_lambda: float = 1e-2
    rank: int = 32
    num_landmarks: int = 32


@dataclass
class KernelRunConfig:
    dataset_name: str = "dolly"
    task: str = "generic"
    base_model: str = "mlx-community/SmolLM2-1.7B-Instruct"
    data_dir: str = "data/dolly"
    adapter_path: str = ""
    output_root: str = DEFAULT_KERNEL_RESULTS_ROOT
    backend: str = "lora_ntk"
    target: str = "score_delta"
    seed: int = 42
    train_limit: int = 128
    valid_limit: int = 128
    test_limit: int = 128
    kernel: KernelMethodConfig = field(default_factory=KernelMethodConfig)
    backend_args: dict[str, Any] = field(default_factory=dict)
    source_config: dict[str, str] = field(default_factory=dict)
    run_tags: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kernel"] = asdict(self.kernel)
        return payload


@dataclass
class KernelRunPaths:
    run_name: str
    run_dir: Path
    metadata_path: Path
    resolved_config_path: Path
    summary_path: Path
    eval_path: Path
    scores_dir: Path
    features_dir: Path
    predictions_dir: Path


def _dataset_key(value: str) -> str:
    return _slugify(value).replace("-", "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    visited = visited or set()
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Config inheritance cycle detected at {path}")
    visited.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. Install `pyyaml` first."
            )
        raw = yaml.safe_load(path.read_text()) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError(
            f"Unsupported config extension: {path.suffix}. Use .yaml/.yml/.json"
        )
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be an object at the top level: {path}")

    extends = raw.get("extends")
    if extends is None:
        return raw

    parent_path = (path.parent / extends).resolve()
    parent = _load_raw_config(parent_path, visited)
    return _deep_merge(parent, raw)


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = lowered.replace("/", "-")
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-")


def default_kernel_run_config() -> KernelRunConfig:
    return KernelRunConfig()


def load_kernel_run_config(path: str | Path) -> KernelRunConfig:
    config_path = resolve_existing_project_path(path)
    raw = _load_raw_config(config_path)
    defaults = default_kernel_run_config().to_dict()
    unknown = sorted(set(raw) - set(defaults) - {"extends"})
    if unknown:
        raise ValueError(f"Unknown kernel config keys: {unknown}")

    merged = dict(defaults)
    merged.update(raw)
    kernel_payload = dict(merged.get("kernel", {}))
    merged["kernel"] = KernelMethodConfig(**kernel_payload)
    merged["backend_args"] = dict(merged.get("backend_args", {}))
    merged["source_config"] = {
        str(key): str(value)
        for key, value in dict(merged.get("source_config", {})).items()
    }
    merged["run_tags"] = [str(tag) for tag in merged.get("run_tags", [])]
    if str(merged["backend"]) not in SUPPORTED_KERNEL_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_KERNEL_BACKENDS))
        raise ValueError(
            f"Unsupported kernel backend `{merged['backend']}`. "
            f"Supported backends: {supported}"
        )
    return KernelRunConfig(**merged)


def save_kernel_run_config(config: KernelRunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to save YAML configs. Install `pyyaml` first."
            )
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2))
        return
    raise ValueError(
        f"Unsupported config extension: {path.suffix}. Use .yaml/.yml/.json"
    )


def build_kernel_run_name(
    config: KernelRunConfig,
    created_at: datetime | None = None,
) -> str:
    created_at = created_at or datetime.now().astimezone()
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    method = _slugify(config.kernel.method)
    backend = _slugify(config.backend)
    dataset = _slugify(config.dataset_name)
    model = _slugify(config.base_model.rsplit("/", 1)[-1])
    target = _slugify(config.target)
    return "__".join(
        [
            timestamp,
            model,
            dataset,
            backend,
            target,
            method,
            f"n{config.train_limit}",
            f"s{config.seed}",
        ]
    )


def _load_lora_run_summaries(
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, Any]]:
    output_root = resolve_project_path(output_root)
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(
        (output_root / "runs").glob("*/summary.json"), reverse=True
    ):
        try:
            summaries.append(json.loads(summary_path.read_text()))
        except json.JSONDecodeError:
            continue
    return summaries


def resolve_adapter_path(
    config: KernelRunConfig,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> str:
    raw = str(config.adapter_path or "").strip()
    if raw and "<" not in raw:
        adapter_path = resolve_project_path(raw)
        if adapter_path.exists():
            return str(adapter_path)

    target_dataset = _dataset_key(config.dataset_name)
    target_model = str(config.base_model).strip()
    candidates: list[dict[str, Any]] = []
    for summary in _load_lora_run_summaries(output_root):
        adapter_dir = str(summary.get("adapter_dir") or "").strip()
        if not adapter_dir or not Path(adapter_dir).exists():
            continue
        if not Path(adapter_dir).joinpath("adapter_config.json").exists():
            continue
        if str(summary.get("status", "")).lower() != "completed":
            continue
        summary_dataset = _dataset_key(str(summary.get("dataset_name", "")))
        if summary_dataset != target_dataset:
            continue
        summary_model = str(summary.get("base_model") or "").strip()
        if target_model and summary_model and summary_model != target_model:
            continue
        candidates.append(summary)

    if not candidates:
        raise FileNotFoundError(
            "Could not resolve a completed adapter directory for dataset "
            f"`{config.dataset_name}` under {resolve_project_path(output_root)}/runs. "
            "Set `adapter_path` explicitly or train the adapter first."
        )

    candidates.sort(
        key=lambda summary: (
            str(summary.get("updated_at") or ""),
            str(summary.get("created_at") or ""),
            str(summary.get("run_name") or ""),
        ),
        reverse=True,
    )
    return str(candidates[0]["adapter_dir"])


def resolve_kernel_run_config(
    config: KernelRunConfig,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> KernelRunConfig:
    resolved_adapter_path = resolve_adapter_path(config, output_root=output_root)
    return replace(config, adapter_path=resolved_adapter_path)


def prepare_kernel_run(
    config: KernelRunConfig,
    run_name: str,
) -> KernelRunPaths:
    output_root = resolve_project_path(config.output_root)
    run_dir = output_root / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    scores_dir = run_dir / "scores"
    features_dir = run_dir / "features"
    predictions_dir = run_dir / "predictions"
    scores_dir.mkdir(parents=True, exist_ok=False)
    features_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=False)
    return KernelRunPaths(
        run_name=run_name,
        run_dir=run_dir,
        metadata_path=run_dir / "metadata.json",
        resolved_config_path=run_dir / "resolved_config.yaml",
        summary_path=run_dir / "summary.json",
        eval_path=run_dir / "eval.json",
        scores_dir=scores_dir,
        features_dir=features_dir,
        predictions_dir=predictions_dir,
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_kernel_metadata(
    config: KernelRunConfig,
    paths: KernelRunPaths,
    config_source: str | Path,
    status: str,
) -> dict[str, Any]:
    return {
        "run_name": paths.run_name,
        "status": status,
        "created_at": now_iso(),
        "config_source": str(Path(config_source)),
        "dataset_name": config.dataset_name,
        "task": config.task,
        "base_model": config.base_model,
        "data_dir": str(resolve_project_path(config.data_dir)),
        "adapter_path": config.adapter_path,
        "backend": config.backend,
        "target": config.target,
        "seed": config.seed,
        "train_limit": config.train_limit,
        "valid_limit": config.valid_limit,
        "test_limit": config.test_limit,
        "kernel": asdict(config.kernel),
        "backend_args": dict(config.backend_args),
        "source_config": dict(config.source_config),
        "run_tags": list(config.run_tags),
        "notes": config.notes,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
