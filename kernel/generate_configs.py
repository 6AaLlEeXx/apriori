from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import re

from data_prep import load_data_prep_config, validate_data_prep_config
from kernel.config import KernelRunConfig, SUPPORTED_KERNEL_BACKENDS
from mlops import LoraRunConfig, load_lora_run_config
from paths import project_root, resolve_existing_project_path, resolve_project_path

try:
    import yaml
except ImportError:  # pragma: no cover - required dependency in pyproject
    yaml = None


DEFAULT_GENERATED_KERNEL_CONFIG_DIR = "configs/kernel/generated"
DEFAULT_BASE_LORA_CONFIG = "configs/base.yaml"
DEFAULT_KERNEL_BACKENDS = ["lora_ntk"]


@dataclass(frozen=True)
class GeneratedKernelConfig:
    output_path: Path
    payload: dict[str, Any]


def _dataset_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _file_component(value: str) -> str:
    component = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    component = re.sub(r"_+", "_", component)
    return component.strip("_") or "dataset"


def _display_path(path: str | Path) -> str:
    resolved = resolve_existing_project_path(path)
    try:
        return str(resolved.relative_to(project_root()))
    except ValueError:
        return str(resolved)


def discover_data_configs(config_root: str | Path = "configs/data") -> list[Path]:
    root = resolve_project_path(config_root)
    paths = [*root.rglob("*.yaml"), *root.rglob("*.yml")]
    return sorted(path for path in paths if path.is_file())


def _dedupe_paths(paths: Sequence[str | Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = resolve_existing_project_path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _load_training_configs(
    training_configs: Sequence[str | Path],
) -> dict[str, tuple[Path, LoraRunConfig]]:
    by_dataset: dict[str, tuple[Path, LoraRunConfig]] = {}
    for path in _dedupe_paths(training_configs):
        config = load_lora_run_config(path)
        by_dataset.setdefault(_dataset_key(config.dataset_name), (path, config))
    return by_dataset


def _backend_args(backend: str) -> dict[str, str]:
    if backend == "lora_ntk":
        return {"leaf_filter": "lora_b_only"}
    return {}


def _resolve_limit(name: str, value: int | None, default: int) -> int:
    if value is None:
        return default
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"`{name}` must be >= 0.")
    return resolved


def _resolve_base_config(
    base_config_path: str | Path,
) -> tuple[Path | None, LoraRunConfig]:
    resolved = resolve_existing_project_path(base_config_path)
    if not resolved.exists():
        return None, LoraRunConfig()
    return resolved, load_lora_run_config(resolved)


def plan_generated_kernel_configs(
    *,
    data_configs: Sequence[str | Path] = (),
    all_data_configs: bool = False,
    training_configs: Sequence[str | Path] = (),
    base_model: str | None = None,
    backends: Sequence[str] = DEFAULT_KERNEL_BACKENDS,
    output_dir: str | Path = DEFAULT_GENERATED_KERNEL_CONFIG_DIR,
    base_config_path: str | Path = DEFAULT_BASE_LORA_CONFIG,
    train_limit: int | None = None,
    valid_limit: int | None = None,
    test_limit: int | None = None,
) -> list[GeneratedKernelConfig]:
    selected_data_configs: list[str | Path] = list(data_configs)
    if all_data_configs:
        selected_data_configs.extend(discover_data_configs())
    data_paths = _dedupe_paths(selected_data_configs)
    if not data_paths:
        raise ValueError(
            "Set at least one `--data-config` or use `--all-data-configs`."
        )

    output_root = resolve_project_path(output_dir)
    training_by_dataset = _load_training_configs(training_configs)
    base_config_resolved, base_config = _resolve_base_config(base_config_path)
    default_kernel = KernelRunConfig()
    resolved_train_limit = _resolve_limit(
        "train_limit",
        train_limit,
        default_kernel.train_limit,
    )
    resolved_valid_limit = _resolve_limit(
        "valid_limit",
        valid_limit,
        default_kernel.valid_limit,
    )
    resolved_test_limit = _resolve_limit(
        "test_limit",
        test_limit,
        default_kernel.test_limit,
    )

    planned: list[GeneratedKernelConfig] = []
    used_output_paths: set[Path] = set()
    for data_path in data_paths:
        data_config = load_data_prep_config(data_path)
        validate_data_prep_config(data_config)
        dataset_name = str(data_config["dataset_name"])
        dataset_key = _dataset_key(dataset_name)
        training_match = training_by_dataset.get(dataset_key)
        training_path = training_match[0] if training_match else None
        training_config = training_match[1] if training_match else None

        metadata = data_config.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        resolved_base_model = (
            base_model
            or (training_config.base_model if training_config is not None else None)
            or base_config.base_model
        )
        task = (
            training_config.task
            if training_config is not None
            else str(metadata.get("task") or "generic")
        )
        data_dir = (
            training_config.data_dir
            if training_config is not None
            else str(data_config.get("output_dir") or f"data/{dataset_name}")
        )

        source_config = {"data": _display_path(data_path)}
        if training_path is not None:
            source_config["training"] = _display_path(training_path)
        if base_config_resolved is not None and training_path is None:
            source_config["base"] = _display_path(base_config_resolved)

        for backend in backends:
            backend = str(backend)
            if backend not in SUPPORTED_KERNEL_BACKENDS:
                supported = ", ".join(sorted(SUPPORTED_KERNEL_BACKENDS))
                raise ValueError(
                    f"Unsupported kernel backend `{backend}`. "
                    f"Supported backends: {supported}"
                )
            backend_component = _file_component(backend)
            output_path = (
                output_root
                / f"{_file_component(dataset_name)}_{backend_component}.yaml"
            )
            if output_path in used_output_paths:
                data_dir_name = _file_component(Path(data_dir).name)
                output_path = output_root / f"{data_dir_name}_{backend_component}.yaml"
            if output_path in used_output_paths:
                output_path = (
                    output_root
                    / f"{_file_component(data_path.stem)}_{backend_component}.yaml"
                )
            payload = {
                "dataset_name": dataset_name,
                "task": task,
                "base_model": resolved_base_model,
                "data_dir": data_dir,
                "adapter_path": "",
                "output_root": default_kernel.output_root,
                "backend": backend,
                "target": "score_delta",
                "seed": default_kernel.seed,
                "train_limit": resolved_train_limit,
                "valid_limit": resolved_valid_limit,
                "test_limit": resolved_test_limit,
                "kernel": asdict(default_kernel.kernel),
                "backend_args": _backend_args(backend),
                "source_config": source_config,
                "run_tags": ["mlx-lora", "kernel", "generated"],
                "notes": (
                    f"Generated {backend} score-delta kernel config for {dataset_name}."
                ),
            }
            used_output_paths.add(output_path)
            planned.append(
                GeneratedKernelConfig(output_path=output_path, payload=payload)
            )
    return planned


def write_generated_kernel_configs(
    planned: Sequence[GeneratedKernelConfig],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write kernel configs.")

    output_paths: list[Path] = []
    for item in planned:
        output_paths.append(item.output_path)
        if dry_run:
            continue
        if item.output_path.exists() and not force:
            raise FileExistsError(
                f"Kernel config already exists: {item.output_path}. "
                "Pass `--force` to overwrite it."
            )
        item.output_path.parent.mkdir(parents=True, exist_ok=True)
        item.output_path.write_text(yaml.safe_dump(item.payload, sort_keys=False))
    return output_paths
