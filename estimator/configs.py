from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import re

from paths import (
    resolve_existing_project_path,
    resolve_project_path,
)

try:
    import yaml
except ImportError: 
    yaml = None


SUPPORTED_KERNEL_BACKENDS = {"lora_ntk"}
EXTENDS_KEY = "extends"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Take a pair of config dictionaries and merge them, where 'override' holds
    an upper hand over 'base'. """
    merged = dict(base)
    for key, value in override.items():
        if key == EXTENDS_KEY:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    """
        Traverses the dependencies o given config path, if config 'extends'
        some other config, then deep merge is applied, with 'extends' overriding
        its dependency config.
    """
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

    extends = raw.get(EXTENDS_KEY)
    if extends is None:
        return raw

    parent_path = (path.parent / extends).resolve()
    parent = _load_raw_config(parent_path, visited)
    return _deep_merge(parent, raw)


def load_config(path: str | Path, defaults: dict[str,Any], name: str = "config", validate: Validator | None = None) -> dict[str, Any]:
    config_path = resolve_existing_project_path(path)
    raw = _load_raw_config(config_path)
    unknown = sorted(set(raw) - set(defaults) - {EXTENDS_KEY})
    if unknown:
        raise ValueError(f"Unknown {name} keys: {unknown}")

    merged = dict(defaults)
    merged.update(raw)

    if validate is not None:
        validate(merged)
    
    return merged


Validator = Callable[[dict[str, Any]], None]

def save_config(config, path: str | Path) -> None:
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


def flatten_ignoring_keys(
    d: dict[str, Any],
    ignore: set[str],
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():                
        new_key = k
        if isinstance(v, dict):
            items.update(flatten_ignoring_keys(v, ignore))
        else:
            if new_key in ignore:
                continue 
            items[new_key] = v
    return items