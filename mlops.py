from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.util
import inspect
from pathlib import Path
from typing import Any
import json
import re
import shlex
import shutil
import subprocess

from paths import (
    DEFAULT_RESULTS_ROOT,
    resolve_existing_project_path,
    resolve_project_path,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


TRAIN_RE = re.compile(
    r"^Iter (?P<step>\d+): Train loss (?P<train_loss>[^,]+)"
    r"(?:, Learning Rate (?P<learning_rate>[^,]+))?"
    r"(?:, It/sec (?P<it_per_sec>[^,]+), Tokens/sec (?P<tokens_per_sec>[^,]+))?"
    r"(?:, Trained Tokens (?P<trained_tokens>\d+))?"
    r"(?:, Peak mem (?P<peak_mem_gb>[^ ]+) GB)?$"
)
VAL_RE = re.compile(
    r"^Iter (?P<step>\d+): Val loss (?P<val_loss>[^,]+)"
    r"(?:, Val took (?P<val_seconds>[^s]+)s)?$"
)
TEST_RE = re.compile(r"^Test loss (?P<test_loss>[^,]+), Test ppl (?P<test_ppl>.+?)\.?$")


@dataclass
class LoraRunConfig:
    dataset_name: str = "dolly"
    task: str = "generic"
    base_model: str = "mlx-community/SmolLM2-1.7B-Instruct"
    prepared_data_dir: str = "data/dolly"
    results_root: str = DEFAULT_RESULTS_ROOT
    mlx_command: str = "mlx_lm.lora"
    test_after_train: bool = True
    run_tags: list[str] = field(default_factory=list)
    mlx_lora_args: dict[str, Any] = field(default_factory=dict)
    subset_training: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunPaths:
    run_name: str
    run_dir: Path
    adapter_dir: Path
    logs_dir: Path
    train_log: Path
    test_log: Path
    metrics_path: Path
    metadata_path: Path
    resolved_config_path: Path
    summary_path: Path
    command_path: Path
    mlx_config_path: Path


MLX_ARG_ALIASES = {
    "gradient_accumulation_steps": "grad_accumulation_steps",
}

MLX_CLI_SUPPORTED_ARGS = {
    "fine_tune_type",
    "optimizer",
    "mask_prompt",
    "num_layers",
    "batch_size",
    "iters",
    "val_batches",
    "learning_rate",
    "steps_per_report",
    "steps_per_eval",
    "grad_accumulation_steps",
    "resume_adapter_file",
    "save_every",
    "test_batches",
    "max_seq_length",
    "grad_checkpoint",
    "seed",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if value != value:  # NaN check
            return None
        return value
    text = str(value).strip()
    if text.lower() == "nan":
        return None
    number = float(text)
    if number != number:
        return None
    return number


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = lowered.replace("/", "-")
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-")


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


def default_lora_run_config() -> LoraRunConfig:
    return LoraRunConfig()


def load_lora_run_config(path: str | Path) -> LoraRunConfig:
    config_path = resolve_existing_project_path(path)
    raw = _load_raw_config(config_path)
    defaults = default_lora_run_config().to_dict()
    unknown = sorted(set(raw) - set(defaults) - {"extends"})
    if unknown:
        raise ValueError(f"Unknown LoRA config keys: {unknown}")

    merged = dict(defaults)
    merged.update(raw)
    merged["run_tags"] = [str(tag) for tag in merged.get("run_tags", [])]
    merged["extra_args"] = [str(arg) for arg in merged.get("extra_args", [])]
    merged["mlx_lora_args"] = dict(merged.get("mlx_lora_args", {}))
    merged["subset_training"] = dict(merged.get("subset_training", {}))
    merged["evaluation"] = dict(merged.get("evaluation", {}))
    return LoraRunConfig(**merged)


def save_lora_run_config(config: LoraRunConfig, path: str | Path) -> None:
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


def build_run_name(config: LoraRunConfig, created_at: datetime | None = None) -> str:
    created_at = created_at or datetime.now().astimezone()
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    model_slug = _slugify(config.base_model.rsplit("/", 1)[-1])
    dataset_slug = _slugify(config.dataset_name)
    parts = [timestamp, model_slug, dataset_slug]

    lora_parameters = config.mlx_lora_args.get("lora_parameters", {})
    if isinstance(lora_parameters, dict) and "rank" in lora_parameters:
        rank = _coerce_int(lora_parameters["rank"])
        if rank is not None:
            parts.append(f"r{rank}")

    num_layers = _coerce_int(config.mlx_lora_args.get("num_layers"))
    if num_layers is not None:
        parts.append(f"l{num_layers}")

    max_seq_length = _coerce_int(config.mlx_lora_args.get("max_seq_length"))
    if max_seq_length is not None:
        parts.append(f"seq{max_seq_length}")

    seed = _coerce_int(config.mlx_lora_args.get("seed"))
    if seed is not None:
        parts.append(f"s{seed}")

    return "__".join(parts)


def prepare_run(
    config: LoraRunConfig,
    run_name: str,
) -> RunPaths:
    results_root = resolve_project_path(config.results_root)
    run_dir = results_root / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    adapter_dir = results_root / "adapters" / run_name
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        logs_dir=logs_dir,
        train_log=logs_dir / "train.log",
        test_log=logs_dir / "test.log",
        metrics_path=run_dir / "metrics.jsonl",
        metadata_path=run_dir / "metadata.json",
        resolved_config_path=run_dir / "resolved_config.yaml",
        summary_path=run_dir / "summary.json",
        command_path=run_dir / "command.txt",
        mlx_config_path=run_dir / "mlx_config.yaml",
    )


def build_run_metadata(
    config: LoraRunConfig,
    paths: RunPaths,
    config_source: str | Path,
    status: str,
) -> dict[str, Any]:
    return {
        "run_name": paths.run_name,
        "status": status,
        "created_at": _now_iso(),
        "config_source": str(Path(config_source)),
        "dataset_name": config.dataset_name,
        "task": config.task,
        "base_model": config.base_model,
        "prepared_data_dir": str(resolve_project_path(config.prepared_data_dir)),
        "results_root": config.results_root,
        "adapter_dir": str(paths.adapter_dir),
        "mlx_config_path": str(paths.mlx_config_path),
        "mlx_command": config.mlx_command,
        "run_tags": list(config.run_tags),
        "evaluation": config.evaluation,
        "notes": config.notes,
        "mlx_lora_args": config.mlx_lora_args,
        "subset_training": config.subset_training,
        "extra_args": config.extra_args,
    }


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _flag_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _format_cli_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def normalize_mlx_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw_args.items():
        normalized[MLX_ARG_ALIASES.get(key, key)] = value
    return normalized


def normalize_lora_parameters(raw_parameters: Any) -> Any:
    if not isinstance(raw_parameters, dict):
        return raw_parameters

    normalized = dict(raw_parameters)
    rank = _coerce_int(normalized.get("rank"))
    scale = normalized.get("scale")
    alpha = normalized.pop("alpha", None)

    if scale is None:
        if alpha is not None:
            alpha_value = _coerce_float(alpha)
            if alpha_value is not None:
                # Inference: treat legacy alpha as the common LoRA alpha and
                # convert to MLX's explicit scale parameter.
                normalized["scale"] = (
                    alpha_value / rank
                    if rank is not None and rank != 0
                    else alpha_value
                )
        else:
            normalized["scale"] = 20.0

    if normalized.get("dropout") is None:
        normalized["dropout"] = 0.0

    return normalized


def build_mlx_config_payload(config: LoraRunConfig) -> dict[str, Any]:
    reserved = {"model", "data", "adapter_path", "train", "test", "config"}
    normalized_args = normalize_mlx_args(config.mlx_lora_args)
    payload: dict[str, Any] = {}

    for key, value in normalized_args.items():
        if key in reserved or value is None:
            continue
        if key in MLX_CLI_SUPPORTED_ARGS:
            continue
        if key == "lora_parameters":
            payload[key] = normalize_lora_parameters(value)
            continue
        payload[key] = value
    return payload


def write_mlx_runtime_config(path: str | Path, payload: dict[str, Any]) -> None:
    if not payload:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to save MLX runtime configs. Install `pyyaml` first."
        )
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def build_train_command(
    config: LoraRunConfig,
    paths: RunPaths,
    prepared_data_dir: str | Path | None = None,
) -> list[str]:
    resolved_data_dir = (
        resolve_project_path(prepared_data_dir)
        if prepared_data_dir is not None
        else resolve_project_path(config.prepared_data_dir)
    )
    command = [
        config.mlx_command,
        "--train",
        "--model",
        config.base_model,
        "--data",
        str(resolved_data_dir),
        "--adapter-path",
        str(paths.adapter_dir),
    ]

    normalized_args = normalize_mlx_args(config.mlx_lora_args)
    config_payload = build_mlx_config_payload(config)
    if config_payload:
        command.extend(["-c", str(paths.mlx_config_path)])

    reserved = {"model", "data", "adapter_path", "train", "test", "config"}
    for key, value in normalized_args.items():
        if key in reserved or value is None:
            continue
        if key not in MLX_CLI_SUPPORTED_ARGS:
            continue
        if isinstance(value, bool):
            if value:
                command.append(_flag_name(key))
            continue
        command.extend([_flag_name(key), _format_cli_value(value)])

    command.extend(config.extra_args)
    return command


def build_test_command(
    config: LoraRunConfig,
    paths: RunPaths,
    prepared_data_dir: str | Path | None = None,
) -> list[str]:
    resolved_data_dir = (
        resolve_project_path(prepared_data_dir)
        if prepared_data_dir is not None
        else resolve_project_path(config.prepared_data_dir)
    )
    return [
        config.mlx_command,
        "--test",
        "--model",
        config.base_model,
        "--data",
        str(resolved_data_dir),
        "--adapter-path",
        str(paths.adapter_dir),
    ]


def write_command(path: str | Path, command: list[str]) -> None:
    path = Path(path)
    path.write_text(shlex.join(command) + "\n")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def write_jsonl_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Selector output rows must be JSON objects.")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_sample_selector(selector_path: str | Path) -> Any:
    selector_path = resolve_existing_project_path(selector_path)
    module_name = f"lora_sample_selector_{selector_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, selector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import selector from: {selector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "select_samples"):
        raise ValueError(
            f"Selector file must define `select_samples`: {selector_path}"
        )
    return module.select_samples


def _call_sample_selector(
    selector: Any,
    rows: list[dict[str, Any]],
    *,
    subset_size: int | None,
    context: dict[str, Any],
) -> Any:
    try:
        signature = inspect.signature(selector)
    except (TypeError, ValueError):
        return selector(rows, subset_size=subset_size)

    parameters = signature.parameters.values()
    accepts_context = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "context"
        for parameter in parameters
    )
    if accepts_context:
        return selector(rows, subset_size=subset_size, context=context)
    return selector(rows, subset_size=subset_size)


def prepare_sampled_data_dir(
    *,
    source_data_dir: str | Path,
    run_dir: str | Path,
    sample_selector: str | Path,
    subset_size: int | None = None,
    selector_context: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    source_data_dir = resolve_project_path(source_data_dir)
    run_dir = Path(run_dir)
    sampled_data_dir = run_dir / "data"
    sampled_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = source_data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")
    for split in ("valid", "test"):
        split_path = source_data_dir / f"{split}.jsonl"
        if split_path.exists():
            shutil.copy2(split_path, sampled_data_dir / split_path.name)

    rows = load_jsonl_records(train_path)
    selector = _load_sample_selector(sample_selector)
    context = {
        "source_data_dir": str(source_data_dir),
        "run_dir": str(run_dir),
        "sampled_data_dir": str(sampled_data_dir),
        **dict(selector_context or {}),
    }
    selected_rows = _call_sample_selector(
        selector,
        rows,
        subset_size=subset_size,
        context=context,
    )
    if selected_rows is None:
        raise ValueError("Selector returned None; expected iterable of rows.")
    selected_rows = list(selected_rows)
    write_jsonl_records(sampled_data_dir / "train.jsonl", selected_rows)

    selector_path = resolve_existing_project_path(sample_selector)
    metadata = {
        "selector_path": str(selector_path),
        "subset_size": subset_size,
        "original_train_examples": len(rows),
        "selected_train_examples": len(selected_rows),
        "sampled_data_dir": str(sampled_data_dir),
        "selector_context": context,
    }
    return sampled_data_dir, metadata


def parse_mlx_log_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None

    train_match = TRAIN_RE.match(text)
    if train_match:
        groups = train_match.groupdict()
        return {
            "event": "train",
            "step": _coerce_int(groups["step"]),
            "train_loss": _coerce_float(groups["train_loss"]),
            "learning_rate": _coerce_float(groups.get("learning_rate")),
            "it_per_sec": _coerce_float(groups.get("it_per_sec")),
            "tokens_per_sec": _coerce_float(groups.get("tokens_per_sec")),
            "trained_tokens": _coerce_int(groups.get("trained_tokens")),
            "peak_mem_gb": _coerce_float(groups.get("peak_mem_gb")),
        }

    val_match = VAL_RE.match(text)
    if val_match:
        groups = val_match.groupdict()
        return {
            "event": "val",
            "step": _coerce_int(groups["step"]),
            "val_loss": _coerce_float(groups["val_loss"]),
            "val_seconds": _coerce_float(groups.get("val_seconds")),
        }

    test_match = TEST_RE.match(text)
    if test_match:
        groups = test_match.groupdict()
        return {
            "event": "test",
            "test_loss": _coerce_float(groups["test_loss"]),
            "test_ppl": _coerce_float(groups["test_ppl"]),
        }
    return None


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def run_command_with_logging(
    command: list[str],
    log_path: str | Path,
    metrics_path: str | Path,
    source_phase: str,
) -> int:
    log_path = Path(log_path)
    metrics_path = Path(metrics_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            f"Command not found: {command[0]}. Install the MLX-LM CLI first."
        ) from exc

    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            parsed = parse_mlx_log_line(line)
            if parsed is not None:
                parsed["timestamp"] = _now_iso()
                parsed["source_phase"] = source_phase
                append_jsonl(metrics_path, parsed)
    return process.wait()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_metric_events(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _best_val_loss(events: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    best_loss: float | None = None
    best_step: int | None = None
    for event in events:
        if event.get("event") != "val":
            continue
        value = event.get("val_loss")
        if value is None:
            continue
        if best_loss is None or value < best_loss:
            best_loss = value
            best_step = event.get("step")
    return best_loss, best_step


def summarize_metric_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    train_events = [event for event in events if event.get("event") == "train"]
    test_events = [event for event in events if event.get("event") == "test"]
    best_val, best_val_step = _best_val_loss(events)

    last_train = train_events[-1] if train_events else {}
    last_test = test_events[-1] if test_events else {}
    peak_mem_values = [
        float(event["peak_mem_gb"])
        for event in train_events
        if event.get("peak_mem_gb") is not None
    ]

    return {
        "last_train_step": last_train.get("step"),
        "last_train_loss": last_train.get("train_loss"),
        "last_learning_rate": last_train.get("learning_rate"),
        "last_it_per_sec": last_train.get("it_per_sec"),
        "last_tokens_per_sec": last_train.get("tokens_per_sec"),
        "trained_tokens": last_train.get("trained_tokens"),
        "peak_mem_gb": max(peak_mem_values) if peak_mem_values else None,
        "best_val_loss": best_val,
        "best_val_step": best_val_step,
        "test_loss": last_test.get("test_loss"),
        "test_ppl": last_test.get("test_ppl"),
    }


def build_summary(
    config: LoraRunConfig,
    paths: RunPaths,
    metadata: dict[str, Any],
    status: str,
    train_exit_code: int | None = None,
    test_exit_code: int | None = None,
) -> dict[str, Any]:
    summary = {
        "run_name": paths.run_name,
        "status": status,
        "created_at": metadata["created_at"],
        "updated_at": _now_iso(),
        "dataset_name": config.dataset_name,
        "task": config.task,
        "base_model": config.base_model,
        "prepared_data_dir": metadata.get(
            "train_prepared_data_dir",
            config.prepared_data_dir,
        ),
        "adapter_dir": str(paths.adapter_dir),
        "run_tags": list(config.run_tags),
        "evaluation": config.evaluation,
        "notes": config.notes,
        "train_exit_code": train_exit_code,
        "test_exit_code": test_exit_code,
        "metrics": summarize_metric_events(load_metric_events(paths.metrics_path)),
    }
    if "sampling" in metadata:
        summary["sampling"] = metadata["sampling"]
    return summary


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
