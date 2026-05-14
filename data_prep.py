from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Callable

from paths import resolve_existing_project_path, resolve_project_path

try:
    import yaml
except ImportError:  # pragma: no cover - project depends on pyyaml
    yaml = None


RecordFormatter = Callable[[dict[str, Any]], dict[str, str]]

TOP_LEVEL_KEYS = {
    "dataset_name",
    "upstream_dataset",
    "output_dir",
    "source",
    "split",
    "prompt_completion_mapping",
    "filters",
    "metadata",
}
SOURCE_KEYS = {
    "type",
    "path",
    "dataset",
    "name",
    "config",
    "split",
    "files",
    "splits",
    "max_examples",
    "append",
    "data_files",
    "revision",
    "streaming",
    "trust_remote_code",
}
SPLIT_KEYS = {
    "strategy",
    "seed",
    "valid_ratio",
    "test_ratio",
    "ratios",
    "max_examples_per_split",
}
PROMPT_COMPLETION_MAPPING_KEYS = {
    "type",
    "computed_fields",
    "prompt_template",
    "prompt_parts",
    "prompt_joiner",
    "completion_template",
}
COMPUTED_FIELD_OPS = {"first_non_empty", "format_choices", "lookup_index"}
FORMAT_CHOICES_KEYS = {"text_field", "label_field", "item_format", "joiner"}
LOOKUP_INDEX_KEYS = {"list", "labels", "index", "as_letter"}
_ABC_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FILTER_KEYS = {
    "max_prompt_chars",
    "max_completion_chars",
    "max_total_chars",
    "token_supervision",
    "exclude_prompt_hashes",
}
TOKEN_FILTER_KEYS = {
    "enabled",
    "tokenizer_model",
    "max_seq_length",
    "min_supervised_tokens",
}
EXCLUDE_PROMPT_HASH_KEYS = {"sources"}
SOURCE_TYPES = {"hf", "jsonl", "json", "csv"}
SPLIT_STRATEGIES = {
    "ratios",
    "train_valid_test",
    "train_valid",
    "existing",
    "train_valid_existing_test",
}


@dataclass(frozen=True)
class TokenSupervisionFilter:
    enabled: bool = False
    tokenizer_model: str | None = None
    max_seq_length: int | None = None
    min_supervised_tokens: int | None = None
    tokenizer: Any | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tokenizer_model": self.tokenizer_model,
            "max_seq_length": self.max_seq_length,
            "min_supervised_tokens": self.min_supervised_tokens,
        }


@dataclass(frozen=True)
class WriteFilters:
    max_prompt_chars: int | None = None
    max_completion_chars: int | None = None
    max_total_chars: int | None = None
    excluded_prompt_hashes: frozenset[str] = field(default_factory=frozenset)
    token_supervision: TokenSupervisionFilter = field(
        default_factory=TokenSupervisionFilter
    )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "max_prompt_chars": self.max_prompt_chars,
            "max_completion_chars": self.max_completion_chars,
            "max_total_chars": self.max_total_chars,
            "excluded_prompt_hashes": len(self.excluded_prompt_hashes),
            "token_supervision": self.token_supervision.to_metadata(),
        }


@dataclass(frozen=True)
class WriteResult:
    count: int
    stats: dict[str, int]


class MemoryDataset:
    """Small in-memory dataset with the subset of HF Dataset APIs used here."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def shuffle(self, seed: int) -> MemoryDataset:
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return MemoryDataset(rows)

    def select(self, indices: Any) -> MemoryDataset:
        return MemoryDataset([self.rows[index] for index in indices])

    def train_test_split(
        self,
        test_size: float | int,
        seed: int,
    ) -> dict[str, MemoryDataset]:
        if len(self.rows) < 2:
            raise ValueError("At least two rows are required to split a dataset.")
        if isinstance(test_size, float):
            if not 0 < test_size < 1:
                raise ValueError("test_size as a ratio must be between 0 and 1.")
            test_count = int(round(len(self.rows) * test_size))
        else:
            test_count = int(test_size)
        test_count = max(1, min(len(self.rows) - 1, test_count))
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return {
            "train": MemoryDataset(rows[:-test_count]),
            "test": MemoryDataset(rows[-test_count:]),
        }


def require_datasets() -> tuple[Any, Any]:
    try:
        from datasets import concatenate_datasets, load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        if exc.name == "_lzma":
            raise RuntimeError(
                "Your current Python interpreter is missing the `_lzma` extension, "
                "so Hugging Face `datasets` cannot import. Rebuild the pyenv "
                "Python with xz/lzma support, then recreate `.venv`."
            ) from exc
        raise RuntimeError(
            "The `datasets` package is required for data prep. "
            "Install project dependencies with `uv sync`."
        ) from exc
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "The `datasets` package is required for data prep. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return load_dataset, concatenate_datasets


def require_mlx_tokenizer_loader() -> Any:
    try:
        from mlx_lm.utils import load_tokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "The `mlx-lm` package is required for token-aware filtering. "
            "Install project dependencies with `uv sync`."
        ) from exc
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "The `mlx-lm` package is required for token-aware filtering. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return load_tokenizer


def ensure_dir(path: str | Path) -> Path:
    output_dir = resolve_project_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_text(value: Any | None) -> str:
    return "" if value is None else str(value).strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_prompt_hashes_from_jsonl(paths: list[str | Path]) -> frozenset[str]:
    hashes: set[str] = set()
    for raw_path in paths:
        path = resolve_project_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt exclusion source not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in prompt exclusion source {path}:"
                        f"{line_number}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Prompt exclusion source {path}:{line_number} "
                        "must contain JSON objects."
                    )
                hashes.add(prompt_hash(normalize_text(payload.get("prompt"))))
    return frozenset(hashes)


def normalize_limit(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def count_masked_completion_tokens(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_seq_length: int,
) -> dict[str, int]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full_tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        return_dict=False,
    )
    full_length = len(full_tokens)
    prompt_length = len(prompt_tokens)
    truncated_length = min(full_length, max_seq_length)
    supervised_tokens = max(0, truncated_length - prompt_length)
    return {
        "full_length": full_length,
        "prompt_length": prompt_length,
        "truncated_length": truncated_length,
        "supervised_tokens": supervised_tokens,
    }


def maybe_take_subset(dataset: Any, max_examples: int | None, seed: int) -> Any:
    if max_examples is None:
        return dataset
    max_examples = max(0, int(max_examples))
    if len(dataset) <= max_examples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_examples))


def split_train_valid_test(
    dataset: Any,
    seed: int,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> tuple[Any, Any, Any]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1.")

    first_split = dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_valid = first_split["train"]
    test = first_split["test"]
    adjusted_valid_ratio = valid_ratio / (1.0 - test_ratio)
    second_split = train_valid.train_test_split(
        test_size=adjusted_valid_ratio,
        seed=seed,
    )
    return second_split["train"], second_split["test"], test


def split_train_and_valid(
    dataset: Any,
    seed: int,
    valid_ratio: float = 0.1,
) -> tuple[Any, Any]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


def split_name(dataset_dict: Any, *names: str) -> Any:
    for name in names:
        if name in dataset_dict:
            return dataset_dict[name]
    raise KeyError(f"None of the splits exist: {names}")


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


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be a mapping.")
    return value


def _expect_unknown_keys(
    value: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in `{location}`: {unknown}")


def _load_config_file(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    visited = visited or set()
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Data prep config inheritance cycle detected at {path}")
    visited.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Data prep config not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML data prep configs.")
        raw = yaml.safe_load(path.read_text()) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError(
            f"Unsupported data prep config extension: {path.suffix}. "
            "Use .yaml/.yml/.json."
        )
    if not isinstance(raw, dict):
        raise ValueError(f"Data prep config must be an object: {path}")

    extends = raw.get("extends")
    if extends is None:
        return raw
    parent = _load_config_file(path.parent / str(extends), visited)
    return _deep_merge(parent, raw)


def load_data_prep_config(path: str | Path) -> dict[str, Any]:
    return _load_config_file(resolve_existing_project_path(path))


def _validate_source_config(
    source: dict[str, Any],
    *,
    location: str = "source",
    allow_append: bool = True,
) -> None:
    _expect_unknown_keys(source, SOURCE_KEYS, location)
    source_type = str(source.get("type", "hf")).lower()
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unsupported `{location}.type`: {source_type}")
    if source_type == "hf" and not (source.get("path") or source.get("dataset")):
        raise ValueError(f"`{location}.path` is required for Hugging Face sources.")
    if source_type in {"jsonl", "json", "csv"}:
        has_path = source.get("path") is not None
        has_files = source.get("files") is not None
        if has_path == has_files:
            raise ValueError(f"`{location}` must set exactly one of `path` or `files`.")
        if has_files:
            _expect_mapping(source["files"], f"{location}.files")
    splits = source.get("splits")
    if splits is not None:
        _expect_mapping(splits, f"{location}.splits")
    if "max_examples" in source:
        normalize_limit(source["max_examples"])
    append = source.get("append")
    if append is None:
        return
    if not allow_append:
        raise ValueError(
            f"`{location}.append` is only supported on the primary source."
        )
    append = _expect_mapping(append, f"{location}.append")
    for split, sources in append.items():
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"`{location}.append.{split}` must be a non-empty list.")
        for index, append_source in enumerate(sources):
            append_source = _expect_mapping(
                append_source,
                f"{location}.append.{split}[{index}]",
            )
            _validate_source_config(
                append_source,
                location=f"{location}.append.{split}[{index}]",
                allow_append=False,
            )


def _validate_template_syntax(template: str, location: str) -> None:
    try:
        list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"Invalid format template at `{location}`: {exc}") from exc


def _validate_prompt_completion_mapping_config(
    prompt_completion_mapping: dict[str, Any],
) -> None:
    _expect_unknown_keys(
        prompt_completion_mapping,
        PROMPT_COMPLETION_MAPPING_KEYS,
        "prompt_completion_mapping",
    )
    mapping_type = str(prompt_completion_mapping.get("type", "template"))
    if mapping_type != "template":
        raise ValueError(
            "Only `prompt_completion_mapping.type: template` is supported."
        )
    if not isinstance(prompt_completion_mapping.get("completion_template"), str):
        raise ValueError("`prompt_completion_mapping.completion_template` is required.")
    _validate_template_syntax(
        prompt_completion_mapping["completion_template"],
        "prompt_completion_mapping.completion_template",
    )
    prompt_template = prompt_completion_mapping.get("prompt_template")
    prompt_parts = prompt_completion_mapping.get("prompt_parts")
    if not isinstance(prompt_template, str) and not isinstance(prompt_parts, list):
        raise ValueError(
            "`prompt_completion_mapping` requires either `prompt_template` or `prompt_parts`."
        )
    if isinstance(prompt_template, str):
        _validate_template_syntax(
            prompt_template,
            "prompt_completion_mapping.prompt_template",
        )
    if isinstance(prompt_parts, list):
        for index, part in enumerate(prompt_parts):
            location = f"prompt_completion_mapping.prompt_parts[{index}]"
            if isinstance(part, str):
                _validate_template_syntax(part, location)
                continue
            part = _expect_mapping(part, location)
            _expect_unknown_keys(
                part,
                {"template", "when_field", "when_any", "when_all"},
                location,
            )
            if not isinstance(part.get("template"), str):
                raise ValueError(f"`{location}.template` is required.")
            _validate_template_syntax(part["template"], f"{location}.template")
    computed_fields = prompt_completion_mapping.get("computed_fields") or {}
    computed_fields = _expect_mapping(
        computed_fields,
        "prompt_completion_mapping.computed_fields",
    )
    for name, spec in computed_fields.items():
        _validate_computed_field(name, spec)


def _validate_computed_field(name: str, spec: Any) -> None:
    spec = _expect_mapping(spec, f"prompt_completion_mapping.computed_fields.{name}")
    if len(spec) != 1:
        raise ValueError(
            f"`prompt_completion_mapping.computed_fields.{name}` must specify exactly one operation."
        )
    ((op_name, op_spec),) = spec.items()
    if op_name not in COMPUTED_FIELD_OPS:
        raise ValueError(
            f"Unknown computed_fields op `{op_name}` in "
            f"`prompt_completion_mapping.computed_fields.{name}`. "
            f"Supported: {sorted(COMPUTED_FIELD_OPS)}"
        )
    if op_name == "first_non_empty":
        if not isinstance(op_spec, list) or not op_spec:
            raise ValueError(
                f"`prompt_completion_mapping.computed_fields.{name}.first_non_empty` "
                "must be a non-empty list."
            )
        if not all(isinstance(source, str) for source in op_spec):
            raise ValueError(
                f"`prompt_completion_mapping.computed_fields.{name}.first_non_empty` "
                "must contain only strings."
            )
    elif op_name == "format_choices":
        op_spec = _expect_mapping(
            op_spec, f"prompt_completion_mapping.computed_fields.{name}.format_choices"
        )
        _expect_unknown_keys(
            op_spec,
            FORMAT_CHOICES_KEYS,
            f"prompt_completion_mapping.computed_fields.{name}.format_choices",
        )
        text_field = op_spec.get("text_field")
        if isinstance(text_field, list):
            if not text_field or not all(isinstance(f, str) for f in text_field):
                raise ValueError(
                    f"`prompt_completion_mapping.computed_fields.{name}.format_choices.text_field` "
                    "as a list must be a non-empty list of dotted-path strings."
                )
        elif not isinstance(text_field, str):
            raise ValueError(
                f"`prompt_completion_mapping.computed_fields.{name}.format_choices.text_field` "
                "must be a dotted-path string or a list of dotted-path strings."
            )
    elif op_name == "lookup_index":
        op_spec = _expect_mapping(
            op_spec, f"prompt_completion_mapping.computed_fields.{name}.lookup_index"
        )
        _expect_unknown_keys(
            op_spec,
            LOOKUP_INDEX_KEYS,
            f"prompt_completion_mapping.computed_fields.{name}.lookup_index",
        )
        if not isinstance(op_spec.get("index"), str):
            raise ValueError(
                f"`prompt_completion_mapping.computed_fields.{name}.lookup_index.index` is required."
            )
        has_list = isinstance(op_spec.get("list"), str)
        has_labels = isinstance(op_spec.get("labels"), list)
        if not op_spec.get("as_letter", False) and not (has_list or has_labels):
            raise ValueError(
                f"`prompt_completion_mapping.computed_fields.{name}.lookup_index` requires one of "
                "`list`, `labels`, or `as_letter: true`."
            )


def _validate_filter_config(filters: dict[str, Any]) -> None:
    _expect_unknown_keys(filters, FILTER_KEYS, "filters")
    for key in ("max_prompt_chars", "max_completion_chars", "max_total_chars"):
        if key in filters:
            normalize_limit(filters[key])
    exclude_prompt_hashes = filters.get("exclude_prompt_hashes")
    if exclude_prompt_hashes is not None:
        exclude_prompt_hashes = _expect_mapping(
            exclude_prompt_hashes,
            "filters.exclude_prompt_hashes",
        )
        _expect_unknown_keys(
            exclude_prompt_hashes,
            EXCLUDE_PROMPT_HASH_KEYS,
            "filters.exclude_prompt_hashes",
        )
        sources = exclude_prompt_hashes.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                "`filters.exclude_prompt_hashes.sources` must be a non-empty list."
            )
        for index, source in enumerate(sources):
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "`filters.exclude_prompt_hashes.sources"
                    f"[{index}]` must be a non-empty path string."
                )
    token_filter = filters.get("token_supervision")
    if token_filter is None:
        return
    token_filter = _expect_mapping(token_filter, "filters.token_supervision")
    _expect_unknown_keys(token_filter, TOKEN_FILTER_KEYS, "filters.token_supervision")
    if not token_filter.get("enabled", False):
        return
    if not token_filter.get("tokenizer_model"):
        raise ValueError(
            "`filters.token_supervision.tokenizer_model` is required when enabled."
        )
    if normalize_limit(token_filter.get("max_seq_length")) is None:
        raise ValueError(
            "`filters.token_supervision.max_seq_length` is required when enabled."
        )
    if normalize_limit(token_filter.get("min_supervised_tokens")) is None:
        raise ValueError(
            "`filters.token_supervision.min_supervised_tokens` is required "
            "when enabled."
        )


def validate_data_prep_config(config: dict[str, Any]) -> None:
    _expect_unknown_keys(config, TOP_LEVEL_KEYS, "data prep config")
    if "dataset_name" not in config:
        raise ValueError("`dataset_name` is required.")
    source = _expect_mapping(config.get("source"), "source")
    _validate_source_config(source)
    split_config = _expect_mapping(config.get("split") or {}, "split")
    _expect_unknown_keys(split_config, SPLIT_KEYS, "split")
    if "splits" in split_config:
        raise ValueError("Use `source.splits`; `split.splits` is not supported.")
    strategy = split_config.get("strategy")
    if strategy is not None and str(strategy) not in SPLIT_STRATEGIES:
        raise ValueError(f"Unsupported `split.strategy`: {strategy}")
    for key in ("valid_ratio", "test_ratio"):
        if key in split_config:
            float(split_config[key])
    if "ratios" in split_config:
        ratios = _expect_mapping(split_config["ratios"], "split.ratios")
        for key in ("train", "valid", "test"):
            if key in ratios:
                float(ratios[key])
    if "max_examples_per_split" in split_config:
        _expect_mapping(
            split_config["max_examples_per_split"],
            "split.max_examples_per_split",
        )
    _validate_prompt_completion_mapping_config(
        _expect_mapping(
            config.get("prompt_completion_mapping"),
            "prompt_completion_mapping",
        )
    )
    _validate_filter_config(_expect_mapping(config.get("filters") or {}, "filters"))
    if "metadata" in config:
        _expect_mapping(config["metadata"], "metadata")


def _normalize_template_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _render_template(
    template: str,
    row: dict[str, str],
    template_name: str,
) -> str:
    try:
        return template.format_map(row)
    except KeyError as exc:
        field = exc.args[0]
        raise ValueError(
            f"Missing field `{field}` while rendering `{template_name}`."
        ) from exc


def _should_render_part(part: dict[str, Any], row: dict[str, str]) -> bool:
    when_field = part.get("when_field")
    if when_field is not None and not row.get(str(when_field)):
        return False
    when_any = part.get("when_any")
    if when_any is not None and not any(row.get(str(field)) for field in when_any):
        return False
    when_all = part.get("when_all")
    if when_all is not None and not all(row.get(str(field)) for field in when_all):
        return False
    return True


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _compute_first_non_empty(example: dict[str, Any], sources: list[str]) -> Any:
    for source in sources:
        value = _resolve_path(example, source)
        if not _is_empty(value):
            return value.strip() if isinstance(value, str) else value
    return ""


def _compute_format_choices(example: dict[str, Any], spec: dict[str, Any]) -> str:
    text_field = spec["text_field"]
    if isinstance(text_field, list):
        texts = [_resolve_path(example, path) for path in text_field]
    else:
        texts = _resolve_path(example, text_field)
    if not isinstance(texts, list):
        return ""
    label_field = spec.get("label_field")
    labels: list[Any] | None = None
    if label_field:
        resolved = _resolve_path(example, label_field)
        if isinstance(resolved, list):
            labels = resolved
    if labels is None or len(labels) != len(texts):
        labels = [_ABC_LABELS[i % len(_ABC_LABELS)] for i in range(len(texts))]
    item_format = str(spec.get("item_format", "{label}. {text}"))
    joiner = str(spec.get("joiner", "\n"))
    return joiner.join(
        item_format.format(label=str(label), text=_normalize_template_value(text))
        for label, text in zip(labels, texts)
    )


def _compute_lookup_index(example: dict[str, Any], spec: dict[str, Any]) -> str:
    raw_index = _resolve_path(example, spec["index"])
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        if isinstance(raw_index, str) and len(raw_index) == 1 and raw_index.isalpha():
            index = _ABC_LABELS.index(raw_index.upper())
        else:
            return ""
    if spec.get("as_letter", False):
        if 0 <= index < len(_ABC_LABELS):
            return _ABC_LABELS[index]
        return ""
    inline_labels = spec.get("labels")
    if isinstance(inline_labels, list):
        if 0 <= index < len(inline_labels):
            return _normalize_template_value(inline_labels[index])
        return ""
    list_field = spec.get("list")
    if not list_field:
        return ""
    items = _resolve_path(example, list_field)
    if isinstance(items, list) and 0 <= index < len(items):
        return _normalize_template_value(items[index])
    return ""


def _apply_computed_fields(
    example: dict[str, Any],
    row: dict[str, str],
    computed_fields: dict[str, Any],
) -> dict[str, str]:
    row = dict(row)
    for name, spec in computed_fields.items():
        ((op_name, op_spec),) = spec.items()
        if op_name == "first_non_empty":
            value = _compute_first_non_empty(example, op_spec)
        elif op_name == "format_choices":
            value = _compute_format_choices(example, op_spec)
        elif op_name == "lookup_index":
            value = _compute_lookup_index(example, op_spec)
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"Unknown computed_fields op `{op_name}`.")
        row[str(name)] = _normalize_template_value(value)
    return row


def build_record_formatter(
    prompt_completion_mapping_config: dict[str, Any],
) -> RecordFormatter:
    prompt_completion_mapping_config = _expect_mapping(
        prompt_completion_mapping_config,
        "prompt_completion_mapping",
    )
    if str(prompt_completion_mapping_config.get("type", "template")) != "template":
        raise ValueError(
            "Only `prompt_completion_mapping.type: template` is supported."
        )

    completion_template = str(
        prompt_completion_mapping_config["completion_template"]
    )
    prompt_template = prompt_completion_mapping_config.get("prompt_template")
    prompt_parts = prompt_completion_mapping_config.get("prompt_parts")
    prompt_joiner = str(prompt_completion_mapping_config.get("prompt_joiner", "\n\n"))
    computed_fields = dict(prompt_completion_mapping_config.get("computed_fields") or {})

    def formatter(example: dict[str, Any]) -> dict[str, str]:
        row = {
            str(key): _normalize_template_value(value) for key, value in example.items()
        }
        row = _apply_computed_fields(example, row, computed_fields)
        if isinstance(prompt_template, str):
            prompt = _render_template(prompt_template, row, "prompt_template")
        else:
            rendered_parts: list[str] = []
            for index, part in enumerate(prompt_parts or []):
                if isinstance(part, str):
                    part_template = part
                else:
                    if not _should_render_part(part, row):
                        continue
                    part_template = part["template"]
                rendered_parts.append(
                    _render_template(part_template, row, f"prompt_parts[{index}]")
                )
            prompt = prompt_joiner.join(part for part in rendered_parts if part)

        return {
            "prompt": prompt,
            "completion": _render_template(
                completion_template,
                row,
                "completion_template",
            ),
        }

    return formatter


def _read_jsonl(path: str | Path) -> MemoryDataset:
    input_path = resolve_existing_project_path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number} is not a JSON object.")
            rows.append(row)
    return MemoryDataset(rows)


def _read_json(path: str | Path) -> MemoryDataset:
    input_path = resolve_existing_project_path(path)
    payload = json.loads(input_path.read_text())
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"JSON source must be a list of objects: {input_path}")
    return MemoryDataset(payload)


def _read_csv(path: str | Path) -> MemoryDataset:
    input_path = resolve_existing_project_path(path)
    with input_path.open(newline="", encoding="utf-8") as handle:
        return MemoryDataset([dict(row) for row in csv.DictReader(handle)])


def _load_local_source(source_type: str, source: dict[str, Any]) -> Any:
    loader_by_type = {
        "jsonl": _read_jsonl,
        "json": _read_json,
        "csv": _read_csv,
    }
    loader = loader_by_type[source_type]
    files = source.get("files")
    if files is not None:
        return {str(split): loader(path) for split, path in files.items()}
    return loader(source["path"])


def _load_hf_source(source: dict[str, Any]) -> Any:
    dataset_path = source.get("path") or source.get("dataset")
    load_dataset, _ = require_datasets()
    args = [str(dataset_path)]
    name = source.get("name") or source.get("config")
    if name is not None:
        args.append(str(name))
    kwargs: dict[str, Any] = {}
    for key in ("split", "data_files", "revision", "streaming", "trust_remote_code"):
        if key in source:
            kwargs[key] = source[key]
    return load_dataset(*args, **kwargs)


def load_dataset_from_source(source: dict[str, Any]) -> Any:
    source = _expect_mapping(source, "source")
    source_type = str(source.get("type", "hf")).lower()
    if source_type == "hf":
        return _load_hf_source(source)
    return _load_local_source(source_type, source)


def _is_dataset_mapping(dataset: Any) -> bool:
    return isinstance(dataset, dict) or (
        hasattr(dataset, "keys") and hasattr(dataset, "__getitem__")
    )


def _configured_split_names(source: dict[str, Any]) -> dict[str, Any]:
    splits = source.get("splits") or {}
    return _expect_mapping(splits, "source.splits") if splits else {}


def _select_configured_split(
    dataset_dict: Any,
    split_map: dict[str, Any],
    target: str,
    defaults: tuple[str, ...],
) -> Any:
    names: list[str] = []
    configured = split_map.get(target)
    if configured is not None:
        names.append(str(configured))
    names.extend(default for default in defaults if default not in names)
    return split_name(dataset_dict, *names)


def _concatenate_datasets(datasets: list[Any]) -> Any:
    if all(isinstance(dataset, MemoryDataset) for dataset in datasets):
        rows: list[dict[str, Any]] = []
        for dataset in datasets:
            rows.extend(dataset.rows)
        return MemoryDataset(rows)
    if any(isinstance(dataset, MemoryDataset) for dataset in datasets):
        raise ValueError("Cannot append in-memory datasets to Hugging Face datasets.")
    _, concatenate_datasets = require_datasets()
    return concatenate_datasets(datasets)


def _load_append_dataset(
    source: dict[str, Any],
    target_split: str,
    seed: int,
) -> Any:
    dataset = load_dataset_from_source(source)
    if _is_dataset_mapping(dataset):
        dataset = _select_configured_split(
            dataset,
            _configured_split_names(source),
            target_split,
            (target_split, "train"),
        )
    return maybe_take_subset(dataset, source.get("max_examples"), seed)


def _append_sources_to_splits(
    splits: dict[str, Any],
    source: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    append = source.get("append") or {}
    if not append:
        return splits
    splits = dict(splits)
    for target_split, append_sources in append.items():
        if target_split not in splits:
            raise ValueError(
                f"`source.append.{target_split}` targets a split that does not exist."
            )
        datasets = [splits[target_split]]
        datasets.extend(
            _load_append_dataset(append_source, target_split, seed)
            for append_source in append_sources
        )
        combined = _concatenate_datasets(datasets)
        if hasattr(combined, "shuffle"):
            combined = combined.shuffle(seed=seed)
        splits[target_split] = combined
    return splits


def build_dataset_splits(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    source = _expect_mapping(config.get("source"), "source")
    split_config = dict(config.get("split") or {})
    split_map = _configured_split_names(source)
    strategy = str(
        split_config.get(
            "strategy",
            "existing" if _is_dataset_mapping(dataset) else "ratios",
        )
    )
    seed = int(split_config.get("seed", 42))
    max_examples = source.get("max_examples")

    def split_ratio(key: str, default: float) -> float:
        ratios = split_config.get("ratios") or {}
        legacy_key = f"{key}_ratio"
        if legacy_key in split_config:
            return float(split_config[legacy_key])
        if isinstance(ratios, dict) and key in ratios:
            return float(ratios[key])
        return default

    if strategy in {"ratios", "train_valid_test"}:
        if _is_dataset_mapping(dataset):
            raise ValueError("Ratio splitting requires a single dataset.")
        dataset = maybe_take_subset(dataset, max_examples=max_examples, seed=seed)
        train, valid, test = split_train_valid_test(
            dataset,
            seed=seed,
            valid_ratio=split_ratio("valid", 0.05),
            test_ratio=split_ratio("test", 0.05),
        )
        splits = {"train": train, "valid": valid, "test": test}
    elif strategy == "train_valid":
        if _is_dataset_mapping(dataset):
            raise ValueError("train_valid splitting requires a single dataset.")
        dataset = maybe_take_subset(dataset, max_examples=max_examples, seed=seed)
        train, valid = split_train_and_valid(
            dataset,
            seed=seed,
            valid_ratio=split_ratio("valid", 0.1),
        )
        splits = {"train": train, "valid": valid}
    elif strategy == "existing":
        if not _is_dataset_mapping(dataset):
            raise ValueError("Existing split strategy requires split data.")
        splits = {
            "train": _select_configured_split(dataset, split_map, "train", ("train",)),
            "valid": _select_configured_split(
                dataset,
                split_map,
                "valid",
                ("validation", "valid", "val"),
            ),
            "test": _select_configured_split(dataset, split_map, "test", ("test",)),
        }
        if max_examples is not None:
            splits = {
                name: maybe_take_subset(split_dataset, max_examples, seed)
                for name, split_dataset in splits.items()
            }
    elif strategy == "train_valid_existing_test":
        if not _is_dataset_mapping(dataset):
            raise ValueError("train_valid_existing_test requires split data.")
        train_source = _select_configured_split(
            dataset,
            split_map,
            "train",
            ("train",),
        )
        train_source = maybe_take_subset(
            train_source,
            max_examples=max_examples,
            seed=seed,
        )
        train, valid = split_train_and_valid(
            train_source,
            seed=seed,
            valid_ratio=split_ratio("valid", 0.1),
        )
        splits = {
            "train": train,
            "valid": valid,
            "test": _select_configured_split(
                dataset,
                split_map,
                "test",
                ("test", "validation"),
            ),
        }
    else:
        raise ValueError(f"Unsupported split strategy `{strategy}`.")

    splits = _append_sources_to_splits(splits, source, seed)
    max_examples_per_split = split_config.get("max_examples_per_split") or {}
    return {
        name: maybe_take_subset(
            split_dataset,
            max_examples=max_examples_per_split.get(name),
            seed=seed,
        )
        for name, split_dataset in splits.items()
    }


def _upstream_dataset_name(config: dict[str, Any]) -> str:
    source = _expect_mapping(config.get("source"), "source")
    return str(
        config.get("upstream_dataset")
        or source.get("path")
        or source.get("dataset")
        or source.get("type", "unknown")
    )


def _configured_seed(config: dict[str, Any]) -> int:
    split_config = config.get("split") or {}
    return int(split_config.get("seed", 42))


def _write_filters(config: dict[str, Any]) -> WriteFilters:
    filters = dict(config.get("filters") or {})
    token_filter = dict(filters.get("token_supervision") or {})
    exclude_prompt_hashes = dict(filters.get("exclude_prompt_hashes") or {})
    excluded_prompt_hashes = frozenset()
    if exclude_prompt_hashes:
        excluded_prompt_hashes = load_prompt_hashes_from_jsonl(
            [str(source) for source in exclude_prompt_hashes.get("sources") or []]
        )
    tokenizer = None
    token_supervision = TokenSupervisionFilter()
    if token_filter.get("enabled", False):
        load_tokenizer = require_mlx_tokenizer_loader()
        tokenizer_model = str(token_filter["tokenizer_model"])
        tokenizer = load_tokenizer(tokenizer_model)
        token_supervision = TokenSupervisionFilter(
            enabled=True,
            tokenizer_model=tokenizer_model,
            max_seq_length=normalize_limit(token_filter.get("max_seq_length")),
            min_supervised_tokens=normalize_limit(
                token_filter.get("min_supervised_tokens")
            ),
            tokenizer=tokenizer,
        )
    return WriteFilters(
        max_prompt_chars=normalize_limit(filters.get("max_prompt_chars")),
        max_completion_chars=normalize_limit(filters.get("max_completion_chars")),
        max_total_chars=normalize_limit(filters.get("max_total_chars")),
        excluded_prompt_hashes=excluded_prompt_hashes,
        token_supervision=token_supervision,
    )


def write_jsonl(
    dataset: Any,
    path: str | Path,
    formatter: RecordFormatter,
    filters: WriteFilters | None = None,
) -> WriteResult:
    filters = filters or WriteFilters()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_filter = filters.token_supervision
    stats: dict[str, int] = {}
    count = 0
    tokenizer: Any = token_filter.tokenizer
    token_max_seq_length = 0
    token_min_supervised_tokens = 0
    if token_filter.enabled:
        if tokenizer is None:
            raise ValueError("Token supervision filtering requires a tokenizer.")
        if token_filter.max_seq_length is None:
            raise ValueError("Token supervision filtering requires max_seq_length.")
        if token_filter.min_supervised_tokens is None:
            raise ValueError(
                "Token supervision filtering requires min_supervised_tokens."
            )
        token_max_seq_length = token_filter.max_seq_length
        token_min_supervised_tokens = token_filter.min_supervised_tokens

    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            record = formatter(row)
            prompt = normalize_text(record.get("prompt"))
            completion = normalize_text(record.get("completion"))
            prompt_len = len(prompt)
            completion_len = len(completion)
            total_len = prompt_len + completion_len
            token_stats = None

            if (
                filters.excluded_prompt_hashes
                and prompt_hash(prompt) in filters.excluded_prompt_hashes
            ):
                stats["skipped"] = stats.get("skipped", 0) + 1
                stats["skipped_excluded_prompt_hash"] = (
                    stats.get("skipped_excluded_prompt_hash", 0) + 1
                )
                continue

            stats["max_prompt_chars_seen"] = max(
                stats.get("max_prompt_chars_seen", 0), prompt_len
            )
            stats["max_completion_chars_seen"] = max(
                stats.get("max_completion_chars_seen", 0), completion_len
            )
            stats["max_total_chars_seen"] = max(
                stats.get("max_total_chars_seen", 0), total_len
            )

            exceeds_limits = (
                (
                    filters.max_prompt_chars is not None
                    and prompt_len > filters.max_prompt_chars
                )
                or (
                    filters.max_completion_chars is not None
                    and completion_len > filters.max_completion_chars
                )
                or (
                    filters.max_total_chars is not None
                    and total_len > filters.max_total_chars
                )
            )
            if token_filter.enabled:
                token_stats = count_masked_completion_tokens(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    completion=completion,
                    max_seq_length=token_max_seq_length,
                )
                stats["max_sequence_tokens_seen"] = max(
                    stats.get("max_sequence_tokens_seen", 0),
                    token_stats["full_length"],
                )
                stats["max_prompt_tokens_seen"] = max(
                    stats.get("max_prompt_tokens_seen", 0),
                    token_stats["prompt_length"],
                )
                if "min_supervised_tokens_seen" in stats:
                    stats["min_supervised_tokens_seen"] = min(
                        stats["min_supervised_tokens_seen"],
                        token_stats["supervised_tokens"],
                    )
                else:
                    stats["min_supervised_tokens_seen"] = token_stats[
                        "supervised_tokens"
                    ]
                if token_stats["full_length"] > token_max_seq_length:
                    stats["truncated_examples"] = stats.get("truncated_examples", 0) + 1
                if token_stats["supervised_tokens"] == 0:
                    stats["zero_supervision_examples"] = (
                        stats.get("zero_supervision_examples", 0) + 1
                    )

            below_supervision_limit = (
                token_filter.enabled
                and token_stats is not None
                and token_stats["supervised_tokens"] < token_min_supervised_tokens
            )

            if exceeds_limits or below_supervision_limit:
                stats["skipped"] = stats.get("skipped", 0) + 1
                if exceeds_limits:
                    stats["skipped_char_limits"] = (
                        stats.get("skipped_char_limits", 0) + 1
                    )
                if below_supervision_limit:
                    stats["skipped_token_supervision"] = (
                        stats.get("skipped_token_supervision", 0) + 1
                    )
                continue

            handle.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return WriteResult(count=count, stats=stats)


def write_metadata(
    output_dir: str | Path,
    dataset_name: str,
    upstream_dataset: str,
    split_counts: dict[str, int],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "dataset_name": dataset_name,
        "upstream_dataset": upstream_dataset,
        "seed": seed,
        "split_counts": split_counts,
    }
    if extra:
        payload["extra"] = extra
    metadata_path = ensure_dir(output_dir) / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, default=str))
    return metadata_path


def prepare_dataset_from_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    tokenizer_model: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, int]:
    config = load_data_prep_config(config_path)
    if overrides:
        config = _deep_merge(config, overrides)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        split_config = dict(config.get("split") or {})
        split_config["seed"] = seed
        config["split"] = split_config
    if tokenizer_model is not None:
        filters = dict(config.get("filters") or {})
        token_filter = filters.get("token_supervision")
        if not isinstance(token_filter, dict) or not token_filter.get("enabled"):
            raise ValueError(
                "--tokenizer-model requires `filters.token_supervision.enabled: true`."
            )
        token_filter["tokenizer_model"] = tokenizer_model
        filters["token_supervision"] = token_filter
        config["filters"] = filters

    validate_data_prep_config(config)
    dataset_name = str(config["dataset_name"])
    target_dir = ensure_dir(config.get("output_dir") or f"data/{dataset_name}")
    source_data = load_dataset_from_source(config["source"])
    splits = build_dataset_splits(source_data, config)
    formatter = build_record_formatter(config["prompt_completion_mapping"])
    filters = _write_filters(config)

    split_counts: dict[str, int] = {}
    filter_stats: dict[str, dict[str, int]] = {}
    for split, split_dataset in splits.items():
        result = write_jsonl(
            split_dataset,
            target_dir / f"{split}.jsonl",
            formatter,
            filters,
        )
        split_counts[split] = result.count
        filter_stats[split] = result.stats

    write_metadata(
        output_dir=target_dir,
        dataset_name=dataset_name,
        upstream_dataset=_upstream_dataset_name(config),
        split_counts=split_counts,
        seed=_configured_seed(config),
        extra={
            "config_path": str(resolve_existing_project_path(config_path)),
            "source": config.get("source", {}),
            "split": config.get("split", {}),
            "prompt_completion_mapping": config.get("prompt_completion_mapping", {}),
            "filters": filters.to_metadata(),
            "filter_stats": filter_stats,
            "metadata": config.get("metadata", {}),
        },
    )
    return split_counts
