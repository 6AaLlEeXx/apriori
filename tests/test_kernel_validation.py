from __future__ import annotations

from pathlib import Path
import json

import pytest

from kernel.config import KernelRunConfig
from kernel.run import validate_kernel_run_inputs
from kernel.scoring import tokenize_pair


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_valid_splits(prepared_data_dir: Path) -> None:
    rows = [{"prompt": "Question?", "completion": "Answer."}]
    for split in ("train", "valid", "test"):
        _write_jsonl(prepared_data_dir / f"{split}.jsonl", rows)


def _write_adapter(adapter_dir: Path, *, with_config: bool = True) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if with_config:
        adapter_dir.joinpath("adapter_config.json").write_text("{}")


def _kernel_config(prepared_data_dir: Path, adapter_dir: Path) -> KernelRunConfig:
    return KernelRunConfig(
        dataset_name="toy",
        prepared_data_dir=str(prepared_data_dir),
        adapter_path=str(adapter_dir),
    )


def test_validate_kernel_inputs_reports_missing_split(tmp_path: Path) -> None:
    prepared_data_dir = tmp_path / "data"
    adapter_dir = tmp_path / "adapter"
    _write_jsonl(prepared_data_dir / "train.jsonl", [{"prompt": "p", "completion": "c"}])
    _write_adapter(adapter_dir)

    with pytest.raises(FileNotFoundError, match="Kernel data split not found"):
        validate_kernel_run_inputs(_kernel_config(prepared_data_dir, adapter_dir))


def test_validate_kernel_inputs_reports_empty_split(tmp_path: Path) -> None:
    prepared_data_dir = tmp_path / "data"
    adapter_dir = tmp_path / "adapter"
    _write_valid_splits(prepared_data_dir)
    (prepared_data_dir / "valid.jsonl").write_text("")
    _write_adapter(adapter_dir)

    with pytest.raises(ValueError, match="Kernel data split is empty"):
        validate_kernel_run_inputs(_kernel_config(prepared_data_dir, adapter_dir))


def test_validate_kernel_inputs_reports_empty_prompt_or_completion(
    tmp_path: Path,
) -> None:
    prepared_data_dir = tmp_path / "data"
    adapter_dir = tmp_path / "adapter"
    _write_valid_splits(prepared_data_dir)
    _write_jsonl(prepared_data_dir / "train.jsonl", [{"prompt": "", "completion": "c"}])
    _write_adapter(adapter_dir)

    with pytest.raises(ValueError, match="empty `prompt`"):
        validate_kernel_run_inputs(_kernel_config(prepared_data_dir, adapter_dir))


def test_validate_kernel_inputs_requires_adapter_config(tmp_path: Path) -> None:
    prepared_data_dir = tmp_path / "data"
    adapter_dir = tmp_path / "adapter"
    _write_valid_splits(prepared_data_dir)
    _write_adapter(adapter_dir, with_config=False)

    with pytest.raises(FileNotFoundError, match="Missing adapter config"):
        validate_kernel_run_inputs(_kernel_config(prepared_data_dir, adapter_dir))


def test_tokenize_pair_requires_chat_template_tokenizer() -> None:
    with pytest.raises(ValueError, match="apply_chat_template"):
        tokenize_pair(object(), prompt="p", completion="c")
