from __future__ import annotations

import json

import pytest

import lora.data_prep as data_prep
from lora.data_prep import (
    TokenSupervisionFilter,
    WriteFilters,
    build_record_formatter,
    count_masked_completion_tokens,
    load_data_prep_config,
    prepare_dataset_from_config,
    write_jsonl,
)


class FakeChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        tools=None,
        return_dict=False,
        add_generation_prompt=False,
    ):
        base = 5
        content_length = sum(len(message["content"]) for message in messages)
        if add_generation_prompt:
            base += 3
        return [0] * (base + content_length)


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_template_formatter_supports_conditionals_and_computed_fields() -> None:
    formatter = build_record_formatter(
        {
            "type": "template",
            "computed_fields": {
                "task": {
                    "first_non_empty": ["rewritten_intent", "intent"],
                }
            },
            "prompt_parts": [
                "Write a helpful response.",
                {"template": "Context:\n{context}", "when_field": "context"},
                "Task:\n{task}",
            ],
            "completion_template": "{response}",
        }
    )

    record = formatter(
        {
            "intent": "zip lists",
            "rewritten_intent": "zip two lists into tuples",
            "context": "",
            "response": "zip(a, b)",
        }
    )

    assert "Context:" not in record["prompt"]
    assert "Task:\nzip two lists into tuples" in record["prompt"]
    assert record["completion"] == "zip(a, b)"


def test_format_choices_renders_mcq_options() -> None:
    formatter = build_record_formatter(
        {
            "type": "template",
            "computed_fields": {
                "options": {
                    "format_choices": {
                        "text_field": "choices.text",
                        "label_field": "choices.label",
                    }
                }
            },
            "prompt_template": "Q: {question}\n\n{options}",
            "completion_template": "{answerKey}",
        }
    )
    record = formatter(
        {
            "question": "What is the sky?",
            "choices": {"text": ["Blue", "Green"], "label": ["A", "B"]},
            "answerKey": "A",
        }
    )
    assert "A. Blue\nB. Green" in record["prompt"]
    assert record["completion"] == "A"


def test_format_choices_synthesizes_labels_for_plain_list() -> None:
    formatter = build_record_formatter(
        {
            "type": "template",
            "computed_fields": {
                "options": {"format_choices": {"text_field": "choices"}},
                "answer_letter": {
                    "lookup_index": {"index": "answer", "as_letter": True}
                },
            },
            "prompt_template": "Q: {question}\n{options}",
            "completion_template": "{answer_letter}",
        }
    )
    record = formatter(
        {
            "question": "2+2?",
            "choices": ["3", "4", "5", "22"],
            "answer": 1,
        }
    )
    assert "A. 3\nB. 4\nC. 5\nD. 22" in record["prompt"]
    assert record["completion"] == "B"


def test_lookup_index_supports_inline_labels() -> None:
    formatter = build_record_formatter(
        {
            "type": "template",
            "computed_fields": {
                "label_text": {
                    "lookup_index": {
                        "index": "label",
                        "labels": ["negative", "neutral", "positive"],
                    }
                }
            },
            "prompt_template": "Review: {text}",
            "completion_template": "{label_text}",
        }
    )
    record = formatter({"text": "Great!", "label": 2})
    assert record["completion"] == "positive"


def test_lookup_index_returns_list_element_when_not_as_letter() -> None:
    formatter = build_record_formatter(
        {
            "type": "template",
            "computed_fields": {
                "answer_text": {"lookup_index": {"list": "choices", "index": "answer"}}
            },
            "prompt_template": "{question}",
            "completion_template": "{answer_text}",
        }
    )
    record = formatter({"question": "Q?", "choices": ["A1", "A2", "A3"], "answer": 2})
    assert record["completion"] == "A3"


def test_validation_rejects_unknown_computed_field_op(tmp_path) -> None:
    source_path = tmp_path / "records.jsonl"
    _write_jsonl(source_path, [{"question": "q", "answer": "a"}])
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
dataset_name: bad_op
source:
  type: jsonl
  path: {source_path}
split:
  strategy: train_valid
mapping:
  type: template
  computed_fields:
    x:
      reverse: [a, b]
  prompt_template: "{{question}}"
  completion_template: "{{answer}}"
"""
    )
    with pytest.raises(ValueError, match="Unknown computed_fields op"):
        prepare_dataset_from_config(config_path)


def test_write_jsonl_skips_overlong_records(tmp_path) -> None:
    dataset = [
        {"prompt": "short", "completion": "ok"},
        {"prompt": "p" * 5001, "completion": "ok"},
        {"prompt": "short", "completion": "c" * 3001},
    ]

    result = write_jsonl(
        dataset,
        tmp_path / "train.jsonl",
        formatter=lambda row: row,
        filters=WriteFilters(
            max_prompt_chars=4000,
            max_completion_chars=3000,
            max_total_chars=5000,
        ),
    )

    assert result.count == 1
    assert result.stats["skipped"] == 2
    assert result.stats["max_prompt_chars_seen"] == 5001
    assert result.stats["max_completion_chars_seen"] == 3001


def test_count_masked_completion_tokens_matches_truncation_behavior() -> None:
    stats = count_masked_completion_tokens(
        tokenizer=FakeChatTokenizer(),
        prompt="p" * 20,
        completion="c" * 10,
        max_seq_length=24,
    )
    assert stats == {
        "full_length": 35,
        "prompt_length": 28,
        "truncated_length": 24,
        "supervised_tokens": 0,
    }


def test_write_jsonl_skips_zero_supervision_rows(tmp_path) -> None:
    dataset = [
        {"prompt": "p" * 20, "completion": "ok"},
        {"prompt": "short", "completion": "answer"},
    ]

    result = write_jsonl(
        dataset,
        tmp_path / "train.jsonl",
        formatter=lambda row: row,
        filters=WriteFilters(
            token_supervision=TokenSupervisionFilter(
                enabled=True,
                tokenizer=FakeChatTokenizer(),
                max_seq_length=24,
                min_supervised_tokens=1,
            )
        ),
    )

    assert result.count == 1
    assert result.stats["zero_supervision_examples"] == 1
    assert result.stats["skipped_token_supervision"] == 1
    assert result.stats["min_supervised_tokens_seen"] == 0


def test_prepare_dataset_from_config_uses_local_jsonl_source(tmp_path) -> None:
    source_path = tmp_path / "records.jsonl"
    _write_jsonl(
        source_path,
        [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)],
    )
    config_path = tmp_path / "data.yaml"
    output_dir = tmp_path / "prepared"
    config_path.write_text(
        f"""
dataset_name: local_qa
source_dataset: local-jsonl
output_dir: {output_dir}
source:
  type: jsonl
  path: {source_path}
split:
  strategy: ratios
  seed: 7
  valid_ratio: 0.2
  test_ratio: 0.2
mapping:
  type: template
  prompt_template: "Question: {{question}}"
  completion_template: "{{answer}}"
"""
    )

    counts = prepare_dataset_from_config(config_path)

    assert counts == {"train": 6, "valid": 2, "test": 2}
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "valid.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["dataset_name"] == "local_qa"
    assert metadata["source_dataset"] == "local-jsonl"
    assert metadata["extra"]["split"]["strategy"] == "ratios"


def test_prepare_dataset_from_config_applies_token_filter(
    tmp_path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt": "p" * 20, "completion": "ok"},
            {"prompt": "short", "completion": "answer"},
        ],
    )
    _write_jsonl(valid_path, [{"prompt": "short", "completion": "answer"}])
    _write_jsonl(test_path, [{"prompt": "short", "completion": "answer"}])
    config_path = tmp_path / "data.yaml"
    output_dir = tmp_path / "prepared"
    config_path.write_text(
        f"""
dataset_name: token_filtered
output_dir: {output_dir}
source:
  type: jsonl
  files:
    train: {train_path}
    valid: {valid_path}
    test: {test_path}
split:
  strategy: existing
mapping:
  type: template
  prompt_template: "{{prompt}}"
  completion_template: "{{completion}}"
filters:
  token_supervision:
    enabled: true
    tokenizer_model: fake-model
    max_seq_length: 24
    min_supervised_tokens: 1
"""
    )

    monkeypatch.setattr(
        data_prep,
        "require_mlx_tokenizer_loader",
        lambda: lambda model: FakeChatTokenizer(),
    )

    counts = prepare_dataset_from_config(config_path)

    assert counts == {"train": 1, "valid": 1, "test": 1}
    metadata = json.loads((output_dir / "metadata.json").read_text())
    filter_stats = metadata["extra"]["filter_stats"]["train"]
    assert filter_stats["skipped_token_supervision"] == 1
    assert filter_stats["zero_supervision_examples"] == 1


def test_prepare_dataset_from_config_appends_split_sources(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    extra_path = tmp_path / "extra.jsonl"
    _write_jsonl(train_path, [{"prompt": "train", "completion": "a"}])
    _write_jsonl(valid_path, [{"prompt": "valid", "completion": "b"}])
    _write_jsonl(test_path, [{"prompt": "test", "completion": "c"}])
    _write_jsonl(
        extra_path,
        [
            {"prompt": "extra-1", "completion": "d"},
            {"prompt": "extra-2", "completion": "e"},
        ],
    )
    config_path = tmp_path / "data.yaml"
    output_dir = tmp_path / "prepared"
    config_path.write_text(
        f"""
dataset_name: appended
output_dir: {output_dir}
source:
  type: jsonl
  files:
    train: {train_path}
    valid: {valid_path}
    test: {test_path}
  append:
    train:
      - type: jsonl
        path: {extra_path}
        max_examples: 1
split:
  strategy: existing
mapping:
  type: template
  prompt_template: "{{prompt}}"
  completion_template: "{{completion}}"
"""
    )

    counts = prepare_dataset_from_config(config_path)

    assert counts == {"train": 2, "valid": 1, "test": 1}
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text().splitlines()
    ]
    prompts = sorted(row["prompt"] for row in train_rows)
    assert "train" in prompts
    assert sum(prompt.startswith("extra-") for prompt in prompts) == 1


def test_tokenizer_override_requires_enabled_token_filter(tmp_path) -> None:
    source_path = tmp_path / "records.jsonl"
    _write_jsonl(source_path, [{"prompt": "p", "completion": "c"}])
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
dataset_name: no_token_filter
source:
  type: jsonl
  path: {source_path}
split:
  strategy: train_valid
mapping:
  type: template
  prompt_template: "{{prompt}}"
  completion_template: "{{completion}}"
"""
    )

    with pytest.raises(ValueError, match="token_supervision.enabled"):
        prepare_dataset_from_config(config_path, tokenizer_model="fake-model")


def test_missing_template_field_reports_field_name(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, [{"prompt": "p", "completion": "c"}])
    _write_jsonl(valid_path, [{"prompt": "p", "completion": "c"}])
    _write_jsonl(test_path, [{"prompt": "p", "completion": "c"}])
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
dataset_name: bad_template
source:
  type: jsonl
  files:
    train: {train_path}
    valid: {valid_path}
    test: {test_path}
split:
  strategy: existing
mapping:
  type: template
  prompt_template: "{{missing}}"
  completion_template: "{{completion}}"
"""
    )

    with pytest.raises(ValueError, match="Missing field `missing`"):
        prepare_dataset_from_config(config_path)


def test_validation_rejects_old_split_splits_location(tmp_path) -> None:
    source_path = tmp_path / "records.jsonl"
    _write_jsonl(source_path, [{"prompt": "p", "completion": "c"}])
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
dataset_name: bad_splits
source:
  type: jsonl
  path: {source_path}
split:
  strategy: train_valid
  splits:
    train: train
mapping:
  type: template
  prompt_template: "{{prompt}}"
  completion_template: "{{completion}}"
"""
    )

    with pytest.raises(ValueError, match="Unknown keys in `split`"):
        prepare_dataset_from_config(config_path)


def test_validation_rejects_unknown_mapping_type(tmp_path) -> None:
    source_path = tmp_path / "records.jsonl"
    _write_jsonl(source_path, [{"prompt": "p", "completion": "c"}])
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
dataset_name: bad_mapping
source:
  type: jsonl
  path: {source_path}
split:
  strategy: train_valid
mapping:
  type: registered
  completion_template: "{{completion}}"
"""
    )

    with pytest.raises(ValueError, match="Only `mapping.type: template`"):
        prepare_dataset_from_config(config_path)


def test_load_data_prep_config_detects_extends_cycles(tmp_path) -> None:
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text("extends: right.yaml\n")
    right.write_text("extends: left.yaml\n")

    with pytest.raises(ValueError, match="inheritance cycle"):
        load_data_prep_config(left)
