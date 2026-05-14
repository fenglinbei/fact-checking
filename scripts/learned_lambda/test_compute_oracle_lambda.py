from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).with_name("compute_oracle_lambda.py")
SPEC = importlib.util.spec_from_file_location("compute_oracle_lambda", MODULE_PATH)
assert SPEC is not None
compute_oracle_lambda = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compute_oracle_lambda)

_resolve_vllm_paths = compute_oracle_lambda._resolve_vllm_paths
_validate_lora_adapter = compute_oracle_lambda._validate_lora_adapter
_append_label_prefix = compute_oracle_lambda._append_label_prefix
_build_label_token_ids = compute_oracle_lambda._build_label_token_ids
_extract_required_label_logprobs = compute_oracle_lambda._extract_required_label_logprobs


class _FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.mapping[text])


def test_resolve_vllm_paths_uses_base_tokenizer_for_adapter_without_tokenizer(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    model_path, tokenizer_path = _resolve_vllm_paths(
        model="/models/base",
        tokenizer=None,
        lora_adapter=str(adapter),
    )

    assert model_path == "/models/base"
    assert tokenizer_path == "/models/base"


def test_resolve_vllm_paths_uses_adapter_tokenizer_when_available(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    model_path, tokenizer_path = _resolve_vllm_paths(
        model="/models/base",
        tokenizer=None,
        lora_adapter=str(adapter),
    )

    assert model_path == "/models/base"
    assert tokenizer_path == str(adapter)


def test_resolve_vllm_paths_prefers_explicit_tokenizer(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    model_path, tokenizer_path = _resolve_vllm_paths(
        model="/models/base",
        tokenizer="/models/tokenizer",
        lora_adapter=str(adapter),
    )

    assert model_path == "/models/base"
    assert tokenizer_path == "/models/tokenizer"


def test_validate_lora_adapter_requires_config_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="LoRA adapter weights not found"):
        _validate_lora_adapter(adapter)

    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    _validate_lora_adapter(adapter)


def test_build_label_token_ids_uses_space_prefixed_single_token_choices() -> None:
    tokenizer = _FakeTokenizer({
        " A": [101],
        " B": [102],
        " C": [103],
        " D": [104],
        " E": [105],
        " F": [106],
    })

    assert _build_label_token_ids(tokenizer) == {
        "A": 101,
        "B": 102,
        "C": 103,
        "D": 104,
        "E": 105,
        "F": 106,
    }


def test_build_label_token_ids_rejects_multi_token_label_choice() -> None:
    tokenizer = _FakeTokenizer({
        " A": [101],
        " B": [102],
        " C": [103],
        " D": [104],
        " E": [105],
        " F": [106, 107],
    })

    with pytest.raises(ValueError, match="must be exactly one token"):
        _build_label_token_ids(tokenizer)


def test_extract_required_label_logprobs_requires_every_label_token() -> None:
    label_token_ids = {"A": 101, "B": 102, "C": 103}
    first_token_logprobs = {
        101: SimpleNamespace(logprob=-0.1),
        102: SimpleNamespace(logprob=-0.2),
        103: SimpleNamespace(logprob=-0.3),
    }

    assert _extract_required_label_logprobs(first_token_logprobs, label_token_ids) == {
        "A": -0.1,
        "B": -0.2,
        "C": -0.3,
    }

    with pytest.raises(RuntimeError, match="Missing: C"):
        _extract_required_label_logprobs(
            {
                101: SimpleNamespace(logprob=-0.1),
                102: SimpleNamespace(logprob=-0.2),
            },
            label_token_ids,
        )


def test_append_label_prefix() -> None:
    assert _append_label_prefix("prompt", "Label:") == "promptLabel:"
    assert _append_label_prefix("prompt", "") == "prompt"
