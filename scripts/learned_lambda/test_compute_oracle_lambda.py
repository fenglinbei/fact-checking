from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("compute_oracle_lambda.py")
SPEC = importlib.util.spec_from_file_location("compute_oracle_lambda", MODULE_PATH)
assert SPEC is not None
compute_oracle_lambda = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compute_oracle_lambda)

_resolve_vllm_paths = compute_oracle_lambda._resolve_vllm_paths
_validate_lora_adapter = compute_oracle_lambda._validate_lora_adapter


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
