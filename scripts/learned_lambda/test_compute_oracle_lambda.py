from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
_build_scoring_prompt_token_ids = compute_oracle_lambda._build_scoring_prompt_token_ids
_extract_prompt_token_logprob = compute_oracle_lambda._extract_prompt_token_logprob
_normalize_label_logprobs = compute_oracle_lambda._normalize_label_logprobs


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


def test_build_scoring_prompt_token_ids_requires_final_label_token() -> None:
    tokenizer = _FakeTokenizer({
        "promptLabel: A": [11, 12, 101],
        "promptLabel: B": [11, 12, 102],
    })

    assert _build_scoring_prompt_token_ids(
        tokenizer,
        prompt="prompt",
        label_prefix="Label:",
        letter="A",
        label_token_id=101,
    ) == [11, 12, 101]

    with pytest.raises(ValueError, match="must end with token_id=102"):
        _build_scoring_prompt_token_ids(
            tokenizer,
            prompt="prompt",
            label_prefix="Label:",
            letter="A",
            label_token_id=102,
        )


def test_extract_prompt_token_logprob_reads_actual_final_prompt_token() -> None:
    output = SimpleNamespace(
        prompt_token_ids=[11, 101],
        prompt_logprobs=[
            None,
            {101: SimpleNamespace(logprob=-0.1)},
        ],
    )

    assert _extract_prompt_token_logprob(output, 101, event_id="e1", lam=0.7, letter="A") == -0.1

    output_with_string_key = SimpleNamespace(
        prompt_token_ids=[11, 101],
        prompt_logprobs=[
            None,
            {"101": {"logprob": -0.2}},
        ],
    )
    assert _extract_prompt_token_logprob(output_with_string_key, 101, event_id="e1", lam=0.7, letter="A") == -0.2

    with pytest.raises(RuntimeError, match="did not include the actual label token"):
        _extract_prompt_token_logprob(
            SimpleNamespace(
                prompt_token_ids=[11, 101],
                prompt_logprobs=[
                    None,
                    {102: SimpleNamespace(logprob=-0.3)},
                ],
            ),
            101,
            event_id="e1",
            lam=0.7,
            letter="A",
        )


def test_normalize_label_logprobs_returns_constrained_distribution() -> None:
    label_logprobs = {
        "A": -1.0,
        "B": -2.0,
        "C": -3.0,
        "D": -4.0,
        "E": -5.0,
        "F": -6.0,
    }

    normalized = _normalize_label_logprobs(label_logprobs)

    assert set(normalized) == {"A", "B", "C", "D", "E", "F"}
    assert sum(np.exp(list(normalized.values()))) == pytest.approx(1.0)
    assert normalized["A"] > normalized["B"] > normalized["C"]


def test_append_label_prefix() -> None:
    assert _append_label_prefix("prompt", "Label:") == "promptLabel:"
    assert _append_label_prefix("prompt", "") == "prompt"
