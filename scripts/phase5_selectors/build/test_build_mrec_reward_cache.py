from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

from scripts.phase5_selectors.build import build_mrec_reward_cache as module


def test_init_scorer_auto_falls_back_to_transformers_when_vllm_rejects_architecture(monkeypatch, capsys) -> None:
    sentinel = object()

    def reject_vllm(args, checkpoint):
        raise ValueError("Model architectures ['Mistral3ForConditionalGeneration'] failed to be inspected.")

    def init_transformers(args, checkpoint):
        assert args.scoring_backend == "auto"
        assert checkpoint["checkpoint_dir"] == "/tmp/teacher/best"
        return sentinel

    monkeypatch.setattr(module, "_init_vllm_scorer", reject_vllm)
    monkeypatch.setattr(module, "_init_transformers_scorer", init_transformers)

    scorer = module._init_scorer(
        argparse.Namespace(scoring_backend="auto"),
        {"checkpoint_dir": "/tmp/teacher/best"},
    )

    captured = capsys.readouterr()
    assert scorer is sentinel
    assert "falling back to transformers" in captured.err


def test_init_scorer_vllm_backend_does_not_fallback(monkeypatch) -> None:
    def reject_vllm(args, checkpoint):
        raise ValueError("vllm unavailable")

    def init_transformers(args, checkpoint):  # pragma: no cover - should not be called.
        raise AssertionError("explicit vllm backend should not fallback")

    monkeypatch.setattr(module, "_init_vllm_scorer", reject_vllm)
    monkeypatch.setattr(module, "_init_transformers_scorer", init_transformers)

    with pytest.raises(ValueError, match="vllm unavailable"):
        module._init_scorer(argparse.Namespace(scoring_backend="vllm"), {})


def test_init_vllm_scorer_can_use_merged_lora_without_dynamic_lora_support(tmp_path: Path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_config.write_text(json.dumps({"r": 16}), encoding="utf-8")
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Mistral3ForConditionalGeneration"],
                "text_config": {"architectures": None, "model_type": "ministral3"},
            }
        ),
        encoding="utf-8",
    )
    import torch
    from safetensors.torch import save_file

    save_file({"language_model.lm_head.weight_scale": torch.ones((1,), dtype=torch.float32)}, merged_dir / "model.safetensors")

    captured: dict[str, object] = {}

    fake_vllm = types.ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = dict(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = dict(kwargs)

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.delitem(sys.modules, "vllm.lora.request", raising=False)
    monkeypatch.delenv("MREC_PATCH_VLLM_MISTRAL_COMMON_TOKENIZER", raising=False)
    monkeypatch.setattr(
        module,
        "_merge_lora_for_vllm_scorer",
        lambda **kwargs: merged_dir,
        raising=False,
    )

    args = argparse.Namespace(
        vllm_tokenizer_path="",
        vllm_tensor_parallel_size=2,
        vllm_gpu_memory_utilization=0.75,
        vllm_dtype="auto",
        vllm_max_model_len=1024,
        vllm_enforce_eager=False,
        vllm_prompt_batch_size=8,
        vllm_lora_mode="merged",
        vllm_merge_lora_cache_dir=str(tmp_path / "cache"),
        vllm_merge_lora_force_rebuild=False,
        vllm_tokenizer_mode="mistral",
        vllm_config_format="",
        vllm_load_format="",
        label_schema="liar6",
    )
    checkpoint = {
        "base_model_name_or_path": "/models/base",
        "checkpoint_dir": str(adapter_dir),
        "adapter_config_path": str(adapter_config),
        "label_token_ids": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
        "label_prefix": "Label:",
    }

    scorer = module._init_vllm_scorer(args, checkpoint)

    llm_kwargs = captured["llm_kwargs"]
    assert llm_kwargs["model"] == str(merged_dir)
    assert llm_kwargs["tokenizer"] == str(merged_dir)
    assert llm_kwargs["tokenizer_mode"] == "mistral"
    assert "config_format" not in llm_kwargs
    assert "load_format" not in llm_kwargs
    assert llm_kwargs["limit_mm_per_prompt"] == {"image": 0}
    assert llm_kwargs["quantization"] == "fp8"
    assert "enable_lora" not in llm_kwargs
    assert "max_lora_rank" not in llm_kwargs
    assert scorer._lora_request is None
    assert os.environ["MREC_PATCH_VLLM_MISTRAL_COMMON_TOKENIZER"] == "1"
    merged_config = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    assert merged_config["text_config"]["architectures"] == ["MistralForCausalLM"]
    assert merged_config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "quant_method": "fp8",
    }
    assert merged_config["text_config"]["quantization_config"] == {
        "activation_scheme": "dynamic",
        "quant_method": "fp8",
    }


def test_score_request_cache_key_includes_scoring_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "_apply_mrec_prompt_fields",
        lambda *, claim, candidates, trace: (claim, candidates, {}),
    )
    monkeypatch.setattr(
        module,
        "_build_training_row",
        lambda retrieval_row, tokenizer, prompt_cfg: {
            "prompt": "prompt",
            "target": "",
            "gold_label": retrieval_row["label"],
            "gold_id": 0,
            "prompt_token_count": 1,
            "evidence_count": len(retrieval_row["candidates"]),
        },
    )
    sample = types.SimpleNamespace(event_id="event-1", claim="claim", label="false", explain="")
    common_kwargs = {
        "source_row": {"event_id": "event-1"},
        "sample": sample,
        "candidates": [{"candidate_key": "c0", "text": "evidence"}],
        "claim_atoms": [],
        "selected_indices": [0],
        "mrec_steps": [],
        "candidate_idx": None,
        "role": "initial",
        "step": 0,
        "tokenizer": object(),
        "prompt_cfg": {"label_schema": "liar6"},
    }

    vllm_request = module._score_request_for_indices(
        **common_kwargs,
        checkpoint={
            "teacher_fingerprint": "teacher",
            "scoring_fingerprint": "backend=vllm;lora=merged",
        },
    )
    transformers_request = module._score_request_for_indices(
        **common_kwargs,
        checkpoint={
            "teacher_fingerprint": "teacher",
            "scoring_fingerprint": "backend=transformers;lora=none",
        },
    )

    assert vllm_request.cache_key != transformers_request.cache_key


def test_patch_vllm_mistral_common_tokenizer_kwargs_retries_without_internal_kwargs(monkeypatch) -> None:
    fake_vllm = types.ModuleType("vllm")
    fake_transformers_utils = types.ModuleType("vllm.transformers_utils")
    fake_tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizer")
    fake_group_module = types.ModuleType("vllm.transformers_utils.tokenizer_group")
    calls: list[dict[str, object]] = []

    def fake_get_tokenizer(tokenizer_name, **kwargs):
        calls.append(dict(kwargs))
        if "max_loras" in kwargs or "_from_auto" in kwargs or "tokenizer_revision" in kwargs:
            raise ValueError(
                "Some kwargs in ['max_loras', '_from_auto', 'tokenizer_revision'] are not supported by "
                "`MistralCommonBackend.from_pretrained`."
            )
        return "tokenizer"

    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer
    fake_tokenizer_module.cached_get_tokenizer = fake_get_tokenizer
    fake_group_module.get_tokenizer = fake_get_tokenizer
    fake_transformers_utils.tokenizer = fake_tokenizer_module
    fake_transformers_utils.tokenizer_group = fake_group_module
    fake_vllm.transformers_utils = fake_transformers_utils
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils", fake_transformers_utils)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer_group", fake_group_module)

    module._patch_vllm_mistral_common_tokenizer_kwargs()

    result = fake_group_module.get_tokenizer(
        "model",
        trust_remote_code=True,
        max_loras=0,
        _from_auto=True,
        tokenizer_revision=None,
        truncation_side="left",
    )

    assert result == "tokenizer"
    assert calls == [
        {
            "trust_remote_code": True,
            "max_loras": 0,
            "_from_auto": True,
            "tokenizer_revision": None,
            "truncation_side": "left",
        },
        {
            "trust_remote_code": True,
            "truncation_side": "left",
        },
    ]

    result = fake_tokenizer_module.cached_get_tokenizer(
        "model",
        trust_remote_code=True,
        max_loras=0,
        _from_auto=True,
        tokenizer_revision=None,
        truncation_side="left",
    )

    assert result == "tokenizer"
    assert calls[-2:] == [
        {
            "trust_remote_code": True,
            "max_loras": 0,
            "_from_auto": True,
            "tokenizer_revision": None,
            "truncation_side": "left",
        },
        {
            "trust_remote_code": True,
            "truncation_side": "left",
        },
    ]


def test_patch_vllm_mistral_common_tokenizer_skips_cached_wrapper(monkeypatch) -> None:
    fake_vllm = types.ModuleType("vllm")
    fake_transformers_utils = types.ModuleType("vllm.transformers_utils")
    fake_tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizer")
    fake_group_module = types.ModuleType("vllm.transformers_utils.tokenizer_group")

    class MistralCommonBackend:
        pass

    class OtherTokenizer:
        pass

    def fake_get_tokenizer(tokenizer_name, **kwargs):
        return tokenizer_name

    def fake_get_cached_tokenizer(tokenizer):
        if isinstance(tokenizer, MistralCommonBackend):
            raise AttributeError("MistralCommonBackend has no attribute all_special_tokens_extended")
        return ("cached", tokenizer)

    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer
    fake_tokenizer_module.get_cached_tokenizer = fake_get_cached_tokenizer
    fake_group_module.get_tokenizer = fake_get_tokenizer
    fake_transformers_utils.tokenizer = fake_tokenizer_module
    fake_transformers_utils.tokenizer_group = fake_group_module
    fake_vllm.transformers_utils = fake_transformers_utils
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils", fake_transformers_utils)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer_group", fake_group_module)

    module._patch_vllm_mistral_common_tokenizer_kwargs()

    mistral_tokenizer = MistralCommonBackend()
    other_tokenizer = OtherTokenizer()

    assert fake_tokenizer_module.get_cached_tokenizer(mistral_tokenizer) is mistral_tokenizer
    assert fake_tokenizer_module.get_cached_tokenizer(other_tokenizer) == ("cached", other_tokenizer)
