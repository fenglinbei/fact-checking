from __future__ import annotations

import json
from pathlib import Path

from fact_checking.infer.api import (
    _MERGED_LORA_CACHE_VERSION,
    _MERGED_LORA_IMPL,
    _VLLMServerHandle,
    _argmax_label_logprobs,
    _cleanup_vllm_server,
    _extract_final_prompt_logprob,
    _merged_lora_cache_complete,
    _merged_lora_cache_key,
    _copy_merge_sidecar_artifacts,
    _dequantize_fp8_safetensors_checkpoint,
    _merge_lora_to_cache,
)


def _write_adapter_files(path: Path, *, weight_payload: bytes = b"weights") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(weight_payload)


def test_merged_lora_cache_key_is_stable_for_same_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_files(adapter_dir)

    first = _merged_lora_cache_key(base_model="/models/base", adapter_dir=adapter_dir, dtype="auto")
    second = _merged_lora_cache_key(base_model="/models/base", adapter_dir=adapter_dir, dtype="auto")

    assert first == second


def test_merged_lora_cache_key_changes_when_adapter_weight_changes(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_files(adapter_dir, weight_payload=b"weights")
    first = _merged_lora_cache_key(base_model="/models/base", adapter_dir=adapter_dir, dtype="auto")

    _write_adapter_files(adapter_dir, weight_payload=b"changed weights")
    second = _merged_lora_cache_key(base_model="/models/base", adapter_dir=adapter_dir, dtype="auto")

    assert first != second


def test_merged_lora_cache_complete_requires_model_and_tokenizer(tmp_path: Path) -> None:
    cache_dir = tmp_path / "merged"
    cache_dir.mkdir()
    (cache_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "model.safetensors").write_bytes(b"model")

    assert not _merged_lora_cache_complete(cache_dir)

    (cache_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert _merged_lora_cache_complete(cache_dir)


def test_merged_lora_cache_complete_accepts_tekken_tokenizer(tmp_path: Path) -> None:
    cache_dir = tmp_path / "merged"
    cache_dir.mkdir()
    (cache_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "model.safetensors").write_bytes(b"model")
    (cache_dir / "tekken.json").write_text("{}", encoding="utf-8")

    assert _merged_lora_cache_complete(cache_dir)


def test_copy_merge_sidecar_artifacts_preserves_multimodal_processor_config(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "merged"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "processor_config.json").write_text('{"processor_class": "PixtralProcessor"}', encoding="utf-8")
    (source_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    copied = _copy_merge_sidecar_artifacts(source_dirs=[source_dir], output_dir=output_dir)

    assert "processor_config.json" in copied
    assert "chat_template.jinja" in copied
    assert (output_dir / "processor_config.json").exists()
    assert (output_dir / "chat_template.jinja").exists()


def test_merge_lora_to_cache_creates_persistent_cache(tmp_path: Path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_files(adapter_dir)

    def fake_merge_lora_to_dir(*, output_dir: Path, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        (output_dir / "model.safetensors").write_bytes(b"model")
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return output_dir

    monkeypatch.setattr("fact_checking.infer.api._merge_lora_to_dir", fake_merge_lora_to_dir)

    result = _merge_lora_to_cache(
        base_model="/models/base",
        adapter_dir=adapter_dir,
        tokenizer_dir=adapter_dir,
        dtype="auto",
        cache_dir=tmp_path / "cache",
    )

    assert _merged_lora_cache_complete(result)
    meta_path = result / "merge_cache.json"
    assert meta_path.exists()
    metadata = meta_path.read_text(encoding="utf-8")
    assert f'"cache_version": {_MERGED_LORA_CACHE_VERSION}' in metadata
    assert f'"merge_impl": "{_MERGED_LORA_IMPL}"' in metadata


def test_dequantize_fp8_safetensors_checkpoint_strips_fp8_scales(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"},
                "text_config": {
                    "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"}
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "language_model.lm_head.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn
            ),
            "language_model.lm_head.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32),
            "language_model.model.norm.weight": torch.ones((2,), dtype=torch.float32),
        },
        merged_dir / "model.safetensors",
    )

    assert _dequantize_fp8_safetensors_checkpoint(merged_dir, dtype="bfloat16")

    tensors = load_file(merged_dir / "model.safetensors")
    assert tensors["language_model.lm_head.weight"].dtype == torch.bfloat16
    assert torch.allclose(
        tensors["language_model.lm_head.weight"].float(),
        torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32),
    )
    assert "language_model.lm_head.weight_scale_inv" not in tensors
    assert tensors["language_model.model.norm.weight"].dtype == torch.bfloat16
    config = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert "quantization_config" not in config["text_config"]


def test_cleanup_does_not_delete_persistent_merged_cache(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    handle = _VLLMServerHandle(merged_model_dir=merged_dir, cleanup_merged_model_dir=False)

    _cleanup_vllm_server(handle, {"server": {"stop_after_infer": True}})

    assert merged_dir.exists()


def test_cleanup_keeps_temporary_merge_when_server_is_left_running(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    handle = _VLLMServerHandle(merged_model_dir=merged_dir, cleanup_merged_model_dir=True)

    _cleanup_vllm_server(handle, {"server": {"stop_after_infer": False}})

    assert merged_dir.exists()


def test_extract_final_prompt_logprob_reads_token_id_mapping() -> None:
    prompt_logprobs = [
        None,
        {"2476": {"logprob": -0.1}},
        {"362": {"logprob": -0.25}},
    ]

    assert _extract_final_prompt_logprob(prompt_logprobs, 362) == -0.25


def test_argmax_label_logprobs_uses_label_order() -> None:
    scores = {
        "A": -5.0,
        "B": -3.0,
        "C": -0.2,
        "D": -4.0,
        "E": -2.0,
        "F": -1.0,
    }

    assert _argmax_label_logprobs(scores, letter_order=list("ABCDEF")) == 2
