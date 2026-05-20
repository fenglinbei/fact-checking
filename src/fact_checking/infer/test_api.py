from __future__ import annotations

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

    assert _argmax_label_logprobs(scores) == 2
