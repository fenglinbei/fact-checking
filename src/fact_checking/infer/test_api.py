from __future__ import annotations

from pathlib import Path

from fact_checking.infer.api import (
    _VLLMServerHandle,
    _cleanup_vllm_server,
    _merged_lora_cache_complete,
    _merged_lora_cache_key,
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
