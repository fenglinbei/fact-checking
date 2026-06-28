from __future__ import annotations

import importlib

from sft.runtime import deps


def test_flash_attn2_available_rejects_broken_extension(monkeypatch) -> None:
    original_import_module = importlib.import_module

    def fake_find_spec(name: str):
        if name == "flash_attn":
            return object()
        return None

    def fake_import_module(name: str):
        if name == "flash_attn":
            raise ImportError("undefined symbol: flash_attn_2_cuda")
        return original_import_module(name)

    monkeypatch.setattr(deps.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(deps.importlib, "import_module", fake_import_module)

    assert deps.flash_attn2_available() is False
