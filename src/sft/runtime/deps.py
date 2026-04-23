from __future__ import annotations

import importlib.util


def flash_attn2_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def fla_fast_path_available() -> bool:
    return (
        importlib.util.find_spec("fla") is not None
        and importlib.util.find_spec("causal_conv1d") is not None
    )
