from __future__ import annotations

import functools
import re
from typing import Any


_PATCH_MARKER = "_mrec_mistral_common_patch"
_ORIGINAL_GET_TOKENIZER = "_mrec_original_get_tokenizer"
_ORIGINAL_GET_CACHED_TOKENIZER = "_mrec_original_get_cached_tokenizer"
_UNSUPPORTED_KWARGS_PATTERN = re.compile(r"Some kwargs in \[(?P<keys>[^\]]*)\]")
_DEFAULT_UNSUPPORTED_KWARGS = {"_from_auto", "max_loras", "tokenizer_revision"}


def _mistral_common_unsupported_kwargs(exc: BaseException) -> set[str]:
    message = str(exc)
    if "MistralCommonBackend.from_pretrained" not in message or "Some kwargs" not in message:
        return set()
    match = _UNSUPPORTED_KWARGS_PATTERN.search(message)
    if not match:
        return set(_DEFAULT_UNSUPPORTED_KWARGS)

    keys: set[str] = set()
    for raw_key in match.group("keys").split(","):
        key = raw_key.strip().strip("'\"")
        if key:
            keys.add(key)
    return keys or set(_DEFAULT_UNSUPPORTED_KWARGS)


def apply_vllm_mistral_common_tokenizer_patch() -> None:
    """Patch vLLM tokenizer helpers for transformers' MistralCommonBackend."""

    try:
        import vllm.transformers_utils.tokenizer as tokenizer_module
        import vllm.transformers_utils.tokenizer_group as tokenizer_group_module
    except ImportError:
        return

    original = getattr(tokenizer_module, _ORIGINAL_GET_TOKENIZER, None)
    if original is None:
        original = getattr(tokenizer_module, "get_tokenizer", None)
        if original is None:
            return
        setattr(tokenizer_module, _ORIGINAL_GET_TOKENIZER, original)

    original_cached = getattr(tokenizer_module, _ORIGINAL_GET_CACHED_TOKENIZER, None)
    if original_cached is None:
        original_cached = getattr(tokenizer_module, "get_cached_tokenizer", None)
        setattr(tokenizer_module, _ORIGINAL_GET_CACHED_TOKENIZER, original_cached)

    def patched_get_tokenizer(tokenizer_name: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(tokenizer_name, *args, **kwargs)
        except ValueError as exc:
            unsupported = _mistral_common_unsupported_kwargs(exc)
            if not unsupported:
                raise
            retry_kwargs = dict(kwargs)
            for key in unsupported:
                retry_kwargs.pop(key, None)
            return original(tokenizer_name, *args, **retry_kwargs)

    def patched_get_cached_tokenizer(tokenizer: Any) -> Any:
        if type(tokenizer).__name__ == "MistralCommonBackend":
            return tokenizer
        if original_cached is None:
            return tokenizer
        return original_cached(tokenizer)

    setattr(patched_get_tokenizer, _PATCH_MARKER, True)
    setattr(patched_get_cached_tokenizer, _PATCH_MARKER, True)
    tokenizer_module.get_tokenizer = patched_get_tokenizer
    tokenizer_module.cached_get_tokenizer = functools.lru_cache(patched_get_tokenizer)
    setattr(tokenizer_module.cached_get_tokenizer, _PATCH_MARKER, True)
    tokenizer_module.get_cached_tokenizer = patched_get_cached_tokenizer
    tokenizer_group_module.get_tokenizer = patched_get_tokenizer
