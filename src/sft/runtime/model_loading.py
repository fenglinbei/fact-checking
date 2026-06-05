from __future__ import annotations

import json
from copy import deepcopy
from importlib import import_module
from typing import Any

from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


_CONDITIONAL_GENERATION_LOADERS = {
    "Gemma3ForConditionalGeneration": ("transformers", "Gemma3ForConditionalGeneration"),
    "Gemma3nForConditionalGeneration": ("transformers", "Gemma3nForConditionalGeneration"),
    "Gemma4ForConditionalGeneration": ("transformers", "Gemma4ForConditionalGeneration"),
    "Mistral3ForConditionalGeneration": ("transformers", "Mistral3ForConditionalGeneration"),
}

_REMOTE_CODE_DISABLED_MARKERS = {
    "phi-4-mini-instruct",
}

_MISTRAL_COMMON_TOKENIZER_CLASSES = {
    "mistralcommonbackend",
    "mistralcommontokenizer",
    "tokenizersbackend",
}

_MISTRAL_COMMON_MODEL_TYPES = {
    "mistral3",
    "ministral3",
}


def _read_local_json(model_name_or_path: str, filename: str) -> dict[str, Any] | None:
    path = Path(str(model_name_or_path))
    if not path.is_dir():
        return None
    json_path = path / filename
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _config_requests_phi_remote_code(path: Path) -> bool:
    payload = _read_local_json(str(path), "config.json")
    if payload is None:
        return False
    if str(payload.get("model_type", "")).lower() != "phi3":
        return False
    auto_map = payload.get("auto_map")
    if not isinstance(auto_map, dict):
        return False
    mapped_values = " ".join(str(value).lower() for value in auto_map.values())
    return "configuration_phi3" in mapped_values or "modeling_phi3" in mapped_values


def resolve_trust_remote_code(model_name_or_path: str, requested: bool = True) -> bool:
    """Return the safe trust_remote_code setting for a known local backbone.

    Phi-4-mini-instruct ships remote code that imports newer Transformers
    internals than the project training environment provides. The same model
    loads through the native Phi3 implementation when remote code is disabled.
    """

    raw = str(model_name_or_path)
    path = Path(raw)
    candidates = {raw.lower(), path.name.lower()}
    if any(marker in item for marker in _REMOTE_CODE_DISABLED_MARKERS for item in candidates):
        return False
    if _config_requests_phi_remote_code(path):
        return False
    return bool(requested)


def is_mistral_common_tokenizer(tokenizer: Any) -> bool:
    tokenizer_type = type(tokenizer)
    module = str(getattr(tokenizer_type, "__module__", "")).lower()
    name = str(getattr(tokenizer_type, "__name__", "")).lower()
    return "mistral_common" in module or name.startswith("mistralcommon")


def _uses_mistral_common_tokenizer(model_name_or_path: str) -> bool:
    path = Path(str(model_name_or_path))
    if not path.is_dir():
        return False

    tokenizer_config = _read_local_json(str(path), "tokenizer_config.json") or {}
    tokenizer_class = str(tokenizer_config.get("tokenizer_class", "")).lower()
    if tokenizer_class in _MISTRAL_COMMON_TOKENIZER_CLASSES:
        return True

    config = _read_local_json(str(path), "config.json") or {}
    model_type = str(config.get("model_type", "")).lower()
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    text_model_type = str(text_config.get("model_type", "")).lower()
    return (path / "tekken.json").exists() and (
        model_type in _MISTRAL_COMMON_MODEL_TYPES or text_model_type in _MISTRAL_COMMON_MODEL_TYPES
    )


def _load_mistral_common_tokenizer(model_name_or_path: str, **tokenizer_kwargs: Any) -> Any:
    import_error: Exception | None = None
    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
    except Exception as exc:  # noqa: BLE001 - keep compatibility with older exports.
        import_error = exc
        try:
            from transformers import MistralCommonTokenizer
        except Exception as fallback_exc:  # noqa: BLE001 - provide a direct dependency hint.
            import_error = fallback_exc
            MistralCommonTokenizer = None  # type: ignore[assignment]

    if MistralCommonTokenizer is None:
        raise RuntimeError(
            "This Mistral tokenizer requires `mistral-common`; install "
            "`mistral-common[opencv]>=1.6.3` in the training environment. "
            f"Tokenizer import failed with {type(import_error).__name__}: {import_error}"
        ) from import_error

    allowed_kwargs = {
        "cache_dir",
        "clean_up_tokenization_spaces",
        "force_download",
        "local_files_only",
        "mode",
        "model_input_names",
        "model_max_length",
        "padding_side",
        "revision",
        "token",
        "trust_remote_code",
        "truncation_side",
    }
    common_kwargs = {key: value for key, value in tokenizer_kwargs.items() if key in allowed_kwargs}
    return MistralCommonTokenizer.from_pretrained(model_name_or_path, **common_kwargs)


def load_compatible_tokenizer(model_name_or_path: str, **tokenizer_kwargs: Any) -> Any:
    tokenizer_kwargs = dict(tokenizer_kwargs)
    tokenizer_kwargs["trust_remote_code"] = resolve_trust_remote_code(
        model_name_or_path,
        bool(tokenizer_kwargs.get("trust_remote_code", True)),
    )
    if _uses_mistral_common_tokenizer(model_name_or_path):
        tokenizer = _load_mistral_common_tokenizer(model_name_or_path, **tokenizer_kwargs)
    else:
        tokenizer_kwargs.setdefault("fix_mistral_regex", True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _import_model_class(module_name: str, class_name: str) -> type:
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    if not isinstance(model_cls, type):
        raise TypeError(f"{module_name}.{class_name} is not a model class")
    return model_cls


def _normalize_mistral3_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(payload)
    text_config = fixed.get("text_config")
    if isinstance(text_config, dict) and str(text_config.get("model_type", "")).lower() == "ministral3":
        text_config["model_type"] = "mistral"
    return fixed


def _finegrained_fp8_dequantize_config(model_name_or_path: str) -> Any | None:
    payload = _read_local_json(model_name_or_path, "config.json")
    if payload is None:
        return None
    quantization_config = payload.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return None
    if str(quantization_config.get("quant_method", "")).lower() != "fp8":
        return None

    try:
        from transformers import FineGrainedFP8Config
    except Exception:  # noqa: BLE001 - older Transformers do not expose this config.
        return None

    weight_block_size = quantization_config.get("weight_block_size")
    if isinstance(weight_block_size, list):
        weight_block_size = tuple(weight_block_size)
    return FineGrainedFP8Config(
        activation_scheme=str(quantization_config.get("activation_scheme", "dynamic")),
        weight_block_size=weight_block_size,
        dequantize=True,
        modules_to_not_convert=quantization_config.get("modules_to_not_convert"),
    )


def load_compatible_config(model_name_or_path: str, *, trust_remote_code: bool = True) -> Any:
    try:
        return AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    except KeyError:
        payload = _read_local_json(model_name_or_path, "config.json")
        if payload is None or str(payload.get("model_type", "")).lower() != "mistral3":
            raise
        from transformers import Mistral3Config

        return Mistral3Config.from_dict(_normalize_mistral3_config_payload(payload))


def _conditional_generation_loader(model_name_or_path: str, trust_remote_code: bool) -> tuple[type, Any] | None:
    cfg = load_compatible_config(model_name_or_path, trust_remote_code=trust_remote_code)
    for arch in getattr(cfg, "architectures", None) or []:
        loader_spec = _CONDITIONAL_GENERATION_LOADERS.get(str(arch))
        if loader_spec is not None:
            return _import_model_class(*loader_spec), cfg
    return None


def load_causal_lm_compatible_model(model_name_or_path: str, **model_kwargs: Any) -> Any:
    """Load a text-generation model for label-token logits.

    Most backbones are registered under AutoModelForCausalLM. Some recent
    multimodal families expose a text-generation forward pass only through a
    ForConditionalGeneration class; for text-only label-token scoring that API
    is still compatible because it accepts input_ids/attention_mask and returns
    logits.
    """

    model_kwargs = dict(model_kwargs)
    model_kwargs["trust_remote_code"] = resolve_trust_remote_code(
        model_name_or_path,
        bool(model_kwargs.get("trust_remote_code", True)),
    )
    if "quantization_config" not in model_kwargs:
        dequantize_config = _finegrained_fp8_dequantize_config(model_name_or_path)
        if dequantize_config is not None:
            model_kwargs["quantization_config"] = dequantize_config

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except (KeyError, ValueError) as exc:
        trust_remote_code = bool(model_kwargs.get("trust_remote_code", True))
        loader = _conditional_generation_loader(model_name_or_path, trust_remote_code)
        if loader is None:
            raise
        loader_cls, cfg = loader
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.setdefault("config", cfg)
        try:
            return loader_cls.from_pretrained(model_name_or_path, **fallback_kwargs)
        except Exception as fallback_exc:  # noqa: BLE001 - preserve loader context.
            raise RuntimeError(
                f"Failed to load {model_name_or_path!r} with {loader_cls.__name__} after "
                "AutoModelForCausalLM did not recognize the architecture."
            ) from fallback_exc
