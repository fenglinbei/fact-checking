#!/usr/bin/env python
"""Static compatibility checks for RAWFC v0.6c backbone migration candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


BACKBONES: dict[str, tuple[str, float]] = {
    "qwen25_15b": ("/data/models/Qwen2.5-1.5B-Instruct", 1.5),
    "qwen3_17b": ("/data/models/Qwen3-1.7B", 1.7),
    "qwen25_3b": ("/data/models/Qwen2.5-3B-Instruct", 3.0),
    "qwen3_4b_2507": ("/data/models/Qwen3-4B-Instruct-2507", 4.0),
    "qwen3_8b": ("/data/models/Qwen3-8B", 8.0),
    "dsr1_qwen7b": ("/data/models/DeepSeek-R1-Distill-Qwen-7B", 7.0),
    "llama31_8b": ("/data/models/Meta-Llama-3.1-8B-Instruct", 8.0),
    "phi4_mini": ("/data/models/Phi-4-mini-instruct", 3.8),
    "gemma4_e4b": ("/data/models/gemma-4-E4B-it", 8.0),
    "ministral3_8b": ("/data/models/Ministral-3-8B-Instruct-2512", 8.4),
}

CONDITIONAL_GENERATION_ARCHES = {
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
}


@dataclass
class CompatRow:
    backbone: str
    path: str
    size_b: float
    exists: bool
    model_type: str
    architectures: list[str]
    max_position_embeddings: int | None
    label_single_token: bool
    causal_lm_arch: bool
    conditional_generation_arch: bool
    status: str
    error: str = ""


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def _read_local_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _max_position_embeddings_from_config(cfg: Any) -> int | None:
    max_position_embeddings = getattr(cfg, "max_position_embeddings", None)
    if max_position_embeddings is not None:
        return max_position_embeddings
    text_config = getattr(cfg, "text_config", None)
    return getattr(text_config, "max_position_embeddings", None)


def _max_position_embeddings_from_payload(payload: dict[str, Any]) -> int | None:
    max_position_embeddings = payload.get("max_position_embeddings")
    if isinstance(max_position_embeddings, int):
        return max_position_embeddings
    text_config = payload.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("max_position_embeddings"), int):
        return text_config["max_position_embeddings"]
    return None


def check_backbone(backbone: str, *, label_prefix: str) -> CompatRow:
    from sft.runtime.model_loading import (
        load_compatible_config,
        load_compatible_tokenizer,
        resolve_trust_remote_code,
    )

    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone={backbone!r}; known={sorted(BACKBONES)}")

    model_path, size_b = BACKBONES[backbone]
    path = Path(model_path)
    if not path.exists():
        return CompatRow(
            backbone=backbone,
            path=model_path,
            size_b=size_b,
            exists=False,
            model_type="",
            architectures=[],
            max_position_embeddings=None,
            label_single_token=False,
            causal_lm_arch=False,
            conditional_generation_arch=False,
            status="missing",
            error=f"model path does not exist: {model_path}",
        )

    try:
        trust_remote_code = resolve_trust_remote_code(model_path, True)
        cfg = load_compatible_config(model_path, trust_remote_code=trust_remote_code)
        tokenizer = load_compatible_tokenizer(model_path, trust_remote_code=True)
        architectures = [str(item) for item in (getattr(cfg, "architectures", None) or [])]
        label_single_token = True
        for letter in "ABCDEF":
            token_text = _choice_text(label_prefix, letter)
            ids = tokenizer(token_text, add_special_tokens=False, truncation=False)["input_ids"]
            if len(ids) != 1:
                label_single_token = False
                break
        causal_lm_arch = any("ForCausalLM" in arch for arch in architectures)
        conditional_generation_arch = any(arch in CONDITIONAL_GENERATION_ARCHES for arch in architectures)
        text_generation_arch = causal_lm_arch or conditional_generation_arch
        status = "ok" if label_single_token and text_generation_arch else "incompatible"
        return CompatRow(
            backbone=backbone,
            path=model_path,
            size_b=size_b,
            exists=True,
            model_type=str(getattr(cfg, "model_type", "")),
            architectures=architectures,
            max_position_embeddings=_max_position_embeddings_from_config(cfg),
            label_single_token=label_single_token,
            causal_lm_arch=causal_lm_arch,
            conditional_generation_arch=conditional_generation_arch,
            status=status,
        )
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script.
        raw_config = _read_local_config(path)
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        if str(raw_config.get("model_type", "")).lower() == "gemma4":
            status = "unsupported_transformers"
            error = (
                "current Transformers does not recognize model_type `gemma4`; "
                "use an environment with Gemma4ForConditionalGeneration support"
            )
        architectures = [str(item) for item in (raw_config.get("architectures") or [])]
        return CompatRow(
            backbone=backbone,
            path=model_path,
            size_b=size_b,
            exists=True,
            model_type=str(raw_config.get("model_type", "")),
            architectures=architectures,
            max_position_embeddings=_max_position_embeddings_from_payload(raw_config),
            label_single_token=False,
            causal_lm_arch=any("ForCausalLM" in arch for arch in architectures),
            conditional_generation_arch=any(arch in CONDITIONAL_GENERATION_ARCHES for arch in architectures),
            status=status,
            error=error,
        )


def _format_architectures(value: list[str]) -> str:
    return ",".join(value) if value else "-"


def print_table(rows: list[CompatRow]) -> None:
    headers = [
        "backbone",
        "size_b",
        "exists",
        "model_type",
        "architectures",
        "max_pos",
        "label_tok",
        "causal_lm",
        "cond_gen",
        "status",
    ]
    data: list[list[str]] = []
    for row in rows:
        data.append(
            [
                row.backbone,
                f"{row.size_b:g}",
                str(row.exists).lower(),
                row.model_type or "-",
                _format_architectures(row.architectures),
                str(row.max_position_embeddings) if row.max_position_embeddings is not None else "-",
                str(row.label_single_token).lower(),
                str(row.causal_lm_arch).lower(),
                str(row.conditional_generation_arch).lower(),
                row.status if not row.error else f"{row.status}: {row.error}",
            ]
        )
    widths = [len(header) for header in headers]
    for item in data:
        widths = [max(width, len(cell)) for width, cell in zip(widths, item)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for item in data:
        print("  ".join(cell.ljust(width) for cell, width in zip(item, widths)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Check all registered backbones.")
    group.add_argument("--backbone", choices=sorted(BACKBONES), help="Check one registered backbone.")
    p.add_argument("--label-prefix", default="Label:")
    p.add_argument("--json", action="store_true", help="Print JSON rows instead of a table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = sorted(BACKBONES) if args.all or not args.backbone else [args.backbone]
    rows = [check_backbone(name, label_prefix=str(args.label_prefix)) for name in names]
    if args.json:
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2))
    else:
        print_table(rows)
    failures = [row for row in rows if row.status != "ok"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
