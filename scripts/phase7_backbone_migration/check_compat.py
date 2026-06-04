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
    status: str
    error: str = ""


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def check_backbone(backbone: str, *, label_prefix: str) -> CompatRow:
    from transformers import AutoConfig, AutoTokenizer

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
            status="missing",
            error=f"model path does not exist: {model_path}",
        )

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        architectures = [str(item) for item in (getattr(cfg, "architectures", None) or [])]
        label_single_token = True
        for letter in "ABCDEF":
            token_text = _choice_text(label_prefix, letter)
            ids = tokenizer(token_text, add_special_tokens=False, truncation=False)["input_ids"]
            if len(ids) != 1:
                label_single_token = False
                break
        causal_lm_arch = any("ForCausalLM" in arch for arch in architectures)
        status = "ok" if label_single_token and causal_lm_arch else "incompatible"
        return CompatRow(
            backbone=backbone,
            path=model_path,
            size_b=size_b,
            exists=True,
            model_type=str(getattr(cfg, "model_type", "")),
            architectures=architectures,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", None),
            label_single_token=label_single_token,
            causal_lm_arch=causal_lm_arch,
            status=status,
        )
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script.
        return CompatRow(
            backbone=backbone,
            path=model_path,
            size_b=size_b,
            exists=True,
            model_type="",
            architectures=[],
            max_position_embeddings=None,
            label_single_token=False,
            causal_lm_arch=False,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
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
    group.add_argument("--all", action="store_true", help="Check all registered A-group backbones.")
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
