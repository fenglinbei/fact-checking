#!/usr/bin/env python
"""Prepare a LoRA verifier checkpoint for vLLM OpenAI serving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--merge-cache-dir", default="outputs/cache/merged_lora")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    _require_file(adapter_dir / "adapter_config.json", "adapter config")
    if not (adapter_dir / "adapter_model.safetensors").exists() and not (adapter_dir / "adapter_model.bin").exists():
        raise FileNotFoundError(f"Missing adapter weights under {adapter_dir}")

    from fact_checking.infer.api import _merge_lora_to_cache

    merged_dir = _merge_lora_to_cache(
        base_model=str(args.base_model),
        adapter_dir=adapter_dir,
        tokenizer_dir=str(args.tokenizer_dir),
        dtype=str(args.dtype),
        cache_dir=str(args.merge_cache_dir),
        force_rebuild=bool(args.force_rebuild),
    )
    print(str(merged_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[prepare-lora-vllm-model][fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
