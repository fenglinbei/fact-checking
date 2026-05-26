#!/usr/bin/env python3
"""从已有 build JSONL 生成 prompt_stats 并回填到运行目录。

适用场景：只跑了 build + infer（没有 train）的实验，缺少 train/prompt_stats/。
直接从 build 缓存的 JSONL 文件（已包含 prompt_token_count、evidence_count 等字段）
计算统计信息，写入各子运行的 train/prompt_stats/ 目录。

用法:
    # 试运行：查看哪些子运行需要回填
    PYTHONPATH=src python scripts/phase1_pipeline/backfill_prompt_stats.py \
        --run-dir outputs/runs/mmr_topk_sweep_infer --dry-run

    # 实际回填
    PYTHONPATH=src python scripts/phase1_pipeline/backfill_prompt_stats.py \
        --run-dir outputs/runs/mmr_topk_sweep_infer

    # 指定 max_length（默认从 manifest config 读取，回退到 2048）
    PYTHONPATH=src python scripts/phase1_pipeline/backfill_prompt_stats.py \
        --run-dir outputs/runs/mmr_topk_sweep_infer --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fact_checking.data.io import load_jsonl
from sft.data.types import PreparedSample
from sft.prompting.stats import (
    build_prompt_snapshots,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prebuilt_prompts,
)


def _rows_to_prepared_samples(rows: list[dict]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for row in rows:
        gold_label = str(row.get("gold_label", ""))
        if not gold_label:
            continue
        samples.append(
            PreparedSample(
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
                preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
                gold_id=int(row.get("gold_id", -1)),
                gold_label=gold_label,
                gold_explain=str(row.get("gold_explain", "")),
                prompt_token_count=int(row.get("prompt_token_count", 0)),
                target_token_count=int(row.get("target_token_count", 0)),
                evidence_count=int(row.get("evidence_count", 0)),
                was_truncated=bool(row.get("was_truncated", False)),
                claim=str(row.get("claim", "")),
                no_evidence=int(row.get("evidence_count", 0)) == 0,
                long_claim=len(str(row.get("claim", "")).split()) > 64,
            )
        )
    return samples


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_build_paths(manifest: dict[str, Any]) -> dict[str, Path] | None:
    """Extract train/val build JSONL paths from manifest."""
    outputs = manifest.get("phases", {}).get("build", {}).get("outputs", {})
    train_path = outputs.get("train")
    val_path = outputs.get("val")
    if not train_path or not val_path:
        return None
    train = Path(train_path)
    val = Path(val_path)
    if train.exists() and val.exists():
        return {"train": train, "val": val}
    return None


def _resolve_max_length(manifest: dict[str, Any], sub_run_dir: Path) -> int:
    """Try to read max_length from saved config; fall back to 2048."""
    for config_path in sorted(sub_run_dir.glob("configs/*.yaml")):
        try:
            from fact_checking.config import load_yaml
            cfg = load_yaml(str(config_path))
            return int(cfg.get("sft_train", {}).get("max_length", 2048))
        except Exception:
            continue
    return 2048


def backfill(run_dir: Path, *, max_length: int | None = None, dry_run: bool = False) -> tuple[int, int, int]:
    """为 sweep 目录下所有子运行生成 prompt_stats。

    Returns (ok, skipped, missing) counts.
    """
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        return 0, 0, 0

    sub_runs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    )
    if not sub_runs:
        print(f"No sub-runs with manifest.json found under {run_dir}")
        return 0, 0, 0

    ok = skipped = missing = 0
    for sub_run in sub_runs:
        manifest = _read_manifest(sub_run / "manifest.json")

        target_dir = sub_run / "train" / "prompt_stats"
        if (target_dir / "prompt_stats.json").exists():
            print(f"[SKIP] {sub_run.name}")
            skipped += 1
            continue

        build_paths = _resolve_build_paths(manifest)
        if build_paths is None:
            print(f"[MISSING build paths] {sub_run.name}")
            missing += 1
            continue

        ml = max_length or _resolve_max_length(manifest, sub_run)

        if dry_run:
            train_ok = build_paths["train"].exists()
            val_ok = build_paths["val"].exists()
            status = "OK" if (train_ok and val_ok) else "MISSING"
            print(f"[DRY-RUN {status}] {sub_run.name}  max_length={ml}")
            if train_ok and val_ok:
                ok += 1
            else:
                missing += 1
            continue

        train_rows = load_jsonl(build_paths["train"])
        val_rows = load_jsonl(build_paths["val"])
        train_samples = _rows_to_prepared_samples(train_rows)
        val_samples = _rows_to_prepared_samples(val_rows)

        train_summary = summarize_prebuilt_prompts(train_samples, max_length=ml, split="train")
        val_summary = summarize_prebuilt_prompts(val_samples, max_length=ml, split="val")
        train_snapshots = build_prompt_snapshots(train_samples, split="train")
        val_snapshots = build_prompt_snapshots(val_samples, split="val")

        log_prompt_summary(train_summary)
        log_prompt_summary(val_summary)

        save_prompt_statistics(
            target_dir.parent,
            train_summary=train_summary,
            val_summary=val_summary,
            train_snapshots=train_snapshots,
            val_snapshots=val_snapshots,
        )
        print(f"[OK] {sub_run.name}  max_length={ml}")
        ok += 1

    print()
    print(f"Summary: ok={ok} skipped={skipped} missing={missing}")
    return ok, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="从 build JSONL 生成 prompt_stats")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ok, skipped, missing = backfill(
        Path(args.run_dir),
        max_length=args.max_length,
        dry_run=args.dry_run,
    )
    if missing > 0:
        print(
            "\n缺失的 build JSONL 文件不在本地，请在远端服务器运行此脚本。",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
