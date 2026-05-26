"""Build-time prompt statistics generation."""

from __future__ import annotations

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


def rows_to_prepared_samples(rows: list[dict]) -> list[PreparedSample]:
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


def generate_prompt_stats(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    max_length: int,
    logger: Any,
) -> None:
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    train_samples = rows_to_prepared_samples(train_rows)
    val_samples = rows_to_prepared_samples(val_rows)

    train_summary = summarize_prebuilt_prompts(train_samples, max_length=max_length, split="train")
    val_summary = summarize_prebuilt_prompts(val_samples, max_length=max_length, split="val")

    train_snapshots = build_prompt_snapshots(train_samples, split="train")
    val_snapshots = build_prompt_snapshots(val_samples, split="val")

    log_prompt_summary(train_summary, logger)
    log_prompt_summary(val_summary, logger)

    save_prompt_statistics(
        output_dir,
        train_summary=train_summary,
        val_summary=val_summary,
        train_snapshots=train_snapshots,
        val_snapshots=val_snapshots,
    )
    logger.info("Saved prompt statistics to %s/prompt_stats/", output_dir)
