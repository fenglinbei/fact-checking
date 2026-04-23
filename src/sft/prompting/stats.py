from __future__ import annotations

import json
from logging import Logger
from pathlib import Path

import numpy as np

from fact_checking.utils.logging import init_logger
from sft.data.types import PromptPreparationRecord

module_logger = init_logger(__name__)


def _summarize_lengths(lengths: list[int], max_length: int) -> dict[str, float]:
    if not lengths:
        return {
            "count": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "overflow_count": 0.0,
            "overflow_rate": 0.0,
        }

    arr = np.asarray(lengths, dtype=np.int64)
    overflow = arr > int(max_length)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "overflow_count": float(np.sum(overflow)),
        "overflow_rate": float(np.mean(overflow)),
    }


def log_prompt_summary(summary: dict[str, object], logger: Logger | None = None) -> None:
    before = summary["prompt_length_before_truncation"]
    after = summary["prompt_length_after_truncation"]
    trunc = summary["evidence_truncation"]
    target_logger = logger or module_logger
    target_logger.info(
        "[PROMPT_STATS] split=%s mode=%s strategy=%s pre_mean=%.2f pre_p95=%.0f pre_overflow=%d "
        "post_mean=%.2f post_p95=%.0f post_overflow=%d truncated=%s trunc_rate=%.4f",
        summary["split"],
        summary["output_mode"],
        summary["prompt_truncation_strategy"],
        before["mean"],
        before["p95"],
        int(before["overflow_count"]),
        after["mean"],
        after["p95"],
        int(after["overflow_count"]),
        trunc["truncated_count"],
        trunc["truncation_rate"],
    )


def summarize_prompt_preparation(
    records: list[PromptPreparationRecord],
    max_length: int,
    split: str,
    truncation_strategy_name: str,
    output_mode: str,
) -> dict[str, object]:
    before_lengths = [record.prompt_length_before_trunc for record in records]
    after_lengths = [record.prompt_length_after_trunc for record in records]
    evidence_before = [record.evidence_count_before_trunc for record in records]
    evidence_after = [record.evidence_count_after_trunc for record in records]
    truncated_count = sum(int(record.was_truncated) for record in records)
    overflow_before_count = sum(int(record.overflow_before_trunc) for record in records)
    overflow_after_count = sum(int(record.overflow_after_trunc) for record in records)
    count = len(records)

    return {
        "split": split,
        "max_length": int(max_length),
        "output_mode": output_mode,
        "prompt_truncation_strategy": truncation_strategy_name,
        "prompt_length_before_truncation": _summarize_lengths(before_lengths, max_length=max_length),
        "prompt_length_after_truncation": _summarize_lengths(after_lengths, max_length=max_length),
        "evidence_truncation": {
            "truncated_count": int(truncated_count),
            "truncation_rate": float(truncated_count / count) if count > 0 else 0.0,
            "overflow_before_count": int(overflow_before_count),
            "overflow_before_rate": float(overflow_before_count / count) if count > 0 else 0.0,
            "overflow_after_count": int(overflow_after_count),
            "overflow_after_rate": float(overflow_after_count / count) if count > 0 else 0.0,
            "mean_evidence_count_before": float(np.mean(evidence_before)) if evidence_before else 0.0,
            "mean_evidence_count_after": float(np.mean(evidence_after)) if evidence_after else 0.0,
            "min_evidence_count_after": int(min(evidence_after)) if evidence_after else 0,
            "max_evidence_count_after": int(max(evidence_after)) if evidence_after else 0,
        },
    }


def save_prompt_statistics(
    output_dir: Path,
    train_summary: dict[str, object],
    val_summary: dict[str, object],
) -> Path:
    stats_dir = output_dir / "prompt_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "prompt_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({"train": train_summary, "val": val_summary}, f, ensure_ascii=False, indent=2)
    return stats_path
