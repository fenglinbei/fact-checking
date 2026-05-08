from __future__ import annotations

import json
from logging import Logger
from pathlib import Path

import numpy as np

from fact_checking.utils.logging import init_logger
from sft.data.types import PreparedSample, PromptPreparationRecord

module_logger = init_logger(__name__)

SNAPSHOT_CATEGORIES = [
    "no_evidence",
    "long_claim",
    "duplicate_evidence",
    "long_report",
]


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
        "[PROMPT_STATS] split=%s version=%s mode=%s strategy=%s pre_mean=%.2f pre_p95=%.0f "
        "pre_overflow=%d post_mean=%.2f post_p95=%.0f post_overflow=%d truncated=%s trunc_rate=%.4f",
        summary["split"],
        summary["prompt_version"],
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
    prompt_version: str,
) -> dict[str, object]:
    before_lengths = [record.prompt_length_before_trunc for record in records]
    after_lengths = [record.prompt_length_after_trunc for record in records]
    target_lengths = [record.target_length for record in records]
    sequence_before_lengths = [record.sequence_length_before_trunc for record in records]
    sequence_after_lengths = [record.sequence_length_after_trunc for record in records]
    claim_lengths = [record.claim_token_count for record in records]
    report_lengths = [record.max_report_char_count for record in records]
    evidence_before = [record.evidence_count_before_trunc for record in records]
    evidence_after = [record.evidence_count_after_trunc for record in records]
    truncated_count = sum(int(record.was_truncated) for record in records)
    overflow_before_count = sum(int(record.overflow_before_trunc) for record in records)
    overflow_after_count = sum(int(record.overflow_after_trunc) for record in records)
    count = len(records)

    return {
        "split": split,
        "max_length": int(max_length),
        "prompt_version": prompt_version,
        "output_mode": output_mode,
        "prompt_truncation_strategy": truncation_strategy_name,
        "prompt_length_before_truncation": _summarize_lengths(before_lengths, max_length=max_length),
        "prompt_length_after_truncation": _summarize_lengths(after_lengths, max_length=max_length),
        "target_length": _summarize_lengths(target_lengths, max_length=max_length),
        "sequence_length_before_truncation": _summarize_lengths(sequence_before_lengths, max_length=max_length),
        "sequence_length_after_truncation": _summarize_lengths(sequence_after_lengths, max_length=max_length),
        "claim_length": _summarize_lengths(claim_lengths, max_length=max_length),
        "max_source_report_char_length": _summarize_lengths(report_lengths, max_length=max_length),
        "snapshot_category_counts": _summarize_categories(records),
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


def build_prompt_snapshots(
    samples: list[PreparedSample],
    records: list[PromptPreparationRecord],
    *,
    split: str,
    limit_per_category: int = 3,
    max_prompt_chars: int = 6000,
) -> dict[str, list[dict[str, object]]]:
    snapshots: dict[str, list[dict[str, object]]] = {category: [] for category in SNAPSHOT_CATEGORIES}
    for sample_idx, (sample, record) in enumerate(zip(samples, records)):
        for category in _record_categories(record):
            if len(snapshots[category]) >= limit_per_category:
                continue
            snapshots[category].append(
                {
                    "split": split,
                    "sample_idx": sample_idx,
                    "category": category,
                    "gold_label": sample.gold_label,
                    "prompt": _truncate_text(sample.prompt, max_prompt_chars),
                    "target": sample.target,
                    "prompt_was_truncated_for_snapshot": len(sample.prompt) > max_prompt_chars,
                    "prompt_length_after_trunc": record.prompt_length_after_trunc,
                    "target_length": record.target_length,
                    "sequence_length_after_trunc": record.sequence_length_after_trunc,
                    "evidence_count_before_trunc": record.evidence_count_before_trunc,
                    "evidence_count_after_trunc": record.evidence_count_after_trunc,
                    "claim_token_count": record.claim_token_count,
                    "max_report_char_count": record.max_report_char_count,
                }
            )
    return snapshots


def save_prompt_statistics(
    output_dir: Path,
    train_summary: dict[str, object],
    val_summary: dict[str, object],
    train_snapshots: dict[str, list[dict[str, object]]] | None = None,
    val_snapshots: dict[str, list[dict[str, object]]] | None = None,
) -> Path:
    stats_dir = output_dir / "prompt_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "prompt_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({"train": train_summary, "val": val_summary}, f, ensure_ascii=False, indent=2)
    snapshots_path = stats_dir / "prompt_snapshots.json"
    with snapshots_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_snapshots or {},
                "val": val_snapshots or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return stats_path


def _summarize_categories(records: list[PromptPreparationRecord]) -> dict[str, dict[str, float]]:
    count = len(records)
    summary: dict[str, dict[str, float]] = {}
    for category in SNAPSHOT_CATEGORIES:
        category_count = sum(int(getattr(record, category)) for record in records)
        summary[category] = {
            "count": float(category_count),
            "rate": float(category_count / count) if count > 0 else 0.0,
        }
    return summary


def _record_categories(record: PromptPreparationRecord) -> list[str]:
    return [category for category in SNAPSHOT_CATEGORIES if bool(getattr(record, category))]


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
