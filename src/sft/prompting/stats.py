from __future__ import annotations

import json
from logging import Logger
from pathlib import Path

import numpy as np

from fact_checking.utils.logging import init_logger
from sft.data.types import PreparedSample

module_logger = init_logger(__name__)

SNAPSHOT_CATEGORIES = [
    "no_evidence",
    "long_claim",
    "was_truncated",
]

LONG_CLAIM_TOKEN_THRESHOLD = 64


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
    before = summary.get("prompt_token_count", summary.get("prompt_length_before_truncation", {}))
    after = summary.get("prompt_token_count", summary.get("prompt_length_after_truncation", {}))
    trunc = summary["evidence_truncation"]
    target_logger = logger or module_logger
    target_logger.info(
        "[PROMPT_STATS] split=%s pre_mean=%.2f pre_p95=%.0f "
        "pre_overflow=%d post_mean=%.2f post_p95=%.0f post_overflow=%d truncated=%s trunc_rate=%.4f",
        summary["split"],
        before.get("mean", 0),
        before.get("p95", 0),
        int(before.get("overflow_count", 0)),
        after.get("mean", 0),
        after.get("p95", 0),
        int(after.get("overflow_count", 0)),
        trunc.get("truncated_count", 0),
        trunc.get("truncation_rate", 0.0),
    )


def summarize_prebuilt_prompts(
    samples: list[PreparedSample],
    max_length: int,
    split: str,
) -> dict[str, object]:
    prompt_tokens = [s.prompt_token_count for s in samples]
    target_tokens = [s.target_token_count for s in samples]
    evidence_counts = [s.evidence_count for s in samples]
    truncated_count = sum(int(s.was_truncated) for s in samples)
    count = len(samples)

    return {
        "split": split,
        "max_length": int(max_length),
        "prompt_token_count": _summarize_lengths(prompt_tokens, max_length=max_length),
        "target_token_count": _summarize_lengths(target_tokens, max_length=max_length),
        "snapshot_category_counts": _summarize_categories(samples),
        "evidence_truncation": {
            "truncated_count": int(truncated_count),
            "truncation_rate": float(truncated_count / count) if count > 0 else 0.0,
            "mean_evidence_count": float(np.mean(evidence_counts)) if evidence_counts else 0.0,
            "min_evidence_count": int(min(evidence_counts)) if evidence_counts else 0,
            "max_evidence_count": int(max(evidence_counts)) if evidence_counts else 0,
        },
    }


def build_prompt_snapshots(
    samples: list[PreparedSample],
    *,
    split: str,
    limit_per_category: int = 3,
    max_prompt_chars: int = 6000,
) -> dict[str, list[dict[str, object]]]:
    snapshots: dict[str, list[dict[str, object]]] = {category: [] for category in SNAPSHOT_CATEGORIES}
    for sample_idx, sample in enumerate(samples):
        for category in _record_categories(sample):
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
                    "prompt_token_count": sample.prompt_token_count,
                    "target_token_count": sample.target_token_count,
                    "evidence_count": sample.evidence_count,
                    "was_truncated": sample.was_truncated,
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


def _summarize_categories(samples: list[PreparedSample]) -> dict[str, dict[str, float]]:
    count = len(samples)
    summary: dict[str, dict[str, float]] = {}
    for category in SNAPSHOT_CATEGORIES:
        category_count = sum(int(getattr(s, category)) for s in samples)
        summary[category] = {
            "count": float(category_count),
            "rate": float(category_count / count) if count > 0 else 0.0,
        }
    return summary


def _record_categories(sample: PreparedSample) -> list[str]:
    categories: list[str] = []
    if sample.no_evidence:
        categories.append("no_evidence")
    if len(sample.claim.split()) > LONG_CLAIM_TOKEN_THRESHOLD:
        categories.append("long_claim")
    if sample.was_truncated:
        categories.append("was_truncated")
    return categories


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
