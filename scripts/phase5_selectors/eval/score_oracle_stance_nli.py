"""Score Stage2 oracle candidate pools with an NLI stance model.

This is a diagnostic/cache builder for Step4.2 stance-aware selector work. It
does not use the gold label as model input; the NLI pair is evidence -> claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    candidate_text,
    load_stage2_oracle_examples,
    write_json,
)


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PAIR_ORIENTATION_EVIDENCE_CLAIM = "evidence_claim"
PAIR_ORIENTATION_CLAIM_EVIDENCE = "claim_evidence"


@dataclass(frozen=True)
class CandidateJob:
    row: dict[str, Any]
    premise: str
    hypothesis: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a candidate-level support/refute/neutral NLI cache for Stage2 oracle rows."
    )
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--model-name", default=DEFAULT_NLI_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pair-orientation", default=PAIR_ORIENTATION_EVIDENCE_CLAIM, choices=[
        PAIR_ORIENTATION_EVIDENCE_CLAIM,
        PAIR_ORIENTATION_CLAIM_EVIDENCE,
    ])
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--entailment-label-id", type=int, default=None)
    p.add_argument("--neutral-label-id", type=int, default=None)
    p.add_argument("--contradiction-label-id", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _validate_shard_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path, manifest_path = _output_paths(out_dir, args)

    started_at = time.time()
    existing_keys = _load_existing_keys(scores_path) if args.resume else set()
    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=args.filter_policy,
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    examples = [
        example
        for example in examples
        if _event_in_shard(example.event_id, shard_index=int(args.shard_index), num_shards=int(args.num_shards))
    ]
    if not examples:
        raise ValueError("No examples after Stage2 audit/filtering/sharding.")

    jobs, skipped_resume = _build_jobs(examples, args=args, existing_keys=existing_keys)
    device = _resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    label_indices = _resolve_nli_label_indices(model.config, args)
    model.to(device)
    model.eval()

    n_written = 0
    with scores_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for batch in tqdm(
            _batches(jobs, int(args.batch_size)),
            total=max(math.ceil(len(jobs) / max(int(args.batch_size), 1)), 1),
            desc=f"stance nli [{args.split}]",
            unit="batch",
            dynamic_ncols=True,
            disable=bool(args.no_progress) or not jobs,
        ):
            rows = _score_batch(
                batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=int(args.max_length),
                label_indices=label_indices,
            )
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            fh.flush()

    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "oracle_results": str(args.oracle_results),
        "output_path": str(scores_path),
        "split": str(args.split),
        "model_name": str(args.model_name),
        "revision": args.revision,
        "model_config_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "id2label": _jsonable_id2label(getattr(model.config, "id2label", {})),
        "label2id": {str(k): int(v) for k, v in getattr(model.config, "label2id", {}).items()},
        "stance_label_indices": label_indices,
        "pair_orientation": str(args.pair_orientation),
        "pair_semantics": {
            "premise": "candidate evidence text" if args.pair_orientation == PAIR_ORIENTATION_EVIDENCE_CLAIM else "claim",
            "hypothesis": "claim" if args.pair_orientation == PAIR_ORIENTATION_EVIDENCE_CLAIM else "candidate evidence text",
            "support_score": "P(entailment | premise, hypothesis)",
            "refute_score": "P(contradiction | premise, hypothesis)",
            "neutral_score": "P(neutral | premise, hypothesis)",
            "qualify_proxy_score": "neutral_score * clipped_hybrid_score; proxy only, not a true NLI class",
        },
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "max_candidates": int(args.max_candidates),
        "top_k": int(args.top_k),
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "sample_limit": args.sample_limit,
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "resume": bool(args.resume),
        "n_examples": len(examples),
        "n_candidate_jobs": len(jobs),
        "n_written_this_run": int(n_written),
        "n_skipped_resume": int(skipped_resume),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote stance scores: {scores_path}")
    print(f"Scored {n_written} candidates; skipped {skipped_resume} already-present candidates.")


def _build_jobs(
    examples: list[Any],
    *,
    args: argparse.Namespace,
    existing_keys: set[str],
) -> tuple[list[CandidateJob], int]:
    jobs: list[CandidateJob] = []
    skipped = 0
    for example in examples:
        selected_order = {int(idx): pos for pos, idx in enumerate(example.selected_indices)}
        for position, candidate in enumerate(example.candidates):
            row = _candidate_base_row(example, candidate, position, selected_order, args=args)
            key = _candidate_key(row)
            row["cache_key"] = key
            if key in existing_keys:
                skipped += 1
                continue
            premise, hypothesis = _nli_pair(row, pair_orientation=str(args.pair_orientation))
            jobs.append(CandidateJob(row=row, premise=premise, hypothesis=hypothesis))
    return jobs, skipped


def _candidate_base_row(
    example: Any,
    candidate: dict[str, Any],
    position: int,
    selected_order: dict[int, int],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    score = example.candidate_scores[position] if position < len(example.candidate_scores) else {}
    source_report = candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {}
    idx = _safe_int(candidate.get("candidate_idx", score.get("candidate_idx", position)), position)
    text = candidate_text(candidate)
    selected = idx in selected_order
    row = {
        "event_id": example.event_id,
        "split": str(args.split),
        "claim": example.claim,
        "gold_label": example.gold_label,
        "margin": float(example.margin),
        "is_correct": bool(example.is_correct),
        "chunk_mmr_fingerprint": example.fingerprint,
        "candidate_pool_fingerprint": str(example.raw.get("candidate_pool_fingerprint") or ""),
        "n_candidates": len(example.candidates),
        "top_k": int(args.top_k),
        "candidate_position": int(position),
        "candidate_idx": int(idx),
        "candidate_uid": str(candidate.get("candidate_uid") or score.get("candidate_uid") or f"{example.event_id}:{idx}"),
        "source_index": _nullable_int(candidate.get("source_index", score.get("source_index"))),
        "report_id": _nullable_int(candidate.get("report_id")),
        "sent_idx": _nullable_int(candidate.get("sent_idx")),
        "chunk_sent_indices": candidate.get("chunk_sent_indices") or [],
        "source_link": str(source_report.get("link") or ""),
        "source_domain": str(source_report.get("domain") or ""),
        "text": text,
        "selected": bool(selected),
        "oracle_step": int(selected_order[idx]) if selected else -1,
    }
    for key in ("hybrid_rank", "dense_score", "lexical_score", "bm25_score", "hybrid_score"):
        row[key] = _nullable_float(score.get(key))
    return row


def _nli_pair(row: dict[str, Any], *, pair_orientation: str) -> tuple[str, str]:
    if pair_orientation == PAIR_ORIENTATION_EVIDENCE_CLAIM:
        return str(row.get("text") or ""), str(row.get("claim") or "")
    if pair_orientation == PAIR_ORIENTATION_CLAIM_EVIDENCE:
        return str(row.get("claim") or ""), str(row.get("text") or "")
    raise ValueError(f"Unknown pair_orientation: {pair_orientation}")


@torch.inference_mode()
def _score_batch(
    batch: list[CandidateJob],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    label_indices: dict[str, int],
) -> list[dict[str, Any]]:
    premises = [item.premise for item in batch]
    hypotheses = [item.hypothesis for item in batch]
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded).logits.detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)

    rows: list[dict[str, Any]] = []
    id2label = _jsonable_id2label(getattr(model.config, "id2label", {}))
    for item, logit_row, prob_row in zip(batch, logits, probs):
        row = dict(item.row)
        support = float(prob_row[label_indices["entailment"]])
        neutral = float(prob_row[label_indices["neutral"]])
        refute = float(prob_row[label_indices["contradiction"]])
        support_logit = float(logit_row[label_indices["entailment"]])
        neutral_logit = float(logit_row[label_indices["neutral"]])
        refute_logit = float(logit_row[label_indices["contradiction"]])
        stance_scores = {
            "support": support,
            "neutral": neutral,
            "refute": refute,
        }
        stance_label = max(stance_scores.items(), key=lambda item_: item_[1])[0]
        hybrid = _clip01(_nullable_float(row.get("hybrid_score")), default=1.0)
        row.update(
            {
                "support_score": support,
                "neutral_score": neutral,
                "refute_score": refute,
                "support_logit": support_logit,
                "neutral_logit": neutral_logit,
                "refute_logit": refute_logit,
                "stance_label": stance_label,
                "stance_confidence": float(stance_scores[stance_label]),
                "stance_polarity": support - refute,
                "support_refute_margin": support - refute,
                "support_refute_abs_margin": abs(support - refute),
                "qualify_proxy_score": neutral * hybrid,
                "nli_probs_by_label": {
                    str(id2label.get(int(idx), idx)): float(prob_row[int(idx)])
                    for idx in range(int(prob_row.numel()))
                },
                "nli_logits_by_label": {
                    str(id2label.get(int(idx), idx)): float(logit_row[int(idx)])
                    for idx in range(int(logit_row.numel()))
                },
            }
        )
        rows.append(row)
    return rows


def _resolve_nli_label_indices(config: Any, args: argparse.Namespace) -> dict[str, int]:
    manual = {
        "entailment": args.entailment_label_id,
        "neutral": args.neutral_label_id,
        "contradiction": args.contradiction_label_id,
    }
    if any(value is not None for value in manual.values()):
        if any(value is None for value in manual.values()):
            raise ValueError(
                "When overriding NLI label ids, provide all three of "
                "--entailment-label-id, --neutral-label-id, and --contradiction-label-id."
            )
        return {key: int(value) for key, value in manual.items() if value is not None}

    labels: list[tuple[int, str]] = []
    for raw_idx, raw_label in getattr(config, "id2label", {}).items():
        labels.append((int(raw_idx), str(raw_label)))
    for raw_label, raw_idx in getattr(config, "label2id", {}).items():
        pair = (int(raw_idx), str(raw_label))
        if pair not in labels:
            labels.append(pair)

    resolved: dict[str, int] = {}
    for idx, label in labels:
        norm = _normalize_label_name(label)
        if _looks_like_entailment(norm):
            resolved.setdefault("entailment", int(idx))
        elif "neutral" in norm or norm in {"nei", "neither", "unknown", "not_enough_info"}:
            resolved.setdefault("neutral", int(idx))
        elif "contrad" in norm or "refut" in norm:
            resolved.setdefault("contradiction", int(idx))

    missing = [key for key in ("entailment", "neutral", "contradiction") if key not in resolved]
    if missing:
        raise ValueError(
            "Could not resolve NLI label ids from model config. "
            f"Missing {missing}; id2label={getattr(config, 'id2label', {})}, "
            f"label2id={getattr(config, 'label2id', {})}. "
            "Pass explicit --entailment-label-id/--neutral-label-id/--contradiction-label-id."
        )
    return resolved


def _normalize_label_name(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _looks_like_entailment(norm: str) -> bool:
    if norm in {"entailment", "entails", "entailed", "support", "supports", "supported"}:
        return True
    return "entail" in norm and "not_entail" not in norm and "non_entail" not in norm


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add(_candidate_key(row))
    return keys


def _candidate_key(row: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(row.get("event_id", "")),
            str(row.get("candidate_uid", "")),
            str(row.get("candidate_idx", "")),
        ]
    )


def _event_in_shard(event_id: str, *, shard_index: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    digest = hashlib.sha1(str(event_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_shards) == int(shard_index)


def _validate_shard_args(args: argparse.Namespace) -> None:
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")


def _output_paths(out_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    if int(args.num_shards) <= 1:
        return out_dir / "candidate_stance_scores.jsonl", out_dir / "manifest.json"
    suffix = f"shard_{int(args.shard_index):05d}_of_{int(args.num_shards):05d}"
    return out_dir / f"candidate_stance_scores.{suffix}.jsonl", out_dir / f"manifest.{suffix}.json"


def _resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Use --device cpu.")
    return device


def _batches(items: list[CandidateJob], batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _jsonable_id2label(id2label: Any) -> dict[int, str]:
    return {int(key): str(value) for key, value in dict(id2label or {}).items()}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clip01(value: float | None, *, default: float) -> float:
    if value is None:
        return float(default)
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    main()
