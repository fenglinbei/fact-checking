"""Diagnose whether lightweight claim aspects explain Stage2 oracle selection."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from fact_checking.selectors.aspects import (
    ASPECT_EXTRACTION_VERSION,
    extract_claim_aspects,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
    load_stage2_oracle_examples,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_ASPECT_ENCODER = "microsoft/deberta-v3-base"
ALIGNMENT_FEATURES = (
    "max_aspect_score",
    "mean_aspect_score",
    "aspect_score_entropy",
)
STEP_FEATURES = (
    "uncovered_gain",
    "covered_overlap",
    "max_aspect_score",
    "mean_aspect_score",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build lightweight claim aspects and analyze oracle aspect coverage signal."
    )
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--max-local-aspects", type=int, default=8)
    p.add_argument("--min-retrievability-score", type=int, default=2)
    p.add_argument("--min-aspect-tokens", type=int, default=4)
    p.add_argument("--max-aspect-tokens", type=int, default=24)
    p.add_argument("--claim-aspects-input", default=None)
    p.add_argument("--fallback-full-claim-when-empty", action="store_true", default=True)
    p.add_argument("--no-fallback-full-claim-when-empty", dest="fallback_full_claim_when_empty", action="store_false")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--model-name", default=DEFAULT_ASPECT_ENCODER)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--min-coverage-lift-pp", type=float, default=3.0)
    p.add_argument("--min-uncovered-gain-auroc", type=float, default=0.57)
    p.add_argument("--top-examples", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=args.filter_policy,
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No examples after Stage2 audit/filtering.")

    bundle_rows = _load_or_extract_aspect_rows(args, examples)
    write_jsonl(out_dir / "claim_aspects.jsonl", bundle_rows)
    extraction_summary = _extraction_summary(bundle_rows)
    write_json(out_dir / "aspect_extraction_summary.json", extraction_summary)

    if args.extract_only:
        manifest = _manifest(args, out_dir, started_at, n_examples=len(examples), extract_only=True)
        manifest["aspect_extraction"] = extraction_summary
        write_json(out_dir / "manifest.json", manifest)
        _write_extraction_markdown(out_dir / "analysis.md", extraction_summary=extraction_summary, args=args)
        print(f"Wrote claim aspects: {out_dir / 'claim_aspects.jsonl'}")
        print("Extraction only; skipped encoder alignment.")
        return

    device = _resolve_device(args.device)
    tokenizer, model = _load_encoder(args, device)
    model.eval()

    text_by_key = _collect_texts_for_embedding(examples, bundle_rows, fallback_full_claim=bool(args.fallback_full_claim_when_empty))
    embeddings = _embed_texts(
        text_by_key,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        no_progress=bool(args.no_progress),
    )

    alignment_rows: list[dict[str, Any]] = []
    candidate_probe_rows: list[dict[str, Any]] = []
    step_probe_rows: list[dict[str, Any]] = []
    event_coverage_rows: list[dict[str, Any]] = []
    for example, bundle in tqdm(
        list(zip(examples, bundle_rows)),
        desc="aspect coverage",
        unit="claim",
        dynamic_ncols=True,
        disable=bool(args.no_progress),
    ):
        event_payload = _event_alignment_payload(
            example,
            bundle,
            embeddings,
            fallback_full_claim=bool(args.fallback_full_claim_when_empty),
            top_k=int(args.top_k),
        )
        alignment_rows.extend(event_payload["candidate_alignment_rows"])
        candidate_probe_rows.extend(event_payload["candidate_probe_rows"])
        step_probe_rows.extend(event_payload["step_probe_rows"])
        event_coverage_rows.append(event_payload["event_coverage"])

    write_jsonl(out_dir / "candidate_aspect_alignment.jsonl", alignment_rows)
    candidate_probe = _feature_probe(candidate_probe_rows, label_key="selected", features=ALIGNMENT_FEATURES)
    step_probe = _feature_probe(step_probe_rows, label_key="target", features=STEP_FEATURES)
    set_coverage = _set_coverage_summary(event_coverage_rows)
    decision = _decision_payload(
        step_probe,
        set_coverage,
        min_coverage_lift_pp=float(args.min_coverage_lift_pp),
        min_uncovered_gain_auroc=float(args.min_uncovered_gain_auroc),
    )
    analysis = {
        "oracle_results": str(args.oracle_results),
        "split": str(args.split),
        "model_name": str(args.model_name),
        "claim_aspects_input": str(args.claim_aspects_input) if args.claim_aspects_input else None,
        "aspect_extraction": extraction_summary,
        "candidate_probe": candidate_probe,
        "step_uncovered_gain_probe": step_probe,
        "set_coverage": set_coverage,
        "decision": decision,
        "examples": _top_step_examples(step_probe_rows, limit=int(args.top_examples)),
    }
    write_json(out_dir / "oracle_aspect_coverage_analysis.json", analysis)
    write_json(
        out_dir / "analysis_summary.json",
        {
            "n_events": len(examples),
            "n_candidate_alignment_rows": len(alignment_rows),
            "n_step_probe_rows": len(step_probe_rows),
            "decision": decision,
            "aspect_extraction": extraction_summary,
            "set_coverage": set_coverage,
        },
    )
    manifest = _manifest(args, out_dir, started_at, n_examples=len(examples), extract_only=False)
    manifest["aspect_extraction"] = extraction_summary
    manifest["n_candidate_alignment_rows"] = len(alignment_rows)
    manifest["n_step_probe_rows"] = len(step_probe_rows)
    write_json(out_dir / "manifest.json", manifest)
    _write_analysis_markdown(out_dir / "analysis.md", analysis=analysis)

    print(f"Wrote aspect coverage analysis under: {out_dir}")
    print(
        "Decision={decision}; uncovered_gain_auc={auc:.4f}; oracle_vs_hybrid_lift={lift:.2f}pp".format(
            decision=decision["decision"],
            auc=float(decision["uncovered_gain_auroc"]),
            lift=float(decision["oracle_vs_hybrid_coverage_lift_pp"]),
        )
    )


def _load_encoder(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    tokenizer_kwargs = {
        "revision": args.revision,
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
        "fix_mistral_regex": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    except Exception as exc:
        message = str(exc)
        if "SentencePiece" in message or "sentencepiece" in message or "tiktoken" in message:
            raise RuntimeError(
                "Failed to load the aspect encoder tokenizer. "
                f"Model {args.model_name!r} usually requires the optional dependency `sentencepiece`; "
                "install project dependencies or run `python3 -m pip install sentencepiece`, "
                "then rerun the diagnostic. You can also pass a local encoder path with a tokenizer "
                "that is already supported in this environment."
            ) from exc
        raise

    model = AutoModel.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    return tokenizer, model


def _collect_texts_for_embedding(
    examples: list[Stage2OracleExample],
    bundles: list[dict[str, Any]],
    *,
    fallback_full_claim: bool,
) -> dict[str, str]:
    text_by_key: dict[str, str] = {}
    for example, bundle in zip(examples, bundles):
        for idx, candidate in enumerate(example.candidates):
            text_by_key[_candidate_key(example.event_id, idx)] = candidate_text(candidate)
        for aspect in _coverage_aspects(bundle, fallback_full_claim=fallback_full_claim):
            text_by_key[_aspect_key(aspect)] = str(aspect["text"])
    return text_by_key


def _load_or_extract_aspect_rows(args: argparse.Namespace, examples: list[Stage2OracleExample]) -> list[dict[str, Any]]:
    if args.claim_aspects_input:
        rows = read_jsonl(args.claim_aspects_input)
        by_event: dict[str, dict[str, Any]] = {}
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id not in by_event:
                by_event[event_id] = dict(row)
        missing = [example.event_id for example in examples if example.event_id not in by_event]
        if missing:
            raise ValueError(
                f"claim-aspects-input is missing {len(missing)} audited oracle examples; "
                f"first missing event_id={missing[0]}"
            )
        return [by_event[example.event_id] for example in examples]

    bundles = [
        extract_claim_aspects(
            example.claim,
            event_id=example.event_id,
            max_local_aspects=int(args.max_local_aspects),
            min_retrievability_score=int(args.min_retrievability_score),
            min_tokens=int(args.min_aspect_tokens),
            max_tokens=int(args.max_aspect_tokens),
        )
        for example in examples
    ]
    return [bundle.to_dict() for bundle in bundles]


@torch.inference_mode()
def _embed_texts(
    text_by_key: dict[str, str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
    no_progress: bool,
) -> dict[str, np.ndarray]:
    keys = list(text_by_key)
    embeddings: dict[str, np.ndarray] = {}
    for batch_keys in tqdm(
        _batches(keys, int(batch_size)),
        total=max(math.ceil(len(keys) / max(int(batch_size), 1)), 1),
        desc="aspect/candidate encode",
        unit="batch",
        dynamic_ncols=True,
        disable=no_progress,
    ):
        texts = [text_by_key[key] for key in batch_keys]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded.get("attention_mask"))
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        for key, vector in zip(batch_keys, pooled.detach().cpu().numpy()):
            embeddings[key] = vector.astype(np.float32)
    return embeddings


def _event_alignment_payload(
    example: Stage2OracleExample,
    bundle: dict[str, Any],
    embeddings: dict[str, np.ndarray],
    *,
    fallback_full_claim: bool,
    top_k: int,
) -> dict[str, Any]:
    aspects = _coverage_aspects(bundle, fallback_full_claim=fallback_full_claim)
    candidate_keys = [_candidate_key(example.event_id, idx) for idx in range(len(example.candidates))]
    aspect_keys = [_aspect_key(aspect) for aspect in aspects]
    if not aspects:
        zero_rows = _empty_event_payload(example, top_k=top_k)
        return zero_rows

    candidate_matrix = np.stack([embeddings[key] for key in candidate_keys], axis=0)
    aspect_matrix = np.stack([embeddings[key] for key in aspect_keys], axis=0)
    raw_cosine = candidate_matrix @ aspect_matrix.T
    alignment = np.clip((raw_cosine + 1.0) / 2.0, 0.0, 1.0)

    selected_set = {int(idx) for idx in example.selected_indices[:top_k]}
    selected_order = {int(idx): pos for pos, idx in enumerate(example.selected_indices[:top_k])}
    candidate_alignment_rows: list[dict[str, Any]] = []
    candidate_probe_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(example.candidates):
        score_row = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
        scores = alignment[idx]
        row = {
            "event_id": example.event_id,
            "gold_label": example.gold_label,
            "candidate_idx": int(idx),
            "candidate_uid": str(candidate.get("candidate_uid") or score_row.get("candidate_uid") or f"{example.event_id}:{idx}"),
            "selected": idx in selected_set,
            "oracle_step": int(selected_order[idx]) if idx in selected_order else -1,
            "max_aspect_score": float(np.max(scores)),
            "mean_aspect_score": float(np.mean(scores)),
            "aspect_score_entropy": _normalized_entropy(scores),
            "n_aspects": len(aspects),
            "aspect_scores": [
                {
                    "aspect_id": aspect["aspect_id"],
                    "type": aspect["type"],
                    "text": aspect["text"],
                    "score": float(score),
                }
                for aspect, score in zip(aspects, scores)
            ],
        }
        for key in ("hybrid_rank", "dense_score", "lexical_score", "bm25_score", "hybrid_score"):
            row[key] = _nullable_float(score_row.get(key))
        candidate_alignment_rows.append(row)
        candidate_probe_rows.append({key: row[key] for key in ("selected", *ALIGNMENT_FEATURES)})

    step_probe_rows = _step_probe_rows(example, alignment, aspects, top_k=top_k)
    oracle_indices = [idx for idx in example.selected_indices[:top_k] if 0 <= int(idx) < len(example.candidates)]
    hybrid_indices = _hybrid_top_indices(example, top_k=top_k)
    candidate_order_indices = list(range(min(int(top_k), len(example.candidates))))
    event_coverage = {
        "event_id": example.event_id,
        "gold_label": example.gold_label,
        "n_aspects": len(aspects),
        "aspect_types": [aspect["type"] for aspect in aspects],
        "oracle_coverage": _set_coverage(alignment, oracle_indices),
        "hybrid_topk_coverage": _set_coverage(alignment, hybrid_indices),
        "candidate_order_topk_coverage": _set_coverage(alignment, candidate_order_indices),
        "oracle_indices": [int(idx) for idx in oracle_indices],
        "hybrid_indices": [int(idx) for idx in hybrid_indices],
    }
    return {
        "candidate_alignment_rows": candidate_alignment_rows,
        "candidate_probe_rows": candidate_probe_rows,
        "step_probe_rows": step_probe_rows,
        "event_coverage": event_coverage,
    }


def _step_probe_rows(
    example: Stage2OracleExample,
    alignment: np.ndarray,
    aspects: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_prefix: list[int] = []
    n_candidates = alignment.shape[0]
    for step, target_idx in enumerate(example.selected_indices[:top_k]):
        target_idx = int(target_idx)
        if target_idx < 0 or target_idx >= n_candidates:
            continue
        if selected_prefix:
            covered = np.max(alignment[selected_prefix], axis=0)
        else:
            covered = np.zeros((alignment.shape[1],), dtype=np.float64)
        for idx in range(n_candidates):
            if idx in selected_prefix:
                continue
            scores = alignment[idx]
            gain = np.maximum(scores - covered, 0.0)
            overlap = np.minimum(scores, covered)
            rows.append(
                {
                    "event_id": example.event_id,
                    "gold_label": example.gold_label,
                    "step": int(step),
                    "candidate_idx": int(idx),
                    "target": idx == target_idx,
                    "uncovered_gain": float(np.mean(gain)),
                    "covered_overlap": float(np.mean(overlap)),
                    "max_aspect_score": float(np.max(scores)),
                    "mean_aspect_score": float(np.mean(scores)),
                    "target_aspect_texts": [
                        aspects[int(pos)]["text"]
                        for pos in np.argsort(-gain)[: min(3, len(aspects))]
                        if float(gain[int(pos)]) > 0.0
                    ],
                }
            )
        selected_prefix.append(target_idx)
    return rows


def _empty_event_payload(example: Stage2OracleExample, *, top_k: int) -> dict[str, Any]:
    return {
        "candidate_alignment_rows": [],
        "candidate_probe_rows": [],
        "step_probe_rows": [],
        "event_coverage": {
            "event_id": example.event_id,
            "gold_label": example.gold_label,
            "n_aspects": 0,
            "aspect_types": [],
            "oracle_coverage": math.nan,
            "hybrid_topk_coverage": math.nan,
            "candidate_order_topk_coverage": math.nan,
            "oracle_indices": [int(idx) for idx in example.selected_indices[:top_k]],
            "hybrid_indices": _hybrid_top_indices(example, top_k=top_k),
        },
    }


def _coverage_aspects(bundle: dict[str, Any], *, fallback_full_claim: bool) -> list[dict[str, Any]]:
    aspects = [dict(aspect) for aspect in bundle.get("aspects") or []]
    if aspects:
        return aspects
    full = bundle.get("full_claim_anchor")
    if fallback_full_claim and isinstance(full, dict):
        copied = dict(full)
        copied["quality"] = "fallback_full_claim"
        return [copied]
    return []


def _extraction_summary(bundle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    local_counts = [len(row.get("aspects") or []) for row in bundle_rows]
    dropped_counts = [len(row.get("dropped_aspects") or []) for row in bundle_rows]
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    score_values: list[float] = []
    for row in bundle_rows:
        version_counts[str(row.get("extraction_version") or "unknown")] += 1
        for aspect in row.get("aspects") or []:
            type_counts[str(aspect.get("type") or "unknown")] += 1
            source_counts[str(aspect.get("source") or "unknown")] += 1
            score_values.append(float(aspect.get("retrievability_score", math.nan)))
    extraction_version = ASPECT_EXTRACTION_VERSION
    if len(version_counts) == 1:
        extraction_version = next(iter(version_counts.keys()))
    return {
        "extraction_version": extraction_version,
        "extraction_version_counts": dict(sorted(version_counts.items())),
        "n_claims": len(bundle_rows),
        "n_local_aspects": int(sum(local_counts)),
        "n_dropped_aspects": int(sum(dropped_counts)),
        "claims_with_no_local_aspects": int(sum(1 for count in local_counts if count == 0)),
        "local_aspects_per_claim": _numeric_summary(np.asarray(local_counts, dtype=np.float64)),
        "dropped_aspects_per_claim": _numeric_summary(np.asarray(dropped_counts, dtype=np.float64)),
        "aspect_type_counts": dict(sorted(type_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "retrievability_score": _numeric_summary(np.asarray(score_values, dtype=np.float64)),
    }


def _feature_probe(rows: list[dict[str, Any]], *, label_key: str, features: tuple[str, ...]) -> dict[str, Any]:
    labels = np.asarray([1 if bool(row.get(label_key)) else 0 for row in rows], dtype=np.int64)
    output: dict[str, Any] = {}
    for feature in features:
        values = np.asarray([_as_float(row.get(feature)) for row in rows], dtype=np.float64)
        valid = ~np.isnan(values)
        if int(valid.sum()) == 0:
            continue
        y = labels[valid]
        x = values[valid]
        auc = _roc_auc_score(y, x)
        output[feature] = {
            "n": int(valid.sum()),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "positive": _numeric_summary(x[y == 1]),
            "negative": _numeric_summary(x[y == 0]),
            "mean_delta_positive_minus_negative": _safe_mean(x[y == 1]) - _safe_mean(x[y == 0]),
            "auroc_positive": auc,
            "separability_auc": max(auc, 1.0 - auc) if not math.isnan(auc) else math.nan,
        }
    return {"label_key": label_key, "features": output}


def _set_coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = np.asarray([_as_float(row.get("oracle_coverage")) for row in rows], dtype=np.float64)
    hybrid = np.asarray([_as_float(row.get("hybrid_topk_coverage")) for row in rows], dtype=np.float64)
    candidate_order = np.asarray([_as_float(row.get("candidate_order_topk_coverage")) for row in rows], dtype=np.float64)
    valid_hybrid = ~np.isnan(oracle) & ~np.isnan(hybrid)
    valid_candidate = ~np.isnan(oracle) & ~np.isnan(candidate_order)
    return {
        "n_events": len(rows),
        "oracle": _numeric_summary(oracle),
        "hybrid_topk": _numeric_summary(hybrid),
        "candidate_order_topk": _numeric_summary(candidate_order),
        "oracle_vs_hybrid_lift_pp": (
            (_safe_mean(oracle[valid_hybrid]) - _safe_mean(hybrid[valid_hybrid])) * 100.0
            if int(valid_hybrid.sum()) else math.nan
        ),
        "oracle_vs_candidate_order_lift_pp": (
            (_safe_mean(oracle[valid_candidate]) - _safe_mean(candidate_order[valid_candidate])) * 100.0
            if int(valid_candidate.sum()) else math.nan
        ),
        "oracle_beats_hybrid_rate": (
            float(np.mean(oracle[valid_hybrid] > hybrid[valid_hybrid])) if int(valid_hybrid.sum()) else math.nan
        ),
        "oracle_beats_candidate_order_rate": (
            float(np.mean(oracle[valid_candidate] > candidate_order[valid_candidate])) if int(valid_candidate.sum()) else math.nan
        ),
    }


def _decision_payload(
    step_probe: dict[str, Any],
    set_coverage: dict[str, Any],
    *,
    min_coverage_lift_pp: float,
    min_uncovered_gain_auroc: float,
) -> dict[str, Any]:
    gain_auc = float(
        step_probe.get("features", {})
        .get("uncovered_gain", {})
        .get("auroc_positive", math.nan)
    )
    coverage_lift = float(set_coverage.get("oracle_vs_hybrid_lift_pp", math.nan))
    go_by_gain = not math.isnan(gain_auc) and gain_auc >= float(min_uncovered_gain_auroc)
    go_by_coverage = not math.isnan(coverage_lift) and coverage_lift >= float(min_coverage_lift_pp)
    return {
        "decision": "go_selector_ablation" if go_by_gain or go_by_coverage else "stop_or_refine_aspects",
        "min_uncovered_gain_auroc": float(min_uncovered_gain_auroc),
        "min_coverage_lift_pp": float(min_coverage_lift_pp),
        "uncovered_gain_auroc": gain_auc,
        "oracle_vs_hybrid_coverage_lift_pp": coverage_lift,
        "go_by_uncovered_gain": bool(go_by_gain),
        "go_by_coverage_lift": bool(go_by_coverage),
    }


def _top_step_examples(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    positives = [row for row in rows if bool(row.get("target"))]
    positives = sorted(positives, key=lambda row: _sort_float(row.get("uncovered_gain")), reverse=True)
    return {
        "top_target_uncovered_gain": [
            {
                "event_id": row.get("event_id"),
                "gold_label": row.get("gold_label"),
                "step": row.get("step"),
                "candidate_idx": row.get("candidate_idx"),
                "uncovered_gain": row.get("uncovered_gain"),
                "max_aspect_score": row.get("max_aspect_score"),
                "target_aspect_texts": row.get("target_aspect_texts"),
            }
            for row in positives[: max(int(limit), 0)]
        ]
    }


def _set_coverage(alignment: np.ndarray, indices: list[int]) -> float:
    valid = [int(idx) for idx in indices if 0 <= int(idx) < alignment.shape[0]]
    if not valid or alignment.shape[1] == 0:
        return math.nan
    return float(np.mean(np.max(alignment[valid], axis=0)))


def _hybrid_top_indices(example: Stage2OracleExample, *, top_k: int) -> list[int]:
    rows = []
    for idx, score in enumerate(example.candidate_scores):
        rank = _nullable_float(score.get("hybrid_rank"))
        hybrid = _nullable_float(score.get("hybrid_score"))
        rows.append((idx, rank if rank is not None else math.inf, -(hybrid or 0.0)))
    return [idx for idx, _, _ in sorted(rows, key=lambda item: (item[1], item[2]))[: int(top_k)]]


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _normalized_entropy(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x)
    probs = np.exp(x) / np.sum(np.exp(x))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / math.log(float(values.size))


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "max": math.nan}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if values.size else math.nan


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) and not math.isinf(number) else math.nan


def _nullable_float(value: Any) -> float | None:
    number = _as_float(value)
    return None if math.isnan(number) else number


def _sort_float(value: Any) -> float:
    number = _as_float(value)
    return number if not math.isnan(number) else -math.inf


def _candidate_key(event_id: str, idx: int) -> str:
    return f"candidate::{event_id}::{int(idx)}"


def _aspect_key(aspect: dict[str, Any]) -> str:
    return f"aspect::{aspect['aspect_id']}"


def _batches(items: list[Any], batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Use --device cpu.")
    return device


def _manifest(args: argparse.Namespace, out_dir: Path, started_at: float, *, n_examples: int, extract_only: bool) -> dict[str, Any]:
    return {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "oracle_results": str(args.oracle_results),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "extract_only": bool(extract_only),
        "aspect_extraction_version": "precomputed" if args.claim_aspects_input else ASPECT_EXTRACTION_VERSION,
        "claim_aspects_input": str(args.claim_aspects_input) if args.claim_aspects_input else None,
        "model_name": str(args.model_name),
        "revision": args.revision,
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "max_candidates": int(args.max_candidates),
        "top_k": int(args.top_k),
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "sample_limit": args.sample_limit,
        "max_local_aspects": int(args.max_local_aspects),
        "min_retrievability_score": int(args.min_retrievability_score),
        "min_aspect_tokens": int(args.min_aspect_tokens),
        "max_aspect_tokens": int(args.max_aspect_tokens),
        "fallback_full_claim_when_empty": bool(args.fallback_full_claim_when_empty),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "device": str(args.device),
        "n_examples": int(n_examples),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def _write_extraction_markdown(path: Path, *, extraction_summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Oracle Aspect Coverage Diagnostic",
        "",
        "- mode: `extract_only`",
        f"- oracle_results: `{args.oracle_results}`",
        f"- claim_aspects_input: `{args.claim_aspects_input}`" if args.claim_aspects_input else "- claim_aspects_input: `generated_rule_aspects`",
        f"- claims: {extraction_summary['n_claims']}",
        f"- local_aspects: {extraction_summary['n_local_aspects']}",
        f"- dropped_aspects: {extraction_summary['n_dropped_aspects']}",
        f"- claims_with_no_local_aspects: {extraction_summary['claims_with_no_local_aspects']}",
        "",
        "## Aspect Types",
        "",
        "| type | count |",
        "|---|---:|",
    ]
    for key, value in extraction_summary.get("aspect_type_counts", {}).items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis_markdown(path: Path, *, analysis: dict[str, Any]) -> None:
    decision = analysis["decision"]
    extraction = analysis["aspect_extraction"]
    set_coverage = analysis["set_coverage"]
    step_features = analysis["step_uncovered_gain_probe"]["features"]
    gain = step_features.get("uncovered_gain", {})
    lines = [
        "# Oracle Aspect Coverage Diagnostic",
        "",
        f"- decision: `{decision['decision']}`",
        f"- oracle_results: `{analysis['oracle_results']}`",
        f"- model_name: `{analysis['model_name']}`",
        f"- claim_aspects_input: `{analysis['claim_aspects_input']}`" if analysis.get("claim_aspects_input") else "- claim_aspects_input: `generated_rule_aspects`",
        f"- claims: {extraction['n_claims']}",
        f"- local_aspects: {extraction['n_local_aspects']}",
        f"- claims_with_no_local_aspects: {extraction['claims_with_no_local_aspects']}",
        "",
        "## Stop/Go Signals",
        "",
        "| metric | value | threshold |",
        "|---|---:|---:|",
        f"| uncovered_gain AUROC | {float(decision['uncovered_gain_auroc']):.4f} | {float(decision['min_uncovered_gain_auroc']):.4f} |",
        f"| oracle vs hybrid coverage lift pp | {float(decision['oracle_vs_hybrid_coverage_lift_pp']):.2f} | {float(decision['min_coverage_lift_pp']):.2f} |",
        "",
        "## Set Coverage",
        "",
        "| set | mean coverage |",
        "|---|---:|",
        f"| oracle | {float(set_coverage['oracle']['mean']):.6f} |",
        f"| hybrid_topk | {float(set_coverage['hybrid_topk']['mean']):.6f} |",
        f"| candidate_order_topk | {float(set_coverage['candidate_order_topk']['mean']):.6f} |",
        "",
        "## Step Probe",
        "",
        "| feature | positive_mean | negative_mean | auroc | separability_auc |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature, payload in step_features.items():
        lines.append(
            "| {feature} | {pos:.6f} | {neg:.6f} | {auc:.4f} | {sep:.4f} |".format(
                feature=feature,
                pos=float(payload["positive"]["mean"]),
                neg=float(payload["negative"]["mean"]),
                auc=float(payload["auroc_positive"]),
                sep=float(payload["separability_auc"]),
            )
        )
    if gain:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Use this cache for selector training only if the stop/go signals clear the thresholds. "
                "Otherwise refine aspect extraction before enabling `targeted_feature_profile=aspect`.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
