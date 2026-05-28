#!/usr/bin/env python3
"""Build VIG-lite rows from saved Stage2 oracle step scores.

This script is intentionally no-vLLM: it reads the verifier scores already
stored under ``search_steps[*].candidate_scores`` in the old oracle JSONL and
rewrites them into the same row shape consumed by
``analyze_oracle_vig_utility.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.data.constants import LETTER2LABEL
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
    load_stage2_oracle_examples,
    write_json,
)
from scripts.selectors.build.generate_oracle_vig_cache import (
    _candidate_score_payload,
    _completed_event_ids,
    _count_jsonl,
    _event_shard,
    _float_or_nan,
    _gold_letter,
    _safe_max,
    _safe_mean,
    _safe_min,
    _shard_suffix,
    _text_features,
    _validate_shard_args,
)

logger = logging.getLogger("oracle_saved_step_utility")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert saved Stage2 oracle search step scores into VIG-like "
            "utility rows without loading vLLM or a verifier model."
        )
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
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    started_at = time.time()
    _validate_shard_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _shard_suffix(int(args.num_shards), int(args.shard_index))
    records_path = out_dir / f"vig_records_{args.split}{suffix}.jsonl"
    final_path = out_dir / f"vig_final_counterfactuals_{args.split}{suffix}.jsonl"
    events_path = out_dir / f"vig_event_summaries_{args.split}{suffix}.jsonl"
    manifest_path = out_dir / f"manifest_{args.split}{suffix}.json"

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    examples = [
        ex for ex in examples
        if _event_shard(ex.event_id, int(args.num_shards)) == int(args.shard_index)
    ]
    if not examples:
        raise ValueError("No examples after Stage2 filtering and sharding.")

    completed = _completed_event_ids(records_path) if bool(args.resume) else set()
    pending = [ex for ex in examples if ex.event_id not in completed]
    logger.info(
        "Loaded %d example(s); completed=%d pending=%d shard=%d/%d",
        len(examples),
        len(completed),
        len(pending),
        int(args.shard_index),
        int(args.num_shards),
    )

    manifest = _manifest(args, started_at, n_examples=len(examples), n_pending=len(pending))
    if not pending:
        final_path.touch(exist_ok=True)
        manifest["status"] = "completed_noop"
        manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
        manifest["records_path"] = str(records_path)
        manifest["final_counterfactuals_path"] = str(final_path)
        manifest["event_summaries_path"] = str(events_path)
        write_json(manifest_path, manifest)
        _write_markdown(
            out_dir / f"analysis_{args.split}{suffix}.md",
            manifest=manifest,
            n_records=_count_jsonl(records_path),
        )
        print(f"No pending examples. Existing saved-score VIG-lite cache: {records_path}")
        return

    generated_events = 0
    generated_rows = 0
    skipped_missing_step_maps = 0
    skipped_missing_score_rows = 0
    target_rank_misses = 0
    iterator = pending
    if not bool(args.no_progress):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(pending, desc="Saved-score VIG-lite", unit="claim", dynamic_ncols=True)
        except Exception:
            pass

    mode = "a" if bool(args.resume) else "w"
    with (
        records_path.open(mode, encoding="utf-8") as records_fh,
        final_path.open(mode, encoding="utf-8") as final_fh,
        events_path.open(mode, encoding="utf-8") as events_fh,
    ):
        # The file exists for analyzer compatibility, but saved-score VIG-lite
        # has no final-set remove/replace counterfactual rows.
        final_fh.flush()
        for example in iterator:
            rows, summary = _rows_from_saved_scores(
                example,
                split=str(args.split),
                top_k=int(args.top_k),
            )
            for row in rows:
                records_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            events_fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
            records_fh.flush()
            events_fh.flush()

            generated_events += 1
            generated_rows += len(rows)
            skipped_missing_step_maps += int(summary.get("missing_step_score_maps", 0))
            skipped_missing_score_rows += int(summary.get("missing_candidate_score_rows", 0))
            target_rank_misses += int(summary.get("target_rank_misses", 0))

    manifest.update(
        {
            "status": "completed",
            "n_generated_events": int(generated_events),
            "n_generated_rows": int(generated_rows),
            "n_generated_final_counterfactual_rows": 0,
            "missing_step_score_maps": int(skipped_missing_step_maps),
            "missing_candidate_score_rows": int(skipped_missing_score_rows),
            "target_rank_misses": int(target_rank_misses),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "records_path": str(records_path),
            "final_counterfactuals_path": str(final_path),
            "event_summaries_path": str(events_path),
        }
    )
    write_json(manifest_path, manifest)
    _write_markdown(out_dir / f"analysis_{args.split}{suffix}.md", manifest=manifest, n_records=_count_jsonl(records_path))
    print(f"Wrote saved-score VIG-lite cache: {records_path}")
    print(f"Wrote event summaries: {events_path}")


def _rows_from_saved_scores(
    example: Stage2OracleExample,
    *,
    split: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_letter = _gold_letter(example.gold_label)
    selected = [int(idx) for idx in example.selected_indices[: int(top_k)]]
    step_maps = _saved_step_score_maps(example)

    selected_prefix: list[int] = []
    rows: list[dict[str, Any]] = []
    target_delta_margins: list[float] = []
    target_ranks_by_delta: list[int] = []
    missing_step_score_maps = 0
    missing_candidate_score_rows = 0
    target_rank_misses = 0

    for step, target_idx in enumerate(selected):
        step_scores = step_maps.get(step)
        if not step_scores:
            missing_step_score_maps += 1
            selected_prefix.append(target_idx)
            continue

        remaining = [
            idx for idx in range(len(example.candidates))
            if idx not in selected_prefix
        ]
        prefix_texts = [candidate_text(example.candidates[idx]) for idx in selected_prefix]
        base_record = _baseline_record(step, selected, step_maps)
        base_payload = _base_payload(base_record, gold_letter)

        scored_remaining: list[tuple[int, dict[str, Any], float, dict[str, Any]]] = []
        for idx in remaining:
            score_record = step_scores.get(int(idx))
            if not score_record:
                missing_candidate_score_rows += 1
                continue
            after = _score_payload(score_record, gold_letter)
            delta_margin = float(after["margin"] - base_payload["margin"])
            scored_remaining.append((idx, after, delta_margin, score_record))

        sorted_remaining = [
            idx for idx, _after, delta, _score_record in sorted(
                scored_remaining,
                key=lambda item: (float(item[2]), -int(item[0])),
                reverse=True,
            )
        ]
        target_rank = sorted_remaining.index(target_idx) + 1 if target_idx in sorted_remaining else -1
        if target_rank < 0:
            target_rank_misses += 1

        for idx, after, delta_margin, score_record in scored_remaining:
            score_row = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
            text_features = _text_features(
                claim=example.claim,
                candidate=candidate_text(example.candidates[idx]),
                prefix_texts=prefix_texts,
            )
            delta_gold = float(after["gold_logprob"] - base_payload["gold_logprob"])
            delta_wrong = float(after["best_wrong_logprob"] - base_payload["best_wrong_logprob"])
            row = {
                "event_id": example.event_id,
                "split": str(split),
                "gold_label": example.gold_label,
                "gold_letter": gold_letter,
                "step": int(step),
                "prefix_indices": [int(x) for x in selected_prefix],
                "prefix_size": int(len(selected_prefix)),
                "candidate_idx": int(idx),
                "candidate_uid": str(
                    example.candidates[idx].get("candidate_uid")
                    or score_row.get("candidate_uid")
                    or f"{example.event_id}:{idx}"
                ),
                "target": bool(idx == target_idx),
                "oracle_remaining_selected": bool(idx in selected[step:]),
                "oracle_selected_rank": int(selected.index(idx)) if idx in selected else -1,
                "target_rank_by_delta_margin": int(target_rank) if idx == target_idx else -1,
                "base_gold_logprob": float(base_payload["gold_logprob"]),
                "base_best_wrong_logprob": float(base_payload["best_wrong_logprob"]),
                "base_margin": float(base_payload["margin"]),
                "base_pred_letter": str(base_payload["pred_letter"]),
                "base_pred_is_gold": bool(base_payload["pred_letter"] == gold_letter),
                "base_available": bool(base_record is not None),
                "after_gold_logprob": float(after["gold_logprob"]),
                "after_best_wrong_logprob": float(after["best_wrong_logprob"]),
                "after_margin": float(after["margin"]),
                "after_pred_letter": str(after["pred_letter"]),
                "after_pred_label": str(after["pred_label"]),
                "after_pred_is_gold": bool(after["pred_letter"] == gold_letter),
                "delta_gold_logprob": float(delta_gold),
                "delta_best_wrong_logprob": float(delta_wrong),
                "delta_margin": float(delta_margin),
                "harmful_delta_margin": bool(delta_margin < 0.0),
                "prediction_changed": bool(base_record is not None and after["pred_letter"] != base_payload["pred_letter"]),
                "label_logprobs_after": after["label_logprobs"],
                "candidate_text_preview": candidate_text(example.candidates[idx])[:240],
                "utility_source": "saved_oracle_step_scores",
                "utility_semantics": (
                    "saved candidate margin minus a constant step baseline; "
                    "step ranking is equivalent to the original oracle scorer; "
                    "step0 lacks a saved empty-prefix baseline"
                ),
                "step_baseline_mode": "previous_selected_score_else_zero",
                "oracle_step_objective_score": _float_or_nan(score_record.get("objective_score")),
                "oracle_step_margin": float(after["margin"]),
                "oracle_step_gold_logprob": float(after["gold_logprob"]),
                "oracle_step_best_wrong_logprob": float(after["best_wrong_logprob"]),
                **_candidate_score_payload(score_row),
                **text_features,
            }
            rows.append(row)

        if target_rank > 0:
            target_row = next(
                (item for item in scored_remaining if int(item[0]) == int(target_idx)),
                None,
            )
            if target_row is not None:
                target_delta_margins.append(float(target_row[2]))
                target_ranks_by_delta.append(int(target_rank))
        selected_prefix.append(target_idx)

    summary = {
        "event_id": example.event_id,
        "gold_label": example.gold_label,
        "selected_indices": selected,
        "oracle_final_margin": float(example.margin),
        "n_rows": int(len(rows)),
        "n_final_counterfactual_rows": 0,
        "utility_source": "saved_oracle_step_scores",
        "target_delta_margin_mean": _safe_mean(target_delta_margins),
        "target_delta_margin_min": _safe_min(target_delta_margins),
        "target_rank_by_delta_margin_mean": _safe_mean(target_ranks_by_delta),
        "target_rank_by_delta_margin_max": _safe_max(target_ranks_by_delta),
        "all_targets_rank1_by_delta": bool(target_ranks_by_delta and all(rank == 1 for rank in target_ranks_by_delta)),
        "missing_step_score_maps": int(missing_step_score_maps),
        "missing_candidate_score_rows": int(missing_candidate_score_rows),
        "target_rank_misses": int(target_rank_misses),
    }
    return rows, summary


def _saved_step_score_maps(example: Stage2OracleExample) -> dict[int, dict[int, dict[str, Any]]]:
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for raw_step in example.raw.get("search_steps") or []:
        if not isinstance(raw_step, dict):
            continue
        try:
            step = int(raw_step.get("step"))
        except (TypeError, ValueError):
            continue
        scores: dict[int, dict[str, Any]] = {}
        for raw_score in raw_step.get("candidate_scores") or []:
            if not isinstance(raw_score, dict):
                continue
            try:
                candidate_idx = int(raw_score.get("candidate_idx"))
            except (TypeError, ValueError):
                continue
            scores[candidate_idx] = dict(raw_score)
        if scores:
            out[step] = scores
    return out


def _baseline_record(
    step: int,
    selected: list[int],
    step_maps: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any] | None:
    if int(step) <= 0:
        return None
    previous_step = int(step) - 1
    previous_selected = int(selected[previous_step])
    return step_maps.get(previous_step, {}).get(previous_selected)


def _base_payload(record: dict[str, Any] | None, gold_letter: str) -> dict[str, Any]:
    if not record:
        return {
            "gold_logprob": 0.0,
            "best_wrong_logprob": 0.0,
            "margin": 0.0,
            "pred_letter": "",
            "pred_label": "",
            "label_logprobs": {},
        }
    return _score_payload(record, gold_letter)


def _score_payload(record: dict[str, Any], gold_letter: str) -> dict[str, Any]:
    label_logprobs = {
        str(letter): float(value)
        for letter, value in dict(record.get("label_logprobs") or {}).items()
        if _is_finite_number(value)
    }
    gold_logprob = _float_or_nan(record.get("gold_logprob"))
    if math.isnan(gold_logprob) and gold_letter in label_logprobs:
        gold_logprob = float(label_logprobs[gold_letter])
    best_wrong = _float_or_nan(record.get("best_wrong_logprob"))
    if math.isnan(best_wrong) and label_logprobs:
        wrong_scores = [
            float(score)
            for letter, score in label_logprobs.items()
            if str(letter) != str(gold_letter)
        ]
        best_wrong = max(wrong_scores) if wrong_scores else float("-inf")
    margin = _float_or_nan(record.get("margin"))
    if math.isnan(margin) and not math.isnan(gold_logprob) and not math.isnan(best_wrong):
        margin = float(gold_logprob - best_wrong)
    pred_letter = str(record.get("pred_letter") or "")
    if not pred_letter and label_logprobs:
        pred_letter = max(label_logprobs, key=label_logprobs.get)
    return {
        "gold_logprob": float(gold_logprob),
        "best_wrong_logprob": float(best_wrong),
        "margin": float(margin),
        "pred_letter": pred_letter,
        "pred_label": LETTER2LABEL.get(pred_letter, ""),
        "label_logprobs": label_logprobs,
    }


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def _manifest(args: argparse.Namespace, started_at: float, *, n_examples: int, n_pending: int) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "oracle_results": str(args.oracle_results),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "utility_source": "saved_oracle_step_scores",
        "utility_semantics": (
            "Rows are reconstructed from search_steps[*].candidate_scores in "
            "the oracle JSONL. For step>0, base_* is the previously selected "
            "prefix score saved by the oracle. For step0, no empty-prefix "
            "score was saved, so base_* is zero and delta_* is rank-equivalent "
            "within the step but not an absolute empty-prefix VIG delta."
        ),
        "has_vllm_rescore": False,
        "has_final_counterfactuals": False,
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "max_candidates": int(args.max_candidates),
        "top_k": int(args.top_k),
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "resume": bool(args.resume),
        "n_examples": int(n_examples),
        "n_pending": int(n_pending),
        "started_at_monotonic": float(started_at),
    }


def _write_markdown(path: Path, *, manifest: dict[str, Any], n_records: int) -> None:
    lines = [
        "# Oracle Saved-Score VIG-lite Cache",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- oracle_results: `{manifest.get('oracle_results')}`",
        f"- utility_source: `{manifest.get('utility_source')}`",
        f"- has_vllm_rescore: `{manifest.get('has_vllm_rescore')}`",
        f"- examples: {manifest.get('n_examples')}",
        f"- pending_at_start: {manifest.get('n_pending')}",
        f"- total_cache_rows: {int(n_records)}",
        "",
        "Each row is one saved oracle step score for one remaining candidate.",
        "`delta_margin` is rank-equivalent within each step because it subtracts a constant step baseline.",
        "For step 0 the old oracle did not save the empty-prefix verifier score, so `base_*` is zero and `delta_*` is not an absolute empty-prefix VIG delta.",
        "No final-set remove/replace counterfactuals are generated in this no-vLLM path.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
