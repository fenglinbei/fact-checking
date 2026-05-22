#!/usr/bin/env python3
"""Generate verifier information-gain cache for Stage2 oracle examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.data.constants import LABEL2ID, LABEL_LETTERS, LETTER2LABEL, LETTER_ORDER
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
    load_stage2_oracle_examples,
    read_jsonl,
    write_json,
)

logger = logging.getLogger("oracle_vig_cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score each remaining candidate under each oracle prefix and write "
            "verifier information-gain rows for mechanism analysis."
        )
    )
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--config", default="configs/experiment/b3_mmr_topk_sweep_1024.yaml")
    p.add_argument("--config-overrides", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--lora-adapter", default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=1032)
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--score-batch-size",
        type=int,
        default=128,
        help="Number of evidence sets per vLLM score batch before six label expansions.",
    )
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--include-final-counterfactuals", dest="include_final_counterfactuals", action="store_true", default=True)
    p.add_argument("--no-include-final-counterfactuals", dest="include_final_counterfactuals", action="store_false")
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
        filter_policy=args.filter_policy,
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
        manifest["status"] = "completed_noop"
        manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
        write_json(manifest_path, manifest)
        _write_markdown(out_dir / f"analysis_{args.split}{suffix}.md", manifest=manifest, n_records=_count_jsonl(records_path))
        print(f"No pending examples. Existing cache: {records_path}")
        return

    build_prompt_cfg = _load_prompt_config(args)
    scorer = _init_scorer(args, build_prompt_cfg)

    generated_events = 0
    generated_rows = 0
    generated_final_rows = 0
    iterator = pending
    if not bool(args.no_progress):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(pending, desc="VIG cache", unit="claim", dynamic_ncols=True)
        except Exception:
            pass

    with records_path.open("a", encoding="utf-8") as records_fh, final_path.open("a", encoding="utf-8") as final_fh, events_path.open("a", encoding="utf-8") as events_fh:
        for example in iterator:
            rows, final_rows, summary = _score_example(
                example,
                scorer=scorer,
                split=str(args.split),
                top_k=int(args.top_k),
                score_batch_size=int(args.score_batch_size),
                include_final_counterfactuals=bool(args.include_final_counterfactuals),
            )
            for row in rows:
                records_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in final_rows:
                final_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            events_fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
            records_fh.flush()
            final_fh.flush()
            events_fh.flush()
            generated_events += 1
            generated_rows += len(rows)
            generated_final_rows += len(final_rows)

    manifest.update(
        {
            "status": "completed",
            "n_generated_events": int(generated_events),
            "n_generated_rows": int(generated_rows),
            "n_generated_final_counterfactual_rows": int(generated_final_rows),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "prompt_config": build_prompt_cfg,
            "records_path": str(records_path),
            "final_counterfactuals_path": str(final_path),
            "event_summaries_path": str(events_path),
        }
    )
    write_json(manifest_path, manifest)
    _write_markdown(out_dir / f"analysis_{args.split}{suffix}.md", manifest=manifest, n_records=_count_jsonl(records_path))
    print(f"Wrote VIG cache: {records_path}")
    print(f"Wrote event summaries: {events_path}")


def _load_prompt_config(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from scripts.oracle_evidence.search_optimal_evidence import load_build_config
    except Exception as exc:
        logger.warning("Could not import oracle config helper; using prompt defaults: %s", exc)
        build_cfg: dict[str, Any] = {}
    else:
        build_cfg = load_build_config(args.config, args.config_overrides, args.model_base_path)
    prompt = dict(build_cfg.get("prompt") or {})
    return {
        "system_prompt": prompt.get("system_prompt") or None,
        "output_mode": str(prompt.get("output_mode", "label_only")).strip().lower(),
        "label_format": str(prompt.get("label_format", "letter")).strip().lower(),
        "max_prompt_length": int(prompt.get("max_length", 1024)),
    }


def _init_scorer(args: argparse.Namespace, prompt_cfg: dict[str, Any]) -> Any:
    for lib in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Run this in the cppo environment on the target server.") from exc

    llm_kwargs: dict[str, Any] = {}
    lora_request = None
    if args.lora_adapter:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError("LoRA support requires a vLLM build with LoRA.") from exc
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        lora_request = LoRARequest("vig-lora", 1, args.lora_adapter)

    llm = LLM(
        model=args.verifier_model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype=str(args.dtype),
        trust_remote_code=True,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    from fact_checking.oracle_evidence.scorer import VerifierScorer

    prompt_budget = int(args.max_model_len) - 16
    scorer = VerifierScorer(
        llm=llm,
        tokenizer=tokenizer,
        system_prompt=prompt_cfg["system_prompt"],
        output_mode=prompt_cfg["output_mode"],
        label_format=prompt_cfg["label_format"],
        max_prompt_length=min(int(prompt_cfg["max_prompt_length"]), prompt_budget),
        lora_request=lora_request,
    )
    logger.info("Label token IDs: %s", {k: v for k, v in scorer.label_token_ids.items()})
    return scorer


def _score_example(
    example: Stage2OracleExample,
    *,
    scorer: Any,
    split: str,
    top_k: int,
    score_batch_size: int,
    include_final_counterfactuals: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gold_letter = _gold_letter(example.gold_label)
    selected = [int(idx) for idx in example.selected_indices[: int(top_k)]]
    selected_prefix: list[int] = []
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    target_delta_margins: list[float] = []
    target_ranks_by_delta: list[int] = []

    for step, target_idx in enumerate(selected):
        remaining = [
            idx for idx in range(len(example.candidates))
            if idx not in selected_prefix
        ]
        prefix_texts = [candidate_text(example.candidates[idx]) for idx in selected_prefix]
        evidence_sets = [prefix_texts] + [
            prefix_texts + [candidate_text(example.candidates[idx])]
            for idx in remaining
        ]
        label_logprobs = _score_complete_sets_all_labels_batched(
            scorer=scorer,
            claims=[example.claim] * len(evidence_sets),
            evidence_sets=evidence_sets,
            batch_size=max(1, int(score_batch_size)),
        )
        records = [
            _record_from_label_row(row, gold_letter)
            for row in label_logprobs
        ]
        base_record = records[0]
        candidate_records = records[1:]

        deltas = [
            float(cand_record["margin"] - base_record["margin"])
            for cand_record in candidate_records
        ]
        sorted_remaining = [
            idx for idx, _delta in sorted(zip(remaining, deltas), key=lambda item: item[1], reverse=True)
        ]
        target_rank = sorted_remaining.index(target_idx) + 1 if target_idx in sorted_remaining else -1

        for idx, cand_record, delta_margin in zip(remaining, candidate_records, deltas):
            score_row = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
            text_features = _text_features(
                claim=example.claim,
                candidate=candidate_text(example.candidates[idx]),
                prefix_texts=prefix_texts,
            )
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
                "base_gold_logprob": float(base_record["gold_logprob"]),
                "base_best_wrong_logprob": float(base_record["best_wrong_logprob"]),
                "base_margin": float(base_record["margin"]),
                "base_pred_letter": str(base_record["pred_letter"]),
                "base_pred_is_gold": bool(base_record["pred_letter"] == gold_letter),
                "after_gold_logprob": float(cand_record["gold_logprob"]),
                "after_best_wrong_logprob": float(cand_record["best_wrong_logprob"]),
                "after_margin": float(cand_record["margin"]),
                "after_pred_letter": str(cand_record["pred_letter"]),
                "after_pred_is_gold": bool(cand_record["pred_letter"] == gold_letter),
                "delta_gold_logprob": float(cand_record["gold_logprob"] - base_record["gold_logprob"]),
                "delta_best_wrong_logprob": float(cand_record["best_wrong_logprob"] - base_record["best_wrong_logprob"]),
                "delta_margin": float(delta_margin),
                "harmful_delta_margin": bool(delta_margin < 0.0),
                "prediction_changed": bool(cand_record["pred_letter"] != base_record["pred_letter"]),
                "label_logprobs_after": cand_record["label_logprobs"],
                "candidate_text_preview": candidate_text(example.candidates[idx])[:240],
                **_candidate_score_payload(score_row),
                **text_features,
            }
            rows.append(row)

        if target_idx in remaining:
            target_delta_margins.append(float(deltas[remaining.index(target_idx)]))
            target_ranks_by_delta.append(int(target_rank))
        selected_prefix.append(target_idx)

    if include_final_counterfactuals:
        final_rows = _score_final_counterfactuals(
            example,
            selected_indices=selected,
            scorer=scorer,
            split=str(split),
            gold_letter=gold_letter,
            score_batch_size=int(score_batch_size),
        )

    summary = {
        "event_id": example.event_id,
        "gold_label": example.gold_label,
        "selected_indices": selected,
        "oracle_final_margin": float(example.margin),
        "n_rows": int(len(rows)),
        "n_final_counterfactual_rows": int(len(final_rows)),
        "target_delta_margin_mean": _safe_mean(target_delta_margins),
        "target_delta_margin_min": _safe_min(target_delta_margins),
        "target_rank_by_delta_margin_mean": _safe_mean(target_ranks_by_delta),
        "target_rank_by_delta_margin_max": _safe_max(target_ranks_by_delta),
        "all_targets_rank1_by_delta": bool(target_ranks_by_delta and all(rank == 1 for rank in target_ranks_by_delta)),
    }
    return rows, final_rows, summary


def _score_final_counterfactuals(
    example: Stage2OracleExample,
    *,
    selected_indices: list[int],
    scorer: Any,
    split: str,
    gold_letter: str,
    score_batch_size: int,
) -> list[dict[str, Any]]:
    if not selected_indices:
        return []
    selected_set = set(selected_indices)
    nonselected = [idx for idx in range(len(example.candidates)) if idx not in selected_set]
    final_texts = [candidate_text(example.candidates[idx]) for idx in selected_indices]

    evidence_sets: list[list[str]] = [final_texts]
    specs: list[dict[str, Any]] = [{"counterfactual_type": "oracle_final"}]

    for pos, selected_idx in enumerate(selected_indices):
        evidence_sets.append([text for i, text in enumerate(final_texts) if i != pos])
        specs.append(
            {
                "counterfactual_type": "remove_selected",
                "selected_step": int(pos),
                "selected_idx": int(selected_idx),
                "replacement_candidate_idx": -1,
            }
        )

    for pos, selected_idx in enumerate(selected_indices):
        for replacement_idx in nonselected:
            replaced = list(final_texts)
            replaced[pos] = candidate_text(example.candidates[replacement_idx])
            evidence_sets.append(replaced)
            specs.append(
                {
                    "counterfactual_type": "replace_selected",
                    "selected_step": int(pos),
                    "selected_idx": int(selected_idx),
                    "replacement_candidate_idx": int(replacement_idx),
                }
            )

    label_logprobs = _score_complete_sets_all_labels_batched(
        scorer=scorer,
        claims=[example.claim] * len(evidence_sets),
        evidence_sets=evidence_sets,
        batch_size=max(1, int(score_batch_size)),
    )
    records = [_record_from_label_row(row, gold_letter) for row in label_logprobs]
    base = records[0]
    out: list[dict[str, Any]] = []
    for spec, record, evidence in zip(specs[1:], records[1:], evidence_sets[1:]):
        selected_idx = int(spec.get("selected_idx", -1))
        replacement_idx = int(spec.get("replacement_candidate_idx", -1))
        payload = {
            "event_id": example.event_id,
            "split": str(split),
            "gold_label": example.gold_label,
            "gold_letter": gold_letter,
            "counterfactual_type": spec["counterfactual_type"],
            "selected_step": int(spec.get("selected_step", -1)),
            "selected_idx": selected_idx,
            "replacement_candidate_idx": replacement_idx,
            "base_final_gold_logprob": float(base["gold_logprob"]),
            "base_final_best_wrong_logprob": float(base["best_wrong_logprob"]),
            "base_final_margin": float(base["margin"]),
            "base_final_pred_letter": str(base["pred_letter"]),
            "counterfactual_gold_logprob": float(record["gold_logprob"]),
            "counterfactual_best_wrong_logprob": float(record["best_wrong_logprob"]),
            "counterfactual_margin": float(record["margin"]),
            "counterfactual_pred_letter": str(record["pred_letter"]),
            "final_contribution_delta_margin": float(base["margin"] - record["margin"]),
            "final_contribution_delta_gold_logprob": float(base["gold_logprob"] - record["gold_logprob"]),
            "final_contribution_delta_best_wrong_logprob": float(base["best_wrong_logprob"] - record["best_wrong_logprob"]),
            "counterfactual_prediction_changed": bool(record["pred_letter"] != base["pred_letter"]),
            "evidence_set_size": int(len(evidence)),
            "selected_text_preview": candidate_text(example.candidates[selected_idx])[:240] if 0 <= selected_idx < len(example.candidates) else "",
            "replacement_text_preview": candidate_text(example.candidates[replacement_idx])[:240] if 0 <= replacement_idx < len(example.candidates) else "",
        }
        if replacement_idx >= 0:
            payload["replacement_improves_final_margin"] = bool(record["margin"] > base["margin"])
            payload["replacement_delta_margin"] = float(record["margin"] - base["margin"])
            score_row = example.candidate_scores[replacement_idx] if replacement_idx < len(example.candidate_scores) else {}
            payload.update({f"replacement_{k}": v for k, v in _candidate_score_payload(score_row).items()})
        else:
            payload["selected_harmful_in_final_set"] = bool(record["margin"] > base["margin"])
        out.append(payload)
    return out


def _score_complete_sets_all_labels_batched(
    *,
    scorer: Any,
    claims: list[str],
    evidence_sets: list[list[str]],
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total = len(claims)
    for start in range(0, total, int(batch_size)):
        end = min(start + int(batch_size), total)
        outputs.append(
            scorer.score_complete_sets_all_labels(
                claims=claims[start:end],
                evidence_sets=evidence_sets[start:end],
            )
        )
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(LETTER_ORDER)), dtype=np.float32)


def _record_from_label_row(row: np.ndarray, gold_letter: str) -> dict[str, Any]:
    label_scores = {letter: float(row[i]) for i, letter in enumerate(LETTER_ORDER)}
    gold_logprob = float(label_scores[gold_letter])
    wrong_scores = [score for letter, score in label_scores.items() if letter != gold_letter]
    best_wrong = max(wrong_scores) if wrong_scores else float("-inf")
    pred_letter = max(label_scores, key=label_scores.get)
    return {
        "gold_logprob": gold_logprob,
        "best_wrong_logprob": float(best_wrong),
        "margin": float(gold_logprob - best_wrong),
        "label_logprobs": label_scores,
        "pred_letter": pred_letter,
        "pred_label": LETTER2LABEL.get(pred_letter, ""),
    }


def _candidate_score_payload(score_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hybrid_rank": _float_or_nan(score_row.get("hybrid_rank")),
        "dense_score": _float_or_nan(score_row.get("dense_score")),
        "lexical_score": _float_or_nan(score_row.get("lexical_score")),
        "bm25_score": _float_or_nan(score_row.get("bm25_score")),
        "hybrid_score": _float_or_nan(score_row.get("hybrid_score")),
    }


def _text_features(*, claim: str, candidate: str, prefix_texts: list[str]) -> dict[str, Any]:
    claim_tokens = _token_set(claim)
    cand_tokens = _token_set(candidate)
    prefix_tokens = [_token_set(text) for text in prefix_texts]
    prefix_jaccards = [_jaccard(cand_tokens, toks) for toks in prefix_tokens]
    return {
        "candidate_token_len": int(len(_tokens(candidate))),
        "candidate_char_len": int(len(candidate)),
        "candidate_has_number": bool(any(ch.isdigit() for ch in candidate)),
        "claim_candidate_token_jaccard": float(_jaccard(claim_tokens, cand_tokens)),
        "prefix_candidate_max_jaccard": float(max(prefix_jaccards) if prefix_jaccards else 0.0),
        "prefix_candidate_mean_jaccard": float(sum(prefix_jaccards) / len(prefix_jaccards) if prefix_jaccards else 0.0),
        "prefix_token_len_total": int(sum(len(_tokens(text)) for text in prefix_texts)),
    }


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in str(text).lower():
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def _token_set(text: str) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _gold_letter(gold_label: str) -> str:
    letter = LABEL_LETTERS.get(str(gold_label))
    if not letter:
        raise ValueError(f"Unsupported gold label: {gold_label!r}")
    return letter


def _validate_shard_args(args: argparse.Namespace) -> None:
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")


def _event_shard(event_id: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    digest = hashlib.sha1(str(event_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_shards)


def _shard_suffix(num_shards: int, shard_index: int) -> str:
    return "" if int(num_shards) <= 1 else f".shard-{int(shard_index):05d}-of-{int(num_shards):05d}"


def _completed_event_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    event_ids: set[str] = set()
    for row in read_jsonl(path):
        event_id = str(row.get("event_id") or "")
        if event_id:
            event_ids.add(event_id)
    return event_ids


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _manifest(args: argparse.Namespace, started_at: float, *, n_examples: int, n_pending: int) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "oracle_results": str(args.oracle_results),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "verifier_model": str(args.verifier_model),
        "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "dtype": str(args.dtype),
        "score_batch_size": int(args.score_batch_size),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "max_candidates": int(args.max_candidates),
        "top_k": int(args.top_k),
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "include_final_counterfactuals": bool(args.include_final_counterfactuals),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "resume": bool(args.resume),
        "n_examples": int(n_examples),
        "n_pending": int(n_pending),
        "started_at_monotonic": float(started_at),
    }


def _write_markdown(path: Path, *, manifest: dict[str, Any], n_records: int) -> None:
    lines = [
        "# Oracle VIG Cache",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- oracle_results: `{manifest.get('oracle_results')}`",
        f"- verifier_model: `{manifest.get('verifier_model')}`",
        f"- lora_adapter: `{manifest.get('lora_adapter')}`",
        f"- examples: {manifest.get('n_examples')}",
        f"- pending_at_start: {manifest.get('n_pending')}",
        f"- total_cache_rows: {int(n_records)}",
        "",
        "Each row scores one remaining candidate under one oracle prefix.",
        "`delta_margin` is the verifier margin after adding the candidate minus the prefix margin.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) and not math.isinf(number) else math.nan


def _safe_mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else math.nan


def _safe_min(values: list[float] | list[int]) -> float:
    return float(np.min(values)) if values else math.nan


def _safe_max(values: list[float] | list[int]) -> float:
    return float(np.max(values)) if values else math.nan


if __name__ == "__main__":
    main()
