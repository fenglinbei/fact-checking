#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from fact_checking.build.candidates import _build_training_row, _load_prompt_tokenizer
from fact_checking.data.constants import LETTER_ORDER
from fact_checking.data.io import load_split
from fact_checking.infer.api import (
    OpenAICompletionsClient,
    _choice_payload_prompt_logprobs,
    _extract_final_prompt_logprob,
)
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl
from fact_checking.selectors.verifier_proxy import (
    DEFAULT_DIRECT_VERIFIER_RUN_DIR,
    DEFAULT_LABEL_POLICY,
    DEFAULT_VERIFIER_CHECKPOINT,
    append_score_cache_row,
    build_anchor2_delta_rows,
    build_grouped_rows,
    cache_key_for_score,
    evidence_set_hash,
    json_safe,
    load_score_cache,
    prompt_config_fingerprint,
    require_verifier_checkpoint,
    score_margin,
    stable_fingerprint,
    verifier_config_fingerprint,
)
from sft.infer_common import build_label_decoding_prompt, label_choice_text
from sft.data.types import PreparedSample


DEFAULT_OUTPUT_DIR = "outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/b3_oracle_direct_v0"
DEFAULT_CACHE_DIR = "outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/verifier_score_cache"
DEFAULT_TRAIN_UNION_POOL = "outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"
DEFAULT_VAL_UNION_POOL = "outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build verifier-proxy candidate utility labels over QD union pools.")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--union-pool-jsonl", default=None)
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--raw-split-json", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--direct-verifier-run-dir", default=DEFAULT_DIRECT_VERIFIER_RUN_DIR)
    p.add_argument("--verifier-checkpoint", default=DEFAULT_VERIFIER_CHECKPOINT)
    p.add_argument("--verifier-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--verifier-model", default="fact-checking-sft")
    p.add_argument("--api-timeout", type=float, default=120.0)
    p.add_argument("--api-max-retries", type=int, default=5)
    p.add_argument("--retry-initial-delay", type=float, default=1.0)
    p.add_argument("--retry-max-delay", type=float, default=30.0)
    p.add_argument("--prompt-logprobs", type=int, default=0)
    p.add_argument("--label-prefix", default="Label:")
    p.add_argument("--label-policy", default=DEFAULT_LABEL_POLICY, choices=[DEFAULT_LABEL_POLICY])
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--prompt-max-length", type=int, default=1024)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = require_verifier_checkpoint(
        args.direct_verifier_run_dir,
        args.verifier_checkpoint,
        label_prefix=str(args.label_prefix),
    )
    prompt_cfg = _prompt_cfg(args, checkpoint_info.checkpoint_dir)
    verifier_fp = verifier_config_fingerprint(checkpoint_info)
    prompt_fp = prompt_config_fingerprint(prompt_cfg)
    cache_path = cache_dir / f"verifier_score_cache_{args.split}_{verifier_fp}_{prompt_fp}_{args.label_policy}.jsonl"
    score_cache, invalid_lines, duplicate_rows = load_score_cache(cache_path) if args.resume else ({}, 0, 0)

    union_rows = read_jsonl(args.union_pool_jsonl or _default_union_pool(args.split))
    if args.sample_limit is not None:
        union_rows = union_rows[: int(args.sample_limit)]
    oracle_rows = read_jsonl(args.oracle_results or _default_oracle_results(args.split))
    oracle_by_event = {str(row.get("event_id") or ""): row for row in oracle_rows}
    raw_by_event = _load_raw_samples(args.raw_split_json or _default_raw_split(args.split))
    tokenizer = _load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    stats = {
        "n_events": 0,
        "n_candidate_rows": 0,
        "n_score_requests": 0,
        "n_score_cache_hits": 0,
        "n_score_api_generated": 0,
        "cache_invalid_lines": int(invalid_lines),
        "cache_duplicate_rows": int(duplicate_rows),
    }
    api_client: OpenAICompletionsClient | None = None

    def get_client() -> OpenAICompletionsClient:
        nonlocal api_client
        if api_client is None:
            api_client = OpenAICompletionsClient(
                base_url=str(args.verifier_base_url),
                model=str(args.verifier_model),
                timeout=float(args.api_timeout),
            )
        return api_client

    flat_rows: list[dict[str, Any]] = []
    raw_score_rows_by_key: dict[str, dict[str, Any]] = {}
    iterator = tqdm(
        union_rows,
        desc=f"verifier-proxy labels[{args.split}]",
        unit="claim",
        dynamic_ncols=True,
        disable=bool(args.no_progress),
    )
    for union_row in iterator:
        event_id = str(union_row.get("event_id") or "")
        oracle_row = oracle_by_event.get(event_id)
        if oracle_row is None:
            raise ValueError(f"Missing oracle row for event_id={event_id}")
        explain = raw_by_event.get(event_id, "")

        def score_fn(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
            stats["n_score_requests"] += 1
            scored_keys = [str(candidate.get("canonical_text") or "") for candidate in scored_candidates]
            scored_keys = [key for key in scored_keys if key]
            set_hash = evidence_set_hash(scored_keys)
            cache_key = cache_key_for_score(
                split=str(args.split),
                event_id=event_id,
                evidence_set_hash_value=set_hash,
                verifier_fingerprint=verifier_fp,
                prompt_fingerprint=prompt_fp,
                label_policy=str(args.label_policy),
            )
            cached = score_cache.get(cache_key)
            if cached is not None:
                stats["n_score_cache_hits"] += 1
                return cached
            prompt_row = _build_prompt_row(
                union_row=union_row,
                oracle_row=oracle_row,
                explain=explain,
                candidates=scored_candidates,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
            )
            score_row = _score_prompt_with_retries(
                client=get_client(),
                prompt_row=prompt_row,
                gold_label=str(oracle_row.get("gold_label") or ""),
                label_token_ids=checkpoint_info.label_token_ids,
                label_prefix=str(args.label_prefix),
                prompt_logprobs=int(args.prompt_logprobs),
                max_retries=int(args.api_max_retries),
                initial_delay=float(args.retry_initial_delay),
                max_delay=float(args.retry_max_delay),
            )
            score_row.update(
                {
                    "status": "completed",
                    "cache_key": cache_key,
                    "split": str(args.split),
                    "event_id": event_id,
                    "claim": str(union_row.get("claim") or ""),
                    "gold_label": str(oracle_row.get("gold_label") or ""),
                    "label_policy": str(args.label_policy),
                    "evidence_set_hash": set_hash,
                    "scored_candidate_keys": scored_keys,
                    "n_scored_candidates": len(scored_candidates),
                    "prompt_token_count": int(prompt_row.get("prompt_token_count") or 0),
                    "was_truncated": bool(prompt_row.get("was_truncated")),
                    "verifier_config_fingerprint": verifier_fp,
                    "prompt_config_fingerprint": prompt_fp,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            append_score_cache_row(cache_path, json_safe(score_row))
            score_cache[cache_key] = score_row
            stats["n_score_api_generated"] += 1
            return score_row

        rows, raw_scores = build_anchor2_delta_rows(
            split=str(args.split),
            union_row=union_row,
            oracle_row=oracle_row,
            score_fn=score_fn,
            label_policy=str(args.label_policy),
        )
        flat_rows.extend(rows)
        for raw in raw_scores:
            raw_score_rows_by_key[str(raw.get("cache_key") or raw.get("evidence_set_hash") or len(raw_score_rows_by_key))] = raw
        stats["n_events"] += 1
        stats["n_candidate_rows"] += len(rows)
        iterator.set_postfix(
            rows=stats["n_candidate_rows"],
            api=stats["n_score_api_generated"],
            hit=stats["n_score_cache_hits"],
        )

    grouped_rows = build_grouped_rows(flat_rows)
    flat_path = out_dir / f"candidate_utility_{args.split}.jsonl"
    grouped_path = out_dir / f"candidate_utility_grouped_{args.split}.jsonl"
    raw_path = out_dir / f"raw_verifier_scores_{args.split}.jsonl"
    manifest_path = out_dir / f"label_manifest_{args.split}.json"
    write_jsonl(flat_path, [json_safe(row) for row in flat_rows])
    write_jsonl(grouped_path, [json_safe(row) for row in grouped_rows])
    write_jsonl(raw_path, [json_safe(row) for row in raw_score_rows_by_key.values()])
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "split": str(args.split),
        "union_pool_jsonl": str(args.union_pool_jsonl or _default_union_pool(args.split)),
        "oracle_results": str(args.oracle_results or _default_oracle_results(args.split)),
        "raw_split_json": str(args.raw_split_json or _default_raw_split(args.split)),
        "output_dir": str(out_dir),
        "cache_path": str(cache_path),
        "resume": bool(args.resume),
        "label_policy": str(args.label_policy),
        "verifier": checkpoint_info.__dict__,
        "verifier_base_url": str(args.verifier_base_url),
        "verifier_model": str(args.verifier_model),
        "verifier_config_fingerprint": verifier_fp,
        "prompt_config": prompt_cfg,
        "prompt_config_fingerprint": prompt_fp,
        "paths": {
            "candidate_utility": str(flat_path),
            "candidate_utility_grouped": str(grouped_path),
            "raw_verifier_scores": str(raw_path),
            "label_manifest": str(manifest_path),
        },
        **stats,
        "cache_hit_rate": float(stats["n_score_cache_hits"] / max(stats["n_score_requests"], 1)),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(manifest_path, json_safe(manifest))
    print(f"Wrote verifier-proxy labels: {flat_path}")
    print(
        "events={n_events} rows={n_candidate_rows} score_hits={n_score_cache_hits} "
        "score_api={n_score_api_generated}".format(**stats)
    )


def _prompt_cfg(args: argparse.Namespace, checkpoint_dir: str) -> dict[str, Any]:
    return {
        "model_name_or_path": str(args.prompt_model_name_or_path or checkpoint_dir),
        "auto_length": True,
        "max_length": int(args.prompt_max_length),
        "output_mode": "label_only",
        "label_format": "letter",
        "system_prompt": None,
    }


def _build_prompt_row(
    *,
    union_row: dict[str, Any],
    oracle_row: dict[str, Any],
    explain: str,
    candidates: list[dict[str, Any]],
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    retrieval_row = {
        "event_id": str(union_row.get("event_id") or oracle_row.get("event_id") or ""),
        "claim": str(union_row.get("claim") or oracle_row.get("claim") or ""),
        "label": str(oracle_row.get("gold_label") or ""),
        "explain": str(explain or ""),
        "candidates": candidates,
    }
    return _build_training_row(retrieval_row, tokenizer, prompt_cfg)


def _score_prompt_with_retries(
    *,
    client: OpenAICompletionsClient,
    prompt_row: dict[str, Any],
    gold_label: str,
    label_token_ids: dict[str, int],
    label_prefix: str,
    prompt_logprobs: int,
    max_retries: int,
    initial_delay: float,
    max_delay: float,
) -> dict[str, Any]:
    delay = float(initial_delay)
    last_error: BaseException | None = None
    for attempt in range(max(int(max_retries), 1)):
        try:
            return _score_prompt(
                client=client,
                prompt_row=prompt_row,
                gold_label=gold_label,
                label_token_ids=label_token_ids,
                label_prefix=label_prefix,
                prompt_logprobs=int(prompt_logprobs),
            )
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_error = exc
            if attempt >= max(int(max_retries), 1) - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, float(max_delay))
    raise RuntimeError(f"Verifier API scoring failed after {max_retries} retries: {last_error}") from last_error


def _score_prompt(
    *,
    client: OpenAICompletionsClient,
    prompt_row: dict[str, Any],
    gold_label: str,
    label_token_ids: dict[str, int],
    label_prefix: str,
    prompt_logprobs: int,
) -> dict[str, Any]:
    sample = PreparedSample(
        prompt=str(prompt_row["prompt"]),
        target=str(prompt_row.get("target") or ""),
        prompt_add_special_tokens=bool(prompt_row.get("prompt_add_special_tokens", False)),
        preserve_prompt_prefix=bool(prompt_row.get("preserve_prompt_prefix", True)),
        gold_id=int(prompt_row.get("gold_id", -1)),
        gold_label=str(prompt_row.get("gold_label") or gold_label),
        gold_explain=str(prompt_row.get("gold_explain") or ""),
        prompt_token_count=int(prompt_row.get("prompt_token_count") or 0),
        target_token_count=int(prompt_row.get("target_token_count") or 0),
        evidence_count=int(prompt_row.get("evidence_count") or 0),
        was_truncated=bool(prompt_row.get("was_truncated")),
        claim=str(prompt_row.get("claim") or ""),
        no_evidence=int(prompt_row.get("evidence_count") or 0) == 0,
        long_claim=len(str(prompt_row.get("claim") or "").split()) > 64,
    )
    prompts = [
        build_label_decoding_prompt(sample, label_prefix) + label_choice_text(label_prefix, letter)
        for letter in LETTER_ORDER
    ]
    data = client.complete(
        prompts,
        max_tokens=1,
        temperature=0.0,
        extra_body={"prompt_logprobs": int(prompt_logprobs)},
    )
    choices = data.get("choices", [])
    if len(choices) != len(LETTER_ORDER):
        raise RuntimeError(f"Verifier API returned {len(choices)} choices for {len(LETTER_ORDER)} labels.")
    by_index = {int(choice.get("index", idx)): choice for idx, choice in enumerate(choices)}
    label_logprobs: dict[str, float] = {}
    for idx, letter in enumerate(LETTER_ORDER):
        choice = by_index.get(idx)
        if choice is None:
            raise RuntimeError(f"Verifier API response missing choice index={idx} for label={letter}.")
        prompt_logprobs = _choice_payload_prompt_logprobs(choice)
        label_logprobs[letter] = _extract_final_prompt_logprob(prompt_logprobs, int(label_token_ids[letter]))
    margin = score_margin(label_logprobs, gold_label)
    return {
        "label_logprobs": label_logprobs,
        **margin,
    }


def _load_raw_samples(path: str) -> dict[str, str]:
    raw_path = Path(path)
    if not raw_path.exists():
        return {}
    return {sample.event_id: sample.explain for sample in load_split(raw_path)}


def _default_union_pool(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_UNION_POOL
    return DEFAULT_VAL_UNION_POOL


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _default_raw_split(split: str) -> str:
    if split == "train":
        return "data/raw/LIAR-RAW/train.json"
    if split == "test":
        return "data/raw/LIAR-RAW/test.json"
    return "data/raw/LIAR-RAW/val.json"


if __name__ == "__main__":
    main()
