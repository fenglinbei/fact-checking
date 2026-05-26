"""Compute verifier utility for each unique evidence set in trajectory pool.

Loads trajectory JSONL + Chunk-MMR cache, builds prompts for unique evidence
sets, and scores the gold label token logprob via vLLM prompt_logprobs.

Usage:
    PYTHONPATH=src python scripts/phase6_rl_mmr/compute_trajectory_utility.py \\
        --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train.jsonl \\
        --chunk-mmr-cache outputs/cache/chunk_mmr/<fp>/chunk_mmr_train.pkl \\
        --model /data/models/Qwen2.5-7B-Instruct \\
        --output outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import (
    ChunkMMRSample,
    _build_chat_prompt,
    _build_system_message,
    _build_user_content,
    _format_evidence_block,
    _load_pickle,
    _load_prompt_tokenizer,
)
from fact_checking.data.constants import LABEL_LETTERS, LETTER_ORDER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute trajectory utility via vLLM.")
    p.add_argument("--trajectories", type=str, required=True)
    p.add_argument("--chunk-mmr-cache", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--lora-adapter", type=str, default=None)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--prompt-model-name", type=str,
                   default="/data/models/Qwen2.5-7B-Instruct")
    p.add_argument("--score-batch-size", type=int, default=256,
                   help="Number of scoring prompts per vLLM.generate call. Lower if OOM.")
    p.add_argument("--max-prompt-length", type=int, default=1024,
                   help="Max token length for prompts. Evidence truncated from tail if exceeded.")
    p.add_argument("--top-logprobs", type=int, default=20,
                   help="Top generated-token logprobs to request in hybrid prepass.")
    p.add_argument("--label-prefix", type=str, default="Label:")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--reuse-oracle", type=str, default=None,
                   help="Path to existing oracle logprobs JSONL for reusing known utilities.")
    return p.parse_args()


def _build_label_token_ids(tokenizer) -> dict[str, int]:
    letter_token_ids: dict[str, int] = {}
    for letter in LETTER_ORDER:
        text = f" {letter}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label choice {text!r} must be exactly one token, got ids={ids}")
        letter_token_ids[letter] = int(ids[0])
    return letter_token_ids


def _build_scoring_prompt_ids(tokenizer, prompt: str, label_prefix: str, letter: str, label_token_id: int) -> list[int]:
    text = prompt + label_prefix + f" {letter}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or int(ids[-1]) != int(label_token_id):
        raise ValueError(
            f"Scoring prompt for label {letter!r} must end with token_id={label_token_id}, got ids={ids[-5:]}"
        )
    return [int(t) for t in ids]


def _build_prompt_for_evidence(
    sample: ChunkMMRSample,
    selected_ids: list[int],
    tokenizer,
    prompt_model_name: str,
    max_length: int = 1024,
) -> str | None:
    """Build a prompt string with the given evidence selection.

    Truncates evidence items from the tail if the prompt exceeds max_length.
    Returns None if even a single evidence item doesn't fit.
    """
    evidence_texts: list[str] = []
    for idx in selected_ids:
        if idx < len(sample.candidates):
            evidence_texts.append(str(sample.candidates[idx].get("text", "")))
    if not evidence_texts:
        evidence_texts = [str(sample.candidates[0].get("text", ""))] if sample.candidates else [""]

    system_msg = build_system_message(None)

    # Try with all evidence, then progressively drop from tail
    kept = list(evidence_texts)
    while kept:
        user_content = build_user_content(
            claim=sample.claim,
            evidence_texts=kept,
            output_mode="label_only",
            label_format="letter",
        )
        prompt = build_chat_prompt(tokenizer, system_msg, user_content)
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

        if n_tokens <= max_length:
            return prompt
        if len(kept) == 1:
            return None  # even a single evidence item is too long
        kept.pop()

    return None


def _load_oracle_reuse(path: str) -> dict[str, float]:
    """Load existing oracle logprobs for reuse. Returns (event_id, evidence_set_key) -> utility.
    Filters out sentinel values (<= -99) that indicate missing logprobs."""
    reuse: dict[str, float] = {}
    n_skipped_sentinel = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = str(rec.get("event_id", ""))
            lam = rec.get("oracle_lambda")
            if not eid:
                continue
            logprobs = rec.get("logprobs_by_lambda", {})
            for lam_str, lp in logprobs.items():
                lp_f = float(lp)
                if lp_f <= -99.0:
                    n_skipped_sentinel += 1
                    continue
                reuse[f"{eid}||{lam_str}"] = lp_f
    if n_skipped_sentinel:
        print(f"Filtered {n_skipped_sentinel} sentinel values from oracle logprobs")
    return reuse


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    # Load trajectories
    trajectories: list[dict] = []
    with Path(args.trajectories).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories")

    # Load chunk cache
    chunk_samples: list[ChunkMMRSample] = load_pickle(Path(args.chunk_mmr_cache))
    sample_by_eid: dict[str, ChunkMMRSample] = {s.event_id: s for s in chunk_samples}
    print(f"Loaded {len(chunk_samples)} chunk samples")

    # Collect unique (event_id, evidence_set_key) pairs
    unique_pairs: dict[tuple[str, str], list[int]] = {}  # (eid, key) -> trajectory indices
    for i, traj in enumerate(trajectories):
        eid = traj["event_id"]
        key = traj.get("evidence_set_key", "")
        if not key:
            continue
        unique_pairs.setdefault((eid, key), []).append(i)

    print(f"Unique evidence sets: {len(unique_pairs)} (from {len(trajectories)} trajectories)")

    # Try to reuse existing oracle logprobs
    oracle_reuse: dict[str, float] = {}
    if args.reuse_oracle:
        oracle_reuse = _load_oracle_reuse(args.reuse_oracle)
        print(f"Loaded {len(oracle_reuse)} oracle reuse entries")

    # Deduplicate: separate known from unknown evidence sets
    known_utilities: dict[tuple[str, str], float] = {}
    unknown_pairs: list[tuple[str, str]] = []

    # Build mapping: for each unique evidence set, try to match with single-λ oracle
    LAMBDA_ORACLE_MAP = {0.1: "0.10", 0.3: "0.30", 0.5: "0.50", 0.7: "0.70", 0.9: "0.90"}

    for (eid, key) in unique_pairs:
        traj_idx = unique_pairs[(eid, key)][0]
        traj = trajectories[traj_idx]
        sched = traj.get("lambda_schedule", [])

        # If all lambdas are the same, this matches a single-λ oracle entry
        if len(set(sched)) == 1:
            lam_val = sched[0]
            lam_key = LAMBDA_ORACLE_MAP.get(round(lam_val, 2))
            if lam_key:
                reuse = oracle_reuse.get(f"{eid}||{lam_key}")
                if reuse is not None:
                    known_utilities[(eid, key)] = reuse
                    continue

        unknown_pairs.append((eid, key))

    print(f"Known utilities (from oracle reuse): {len(known_utilities)}")
    print(f"Unknown utilities (need vLLM scoring): {len(unknown_pairs)}")

    # Score unknown evidence sets via vLLM
    if unknown_pairs:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed.") from exc

        prompt_tokenizer = load_prompt_tokenizer(args.prompt_model_name)

        model_path = args.model
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": args.dtype,
            "max_model_len": int(args.max_model_len),
            "trust_remote_code": True,
        }
        if args.tokenizer:
            llm_kwargs["tokenizer"] = args.tokenizer
        if args.lora_adapter:
            llm_kwargs.update({"enable_lora": True, "max_lora_rank": int(args.max_lora_rank)})

        llm = LLM(**llm_kwargs)
        tokenizer = llm.get_tokenizer()
        letter_token_ids = _build_label_token_ids(tokenizer)
        # Build reverse map: token_id → letter
        token_id_to_letter = {tid: letter for letter, tid in letter_token_ids.items()}

        # Hybrid scoring: first try generated-token logprobs (lightweight),
        # only fall back to prompt_logprobs for missing labels.
        top_logprobs = int(args.top_logprobs)
        hybrid_sp = SamplingParams(
            max_tokens=1, temperature=0.0, logprobs=top_logprobs, detokenize=False,
        )
        fallback_sp = SamplingParams(
            max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False,
        )

        # Build base prompts (ending with "Label:", ready for generation)
        base_prompts: list[str] = []
        base_meta: list[dict] = []
        n_skipped_overflow = 0
        n_skipped_no_gold = 0
        n_skipped_no_sample = 0
        for eid, key in tqdm(
            unknown_pairs, desc="build prompts", unit="set",
            dynamic_ncols=True, disable=not show_progress,
        ):
            sample = sample_by_eid.get(eid)
            if sample is None:
                n_skipped_no_sample += 1
                continue
            selected_ids = [int(x) for x in key.split("_") if x]
            prompt_str = _build_prompt_for_evidence(
                sample, selected_ids, prompt_tokenizer, args.prompt_model_name,
                max_length=args.max_prompt_length,
            )
            if prompt_str is None:
                n_skipped_overflow += 1
                continue
            gold_label = sample.label
            gold_letter = LABEL_LETTERS.get(gold_label)
            if gold_letter is None:
                n_skipped_no_gold += 1
                continue
            # Append "Label:" prefix so the model generates the label token
            base_prompts.append(prompt_str + args.label_prefix)
            base_meta.append({
                "event_id": eid, "key": key,
                "gold_letter": gold_letter,
                "gold_token_id": letter_token_ids[gold_letter],
            })

        if n_skipped_overflow or n_skipped_no_gold or n_skipped_no_sample:
            print(f"Skipped: {n_skipped_overflow} overflow, {n_skipped_no_gold} no gold label, {n_skipped_no_sample} no sample")
        print(f"Running vLLM hybrid scoring on {len(base_prompts)} prompts...")

        # ---- Phase 1: lightweight generated-token logprobs ----
        fallback_indices: list[int] = []
        score_batch_size = int(args.score_batch_size)

        for batch_start in tqdm(
            range(0, len(base_prompts), score_batch_size),
            desc="hybrid scan", unit="batch",
            dynamic_ncols=True, disable=not show_progress,
        ):
            batch_end = min(batch_start + score_batch_size, len(base_prompts))
            batch_prompts = base_prompts[batch_start:batch_end]
            batch_meta = base_meta[batch_start:batch_end]

            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=hybrid_sp,
                use_tqdm=False,
            )

            for _i, (output, meta) in enumerate(zip(outputs, batch_meta)):
                global_idx = batch_start + _i
                if not output.outputs or not output.outputs[0].logprobs:
                    fallback_indices.append(global_idx)
                    continue

                first_token_lps = output.outputs[0].logprobs[0]
                gold_tid = meta["gold_token_id"]
                entry = first_token_lps.get(gold_tid) or first_token_lps.get(str(gold_tid))
                if entry is not None:
                    lp = float(entry.logprob) if hasattr(entry, "logprob") else float(entry)
                    known_utilities[(meta["event_id"], meta["key"])] = lp
                else:
                    fallback_indices.append(global_idx)

        print(f"Hybrid scan: {len(base_prompts) - len(fallback_indices)} scored, {len(fallback_indices)} need fallback")

        # ---- Phase 2: fallback prompt_logprobs scoring ----
        if fallback_indices:
            fallback_prompt_ids: list[list[int]] = []
            fallback_meta: list[dict] = []
            for idx in fallback_indices:
                meta = base_meta[idx]
                prompt_str = base_prompts[idx]
                gold_letter = meta["gold_letter"]
                gold_tid = meta["gold_token_id"]
                prompt_ids = _build_scoring_prompt_ids(
                    tokenizer, prompt_str, "", gold_letter, gold_tid,
                )
                fallback_prompt_ids.append(prompt_ids)
                fallback_meta.append(meta)

            fb_batch_size = max(32, score_batch_size // 4)
            print(f"Running fallback prompt_logprobs on {len(fallback_prompt_ids)} prompts (batch={fb_batch_size})...")
            for batch_start in tqdm(
                range(0, len(fallback_prompt_ids), fb_batch_size),
                desc="fallback scoring", unit="batch",
                dynamic_ncols=True, disable=not show_progress,
            ):
                batch_end = min(batch_start + fb_batch_size, len(fallback_prompt_ids))
                batch_ids = fallback_prompt_ids[batch_start:batch_end]
                batch_meta = fallback_meta[batch_start:batch_end]

                outputs = llm.generate(
                    prompt_token_ids=batch_ids,
                    sampling_params=fallback_sp,
                    use_tqdm=False,
                )

                for output, meta in zip(outputs, batch_meta):
                    pts_out = getattr(output, "prompt_token_ids", [])
                    prompt_logprobs = getattr(output, "prompt_logprobs", [])
                    if not prompt_logprobs or not pts_out:
                        continue
                    gold_tid = meta["gold_token_id"]
                    if int(pts_out[-1]) != int(gold_tid):
                        continue
                    last_lps = prompt_logprobs[-1]
                    entry = last_lps.get(gold_tid) or last_lps.get(str(gold_tid))
                    if entry is not None:
                        lp = float(entry.logprob) if hasattr(entry, "logprob") else float(entry)
                        known_utilities[(meta["event_id"], meta["key"])] = lp

        print(f"Scored {len(known_utilities)} total evidence sets")

    # Write scored trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            eid = traj["event_id"]
            key = traj.get("evidence_set_key", "")
            utility = known_utilities.get((eid, key))
            if utility is not None:
                traj["utility"] = utility
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    n_scored = sum(1 for t in trajectories if t.get("utility") is not None)
    print(f"Wrote {len(trajectories)} trajectories ({n_scored} with utility) to {output_path}")


if __name__ == "__main__":
    main()
