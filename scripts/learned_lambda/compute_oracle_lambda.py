"""Step 2: Compute oracle λ per claim via vLLM batch inference.

Reads per-λ prompt JSONL files (from generate_oracle_prompts.py), runs the base
LLM (zero-shot) with logprobs, and picks the λ that maximises the log-probability
of the correct label token.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/compute_oracle_lambda.py \
        --prompts-dir outputs/learned_lambda/prompts/ \
        --model /data/models/Qwen2.5-7B-Instruct \
        --output outputs/learned_lambda/oracle_lambda_train.jsonl \
        --split-name train
"""
from __future__ import annotations

import argparse
import inspect
import json
import glob
import re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from fact_checking.data.constants import LABEL_LETTERS, LABEL2ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute oracle λ via vLLM batch inference.")
    p.add_argument("--prompts-dir", type=str, required=True, help="Directory with per-λ JSONL files")
    p.add_argument("--model", type=str, required=True, help="Model path for vLLM")
    p.add_argument("--output", type=str, required=True, help="Output JSONL path")
    p.add_argument("--split-name", type=str, default="train")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--default-lambda", type=float, default=0.7, help="Tie-break preference")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def _load_prompts_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_lambda_from_filename(name: str) -> float | None:
    m = re.search(r"lambda_([\d.]+)", name)
    return float(m.group(1)) if m else None


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install vllm first.") from exc

    prompts_dir = Path(args.prompts_dir)
    pattern = f"lambda_*_{args.split_name}.jsonl"
    jsonl_files = sorted(glob.glob(str(prompts_dir / pattern)))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files matching {pattern} in {prompts_dir}")

    lambda_to_rows: dict[float, list[dict]] = {}
    for fpath in tqdm(
        jsonl_files,
        desc="load prompt files",
        unit="file",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        lam = _extract_lambda_from_filename(Path(fpath).name)
        if lam is None:
            continue
        rows = _load_prompts_jsonl(Path(fpath))
        lambda_to_rows[lam] = rows
        print(f"Loaded λ={lam:.1f}: {len(rows)} prompts", flush=True)

    lambda_grid = sorted(lambda_to_rows.keys())
    n_claims = len(next(iter(lambda_to_rows.values())))
    total_prompts = sum(len(v) for v in lambda_to_rows.values())
    print(f"Lambda grid: {lambda_grid}", flush=True)
    print(f"Claims per λ: {n_claims}, total prompts: {total_prompts}", flush=True)

    # Build flat prompt list with metadata
    all_prompts: list[str] = []
    all_meta: list[dict] = []  # {event_id, gold_label, gold_id, lambda_val, idx_in_lambda}
    with tqdm(
        total=total_prompts,
        desc="build inference batch",
        unit="prompt",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as build_progress:
        for lam in lambda_grid:
            for i, row in enumerate(lambda_to_rows[lam]):
                prompt = row.get("prompt", "")
                build_progress.update(1)
                if not prompt:
                    continue
                all_prompts.append(prompt)
                all_meta.append({
                    "event_id": row.get("event_id", ""),
                    "gold_label": row.get("gold_label", ""),
                    "gold_id": int(row.get("gold_id", -1)),
                    "lambda_val": lam,
                    "idx": i,
                })

    print(f"Total prompts to process: {len(all_prompts)}", flush=True)

    # Initialize vLLM
    print(f"Loading model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Map label letters to token IDs
    letter_to_label = {v: k for k, v in LABEL_LETTERS.items()}
    letter_token_ids: dict[str, int] = {}
    for letter in LABEL_LETTERS.values():
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[0]
    print(f"Letter token IDs: {letter_token_ids}", flush=True)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(letter_token_ids),
    )

    print("Running batch inference...", flush=True)
    generate_params = inspect.signature(llm.generate).parameters
    if "use_tqdm" in generate_params:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=show_progress)
    else:
        outputs = llm.generate(all_prompts, sampling_params)
    print(f"Inference complete: {len(outputs)} outputs", flush=True)

    # Extract log-probs and compute oracle λ
    # Structure: event_id -> {lambda_val -> logprob_of_correct_label}
    event_logprobs: dict[str, dict[float, float]] = {}
    event_gold: dict[str, str] = {}

    for output, meta in tqdm(
        zip(outputs, all_meta),
        total=len(all_meta),
        desc="extract label logprobs",
        unit="prompt",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        event_id = meta["event_id"]
        lam = meta["lambda_val"]
        gold_label = meta["gold_label"]
        event_gold[event_id] = gold_label

        correct_letter = LABEL_LETTERS.get(gold_label, "")
        correct_token_id = letter_token_ids.get(correct_letter)

        logprob = -100.0  # default: very low
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0]
            if correct_token_id is not None and correct_token_id in first_token_logprobs:
                logprob = first_token_logprobs[correct_token_id].logprob

        event_logprobs.setdefault(event_id, {})[lam] = logprob

    # Select oracle λ per claim
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oracle_lambdas: list[float] = []
    event_ids = sorted(event_logprobs.keys())
    with output_path.open("w") as f:
        for event_id in tqdm(
            event_ids,
            desc="write oracle labels",
            unit="claim",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            lp_by_lam = event_logprobs[event_id]
            best_lp = max(lp_by_lam.values())

            # Tie-break: among lambdas within 0.01 of the best, pick closest to default
            candidates = [l for l, lp in lp_by_lam.items() if best_lp - lp < 0.01]
            oracle_lam = min(candidates, key=lambda l: abs(l - args.default_lambda))

            oracle_lambdas.append(oracle_lam)
            record = {
                "event_id": event_id,
                "gold_label": event_gold.get(event_id, ""),
                "oracle_lambda": oracle_lam,
                "best_logprob": best_lp,
                "logprobs_by_lambda": {f"{l:.1f}": lp for l, lp in sorted(lp_by_lam.items())},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    oracle_arr = np.array(oracle_lambdas)
    print(f"\nOracle λ statistics:", flush=True)
    print(f"  N claims: {len(oracle_arr)}", flush=True)
    print(f"  Mean: {oracle_arr.mean():.3f}", flush=True)
    print(f"  Std:  {oracle_arr.std():.3f}", flush=True)
    print(f"  Min:  {oracle_arr.min():.1f}", flush=True)
    print(f"  Max:  {oracle_arr.max():.1f}", flush=True)

    # Distribution
    for lam in lambda_grid:
        count = int(np.sum(oracle_arr == lam))
        pct = 100.0 * count / len(oracle_arr)
        print(f"  λ={lam:.1f}: {count:5d} ({pct:5.1f}%)", flush=True)

    print(f"\nOutput: {output_path}", flush=True)


if __name__ == "__main__":
    main()
