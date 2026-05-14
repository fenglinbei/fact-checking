"""Step 2: Compute oracle λ per claim via vLLM batch inference.

Reads per-λ prompt JSONL files (from generate_oracle_prompts.py), runs a vLLM
model with optional LoRA adapter weights and prompt logprobs, then picks the λ
that maximises the log-probability of the correct label token.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/compute_oracle_lambda.py \
        --prompts-dir outputs/learned_lambda/prompts/ \
        --model /data/models/Qwen2.5-7B-Instruct \
        --output outputs/learned_lambda/oracle_lambda_train.jsonl \
        --split-name train

    PYTHONPATH=src python scripts/learned_lambda/compute_oracle_lambda.py \
        --prompts-dir outputs/learned_lambda/prompts/ \
        --model /data/models/Qwen2.5-7B-Instruct \
        --lora-adapter outputs/runs/.../train/best \
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

from fact_checking.data.constants import LABEL_LETTERS, LETTER_ORDER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute oracle λ via vLLM batch inference.")
    p.add_argument("--prompts-dir", type=str, required=True, help="Directory with per-λ JSONL files")
    p.add_argument("--model", type=str, required=True, help="Model path for vLLM")
    p.add_argument("--output", type=str, required=True, help="Output JSONL path")
    p.add_argument("--split-name", type=str, default="train")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path for vLLM")
    p.add_argument("--lora-adapter", type=str, default=None, help="PEFT LoRA adapter checkpoint directory")
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--default-lambda", type=float, default=0.7, help="Tie-break preference")
    p.add_argument(
        "--label-prefix",
        type=str,
        default="Label:",
        help="Prefix appended before scoring the single-token label choice. "
             "The default matches the normal label-decoding inference path.",
    )
    p.add_argument(
        "--score-batch-size",
        type=int,
        default=1024,
        help="Number of label-continuation scoring requests per vLLM.generate call. "
             "Each original prompt creates six scoring requests.",
    )
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


def _validate_lora_adapter(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter config not found: {adapter_dir / 'adapter_config.json'}")
    if not (adapter_dir / "adapter_model.safetensors").exists() and not (adapter_dir / "adapter_model.bin").exists():
        raise FileNotFoundError(f"LoRA adapter weights not found in {adapter_dir}")


def _resolve_vllm_paths(
    *, model: str, tokenizer: str | None, lora_adapter: str | None
) -> tuple[str, str | None]:
    if not lora_adapter:
        return model, tokenizer

    adapter_dir = Path(lora_adapter)
    if tokenizer:
        return model, tokenizer
    if (adapter_dir / "tokenizer_config.json").exists():
        return model, str(adapter_dir)
    return model, model


def _label_choice_text(letter: str) -> str:
    """Return the one-token label completion used after ``Label:``."""
    return f" {letter}"


def _build_label_token_ids(tokenizer) -> dict[str, int]:
    """Map A-F label letters to their single-token completion ids.

    This mirrors the normal constrained decoding path, which appends
    ``Label:`` to the prompt and then generates one of ``" A"`` ... ``" F"``.
    """
    letter_token_ids: dict[str, int] = {}
    for letter in LETTER_ORDER:
        token_text = _label_choice_text(letter)
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label choice {token_text!r} must be exactly one token, got ids={ids}. "
                "Oracle scoring requires single-token label choices."
            )
        letter_token_ids[letter] = int(ids[0])
    return letter_token_ids


def _append_label_prefix(prompt: str, label_prefix: str) -> str:
    if not label_prefix:
        return prompt
    return prompt + label_prefix


def _build_scoring_prompt_token_ids(
    tokenizer,
    *,
    prompt: str,
    label_prefix: str,
    letter: str,
    label_token_id: int,
) -> list[int]:
    text = _append_label_prefix(prompt, label_prefix) + _label_choice_text(letter)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or int(ids[-1]) != int(label_token_id):
        raise ValueError(
            f"Scoring prompt for label {letter!r} must end with token_id={label_token_id}, got ids={ids[-5:]}. "
            "Check that --label-prefix and tokenizer match the model used for inference."
        )
    return [int(token_id) for token_id in ids]


def _extract_logprob_value(entry) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    return float(entry)


def _extract_prompt_token_logprob(output, token_id: int, *, event_id: str, lam: float, letter: str) -> float:
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    if not prompt_token_ids:
        raise RuntimeError(f"Missing prompt_token_ids for event_id={event_id}, lambda={lam:.2f}, label={letter}")
    if int(prompt_token_ids[-1]) != int(token_id):
        raise RuntimeError(
            f"Scoring output ended with token_id={prompt_token_ids[-1]}, expected {token_id} "
            f"for event_id={event_id}, lambda={lam:.2f}, label={letter}"
        )

    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if not prompt_logprobs:
        raise RuntimeError(f"Missing prompt_logprobs for event_id={event_id}, lambda={lam:.2f}, label={letter}")
    last_token_logprobs = prompt_logprobs[-1]
    if not last_token_logprobs:
        raise RuntimeError(
            f"Empty prompt logprobs for final label token event_id={event_id}, lambda={lam:.2f}, label={letter}"
        )

    entry = last_token_logprobs.get(token_id)
    if entry is None:
        entry = last_token_logprobs.get(str(token_id))
    if entry is None:
        keys = list(last_token_logprobs.keys())[:10]
        raise RuntimeError(
            f"vLLM prompt_logprobs did not include the actual label token {letter}(token_id={token_id}) "
            f"for event_id={event_id}, lambda={lam:.2f}; returned keys sample={keys}. "
            "This should not happen with prompt_logprobs enabled."
        )
    return _extract_logprob_value(entry)


def _normalize_label_logprobs(label_logprobs: dict[str, float]) -> dict[str, float]:
    values = np.array([float(label_logprobs[letter]) for letter in LETTER_ORDER], dtype=np.float64)
    max_value = float(np.max(values))
    log_z = max_value + float(np.log(np.exp(values - max_value).sum()))
    return {letter: float(label_logprobs[letter] - log_z) for letter in LETTER_ORDER}


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install vllm first.") from exc

    lora_request = None
    llm_kwargs = {}
    if args.lora_adapter:
        adapter_dir = Path(args.lora_adapter)
        _validate_lora_adapter(adapter_dir)
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError("vLLM LoRA inference requires a vLLM build with LoRA support.") from exc

        llm_kwargs.update({"enable_lora": True, "max_lora_rank": int(args.max_lora_rank)})
        lora_request = LoRARequest("oracle-lambda-lora", 1, str(adapter_dir))

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
        print(f"Loaded λ={lam:.2f}: {len(rows)} prompts", flush=True)

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

    # Initialize vLLM. For LoRA, args.model is the base model and args.lora_adapter is applied per request.
    model_path, tokenizer_path = _resolve_vllm_paths(
        model=args.model,
        tokenizer=args.tokenizer,
        lora_adapter=args.lora_adapter,
    )
    print(f"Loading model: {model_path}", flush=True)
    if tokenizer_path:
        print(f"Tokenizer: {tokenizer_path}", flush=True)
    if args.lora_adapter:
        print(f"LoRA adapter: {args.lora_adapter}", flush=True)

    llm_init_kwargs = {
        "model": model_path,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": args.dtype,
        "max_model_len": int(args.max_model_len),
        "trust_remote_code": True,
        **llm_kwargs,
    }
    if tokenizer_path:
        llm_init_kwargs["tokenizer"] = tokenizer_path
    llm = LLM(**llm_init_kwargs)
    tokenizer = llm.get_tokenizer()

    # Map label letters to the same one-token choices used by normal inference:
    # prompt + "Label:" -> generate one of " A" ... " F".
    letter_token_ids = _build_label_token_ids(tokenizer)
    print(f"Label prefix: {args.label_prefix!r}", flush=True)
    print(f"Letter token IDs: {letter_token_ids}", flush=True)

    sampling_param_names = set(inspect.signature(SamplingParams).parameters)
    if "prompt_logprobs" not in sampling_param_names:
        raise RuntimeError(
            "This vLLM build does not support SamplingParams.prompt_logprobs. "
            "Upgrade vLLM or use a version that can score prompt continuations."
        )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,
        detokenize=False,
    )

    print("Running label-continuation scoring...", flush=True)
    generate_params = inspect.signature(llm.generate).parameters

    # Extract log-probs and compute oracle λ
    # Structure: event_id -> {lambda_val -> logprob_of_correct_label}
    event_logprobs: dict[str, dict[float, float]] = {}
    event_label_logprobs: dict[str, dict[float, dict[str, float]]] = {}
    event_gold: dict[str, str] = {}

    for meta in all_meta:
        event_gold[meta["event_id"]] = meta["gold_label"]

    score_batch_size = int(args.score_batch_size)
    if score_batch_size <= 0:
        score_batch_size = max(1, len(all_meta) * len(letter_token_ids))
    total_scoring_requests = len(all_meta) * len(letter_token_ids)

    batch_token_ids: list[list[int]] = []
    batch_meta: list[dict] = []

    def flush_scoring_batch() -> int:
        nonlocal batch_token_ids, batch_meta
        if not batch_token_ids:
            return 0
        generate_kwargs = {
            "prompt_token_ids": batch_token_ids,
            "sampling_params": sampling_params,
        }
        if "use_tqdm" in generate_params:
            generate_kwargs["use_tqdm"] = False
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        outputs = llm.generate(**generate_kwargs)
        if len(outputs) != len(batch_meta):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch_meta)} scoring requests")

        for output, meta in zip(outputs, batch_meta):
            event_id = meta["event_id"]
            lam = meta["lambda_val"]
            letter = meta["letter"]
            token_id = meta["label_token_id"]
            logprob = _extract_prompt_token_logprob(
                output,
                token_id,
                event_id=event_id,
                lam=lam,
                letter=letter,
            )
            event_label_logprobs.setdefault(event_id, {}).setdefault(lam, {})[letter] = logprob

        processed = len(batch_meta)
        batch_token_ids = []
        batch_meta = []
        return processed

    with tqdm(
        total=total_scoring_requests,
        desc="score label continuations",
        unit="choice",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as score_progress:
        for prompt, meta in zip(all_prompts, all_meta):
            for letter, token_id in letter_token_ids.items():
                batch_token_ids.append(
                    _build_scoring_prompt_token_ids(
                        tokenizer,
                        prompt=prompt,
                        label_prefix=args.label_prefix,
                        letter=letter,
                        label_token_id=token_id,
                    )
                )
                batch_meta.append({
                    "event_id": meta["event_id"],
                    "lambda_val": meta["lambda_val"],
                    "letter": letter,
                    "label_token_id": token_id,
                })
                if len(batch_token_ids) >= score_batch_size:
                    score_progress.update(flush_scoring_batch())
        score_progress.update(flush_scoring_batch())

    print(f"Scoring complete: {total_scoring_requests} label continuations", flush=True)

    for meta in tqdm(
        all_meta,
        total=len(all_meta),
        desc="extract correct-label logprobs",
        unit="prompt",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        event_id = meta["event_id"]
        lam = meta["lambda_val"]
        gold_label = meta["gold_label"]
        correct_letter = LABEL_LETTERS.get(gold_label, "")
        if not correct_letter:
            raise ValueError(f"Unknown gold_label={gold_label!r} for event_id={event_id}")

        label_logprobs = event_label_logprobs.get(event_id, {}).get(lam, {})
        missing = [letter for letter in LETTER_ORDER if letter not in label_logprobs]
        if missing:
            raise RuntimeError(
                f"Missing scored label logprobs for event_id={event_id}, lambda={lam:.2f}: {missing}"
            )
        normalized_label_logprobs = _normalize_label_logprobs(label_logprobs)
        event_label_logprobs[event_id][lam] = normalized_label_logprobs
        event_logprobs.setdefault(event_id, {})[lam] = normalized_label_logprobs[correct_letter]

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
                "logprobs_by_lambda": {f"{l:.2f}": lp for l, lp in sorted(lp_by_lam.items())},
                "label_logprobs_by_lambda": {
                    f"{l:.2f}": {
                        letter: lp
                        for letter, lp in sorted(event_label_logprobs[event_id][l].items())
                    }
                    for l in sorted(event_label_logprobs[event_id])
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    oracle_arr = np.array(oracle_lambdas)
    print(f"\nOracle λ statistics:", flush=True)
    print(f"  N claims: {len(oracle_arr)}", flush=True)
    print(f"  Mean: {oracle_arr.mean():.3f}", flush=True)
    print(f"  Std:  {oracle_arr.std():.3f}", flush=True)
    print(f"  Min:  {oracle_arr.min():.2f}", flush=True)
    print(f"  Max:  {oracle_arr.max():.2f}", flush=True)

    # Distribution
    for lam in lambda_grid:
        count = int(np.sum(oracle_arr == lam))
        pct = 100.0 * count / len(oracle_arr)
        print(f"  λ={lam:.2f}: {count:5d} ({pct:5.1f}%)", flush=True)

    print(f"\nOutput: {output_path}", flush=True)


if __name__ == "__main__":
    main()
