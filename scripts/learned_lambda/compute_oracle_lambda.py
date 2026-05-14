"""Step 2: Compute oracle λ per claim via vLLM batch inference.

Reads per-λ prompt JSONL files (from generate_oracle_prompts.py), runs a vLLM
model with optional LoRA adapter weights and logprobs, then picks the λ that
maximises the log-probability of the correct label token.

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


def _extract_logprob_value(entry) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    return float(entry)


def _extract_required_label_logprobs(
    first_token_logprobs: dict,
    letter_token_ids: dict[str, int],
) -> dict[str, float]:
    label_logprobs: dict[str, float] = {}
    missing: list[str] = []
    for letter, token_id in letter_token_ids.items():
        entry = first_token_logprobs.get(token_id)
        if entry is None:
            entry = first_token_logprobs.get(str(token_id))
        if entry is None:
            missing.append(f"{letter}(token_id={token_id})")
            continue
        label_logprobs[letter] = _extract_logprob_value(entry)
    if missing:
        raise RuntimeError(
            "vLLM did not return all required A-F label token logprobs. "
            f"Missing: {', '.join(missing)}. "
            "This should not happen when allowed_token_ids is set to the six label tokens."
        )
    return label_logprobs


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
    if "allowed_token_ids" not in sampling_param_names:
        raise RuntimeError(
            "This vLLM build does not support SamplingParams.allowed_token_ids. "
            "Upgrade vLLM or use a version that can constrain oracle scoring to A-F label tokens."
        )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(letter_token_ids),
        allowed_token_ids=list(letter_token_ids.values()),
    )

    print("Running batch inference...", flush=True)
    generate_params = inspect.signature(llm.generate).parameters
    generate_kwargs = {
        "prompts": [_append_label_prefix(prompt, args.label_prefix) for prompt in all_prompts],
        "sampling_params": sampling_params,
    }
    if "use_tqdm" in generate_params:
        generate_kwargs["use_tqdm"] = show_progress
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(**generate_kwargs)
    print(f"Inference complete: {len(outputs)} outputs", flush=True)

    # Extract log-probs and compute oracle λ
    # Structure: event_id -> {lambda_val -> logprob_of_correct_label}
    event_logprobs: dict[str, dict[float, float]] = {}
    event_label_logprobs: dict[str, dict[float, dict[str, float]]] = {}
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
        if not correct_letter:
            raise ValueError(f"Unknown gold_label={gold_label!r} for event_id={event_id}")
        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError(f"Missing vLLM output logprobs for event_id={event_id}, lambda={lam:.2f}")

        first_token_logprobs = output.outputs[0].logprobs[0]
        if not first_token_logprobs:
            raise RuntimeError(f"Empty vLLM token logprobs for event_id={event_id}, lambda={lam:.2f}")
        label_logprobs = _extract_required_label_logprobs(first_token_logprobs, letter_token_ids)
        logprob = label_logprobs[correct_letter]

        event_logprobs.setdefault(event_id, {})[lam] = logprob
        event_label_logprobs.setdefault(event_id, {})[lam] = label_logprobs

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
