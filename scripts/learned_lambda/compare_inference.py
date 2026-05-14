"""Compare inference accuracy between two build JSONLs using vLLM offline API.

Uses vLLM's LLM class directly (same approach as compute_oracle_lambda.py)
to avoid API server overhead and HAMI memory limitations.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/compare_inference.py \
        --baseline-build outputs/learned_lambda/build_fixed_predictor_val.jsonl \
        --experiment-build outputs/learned_lambda/build_oracle_predictor_val.jsonl \
        --baseline-label "fixed_0.70" \
        --experiment-label "oracle" \
        --model /data/models/Qwen2.5-7B-Instruct \
        --lora-adapter outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
        --output-dir outputs/learned_lambda/comparison/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from fact_checking.data.constants import LABELS, LABEL_LETTERS, LETTER_ORDER
from sft.infer_common import _load_prebuilt_samples
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare inference accuracy between two build JSONLs.")
    p.add_argument("--baseline-build", type=str, required=True)
    p.add_argument("--experiment-build", type=str, required=True)
    p.add_argument("--baseline-label", type=str, default="baseline")
    p.add_argument("--experiment-label", type=str, default="experiment")
    p.add_argument("--model", type=str, default="/data/models/Qwen2.5-7B-Instruct")
    p.add_argument("--lora-adapter", type=str, default=None)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="outputs/learned_lambda/comparison")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def load_build_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_name_from_id(label_id: int) -> str:
    if 0 <= label_id < len(LABELS):
        return LABELS[label_id]
    return "parse_error"


def run_inference_offline(
    samples,
    llm,
    letter_token_ids: dict[str, int],
    label: str,
    show_progress: bool,
    lora_request=None,
) -> dict:
    """Run inference using vLLM offline LLM API with guided_choice for label decoding."""
    from vllm import SamplingParams

    # Build prompts with "Label:" prefix for constrained decoding
    label_choices = [f" {letter}" for letter in LETTER_ORDER]
    prompts = [s.prompt + "Label:" for s in samples]

    # Try to use guided_choice if available, otherwise fall back to plain greedy
    sampling_kwargs = dict(max_tokens=1, temperature=0.0)
    try:
        sampling_params = SamplingParams(**sampling_kwargs, guided_choice=label_choices)
    except (TypeError, ValueError):
        # guided_choice not supported in offline API for this vLLM version
        sampling_params = SamplingParams(**sampling_kwargs)

    if not show_progress:
        tqdm.write(f"  Generating predictions for {len(prompts)} samples...")
    gen_kwargs = dict(prompts=prompts, sampling_params=sampling_params)
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    outputs = llm.generate(**gen_kwargs)

    prediction_records: list[dict] = []
    correct = 0
    parse_errors = 0
    # Map letter to label
    letter_to_label = {v: k for k, v in LABEL_LETTERS.items()}

    for sample_idx, (sample, output) in enumerate(zip(samples, outputs)):
        raw_completion = ""
        if output.outputs:
            raw_completion = output.outputs[0].text.strip()
        raw_output = f"Label: {raw_completion}" if raw_completion else "Label:"

        # Parse: look for A-F
        pred_label = "parse_error"
        pred_id = -1
        for letter in LETTER_ORDER:
            if letter in raw_completion:
                pred_label = letter_to_label.get(letter, "parse_error")
                pred_id = LABELS.index(pred_label) if pred_label in LABELS else -1
                break

        if pred_id == int(sample.gold_id):
            correct += 1
        if pred_id < 0:
            parse_errors += 1

        if show_progress and sample_idx % 100 == 0:
            tqdm.write(
                f"  [{label}] {sample_idx + 1}/{len(samples)}: "
                f"acc={correct / (sample_idx + 1):.3f}"
            )

        prediction_records.append({
            "sample_idx": sample_idx,
            "raw_output": raw_output,
            "raw_completion": raw_completion,
            "pred_id": int(pred_id),
            "pred_label": pred_label,
            "gold_id": int(sample.gold_id),
            "gold_label": sample.gold_label,
        })

    pred_ids = np.asarray([int(r["pred_id"]) for r in prediction_records], dtype=np.int64)
    gold_ids = np.asarray([int(r["gold_id"]) for r in prediction_records], dtype=np.int64)
    metrics = _compute_classification_metrics(pred_ids, gold_ids)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_ids, gold_ids)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels
    metrics["prediction_records"] = prediction_records
    return metrics


def _save_results(
    output_dir: Path,
    metrics: dict,
    predictions: list[dict],
    label: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        "num_samples": int(len(predictions)),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "parse_error_rate": float(metrics["parse_error_rate"]),
        "per_class": metrics.get("per_class", {}),
    }

    metrics_path = output_dir / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions_path = output_dir / f"{label}_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    confusion_path = output_dir / f"{label}_confusion.json"
    confusion_path.write_text(
        json.dumps({
            "gold_labels": LABELS,
            "pred_labels": metrics["confusion_labels"],
            "matrix": metrics["confusion_matrix"].tolist(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "confusion": str(confusion_path),
    }


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    # Load both builds
    baseline_rows = load_build_jsonl(args.baseline_build)
    experiment_rows = load_build_jsonl(args.experiment_build)
    print(f"Baseline ({args.baseline_label}): {len(baseline_rows)} samples")
    print(f"Experiment ({args.experiment_label}): {len(experiment_rows)} samples")

    baseline_samples = _load_prebuilt_samples(baseline_rows)
    experiment_samples = _load_prebuilt_samples(experiment_rows)

    if len(baseline_samples) != len(experiment_samples):
        print(f"WARNING: sample counts differ ({len(baseline_samples)} vs {len(experiment_samples)})")

    # Initialize vLLM
    print(f"\nLoading vLLM model: {args.model}")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed.") from exc

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": args.dtype,
        "max_model_len": int(args.max_model_len),
        "trust_remote_code": True,
    }

    lora_request = None
    if args.lora_adapter:
        adapter_dir = Path(args.lora_adapter)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise RuntimeError("vLLM LoRA support required.")

        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = int(args.max_lora_rank)
        lora_request = LoRARequest("lambda-compare", 1, str(adapter_dir))
        print(f"  LoRA adapter: {adapter_dir}")

    print(f"  TP size: {args.tensor_parallel_size}, max_len: {args.max_model_len}")
    llm = LLM(**llm_kwargs)

    # Map label letters to token IDs (for reference)
    tokenizer = llm.get_tokenizer()
    letter_token_ids: dict[str, int] = {}
    for letter in LABEL_LETTERS.values():
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[0]
    print(f"  Letter token IDs: {letter_token_ids}")

    # Run baseline inference
    print(f"\n{'='*50}")
    print(f"Running inference: {args.baseline_label}")
    print(f"{'='*50}")
    baseline_metrics = run_inference_offline(
        baseline_samples, llm, letter_token_ids,
        label=args.baseline_label, show_progress=show_progress,
        lora_request=lora_request,
    )

    # Run experiment inference
    print(f"\n{'='*50}")
    print(f"Running inference: {args.experiment_label}")
    print(f"{'='*50}")
    experiment_metrics = run_inference_offline(
        experiment_samples, llm, letter_token_ids,
        label=args.experiment_label, show_progress=show_progress,
        lora_request=lora_request,
    )

    # Compare
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")

    b_acc = float(baseline_metrics["accuracy"])
    e_acc = float(experiment_metrics["accuracy"])
    b_f1 = float(baseline_metrics["macro_f1"])
    e_f1 = float(experiment_metrics["macro_f1"])
    delta_acc = e_acc - b_acc
    delta_f1 = e_f1 - b_f1

    print(f"{'Metric':<25} {args.baseline_label:>15} {args.experiment_label:>15} {'Δ':>10}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    print(f"{'Accuracy':<25} {b_acc:>15.4f} {e_acc:>15.4f} {delta_acc:>+10.4f}")
    print(f"{'Macro F1':<25} {b_f1:>15.4f} {e_f1:>15.4f} {delta_f1:>+10.4f}")
    print(f"{'Macro Precision':<25} {float(baseline_metrics['macro_precision']):>15.4f} {float(experiment_metrics['macro_precision']):>15.4f}")
    print(f"{'Macro Recall':<25} {float(baseline_metrics['macro_recall']):>15.4f} {float(experiment_metrics['macro_recall']):>15.4f}")
    print(f"{'Parse Error Rate':<25} {float(baseline_metrics['parse_error_rate']):>15.4f} {float(experiment_metrics['parse_error_rate']):>15.4f}")

    # Per-class
    print(f"\n{'Class':<15} {'Baseline F1':>12} {'Experiment F1':>13} {'Δ':>10}")
    print(f"{'-'*15} {'-'*12} {'-'*13} {'-'*10}")
    for label_name in LABELS:
        b_pc = baseline_metrics["per_class"].get(label_name, {})
        e_pc = experiment_metrics["per_class"].get(label_name, {})
        b_f1_pc = float(b_pc.get("f1", 0))
        e_f1_pc = float(e_pc.get("f1", 0))
        print(f"{label_name:<15} {b_f1_pc:>12.4f} {e_f1_pc:>13.4f} {e_f1_pc - b_f1_pc:>+10.4f}")

    # Save
    output_dir = Path(args.output_dir)
    _save_results(output_dir, baseline_metrics, baseline_metrics["prediction_records"], args.baseline_label)
    _save_results(output_dir, experiment_metrics, experiment_metrics["prediction_records"], args.experiment_label)

    comparison = {
        "baseline_label": args.baseline_label,
        "experiment_label": args.experiment_label,
        "baseline_build": args.baseline_build,
        "experiment_build": args.experiment_build,
        "baseline": {"accuracy": b_acc, "macro_f1": b_f1},
        "experiment": {"accuracy": e_acc, "macro_f1": e_f1},
        "delta": {"accuracy": delta_acc, "macro_f1": delta_f1},
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {summary_path}")

    # Assessment
    print(f"\n{'='*60}")
    if delta_acc > 0.02:
        print(f"✓ Oracle λ improves accuracy by {delta_acc:+.4f} (>2%). Direction is promising.")
    elif delta_acc > 0.005:
        print(f"~ Oracle λ gives marginal accuracy gain ({delta_acc:+.4f}). Benefit may be too small.")
    else:
        print(f"✗ Oracle λ does not improve accuracy ({delta_acc:+.4f}). Learned lambda may not be viable.")


if __name__ == "__main__":
    main()
