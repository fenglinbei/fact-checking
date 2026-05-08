from __future__ import annotations

import argparse

from fact_checking.data.constants import LABELS
from fact_checking.utils.logging import init_logger
from sft.data.io import save_eval_artifacts
from sft.eval import summarize_prediction_records
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.parser import _parse_label_id

logger = init_logger(__name__)


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SFT checkpoint with vLLM.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--log-predictions", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install vllm in this environment before using python -m sft.vllm_infer."
        ) from exc

    args = parse_args()
    context = build_inference_context(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        config_path=args.config,
    )

    model_path = str(context.checkpoint_dir)
    tokenizer_path = str(context.checkpoint_dir)
    if context.is_peft_adapter and not (context.checkpoint_dir / "tokenizer_config.json").exists():
        tokenizer_path = context.model_name_or_path
    llm_kwargs = {}
    lora_request = None
    if context.is_peft_adapter:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError("vLLM LoRA inference requires a vLLM build with LoRA support.") from exc

        model_path = context.model_name_or_path
        lora_cfg = context.train_cfg.get("lora", {}) or {}
        max_lora_rank = int(lora_cfg.get("r", 16)) if isinstance(lora_cfg, dict) else 16
        llm_kwargs.update({"enable_lora": True, "max_lora_rank": max_lora_rank})
        lora_request = LoRARequest("sft-lora", 1, str(context.checkpoint_dir))

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype=args.dtype,
        max_model_len=context.max_length,
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        max_tokens=int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(context.train_cfg.get("max_new_tokens", context.baseline_cfg.get("max_new_tokens", 24)))
        ),
        temperature=float(context.train_cfg.get("temperature", context.baseline_cfg.get("temperature", 0.0))),
    )

    generate_kwargs = {
        "prompts": [sample.prompt for sample in context.samples],
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(**generate_kwargs)

    prediction_records: list[dict[str, object]] = []
    for sample_idx, (sample, output) in enumerate(zip(context.samples, outputs)):
        raw_output = output.outputs[0].text if output.outputs else ""
        pred_id = _parse_label_id(raw_output)

        prediction_records.append(
            {
                "sample_idx": sample_idx,
                "prompt": sample.prompt,
                "target": sample.target,
                "raw_output": raw_output,
                "pred_id": int(pred_id),
                "pred_label": _label_name_from_id(int(pred_id)),
                "gold_id": int(sample.gold_id),
                "gold_label": sample.gold_label,
                "gold_explain": sample.gold_explain,
            }
        )

    eval_metrics = summarize_prediction_records(
        prediction_records,
        eval_logger=logger,
        log_predictions_limit=int(args.log_predictions),
    )
    artifacts = save_eval_artifacts(
        eval_dir=context.eval_output_dir / "vllm",
        metrics=build_serializable_metrics(eval_metrics),
        confusion_matrix=eval_metrics["confusion_matrix"],
        confusion_labels=eval_metrics["confusion_labels"],
        prediction_records=eval_metrics.get("prediction_records", []),
        predictions_filename=f"{context.split}_predictions.jsonl",
        title=f"Confusion Matrix ({context.split}/{context.checkpoint_name}, vLLM)",
    )
    logger.info(
        "[INFO] vLLM %s eval for %s saved to %s (metrics=%s, predictions=%s)",
        context.split,
        context.checkpoint_name,
        context.eval_output_dir / "vllm",
        artifacts["metrics_path"],
        artifacts["predictions_path"],
    )


if __name__ == "__main__":
    main()
