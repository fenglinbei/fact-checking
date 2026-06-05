from __future__ import annotations

import argparse

from fact_checking.utils.logging import init_logger
from sft.data.io import save_eval_artifacts
from sft.eval import log_eval_summary, summarize_prediction_records
from sft.infer_common import (
    build_inference_context,
    build_label_decoding_input_ids,
    build_label_decoding_prompt,
    build_serializable_metrics,
    build_vllm_prediction_record,
    create_vllm_logit_processors,
)
from sft.logit_adjust import build_logit_adjust_cfg_from_train_config, load_logit_adjust_cfg

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SFT checkpoint with vLLM.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--log-predictions", type=int, default=0)
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
    logit_adjust_cfg = load_logit_adjust_cfg(context.run_dir)
    if logit_adjust_cfg is None:
        logit_adjust_cfg = build_logit_adjust_cfg_from_train_config(context.cfg, context.tokenizer)
    logits_processors = create_vllm_logit_processors(logit_adjust_cfg)
    use_label_decoding = bool(logits_processors)

    sampling_params = SamplingParams(
        max_tokens=1
        if use_label_decoding
        else int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(context.train_cfg.get("max_new_tokens", context.baseline_cfg.get("max_new_tokens", 24)))
        ),
        temperature=float(context.train_cfg.get("temperature", context.baseline_cfg.get("temperature", 0.0))),
        logits_processors=logits_processors if logits_processors else None,
    )

    label_prefix = "Label:"
    prompt_token_ids = [
        build_label_decoding_input_ids(sample, context.tokenizer, label_prefix)
        if use_label_decoding
        else sample.prompt_input_ids
        for sample in context.samples
    ]
    if any(ids is not None for ids in prompt_token_ids) and not all(ids is not None for ids in prompt_token_ids):
        raise ValueError("Mixed pre-tokenized and string prompts are not supported in one vLLM batch.")

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if all(ids is not None for ids in prompt_token_ids):
        generate_kwargs["prompt_token_ids"] = prompt_token_ids
    else:
        generate_kwargs["prompts"] = [
            build_label_decoding_prompt(sample, label_prefix) if use_label_decoding else sample.prompt
            for sample in context.samples
        ]
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(**generate_kwargs)

    prediction_records: list[dict[str, object]] = []
    for sample_idx, (sample, output) in enumerate(zip(context.samples, outputs)):
        raw_completion = output.outputs[0].text if output.outputs else ""
        prediction_records.append(
            build_vllm_prediction_record(
                sample_idx, sample, raw_completion,
                use_label_decoding=use_label_decoding,
            )
        )

    eval_metrics = summarize_prediction_records(
        prediction_records,
        eval_logger=logger,
        log_predictions_limit=int(args.log_predictions),
    )
    log_eval_summary(
        eval_metrics,
        eval_logger=logger,
        split=context.split,
        checkpoint=context.checkpoint_name,
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
