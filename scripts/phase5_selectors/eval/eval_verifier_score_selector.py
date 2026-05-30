#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.build.candidates import _load_prompt_tokenizer
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
)
from fact_checking.selectors.verifier_proxy import (
    DEFAULT_DIRECT_VERIFIER_RUN_DIR,
    DEFAULT_VERIFIER_CHECKPOINT,
    prompt_config_fingerprint,
    require_verifier_checkpoint,
    verifier_config_fingerprint,
)
from fact_checking.selectors.verifier_score_selector import (
    run_chunked_selector,
)


DEFAULT_VAL_ORACLE_RESULTS = (
    "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/selectors/verifier_score_selector/b3_oracle_direct_v0/val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate deployable verifier-score selectors with chunked resumable local vLLM inference."
    )
    p.add_argument("--oracle-results", default=DEFAULT_VAL_ORACLE_RESULTS)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)

    p.add_argument("--selection-mode", default="both", help="static_top5, greedy_stepwise_top5, or both.")
    p.add_argument(
        "--score-modes",
        default="pred_margin,entropy_neg,base_pred_margin,gold_margin",
        help="Comma-separated score modes.",
    )
    p.add_argument("--claim-batch-size", type=int, default=8)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--fsync-cache", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--run-instance-id", default="")

    p.add_argument("--direct-verifier-run-dir", default=DEFAULT_DIRECT_VERIFIER_RUN_DIR)
    p.add_argument("--verifier-checkpoint", default=DEFAULT_VERIFIER_CHECKPOINT)
    p.add_argument("--label-prefix", default="Label:")
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--prompt-max-length", type=int, default=1024)

    p.add_argument("--vllm-model-path", default=None)
    p.add_argument("--vllm-tokenizer-path", default=None)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=4)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--vllm-dtype", default="auto")
    p.add_argument("--vllm-max-model-len", type=int, default=None)
    p.add_argument("--vllm-prompt-batch-size", type=int, default=6000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=str(args.expected_chunk_mmr_fingerprint),
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No Stage2 oracle examples after audit/filtering.")

    checkpoint_info = require_verifier_checkpoint(
        args.direct_verifier_run_dir,
        args.verifier_checkpoint,
        label_prefix=str(args.label_prefix),
    )
    prompt_cfg = _prompt_cfg(args, checkpoint_info.checkpoint_dir)
    verifier_fp = verifier_config_fingerprint(checkpoint_info)
    prompt_fp = prompt_config_fingerprint(prompt_cfg)

    tokenizer = None
    scorer = None
    if not bool(args.finalize_only):
        tokenizer = _load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))
        scorer = _init_vllm_scorer(args, checkpoint_info)

    final = run_chunked_selector(
        examples=examples,
        output_dir=args.output_dir,
        split=str(args.split),
        top_k=int(args.top_k),
        selection_modes=str(args.selection_mode),
        score_modes=str(args.score_modes),
        verifier_fingerprint=verifier_fp,
        prompt_fingerprint=prompt_fp,
        scorer=scorer,
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        claim_batch_size=int(args.claim_batch_size),
        resume=bool(args.resume),
        fsync_cache=bool(args.fsync_cache),
        finalize_only=bool(args.finalize_only),
        run_instance_id=str(args.run_instance_id or ""),
        no_progress=bool(args.no_progress),
        run_metadata={
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "oracle_results": str(args.oracle_results),
            "output_dir": str(args.output_dir),
            "filter_policy": str(args.filter_policy),
            "min_margin": float(args.min_margin),
            "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
            "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
            "verifier": checkpoint_info.__dict__,
            "verifier_config_fingerprint": verifier_fp,
            "prompt_config": prompt_cfg,
            "prompt_config_fingerprint": prompt_fp,
            "vllm": {
                "model_path": str(args.vllm_model_path or checkpoint_info.base_model_name_or_path),
                "tokenizer_path": str(
                    args.vllm_tokenizer_path
                    or args.vllm_model_path
                    or checkpoint_info.base_model_name_or_path
                ),
                "tensor_parallel_size": int(args.vllm_tensor_parallel_size),
                "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
                "dtype": str(args.vllm_dtype),
                "max_model_len": int(args.vllm_max_model_len)
                if args.vllm_max_model_len is not None
                else None,
                "prompt_batch_size": int(args.vllm_prompt_batch_size),
            },
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )
    print(f"Wrote verifier-score selector outputs: {Path(args.output_dir) / 'comparison_table.json'}")
    print(f"finalized_events={final.get('n_finalized_events')} run_fingerprint={final.get('run_fingerprint')}")
    for row in final.get("comparison", []):
        print(
            "{selector}: jaccard@5={jac:.4f} recall@5={rec:.4f} delta_jaccard_vs_hybrid={delta:.4f}".format(
                selector=row.get("selector_name"),
                jac=float(row.get("jaccard@5", 0.0)),
                rec=float(row.get("recall@5", 0.0)),
                delta=float(row.get("delta_vs_hybrid_jaccard@5", 0.0)),
            )
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


def _init_vllm_scorer(args: argparse.Namespace, checkpoint_info: Any) -> Any:
    for lib in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker"):
        import logging

        logging.getLogger(lib).setLevel(logging.WARNING)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Run this in the cppo environment on the target server.") from exc
    from fact_checking.selectors.verifier_scorer import LLMVerifierScorer

    base_model = str(args.vllm_model_path or checkpoint_info.base_model_name_or_path)
    tokenizer_path = str(args.vllm_tokenizer_path or base_model)
    llm_kwargs: dict[str, Any] = {
        "model": base_model,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": int(args.vllm_tensor_parallel_size),
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "dtype": str(args.vllm_dtype),
        "trust_remote_code": True,
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = int(args.vllm_max_model_len)

    lora_request = None
    adapter_config_path = Path(checkpoint_info.adapter_config_path)
    if adapter_config_path.exists():
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError("vLLM LoRA support is required for this checkpoint.") from exc
        with adapter_config_path.open(encoding="utf-8") as fh:
            adapter_cfg = json.load(fh)
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = int(adapter_cfg.get("r", 16))
        lora_request = LoRARequest("verifier-score-selector", 1, checkpoint_info.checkpoint_dir)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    return LLMVerifierScorer(
        llm=llm,
        sampling_params=sampling_params,
        label_token_ids=checkpoint_info.label_token_ids,
        label_prefix=str(args.label_prefix),
        lora_request=lora_request,
        prompt_batch_size=int(args.vllm_prompt_batch_size),
    )


if __name__ == "__main__":
    main()
