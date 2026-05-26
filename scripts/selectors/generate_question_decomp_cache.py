#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fact_checking.selectors.question_decomp import (
    QuestionGenerationSettings,
    generate_or_load_questions,
    make_openai_chat_client_factory,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
)


DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_TEST_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and resume a stable question-decomposition cache for retrieval experiments."
    )
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.0)
    p.add_argument("--sample-limit", type=int, default=None)

    p.add_argument("--question-cache-dir", default="outputs/selectors/question_decomp_retrieval/question_cache")
    p.add_argument("--question-cache-id", default=None)
    p.add_argument("--resume-questions", dest="resume_questions", action="store_true", default=True)
    p.add_argument("--no-resume-questions", dest="resume_questions", action="store_false")

    p.add_argument("--question-base-url", default=os.environ.get("QUESTION_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--question-model", default=os.environ.get("QUESTION_MODEL", "/data/models/Qwen2.5-7B-Instruct"))
    p.add_argument("--question-api-key-env", default=os.environ.get("QUESTION_API_KEY_ENV", "QUESTION_API_KEY"))
    p.add_argument("--api-timeout", type=float, default=float(os.environ.get("QUESTION_API_TIMEOUT", "120")))
    p.add_argument("--api-max-retries", type=int, default=int(os.environ.get("API_MAX_RETRIES", "5")))
    p.add_argument("--retry-initial-delay", type=float, default=float(os.environ.get("API_RETRY_INITIAL_DELAY", "1.0")))
    p.add_argument("--retry-max-delay", type=float, default=float(os.environ.get("API_RETRY_MAX_DELAY", "30.0")))

    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "384")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "20260526")))
    p.add_argument("--guided-json", dest="guided_json", action="store_true", default=True)
    p.add_argument("--no-guided-json", dest="guided_json", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    oracle_results = args.oracle_results or _default_oracle_results(args.split)
    if not Path(oracle_results).exists():
        raise FileNotFoundError(
            f"Oracle results not found: {oracle_results}. Set ORACLE_RESULTS or pass --oracle-results."
        )

    examples = load_stage2_oracle_examples(
        oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No examples after Stage2 audit/filtering.")

    settings = QuestionGenerationSettings(
        model=str(args.question_model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
        guided_json=bool(args.guided_json),
    )
    client_factory = make_openai_chat_client_factory(
        base_url=str(args.question_base_url),
        model=str(args.question_model),
        api_key_env=str(args.question_api_key_env) if args.question_api_key_env else None,
        timeout=float(args.api_timeout),
    )
    result = generate_or_load_questions(
        examples=examples,
        split=str(args.split),
        output_dir=args.output_dir,
        question_cache_dir=args.question_cache_dir,
        settings=settings,
        client_factory=client_factory,
        cache_id=args.question_cache_id,
        resume_questions=bool(args.resume_questions),
        api_max_retries=int(args.api_max_retries),
        retry_initial_delay=float(args.retry_initial_delay),
        retry_max_delay=float(args.retry_max_delay),
        run_metadata={
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "oracle_results": str(oracle_results),
            "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
            "question_base_url": str(args.question_base_url),
            "api_timeout": float(args.api_timeout),
            "max_candidates": int(args.max_candidates),
            "top_k": int(args.top_k),
            "filter_policy": str(args.filter_policy),
            "min_margin": float(args.min_margin),
            "expected_chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        },
    )
    manifest = result.manifest
    print(f"Wrote questions: {manifest['questions_path']}")
    print(f"Question cache: {manifest['question_cache_path']}")
    print(
        "examples={n_examples_requested} cache_hits={n_loaded_from_cache} "
        "api_generated={n_api_generated} fingerprint={question_config_fingerprint}".format(**manifest)
    )


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    if split == "test":
        return DEFAULT_TEST_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


if __name__ == "__main__":
    main()
