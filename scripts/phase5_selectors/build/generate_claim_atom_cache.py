#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fact_checking.selectors.claim_atomization import (
    AtomGenerationSettings,
    AtomInputExample,
    fallback_atoms_for_claim,
    generate_or_load_claim_atoms,
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
    p = argparse.ArgumentParser(description="Generate and resume a stable claim-atom cache.")
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--raw-path", default=None)
    p.add_argument("--input-mode", default="raw_split", choices=["oracle_results", "raw_split"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--label-schema", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.0)
    p.add_argument("--sample-limit", type=int, default=None)

    p.add_argument("--atom-cache-dir", default="outputs/selectors/atom_anchor/claim_atom_cache")
    p.add_argument("--atom-cache-id", default=None)
    p.add_argument("--resume-atoms", dest="resume_atoms", action="store_true", default=True)
    p.add_argument("--no-resume-atoms", dest="resume_atoms", action="store_false")

    p.add_argument("--atom-base-url", default=os.environ.get("ATOM_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--atom-model", default=os.environ.get("ATOM_MODEL", "deepseek-v4-flash"))
    p.add_argument("--atom-api-key-env", default=os.environ.get("ATOM_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    p.add_argument("--api-timeout", type=float, default=float(os.environ.get("ATOM_API_TIMEOUT", "120")))
    p.add_argument("--api-max-retries", type=int, default=int(os.environ.get("ATOM_API_MAX_RETRIES", "5")))
    p.add_argument("--api-concurrency", type=int, default=int(os.environ.get("ATOM_API_CONCURRENCY", "128")))
    p.add_argument("--retry-initial-delay", type=float, default=float(os.environ.get("ATOM_API_RETRY_INITIAL_DELAY", "1.0")))
    p.add_argument("--retry-max-delay", type=float, default=float(os.environ.get("ATOM_API_RETRY_MAX_DELAY", "30.0")))

    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("ATOM_MAX_TOKENS", "2048")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("ATOM_TEMPERATURE", "0.0")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("ATOM_TOP_P", "1.0")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("ATOM_SEED", "20260526")))
    p.add_argument("--thinking-type", default=os.environ.get("ATOM_THINKING_TYPE", "disabled"))
    p.add_argument("--api-parse-max-retries", type=int, default=int(os.environ.get("ATOM_API_PARSE_MAX_RETRIES", "2")))
    p.add_argument("--guided-json", dest="guided_json", action="store_true", default=True)
    p.add_argument("--no-guided-json", dest="guided_json", action="store_false")
    p.add_argument("--mock-atoms", action="store_true", help="Use deterministic fallback atoms without an API call.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    oracle_results = ""
    if str(args.input_mode) == "raw_split":
        if not args.raw_path:
            raise ValueError("--raw-path is required when --input-mode=raw_split.")
        examples = _load_raw_split_examples(
            args.raw_path,
            dataset=args.dataset,
            label_schema=args.label_schema,
            sample_limit=args.sample_limit,
        )
    else:
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
        raise ValueError("No examples loaded for atomization.")

    thinking_type = args.thinking_type
    if isinstance(thinking_type, str) and not thinking_type.strip():
        thinking_type = None
    settings = AtomGenerationSettings(
        model=str(args.atom_model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
        guided_json=bool(args.guided_json),
        thinking_type=thinking_type,
    )
    client_factory = _mock_atom_client_factory() if bool(args.mock_atoms) else make_openai_chat_client_factory(
        base_url=str(args.atom_base_url),
        model=str(args.atom_model),
        api_key_env=str(args.atom_api_key_env) if args.atom_api_key_env else None,
        timeout=float(args.api_timeout),
    )
    result = generate_or_load_claim_atoms(
        examples=examples,
        split=str(args.split),
        output_dir=args.output_dir,
        atom_cache_dir=args.atom_cache_dir,
        settings=settings,
        client_factory=client_factory,
        cache_id=args.atom_cache_id,
        resume_atoms=bool(args.resume_atoms),
        api_max_retries=int(args.api_max_retries),
        retry_initial_delay=float(args.retry_initial_delay),
        retry_max_delay=float(args.retry_max_delay),
        api_parse_max_retries=int(args.api_parse_max_retries),
        api_concurrency=int(args.api_concurrency),
        no_progress=bool(args.no_progress),
        run_metadata={
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "oracle_results": str(oracle_results),
            "raw_path": str(args.raw_path or ""),
            "input_mode": str(args.input_mode),
            "dataset": str(args.dataset or ""),
            "label_schema": str(args.label_schema or ""),
            "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
            "atom_base_url": str(args.atom_base_url),
            "api_timeout": float(args.api_timeout),
            "api_concurrency": int(args.api_concurrency),
            "mock_atoms": bool(args.mock_atoms),
            "thinking_type": thinking_type,
            "max_candidates": int(args.max_candidates),
            "top_k": int(args.top_k),
            "filter_policy": str(args.filter_policy),
            "min_margin": float(args.min_margin),
            "expected_chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        },
    )
    manifest = result.manifest
    print(f"Wrote claim atoms: {manifest['claim_atoms_path']}")
    print(f"Atom cache: {manifest['atom_cache_path']}")
    print(
        "examples={n_examples_requested} cache_hits={n_loaded_from_cache} "
        "api_generated={n_api_generated} fingerprint={atom_config_fingerprint}".format(**manifest)
    )


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    if split == "test":
        return DEFAULT_TEST_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _load_raw_split_examples(
    raw_path: str,
    *,
    dataset: str | None,
    label_schema: str | None,
    sample_limit: int | None,
) -> list[AtomInputExample]:
    from fact_checking.data.io import load_split

    samples = load_split(raw_path, dataset=dataset, label_schema=label_schema)
    if sample_limit is not None:
        samples = samples[: int(sample_limit)]
    return [
        AtomInputExample(
            event_id=sample.event_id,
            claim=sample.claim,
            gold_label=sample.label,
        )
        for sample in samples
    ]


class _MockAtomClient:
    def __init__(self) -> None:
        self.last_response_metadata: dict[str, object] = {"finish_reason": "stop", "mock": True}

    def generate(self, *, system_prompt: str, user_prompt: str, settings: AtomGenerationSettings) -> str:
        del system_prompt, settings
        claim = _claim_from_user_prompt(user_prompt)
        return json.dumps(
            {
                "complexity": "simple",
                "claim_atoms": fallback_atoms_for_claim(claim),
            },
            ensure_ascii=False,
        )


def _mock_atom_client_factory():
    return _MockAtomClient


def _claim_from_user_prompt(user_prompt: str) -> str:
    lines = str(user_prompt or "").splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "claim:" and idx + 1 < len(lines):
            return lines[idx + 1].strip()
    return ""


if __name__ == "__main__":
    main()
