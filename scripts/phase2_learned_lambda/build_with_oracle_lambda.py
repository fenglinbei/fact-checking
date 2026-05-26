"""Build a single JSONL with per-claim oracle λ overrides.

Loads a chunk-MMR cache and an oracle-λ JSONL, then calls the standard MMR
phase with per-claim lambda values. The resulting build JSONL is identical
in format to the main pipeline output, except each claim uses its oracle λ
during MMR candidate selection.

Usage:
    PYTHONPATH=src python scripts/phase2_learned_lambda/build_with_oracle_lambda.py \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --experiment b3_mmr_topk_sweep_1024 \
        --split-name train \
        --output outputs/learned_lambda/build_oracle_train.jsonl

    # Only build for a subset of event-ids (e.g. predictor val split):
    PYTHONPATH=src python scripts/phase2_learned_lambda/build_with_oracle_lambda.py \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --experiment b3_mmr_topk_sweep_1024 \
        --split-name train \
        --output outputs/learned_lambda/build_oracle_val.jsonl \
        --event-id-file outputs/learned_lambda/predictor_val_eids.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fact_checking.build.candidates import (
    _chunk_mmr_config_fingerprint,
    _load_pickle,
    _load_prompt_tokenizer,
    _mmr_phase_from_chunk_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build JSONL with per-claim oracle λ overrides.")
    p.add_argument("--oracle-lambdas", type=str, required=True)
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--chunk-mmr-cache", type=str, default=None)
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--cpu-workers", type=int, default=1)
    p.add_argument("--event-id-file", type=str, default=None,
                   help="Optional JSON file with a list of event_ids to include.")
    p.add_argument("--mmr-lambda-fallback", type=float, default=0.70,
                   help="Fallback λ for claims missing from oracle data.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _resolve_build_cfg(experiment: str, config_overrides: list[str]) -> dict[str, Any]:
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        overrides = [f"experiment={experiment}"] + list(config_overrides)
        cfg = compose(config_name="pipeline/default", overrides=overrides)
        return OmegaConf.to_container(cfg["build"], resolve=True)  # type: ignore[return-value]


def _resolve_chunk_mmr_cache(
    build_cfg: dict[str, Any],
    split_name: str,
    cache_root: str,
    chunk_mmr_cache: str | None,
) -> Path:
    if chunk_mmr_cache:
        return Path(chunk_mmr_cache)
    retrieval_cfg = build_cfg.get("retrieval", {})
    fp = chunk_mmr_config_fingerprint(build_cfg)
    return Path(cache_root) / fp / f"{split_name}.pkl"


def _pick_retrieval_value(
    cli_val: float | None,
    retrieval_cfg: dict[str, Any],
    key: str,
    default: float,
) -> float:
    if cli_val is not None:
        return float(cli_val)
    return float(retrieval_cfg.get(key, default))


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    build_cfg = _resolve_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    prompt_cfg = build_cfg.get("prompt", {})

    chunk_mmr_path = _resolve_chunk_mmr_cache(
        build_cfg, args.split_name, args.chunk_mmr_cache_root, args.chunk_mmr_cache,
    )
    if not chunk_mmr_path.exists():
        raise FileNotFoundError(f"Chunk-MMR cache not found: {chunk_mmr_path}")

    top_k = int(args.top_k if args.top_k is not None else retrieval_cfg.get("top_k", 16))
    alpha_dense = _pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70)
    alpha_lexical = _pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20)
    alpha_bm25 = _pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10)

    model_name_or_path = str(prompt_cfg.get("model_name_or_path", ""))
    tokenizer = load_prompt_tokenizer(model_name_or_path)
    prompt_cfg_local = {
        "auto_length": bool(prompt_cfg.get("auto_length", True)),
        "max_length": int(prompt_cfg.get("max_length", 2048)),
        "output_mode": str(prompt_cfg.get("output_mode", "label_only")).strip().lower(),
        "label_format": str(prompt_cfg.get("label_format", "name")).strip().lower(),
        "system_prompt": prompt_cfg.get("system_prompt") or None,
    }

    # Load chunk-MMR cache
    chunk_samples = load_pickle(chunk_mmr_path)
    print(f"Loaded {len(chunk_samples)} chunk-MMR samples from {chunk_mmr_path}", flush=True)

    # Load oracle λ
    oracle_by_eid: dict[str, float] = {}
    with open(args.oracle_lambdas) as f:
        for line in f:
            rec = json.loads(line.strip())
            oracle_by_eid[rec["event_id"]] = float(rec["oracle_lambda"])
    print(f"Loaded {len(oracle_by_eid)} oracle λ values", flush=True)

    # Optional: filter to a subset of event_ids
    allowed_eids: set[str] | None = None
    if args.event_id_file:
        with open(args.event_id_file) as f:
            allowed_eids = set(json.load(f))
        print(f"Filtering to {len(allowed_eids)} event_ids from {args.event_id_file}", flush=True)

    # Build lambda_overrides
    lambda_overrides: dict[str, float] = {}
    matched = 0
    missing = 0
    filtered = 0
    for sample in chunk_samples:
        if allowed_eids is not None and sample.event_id not in allowed_eids:
            filtered += 1
            continue
        if sample.event_id in oracle_by_eid:
            lambda_overrides[sample.event_id] = oracle_by_eid[sample.event_id]
            matched += 1
        else:
            missing += 1

    print(
        f"lambda_overrides: {matched} matched, {missing} missing (will use fallback={args.mmr_lambda_fallback}), "
        f"{filtered} filtered out",
        flush=True,
    )

    if allowed_eids is not None:
        # Only process samples that are in allowed_eids
        chunk_samples = [s for s in chunk_samples if s.event_id in allowed_eids]
        print(f"Processing {len(chunk_samples)} filtered samples", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _mmr_phase_from_chunk_cache(
        chunk_samples=chunk_samples,
        mmr_lambda=args.mmr_lambda_fallback,
        top_k=top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg_local,
        output_path=output_path,
        cpu_workers=args.cpu_workers,
        lambda_overrides=lambda_overrides,
        show_progress=show_progress,
        progress_desc="MMR (oracle λ)",
    )
    print(f"Wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
