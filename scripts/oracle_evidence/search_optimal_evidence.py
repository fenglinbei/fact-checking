#!/usr/bin/env python3
"""Search for the theoretically optimal evidence set for each claim.

Given a fixed chunking strategy, top-K, and trained verifier, this script
finds the K-subset of candidate evidence that maximizes the verifier's
probability of the correct label.  The result is an upper bound on what
any evidence selection policy (fixed-MMR, learned-λ MMR, RL-MMR) can
achieve with the same candidate pool and verifier.

Usage::

    PYTHONPATH=src python scripts/oracle_evidence/search_optimal_evidence.py \\
        --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \\
        --verifier-model /path/to/trained/model \\
        --split val --top-k 5 --search-method greedy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("oracle_evidence")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.  Dicts are merged; scalars
    and lists are replaced."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_model_path(raw: str | None, base_path: str | None) -> str | None:
    """Replace ``/data/models/`` prefix with *base_path* if given."""
    if raw is None:
        return None
    if base_path and raw.startswith("/data/models/"):
        return raw.replace("/data/models/", base_path.rstrip("/") + "/", 1)
    return raw


def load_build_config(config_path: str, config_overrides: str | None, model_base_path: str | None) -> dict:
    """Load and merge Hydra build config from experiment file + defaults."""
    project_root = Path(__file__).resolve().parents[2]

    # Load defaults
    default_path = project_root / "configs" / "build" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default build config not found: {default_path}")
    default_cfg = OmegaConf.to_container(
        OmegaConf.load(default_path), resolve=False
    )
    build_default = default_cfg.get("build", {})

    # Load experiment config
    exp_path = Path(config_path)
    if not exp_path.is_absolute():
        exp_path = project_root / exp_path
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    exp_cfg = OmegaConf.to_container(OmegaConf.load(str(exp_path)), resolve=False)
    build_exp = exp_cfg.get("build", {})

    # Merge: default → experiment
    build_cfg = _deep_merge(build_default, build_exp)

    # Apply CLI overrides
    if config_overrides:
        for override in config_overrides.split(","):
            override = override.strip()
            if "=" not in override:
                continue
            key_path, value = override.split("=", 1)
            # Navigate to nested key
            keys = key_path.split(".")
            if keys and keys[0] == "build":
                keys = keys[1:]
            target = build_cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            # Try to cast value
            try:
                target[keys[-1]] = int(value)
            except ValueError:
                try:
                    target[keys[-1]] = float(value)
                except ValueError:
                    target[keys[-1]] = value

    # Resolve model paths
    if model_base_path:
        retrieval = build_cfg.get("retrieval", {})
        if retrieval.get("embedder_model"):
            retrieval["embedder_model"] = _resolve_model_path(
                retrieval["embedder_model"], model_base_path
            )
        chunking = retrieval.get("chunking", {})
        if chunking.get("embedder_model"):
            chunking["embedder_model"] = _resolve_model_path(
                chunking["embedder_model"], model_base_path
            )
        prompt = build_cfg.get("prompt", {})
        if prompt.get("model_name_or_path"):
            prompt["model_name_or_path"] = _resolve_model_path(
                prompt["model_name_or_path"], model_base_path
            )

    return build_cfg


# ---------------------------------------------------------------------------
# Cache fingerprint & management
# ---------------------------------------------------------------------------


CHUNK_MMR_CACHE_VERSION = "chunk-text-embedding-v1"


def _stable_json(obj: Any) -> str:
    import hashlib

    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _fingerprint(payload: Any, length: int = 12) -> str:
    import hashlib

    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:length]


def _premmr_config_fingerprint(cfg: dict) -> str:
    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {"data": cfg.get("data", {}), "retrieval": retrieval}
    return _fingerprint(payload)


def _chunk_mmr_config_fingerprint(cfg: dict) -> str:
    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {
        "version": CHUNK_MMR_CACHE_VERSION,
        "data": cfg.get("data", {}),
        "retrieval": retrieval,
        "chunking": retrieval_cfg.get("chunking", {}),
    }
    return _fingerprint(payload)


def _candidate_uid(candidate: dict) -> str:
    payload = {
        "report_id": candidate.get("report_id"),
        "sent_idx": candidate.get("sent_idx"),
        "chunk_sent_indices": candidate.get("chunk_sent_indices"),
        "text": " ".join(str(candidate.get("text", "")).lower().strip().split()),
    }
    return _fingerprint(payload)


def _candidate_pool_fingerprint(candidates: list[dict], metadata: dict) -> str:
    payload = {
        "metadata": metadata,
        "candidates": [
            {
                "candidate_uid": c.get("candidate_uid"),
                "source_index": c.get("source_index"),
                "text": c.get("text"),
                "dense_score": c.get("dense_score"),
                "lexical_score": c.get("lexical_score"),
                "bm25_score": c.get("bm25_score"),
                "hybrid_score": c.get("hybrid_score"),
            }
            for c in candidates
        ],
    }
    return _fingerprint(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _serialize_candidate_pool(candidates: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, candidate in enumerate(candidates):
        rows.append({
            "candidate_idx": idx,
            "candidate_uid": candidate.get("candidate_uid"),
            "source_index": candidate.get("source_index"),
            "report_id": candidate.get("report_id"),
            "sent_idx": candidate.get("sent_idx"),
            "chunk_sent_indices": _jsonable(candidate.get("chunk_sent_indices", [])),
            "text": str(candidate.get("text", "")),
            "source_report": _jsonable(candidate.get("source_report", {})),
        })
    return rows


def _serialize_candidate_scores(candidates: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, candidate in enumerate(candidates):
        rows.append({
            "candidate_idx": idx,
            "candidate_uid": candidate.get("candidate_uid"),
            "source_index": candidate.get("source_index"),
            "hybrid_rank": idx,
            "dense_score": float(candidate.get("dense_score", 0.0)),
            "lexical_score": float(candidate.get("lexical_score", 0.0)),
            "bm25_score": float(candidate.get("bm25_score", 0.0)),
            "hybrid_score": float(candidate.get("hybrid_score", 0.0)),
        })
    return rows


def _candidate_pool_metadata(
    *,
    task: dict,
    candidates: list[dict],
    build_cfg: dict,
    chunk_fp: str,
    pre_fp: str,
    chunk_cache_path: Path,
    two_stage: bool,
    two_stage_limit: int,
    top_k: int,
    multiplier: int,
) -> dict:
    retrieval_cfg = build_cfg.get("retrieval", {})
    return {
        "candidate_pool_version": "oracle-search-candidate-pool-v1",
        "chunk_mmr_fingerprint": chunk_fp,
        "pre_mmr_fingerprint": pre_fp,
        "chunk_mmr_cache_path": str(chunk_cache_path),
        "n_original": int(task.get("n_original", 0)),
        "n_scored": int(task.get("n_scored", 0)),
        "n_dedup": int(task.get("n_dedup", 0)),
        "n_candidates": len(candidates),
        "two_stage": bool(two_stage),
        "two_stage_limit": int(two_stage_limit),
        "two_stage_multiplier": int(multiplier),
        "top_k": int(top_k),
        "score_config": {
            "alpha_dense": float(retrieval_cfg.get("alpha_dense", 0.70)),
            "alpha_lexical": float(retrieval_cfg.get("alpha_lexical", 0.20)),
            "alpha_bm25": float(retrieval_cfg.get("alpha_bm25", 0.10)),
        },
        "candidate_order": "hybrid_score_desc" if two_stage else "dedup_source_order",
    }


def _scored_dedup_candidates(sample, retrieval_cfg: dict, canonicalize_sentence_fn) -> list[dict]:
    from fact_checking.build.candidates import compute_hybrid_scores

    alpha_dense = float(retrieval_cfg.get("alpha_dense", 0.70))
    alpha_lexical = float(retrieval_cfg.get("alpha_lexical", 0.20))
    alpha_bm25 = float(retrieval_cfg.get("alpha_bm25", 0.10))
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    dedup: dict[str, dict] = {}
    for idx in range(n):
        base = dict(sample.candidates[idx])
        base.update({
            "source_index": int(idx),
            "candidate_uid": _candidate_uid(base),
            "dense_score": float(scored["dense_scores"][idx]),
            "lexical_score": float(scored["lexical_scores"][idx]),
            "bm25_score": float(scored["bm25_scores"][idx]),
            "hybrid_score": float(scored["hybrid_scores"][idx]),
        })
        canon = canonicalize_sentence_fn(str(base.get("text", "")))
        if not canon:
            continue
        old = dedup.get(canon)
        if old is None or base["hybrid_score"] > float(old.get("hybrid_score", 0.0)):
            dedup[canon] = base
    return list(dedup.values())


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Cache auto-build
# ---------------------------------------------------------------------------


def _build_run_summary(build_cfg: dict) -> dict:
    """Build a run_summary dict matching the one in ``run_build()``.

    Only the keys consumed by ``_compute_pre_mmr_split`` and
    ``_compute_chunk_mmr_split`` are required; the rest are filled
    with sensible defaults.
    """
    retrieval_cfg = build_cfg.get("retrieval", {})
    return {
        "embedder_model": retrieval_cfg["embedder_model"],
        "device": retrieval_cfg.get("device", "cuda"),
        "top_k": int(retrieval_cfg.get("top_k", 24)),
        "batch_size": int(retrieval_cfg.get("batch_size", 64)),
        "max_length": int(retrieval_cfg.get("max_length", 256)),
        "precision": retrieval_cfg.get("precision", "fp32"),
        "prefetch_size": int(retrieval_cfg.get("prefetch_size", 1)),
        "num_gpus": int(retrieval_cfg.get("num_gpus", 1)),
        "output_dir": "",
    }


def ensure_chunk_mmr_cache(
    build_cfg: dict, split: str, project_root: Path
) -> tuple[list, Path]:
    """Ensure chunk-MMR cache exists for *split*; build it if missing.

    Returns (chunk_mmr_samples, cache_path).
    """
    chunk_fp = _chunk_mmr_config_fingerprint(build_cfg)
    chunk_dir = project_root / "outputs" / "cache" / "chunk_mmr" / chunk_fp
    cache_path = chunk_dir / f"{split}.pkl"

    if cache_path.exists():
        logger.info("Chunk-MMR cache found: %s", cache_path)
        return _load_pickle(cache_path), cache_path

    # Need to build
    from fact_checking.build.candidates import (
        _compute_chunk_mmr_split,
        _compute_pre_mmr_split,
    )

    pre_fp = _premmr_config_fingerprint(build_cfg)
    pre_dir = project_root / "outputs" / "cache" / "pre_mmr" / pre_fp
    pre_path = pre_dir / f"{split}.pkl"

    data_cfg = build_cfg.get("data", {})
    retrieval_cfg = build_cfg.get("retrieval", {})
    run_summary = _build_run_summary(build_cfg)
    num_gpus = int(retrieval_cfg.get("num_gpus", 1))

    if not pre_path.exists():
        logger.info("Pre-MMR cache not found, building: %s", pre_path)
        pre_dir.mkdir(parents=True, exist_ok=True)
        _compute_pre_mmr_split(
            split_name=split,
            data_cfg=data_cfg,
            retrieval_cfg=retrieval_cfg,
            run_summary=run_summary,
            cache_dir=pre_dir,
            num_gpus=num_gpus,
        )

    logger.info("Chunk-MMR cache not found, building: %s", cache_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    _compute_chunk_mmr_split(
        split_name=split,
        retrieval_cfg=retrieval_cfg,
        run_summary=run_summary,
        pre_mmr_path=pre_path,
        cache_dir=chunk_dir,
        num_gpus=num_gpus,
    )

    return _load_pickle(cache_path), cache_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search for optimal evidence sets (oracle upper bound)",
    )
    p.add_argument(
        "--config",
        default="configs/experiment/b3_mmr_topk_sweep_1024.yaml",
        help="Hydra experiment config path (default: configs/experiment/b3_mmr_topk_sweep_1024.yaml)",
    )
    p.add_argument(
        "--config-overrides",
        default=None,
        help="Extra Hydra overrides, comma-separated (e.g. build.retrieval.top_k=5)",
    )
    p.add_argument(
        "--model-base-path",
        default=None,
        help="Replace /data/models/ prefix in config paths (e.g. ~/project/hateSpeechDetection/models/base/)",
    )
    p.add_argument(
        "--verifier-model",
        required=True,
        help="Path to trained verifier model for vLLM inference",
    )
    p.add_argument(
        "--lora-adapter",
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    p.add_argument("--top-k", type=int, default=5, help="Target evidence set size")
    p.add_argument(
        "--search-method",
        default="greedy",
        choices=["greedy", "exhaustive", "beam"],
        help="Search strategy (default: greedy)",
    )
    p.add_argument(
        "--objective",
        default="gold_logprob",
        choices=["gold_logprob", "margin"],
        help=(
            "Oracle search objective. gold_logprob maximizes log P(gold); "
            "margin maximizes log P(gold) - max_y!=gold log P(y)."
        ),
    )
    p.add_argument("--beam-width", type=int, default=3, help="Beam width for beam search")
    p.add_argument(
        "--max-exhaustive-n",
        type=int,
        default=20,
        help="Max candidate pool size for exhaustive search",
    )
    p.add_argument(
        "--tensor-parallel-size", type=int, default=4, help="Number of GPUs for vLLM"
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization",
    )
    p.add_argument(
        "--max-model-len", type=int, default=1024, help="Max model sequence length"
    )
    p.add_argument("--dtype", default="auto", help="Model dtype")
    p.add_argument(
        "--score-batch-size",
        type=int,
        default=256,
        help="Max prompts per llm.generate call",
    )
    p.add_argument(
        "--max-samples", type=int, default=0, help="Max samples to process (0=all)"
    )
    p.add_argument(
        "--split", default="val", help="Data split: train / val / test"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/oracle_evidence/<timestamp>/)",
    )
    p.add_argument(
        "--two-stage",
        action="store_true",
        default=True,
        help="Enable two-stage pruning: hybrid_score top-M first",
    )
    p.add_argument(
        "--no-two-stage",
        action="store_true",
        help="Disable two-stage pruning",
    )
    p.add_argument(
        "--two-stage-multiplier",
        type=int,
        default=3,
        help="Retain top (top_k * M) candidates in two-stage mode",
    )
    p.add_argument(
        "--save-candidate-pool",
        dest="save_candidate_pool",
        action="store_true",
        default=True,
        help="Save full effective candidate_pool, candidate_scores, and candidate_pool_fingerprint in each result row.",
    )
    p.add_argument(
        "--no-save-candidate-pool",
        dest="save_candidate_pool",
        action="store_false",
        help="Do not save candidate_pool/candidate_scores in result rows.",
    )
    p.add_argument(
        "--save-search-step-scores",
        action="store_true",
        help="Save per-step candidate oracle logprobs where supported. This can make JSONL much larger.",
    )
    p.add_argument(
        "--verify-config-only",
        action="store_true",
        help="Load config, check cache, print stats, then exit (no vLLM)",
    )
    p.add_argument(
        "--no-progress", action="store_true", help="Suppress progress bars"
    )
    return p.parse_args()


def _collect_candidate_stats(samples: list) -> dict:
    """Compute candidate pool size distribution across samples."""
    sizes = [len(s.candidates) for s in samples]
    if not sizes:
        return {}
    arr = np.array(sizes, dtype=np.int32)
    return {
        "n_samples": int(len(arr)),
        "min": int(arr.min()),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "n_le_15": int((arr <= 15).sum()),
        "n_le_20": int((arr <= 20).sum()),
    }


def _collect_effective_candidate_stats(results: list) -> dict:
    sizes = [int(r.n_candidates) for r in results]
    if not sizes:
        return {}
    arr = np.array(sizes, dtype=np.int32)
    return {
        "n_samples": int(len(arr)),
        "min": int(arr.min()),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "n_le_15": int((arr <= 15).sum()),
        "n_le_20": int((arr <= 20).sum()),
    }


def _compute_metrics(results: list) -> dict:
    """Compute classification metrics from search results."""
    from sft.metrics import _compute_classification_metrics

    pred_ids = np.array([r.final_prediction for r in results], dtype=np.int32)
    gold_ids = np.array([r.gold_id for r in results], dtype=np.int32)
    return _compute_classification_metrics(pred_ids, gold_ids)


def run_search(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]

    # ---- Load config -------------------------------------------------------
    logger.info("Loading config: %s", args.config)
    build_cfg = load_build_config(args.config, args.config_overrides, args.model_base_path)
    retrieval_cfg = build_cfg.get("retrieval", {})
    prompt_cfg = build_cfg.get("prompt", {})

    prompt_model_path = prompt_cfg.get("model_name_or_path", "")
    if args.model_base_path:
        prompt_model_path = _resolve_model_path(prompt_model_path, args.model_base_path)
    max_prompt_length = int(prompt_cfg.get("max_length", 1024))
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "letter")).strip().lower()
    system_prompt = prompt_cfg.get("system_prompt") or None

    logger.info("Prompt model (tokenizer): %s", prompt_model_path)
    logger.info("Max prompt length: %d", max_prompt_length)
    logger.info("Output mode: %s, label format: %s", output_mode, label_format)

    # ---- Compute fingerprints -----------------------------------------------
    chunk_fp = _chunk_mmr_config_fingerprint(build_cfg)
    chunk_dir = project_root / "outputs" / "cache" / "chunk_mmr" / chunk_fp
    chunk_cache_path = chunk_dir / f"{args.split}.pkl"

    pre_fp = _premmr_config_fingerprint(build_cfg)
    pre_dir = project_root / "outputs" / "cache" / "pre_mmr" / pre_fp
    pre_cache_path = pre_dir / f"{args.split}.pkl"

    logger.info("Chunk-MMR fingerprint: %s", chunk_fp)
    logger.info("Chunk-MMR cache path:  %s", chunk_cache_path)
    logger.info("Chunk-MMR exists:      %s", chunk_cache_path.exists())
    logger.info("Pre-MMR fingerprint:   %s", pre_fp)
    logger.info("Pre-MMR exists:        %s", pre_cache_path.exists())

    if args.verify_config_only:
        logger.info("Config summary:")
        logger.info("  chunking: %s (theta=%s)",
                    retrieval_cfg.get("chunking", {}).get("strategy", "?"),
                    retrieval_cfg.get("chunking", {}).get("theta", "?"))
        logger.info("  embedder: %s", retrieval_cfg.get("embedder_model", "?"))
        logger.info("  num_gpus: %s", retrieval_cfg.get("num_gpus", "?"))
        logger.info("  top_k (default): %s", retrieval_cfg.get("top_k", "?"))
        if chunk_cache_path.exists():
            chunk_samples = _load_pickle(chunk_cache_path)
            stats = _collect_candidate_stats(chunk_samples)
            logger.info("Candidate pool stats:\n%s", json.dumps(stats, indent=2))
        else:
            logger.info("Cache not available — run without --verify-config-only on a machine with model files to build it.")
        logger.info("--verify-config-only: exiting.")
        return

    # ---- Cache build (if needed) --------------------------------------------
    if not chunk_cache_path.exists():
        logger.info("Chunk-MMR cache not found; building...")
        from fact_checking.build.candidates import (
            _compute_chunk_mmr_split,
            _compute_pre_mmr_split,
        )
        data_cfg = build_cfg.get("data", {})
        run_summary = _build_run_summary(build_cfg)
        num_gpus = int(retrieval_cfg.get("num_gpus", 1))

        if not pre_cache_path.exists():
            logger.info("Pre-MMR cache not found; building...")
            pre_dir.mkdir(parents=True, exist_ok=True)
            _compute_pre_mmr_split(
                split_name=args.split,
                data_cfg=data_cfg,
                retrieval_cfg=retrieval_cfg,
                run_summary=run_summary,
                cache_dir=pre_dir,
                num_gpus=num_gpus,
            )

        chunk_dir.mkdir(parents=True, exist_ok=True)
        _compute_chunk_mmr_split(
            split_name=args.split,
            retrieval_cfg=retrieval_cfg,
            run_summary=run_summary,
            pre_mmr_path=pre_cache_path,
            cache_dir=chunk_dir,
            num_gpus=num_gpus,
        )

    # ---- Load cache ---------------------------------------------------------
    chunk_samples = _load_pickle(chunk_cache_path)
    logger.info("Loaded %d samples from cache", len(chunk_samples))

    # Print candidate pool statistics
    stats = _collect_candidate_stats(chunk_samples)
    logger.info(
        "Candidate pool sizes: min=%d p25=%d median=%d p75=%d p90=%d p95=%d max=%d mean=%.1f",
        stats["min"], stats["p25"], stats["median"], stats["p75"],
        stats["p90"], stats["p95"], stats["max"], stats["mean"],
    )
    logger.info(
        "Samples with N≤15: %d/%d (%.1f%%)",
        stats["n_le_15"], stats["n_samples"],
        100.0 * stats["n_le_15"] / max(1, stats["n_samples"]),
    )

    # ---- Score and deduplicate candidates within each sample ---------------
    from fact_checking.build.candidates import canonicalize_sentence

    top_k = args.top_k
    two_stage = args.two_stage and not args.no_two_stage
    multiplier = args.two_stage_multiplier
    max_exhaustive_n = args.max_exhaustive_n

    # ---- vLLM init ---------------------------------------------------------
    logger.info("Initializing vLLM...")
    # Suppress vLLM engine log spam during init / generate
    for _lib in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install vllm first.") from exc

    llm_kwargs: dict = {}
    lora_request = None
    if args.lora_adapter:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError(
                "LoRA support requires a vLLM build with LoRA."
            ) from exc
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        lora_request = LoRARequest("oracle-lora", 1, args.lora_adapter)

    llm = LLM(
        model=args.verifier_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    logger.info("vLLM initialized. Tokenizer vocab size: %d", tokenizer.vocab_size)

    # ---- Scorer ------------------------------------------------------------
    from fact_checking.oracle_evidence.scorer import VerifierScorer

    # Use the actual vLLM max_model_len (minus margin) as the prompt budget,
    # not the training config's max_length, so truncation respects the real limit.
    prompt_budget = args.max_model_len - 16
    scorer = VerifierScorer(
        llm=llm,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        output_mode=output_mode,
        label_format=label_format,
        max_prompt_length=min(max_prompt_length, prompt_budget),
        lora_request=lora_request,
    )
    logger.info(
        "Label token IDs: %s",
        {k: v for k, v in scorer.label_token_ids.items()},
    )

    # ---- Search ------------------------------------------------------------
    from fact_checking.data.constants import ID2LABEL, LABEL2ID, LABEL_LETTERS
    from fact_checking.oracle_evidence.search import (
        SearchResult,
        beam_search,
        exhaustive_search,
        greedy_search,
    )

    # Build task list
    tasks: list[dict] = []
    for sample in chunk_samples:
        if not sample.candidates:
            continue
        dedup = _scored_dedup_candidates(sample, retrieval_cfg, canonicalize_sentence)
        if not dedup:
            continue

        gold_label = str(sample.label or "").strip().lower()
        gold_letter = LABEL_LETTERS.get(gold_label, "")
        if not gold_letter:
            continue

        tasks.append({
            "event_id": str(sample.event_id),
            "claim": str(sample.claim),
            "gold_label": gold_label,
            "gold_letter": gold_letter,
            "candidates": dedup,
            "n_original": len(sample.candidates),
            "n_scored": min(len(sample.candidates), int(sample.chunk_emb.shape[0])),
            "n_dedup": len(dedup),
        })

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]

    logger.info("Search tasks: %d", len(tasks))

    # Determine per-sample search strategy
    results: list[SearchResult] = []
    search_start = time.monotonic()

    # Progress display: tqdm when available, otherwise periodic logging
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    task_iter = tasks
    pbar = None
    if not args.no_progress:
        if _tqdm is not None:
            pbar = _tqdm(total=len(tasks), desc="Oracle search", unit="sample",
                         dynamic_ncols=True)
        else:
            logger.info("Searching %d samples (tqdm not available, logging every 100)...",
                        len(tasks))

    for i, task in enumerate(task_iter):
        candidates = task["candidates"]
        n_full = len(candidates)

        # Two-stage pruning
        two_stage_limit = n_full
        if two_stage:
            limit = min(n_full, top_k * multiplier)
            two_stage_limit = limit
            candidates = sorted(
                candidates,
                key=lambda c: float(c.get("hybrid_score", 0.0)),
                reverse=True,
            )[:limit]

        n = len(candidates)
        pool_metadata = _candidate_pool_metadata(
            task=task,
            candidates=candidates,
            build_cfg=build_cfg,
            chunk_fp=chunk_fp,
            pre_fp=pre_fp,
            chunk_cache_path=chunk_cache_path,
            two_stage=two_stage,
            two_stage_limit=two_stage_limit,
            top_k=top_k,
            multiplier=multiplier,
        )
        candidate_pool_fingerprint = _candidate_pool_fingerprint(candidates, pool_metadata)
        method = args.search_method

        # Force exhaustive for small pools, downgrade for large pools
        if method == "exhaustive" and n > max_exhaustive_n:
            logger.debug(
                "Sample %s: N=%d > %d, falling back to greedy",
                task["event_id"], n, max_exhaustive_n,
            )
            method = "greedy"

        common = dict(
            claim=task["claim"],
            candidates=candidates,
            top_k=min(top_k, n),
            gold_label_letter=task["gold_letter"],
            scorer=scorer,
            score_batch_size=args.score_batch_size,
            record_step_scores=args.save_search_step_scores,
            objective=args.objective,
        )

        if method == "exhaustive":
            result = exhaustive_search(**common)
        elif method == "beam":
            result = beam_search(beam_width=args.beam_width, **common)
        else:
            result = greedy_search(**common)

        result.event_id = task["event_id"]
        result.gold_label = task["gold_label"]
        result.gold_id = LABEL2ID.get(task["gold_label"], -1)
        result.is_correct = (result.final_prediction == result.gold_id)
        result.candidate_pool_fingerprint = candidate_pool_fingerprint
        result.candidate_pool_metadata = pool_metadata
        if args.save_candidate_pool:
            result.candidate_pool = _serialize_candidate_pool(candidates)
            result.candidate_scores = _serialize_candidate_scores(candidates)

        results.append(result)

        if pbar is not None:
            correct = sum(1 for r in results if r.is_correct)
            pbar.set_postfix({"acc": f"{correct / len(results):.3f}"})
            pbar.update(1)
        elif not args.no_progress and _tqdm is None and (i + 1) % 100 == 0:
            correct = sum(1 for r in results if r.is_correct)
            elapsed = time.monotonic() - search_start
            logger.info(
                "Progress: %d/%d | acc=%.3f | elapsed=%.0fs",
                i + 1, len(tasks),
                correct / len(results),
                elapsed,
            )

    if pbar is not None:
        pbar.close()

    search_elapsed = time.monotonic() - search_start
    logger.info("Search completed in %.0fs (%.1f min)", search_elapsed, search_elapsed / 60.0)

    # ---- Metrics -----------------------------------------------------------
    if not results:
        logger.warning("No results to evaluate.")
        return

    metrics = _compute_metrics(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / len(results)

    logger.info("Oracle Accuracy: %.4f (%d/%d)", accuracy, correct, len(results))
    logger.info("Macro F1: %.4f", metrics.get("macro_f1", 0.0))

    # ---- Output ------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "oracle_evidence" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results JSONL
    results_path = output_dir / f"oracle_results_{args.split}.jsonl"
    with open(results_path, "w", encoding="utf-8") as fh:
        for r in results:
            entry = {
                "event_id": r.event_id,
                "claim": r.claim,
                "gold_label": r.gold_label,
                "gold_id": r.gold_id,
                "n_candidates": r.n_candidates,
                "top_k": r.top_k,
                "selected_indices": r.selected_indices,
                "selected_texts": r.selected_texts,
                "final_logprob": r.final_logprob,
                "final_objective": r.final_objective,
                "gold_logprob": r.gold_logprob,
                "best_wrong_logprob": r.best_wrong_logprob,
                "margin": r.margin,
                "label_logprobs": r.label_logprobs,
                "final_prediction": int(r.final_prediction),
                "pred_label": ID2LABEL.get(int(r.final_prediction), "parse_error"),
                "prediction_source": r.prediction_source,
                "is_correct": r.is_correct,
                "search_method": r.search_method,
                "search_objective": r.search_objective,
                "search_steps": r.search_steps,
                "candidate_pool_fingerprint": r.candidate_pool_fingerprint,
                "candidate_pool_metadata": r.candidate_pool_metadata,
            }
            if args.save_candidate_pool:
                entry["candidate_pool"] = r.candidate_pool
                entry["candidate_scores"] = r.candidate_scores
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Results written to: %s", results_path)

    # Metrics JSON
    metrics_path = output_dir / f"oracle_metrics_{args.split}.json"
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            serializable_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v
    serializable_metrics["accuracy"] = accuracy
    serializable_metrics["n_samples"] = len(results)
    serializable_metrics["search_method"] = args.search_method
    serializable_metrics["search_objective"] = args.objective
    serializable_metrics["top_k"] = top_k
    serializable_metrics["split"] = args.split
    serializable_metrics["search_elapsed_s"] = search_elapsed
    serializable_metrics["candidate_pool_stats"] = stats
    serializable_metrics["effective_candidate_pool_stats"] = _collect_effective_candidate_stats(results)
    serializable_metrics["output_contract"] = {
        "version": "oracle-results-v3",
        "save_candidate_pool": bool(args.save_candidate_pool),
        "save_search_step_scores": bool(args.save_search_step_scores),
        "search_objective": args.objective,
        "gold_logprob": "log P(gold label token | claim, selected evidence)",
        "best_wrong_logprob": "max log P(non-gold label token | claim, selected evidence)",
        "margin": "gold_logprob - best_wrong_logprob",
        "final_objective": "gold_logprob for objective=gold_logprob; margin for objective=margin",
        "candidate_pool_fingerprint": "per-row fingerprint over effective candidate pool metadata, order, text, and retrieval scores",
        "selected_indices_coordinate": "indices into per-row effective candidate_pool after deduplication and optional two-stage pruning",
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(serializable_metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Metrics written to: %s", metrics_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Oracle Evidence Selection Summary")
    logger.info("=" * 60)
    logger.info("Split: %s", args.split)
    logger.info("Samples: %d", len(results))
    logger.info("Top-K: %d", top_k)
    logger.info("Search method: %s", args.search_method)
    logger.info("Search objective: %s", args.objective)
    logger.info("Oracle Accuracy: %.4f", accuracy)
    logger.info("Macro F1: %.4f", metrics.get("macro_f1", 0.0))
    logger.info("Search time: %.0fs (%.1f min)", search_elapsed, search_elapsed / 60.0)
    logger.info("Output: %s", output_dir)

    # Explicit vLLM cleanup to avoid NCCL OOM errors during Python teardown.
    # Without this, vLLM multiprocess workers may crash on exit when CUDA
    # resources are released in an uncontrolled order.
    logger.info("Shutting down vLLM engine...")
    try:
        del scorer
        del llm
    except Exception:
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Cleanup complete.")


if __name__ == "__main__":
    run_search(parse_args())
