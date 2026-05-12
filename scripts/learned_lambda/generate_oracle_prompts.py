"""Step 1: Generate build JSONL for each λ value from PreMMR cache.

Reuses the existing _mmr_phase_from_premmr to produce prompts identical to
those the main pipeline would generate, one file per λ.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/generate_oracle_prompts.py \
        --experiment b3_mmr_topk_sweep_1024 \
        --rebuild-premmr-cache \
        --output-dir outputs/learned_lambda/prompts/ \
        --top-k 12
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fact_checking.build.chunking import build_chunking_strategy
from fact_checking.build.candidates import (
    _can_reuse_chunk_embeddings,
    _compute_pre_mmr_split,
    _load_pickle,
    _load_prompt_tokenizer,
    _mmr_phase_from_premmr,
    _premmr_config_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate oracle-λ prompts for each λ value.")
    p.add_argument(
        "--premmr-cache",
        type=str,
        default=None,
        help="Path to PreMMR cache pickle. If omitted, resolve it from the experiment build fingerprint.",
    )
    p.add_argument(
        "--premmr-cache-root",
        type=str,
        default="outputs/cache/pre_mmr",
        help="Root directory for fingerprinted PreMMR caches.",
    )
    p.add_argument(
        "--rebuild-premmr-cache",
        action="store_true",
        help="Rebuild the fingerprinted PreMMR cache for --split-name before generating prompts.",
    )
    p.add_argument("--output-dir", type=str, required=True, help="Directory for per-λ JSONL outputs")
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional Hydra experiment name whose build.retrieval/build.prompt settings should be reused.",
    )
    p.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        help="Additional Hydra overrides used with --experiment, e.g. build.retrieval.chunking.theta=0.6",
    )
    p.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Tokenizer model path for prompt construction. Overrides build.prompt.model_name_or_path.",
    )
    p.add_argument("--top-k", type=int, default=None, help="Overrides build.retrieval.top_k.")
    p.add_argument("--alpha-dense", type=float, default=None, help="Overrides build.retrieval.alpha_dense.")
    p.add_argument("--alpha-lexical", type=float, default=None, help="Overrides build.retrieval.alpha_lexical.")
    p.add_argument("--alpha-bm25", type=float, default=None, help="Overrides build.retrieval.alpha_bm25.")
    p.add_argument("--cpu-workers", type=int, default=None, help="Overrides build.retrieval.cpu_workers.")
    p.add_argument("--lambda-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--prompt-max-length", type=int, default=None, help="Overrides build.prompt.max_length.")
    p.add_argument("--prompt-output-mode", type=str, default=None, help="Overrides build.prompt.output_mode.")
    p.add_argument("--prompt-label-format", type=str, default=None, help="Overrides build.prompt.label_format.")
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    return p.parse_args()


def _load_experiment_build_cfg(experiment: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="pipeline/default",
            overrides=[f"experiment={experiment}", *overrides],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra config did not resolve to a dictionary.")
    build_cfg = resolved.get("build", {})
    if not isinstance(build_cfg, dict):
        raise TypeError("Resolved Hydra config does not contain a build dictionary.")
    return build_cfg


def _pick(cli_value, cfg: dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    value = cfg.get(key)
    if value is not None:
        return value
    return default


def _ensure_premmr_cache(
    build_cfg: dict[str, Any],
    *,
    split_name: str,
    cache_root: Path,
    rebuild: bool,
) -> Path:
    if not build_cfg:
        raise ValueError("--experiment is required when --premmr-cache is omitted or --rebuild-premmr-cache is set.")

    data_cfg = build_cfg.get("data")
    retrieval_cfg = build_cfg.get("retrieval")
    if not isinstance(data_cfg, dict) or not isinstance(retrieval_cfg, dict):
        raise ValueError("Resolved experiment build config must contain build.data and build.retrieval dictionaries.")

    fp = _premmr_config_fingerprint(build_cfg)
    cache_dir = cache_root / fp
    cache_path = cache_dir / f"{split_name}.pkl"

    if rebuild:
        if cache_path.exists():
            print(f"Removing existing PreMMR cache: {cache_path}", flush=True)
            cache_path.unlink()
        if cache_dir.exists():
            for chunk_path in cache_dir.glob(f"{split_name}_gpu*.pkl"):
                print(f"Removing stale PreMMR worker cache: {chunk_path}", flush=True)
                chunk_path.unlink()

    run_summary = {
        "embedder_model": retrieval_cfg["embedder_model"],
        "max_length": int(retrieval_cfg.get("max_length", 256)),
        "batch_size": int(retrieval_cfg.get("batch_size", 64)),
        "precision": retrieval_cfg.get("precision", "fp32"),
        "prefetch_size": int(retrieval_cfg.get("prefetch_size", 1)),
        "num_gpus": int(retrieval_cfg.get("num_gpus", 1)),
        "device": retrieval_cfg.get("device", "cuda"),
    }

    print(f"PreMMR fingerprint: {fp}", flush=True)
    print(f"Ensuring PreMMR cache: {cache_path}", flush=True)
    return _compute_pre_mmr_split(
        split_name=split_name,
        data_cfg=data_cfg,
        retrieval_cfg=retrieval_cfg,
        run_summary=run_summary,
        cache_dir=cache_dir,
        num_gpus=run_summary["num_gpus"],
    )


def main() -> None:
    args = parse_args()
    if args.config_overrides and not args.experiment:
        raise ValueError("--config-overrides requires --experiment.")

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_cfg: dict[str, Any] = {}
    if args.experiment:
        build_cfg = _load_experiment_build_cfg(args.experiment, args.config_overrides)
        print(f"Loaded build config from experiment={args.experiment}", flush=True)

    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    prompt_cfg_from_config = dict(build_cfg.get("prompt", {}) or {})

    model_name_or_path = (
        args.model_name_or_path
        or str(prompt_cfg_from_config.get("model_name_or_path", "") or "").strip()
    )
    if not model_name_or_path:
        raise ValueError(
            "--model-name-or-path is required unless --experiment provides build.prompt.model_name_or_path"
        )

    top_k = int(_pick(args.top_k, retrieval_cfg, "top_k", 16))
    alpha_dense = float(_pick(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(_pick(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(_pick(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))
    cpu_workers = int(_pick(args.cpu_workers, retrieval_cfg, "cpu_workers", 1))
    prompt_cfg = {
        "auto_length": bool(prompt_cfg_from_config.get("auto_length", True)),
        "max_length": int(_pick(args.prompt_max_length, prompt_cfg_from_config, "max_length", 1024)),
        "output_mode": str(_pick(args.prompt_output_mode, prompt_cfg_from_config, "output_mode", "label_only")),
        "label_format": str(_pick(args.prompt_label_format, prompt_cfg_from_config, "label_format", "letter")),
        "system_prompt": prompt_cfg_from_config.get("system_prompt") or None,
    }
    strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
    reuse_chunk_embeddings = _can_reuse_chunk_embeddings(strategy, retrieval_cfg)

    print(
        "Build settings: "
        f"top_k={top_k}, alpha_dense={alpha_dense}, alpha_lexical={alpha_lexical}, "
        f"alpha_bm25={alpha_bm25}, cpu_workers={cpu_workers}",
        flush=True,
    )
    print(
        f"Prompt settings: model={model_name_or_path}, max_length={prompt_cfg['max_length']}, "
        f"output_mode={prompt_cfg['output_mode']}, label_format={prompt_cfg['label_format']}",
        flush=True,
    )
    print(
        f"Chunking strategy: {type(strategy).__name__}, "
        f"reuse_pre_mmr_embeddings={reuse_chunk_embeddings}",
        flush=True,
    )

    if args.rebuild_premmr_cache or args.premmr_cache is None:
        premmr_cache_path = _ensure_premmr_cache(
            build_cfg,
            split_name=args.split_name,
            cache_root=Path(args.premmr_cache_root),
            rebuild=args.rebuild_premmr_cache,
        )
    else:
        premmr_cache_path = Path(args.premmr_cache)

    print(f"Loading PreMMR cache: {premmr_cache_path}", flush=True)
    pre_samples = _load_pickle(premmr_cache_path)
    print(f"Loaded {len(pre_samples)} samples", flush=True)

    tokenizer = _load_prompt_tokenizer(model_name_or_path)

    for lam in lambda_grid:
        output_path = output_dir / f"lambda_{lam:.1f}_{args.split_name}.jsonl"
        print(f"Generating prompts for λ={lam:.1f} → {output_path}", flush=True)
        _mmr_phase_from_premmr(
            pre_samples=pre_samples,
            mmr_lambda=lam,
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            strategy=strategy,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            output_path=output_path,
            cpu_workers=cpu_workers,
            reuse_chunk_embeddings=reuse_chunk_embeddings,
        )
        print(f"  Done: {output_path}", flush=True)

    print(f"All {len(lambda_grid)} λ values processed. Output dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
