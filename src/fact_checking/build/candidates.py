from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from fact_checking.build.chunking import ChunkingStrategy, SentenceChunking, build_chunking_strategy
from fact_checking.config import load_yaml
from fact_checking.data.io import iter_sentences, load_split
from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.retrieval.text_utils import bm25_like_score, lexical_overlap_f1
from fact_checking.utils.logging import init_logger


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    split_paths: dict[str, Path]


def canonicalize_sentence(text: str) -> str:
    return " ".join(text.lower().strip().split())


def minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def build_candidates_for_sample(
    sample,
    embedder: TextEmbedder,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
    chunking_strategy: ChunkingStrategy | None = None,
) -> dict[str, Any]:
    sentences = list(iter_sentences(sample))
    if not sentences:
        return {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }

    sent_texts = [s.text for s in sentences]
    sent_emb = embedder.encode(sent_texts, is_query=False)
    claim_emb = embedder.encode([sample.claim], is_query=True)[0]
    dense_scores = sent_emb @ claim_emb

    lexical_scores = np.asarray([lexical_overlap_f1(sample.claim, s) for s in sent_texts], dtype=np.float32)
    bm25_scores = np.asarray([bm25_like_score(sample.claim, s) for s in sent_texts], dtype=np.float32)

    dense_scaled = minmax_scale(dense_scores)
    lexical_scaled = minmax_scale(lexical_scores)
    bm25_scaled = minmax_scale(bm25_scores)

    hybrid_scores = alpha_dense * dense_scaled + alpha_lexical * lexical_scaled + alpha_bm25 * bm25_scaled

    keep_indices = maximal_marginal_relevance(
        query_scores=hybrid_scores,
        sentence_vectors=sent_emb,
        top_k=min(top_k, len(sentences)),
        lambda_weight=mmr_lambda,
    )

    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in keep_indices:
        sent = sentences[idx]
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        evidence_text = strategy.chunk(content, sent.sent_idx) if content else sent.text
        candidate = {
            "report_id": sent.report_id,
            "sent_idx": sent.sent_idx,
            "text": evidence_text,
            "dense_score": float(dense_scores[idx]),
            "lexical_score": float(lexical_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
            "source_report": {
                "report_id": sent.report_id,
                "link": sent.link,
                "domain": sent.domain,
            },
        }
        dedup_key = canonicalize_sentence(evidence_text)
        old_candidate = deduped_by_text.get(dedup_key)
        if old_candidate is None or candidate["hybrid_score"] > old_candidate["hybrid_score"]:
            deduped_by_text[dedup_key] = candidate

    candidates = list(deduped_by_text.values())
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    candidates = candidates[:top_k]
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": candidates,
    }


def run_build(cfg: dict[str, Any], *, output_dir: str | Path | None = None, split: str | None = None) -> BuildResult:
    logger = init_logger(__name__)
    data_cfg = cfg["data"]
    retrieval_cfg = cfg["retrieval"]
    target_dir = Path(output_dir or cfg.get("output_dir", "outputs/cache/build/manual"))
    target_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "output_dir": str(target_dir),
        "embedder_model": retrieval_cfg["embedder_model"],
        "device": retrieval_cfg.get("device", "cuda"),
        "cuda_available": torch.cuda.is_available(),
        "top_k": int(retrieval_cfg.get("top_k", 24)),
        "batch_size": int(retrieval_cfg.get("batch_size", 64)),
        "max_length": int(retrieval_cfg.get("max_length", 256)),
        "alpha_dense": float(retrieval_cfg.get("alpha_dense", 0.70)),
        "alpha_lexical": float(retrieval_cfg.get("alpha_lexical", 0.20)),
        "alpha_bm25": float(retrieval_cfg.get("alpha_bm25", 0.10)),
        "mmr_lambda": float(retrieval_cfg.get("mmr_lambda", 0.70)),
    }
    logger.info("Build run summary: %s", run_summary)

    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=run_summary["embedder_model"],
            device=run_summary["device"],
            max_length=run_summary["max_length"],
            batch_size=run_summary["batch_size"],
        )
    )

    chunking_cfg = retrieval_cfg.get("chunking")
    chunking_strategy = build_chunking_strategy(chunking_cfg)
    logger.info("Chunking strategy: %s", type(chunking_strategy).__name__)

    split_names = [split] if split else ["train", "val", "test"]
    split_paths: dict[str, Path] = {}
    for split_name in split_names:
        input_path = data_cfg[f"{split_name}_path"]
        samples = load_split(input_path)
        output_path = target_dir / f"build_{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as writer:
            for sample in tqdm(samples, desc=f"Build [{split_name}]"):
                row = build_candidates_for_sample(
                    sample=sample,
                    embedder=embedder,
                    top_k=run_summary["top_k"],
                    alpha_dense=run_summary["alpha_dense"],
                    alpha_lexical=run_summary["alpha_lexical"],
                    alpha_bm25=run_summary["alpha_bm25"],
                    mmr_lambda=run_summary["mmr_lambda"],
                    chunking_strategy=chunking_strategy,
                )
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_paths[split_name] = output_path
        logger.info("Wrote build file: %s", output_path)

    return BuildResult(output_dir=target_dir, split_paths=split_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate evidence files.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    build_cfg = cfg.get("build", cfg)
    run_build(build_cfg, output_dir=args.output_dir, split=args.split)


if __name__ == "__main__":
    main()
