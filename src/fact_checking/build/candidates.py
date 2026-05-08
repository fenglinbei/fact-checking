from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
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
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)
from fact_checking.utils.logging import init_logger
from fact_checking.utils.text import robust_sentence_split


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

    q_ctr, q_len = content_tokens_counter(sample.claim)
    lexical_scores = np.empty(len(sent_texts), dtype=np.float32)
    bm25_scores = np.empty(len(sent_texts), dtype=np.float32)
    for i, s in enumerate(sent_texts):
        s_ctr, s_len = content_tokens_counter(s)
        lexical_scores[i] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[i] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

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
    content_splits: dict[str, list[str]] = {}
    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in keep_indices:
        sent = sentences[idx]
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if content:
            if content not in content_splits:
                content_splits[content] = robust_sentence_split(content)
            evidence_text = strategy.chunk_from_presplit(content_splits[content], sent.sent_idx)
        else:
            evidence_text = sent.text
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


def _process_sample_post_embed(
    sample,
    sents: list,
    sent_texts: list[str],
    sent_emb: np.ndarray,
    claim_emb: np.ndarray,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
    strategy: ChunkingStrategy,
) -> dict[str, Any]:
    """CPU-only portion: scoring, MMR, chunking, dedup for a single sample."""
    dense_scores = sent_emb @ claim_emb

    q_ctr, q_len = content_tokens_counter(sample.claim)
    n = len(sent_texts)
    lexical_scores = np.empty(n, dtype=np.float32)
    bm25_scores = np.empty(n, dtype=np.float32)
    for j, s in enumerate(sent_texts):
        s_ctr, s_len = content_tokens_counter(s)
        lexical_scores[j] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[j] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

    dense_scaled = minmax_scale(dense_scores)
    lexical_scaled = minmax_scale(lexical_scores)
    bm25_scaled = minmax_scale(bm25_scores)
    hybrid_scores = alpha_dense * dense_scaled + alpha_lexical * lexical_scaled + alpha_bm25 * bm25_scaled

    keep_indices = maximal_marginal_relevance(
        query_scores=hybrid_scores,
        sentence_vectors=sent_emb,
        top_k=min(top_k, len(sents)),
        lambda_weight=mmr_lambda,
    )

    content_splits: dict[str, list[str]] = {}
    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in keep_indices:
        sent = sents[idx]
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if content:
            if content not in content_splits:
                content_splits[content] = robust_sentence_split(content)
            evidence_text = strategy.chunk_from_presplit(content_splits[content], sent.sent_idx)
        else:
            evidence_text = sent.text
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


def _build_candidates_batch(
    samples: list,
    embedder: TextEmbedder,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
    chunking_strategy: ChunkingStrategy | None = None,
    cpu_workers: int = 1,
) -> list[dict[str, Any]]:
    """Build candidates for multiple samples with batched embeddings."""
    all_sent_texts: list[str] = []
    sample_boundaries: list[tuple[int, int]] = []
    claims: list[str] = []
    per_sample: list[tuple[list, list[str]]] = []  # (sentences, sent_texts) per sample

    for sample in samples:
        sents = list(iter_sentences(sample))
        if not sents:
            sample_boundaries.append((0, 0))
            claims.append(sample.claim)
            per_sample.append(([], []))
            continue
        start = len(all_sent_texts)
        sent_texts = [s.text for s in sents]
        all_sent_texts.extend(sent_texts)
        end = len(all_sent_texts)
        sample_boundaries.append((start, end))
        claims.append(sample.claim)
        per_sample.append((sents, sent_texts))

    if all_sent_texts:
        all_sent_emb = embedder.encode(all_sent_texts, is_query=False)
    else:
        all_sent_emb = np.zeros((0,), dtype=np.float32)
    all_claim_emb = embedder.encode(claims, is_query=True)

    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    n_samples = len(samples)

    def _task(i: int) -> dict[str, Any]:
        sample = samples[i]
        sents, sent_texts = per_sample[i]
        if not sents:
            return {
                "event_id": sample.event_id,
                "claim": sample.claim,
                "label": sample.label,
                "explain": sample.explain,
                "candidates": [],
            }
        start, end = sample_boundaries[i]
        return _process_sample_post_embed(
            sample=sample,
            sents=sents,
            sent_texts=sent_texts,
            sent_emb=all_sent_emb[start:end],
            claim_emb=all_claim_emb[i],
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            mmr_lambda=mmr_lambda,
            strategy=strategy,
        )

    if cpu_workers > 1:
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            results = list(pool.map(_task, range(n_samples)))
    else:
        results = [_task(i) for i in range(n_samples)]

    return results


def _build_worker(
    gpu_id: int,
    run_summary: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    """Load split data, process the slice for this GPU, and write results (runs in a child process)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    samples = load_split(data_cfg[f"{split_name}_path"])
    num_gpus = run_summary["num_gpus"]
    chunk_size = (len(samples) + num_gpus - 1) // num_gpus
    samples_chunk = samples[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
    if not samples_chunk:
        output_path.write_text("", encoding="utf-8")
        return

    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=run_summary["embedder_model"],
            device="cuda",
            max_length=run_summary["max_length"],
            batch_size=run_summary["batch_size"],
            precision=run_summary["precision"],
        )
    )

    chunking_cfg = retrieval_cfg.get("chunking")
    chunking_strategy = build_chunking_strategy(chunking_cfg)
    prefetch_size = run_summary["prefetch_size"]
    cpu_workers = run_summary["cpu_workers"]

    with output_path.open("w", encoding="utf-8") as writer:
        if prefetch_size > 1:
            for start in range(0, len(samples_chunk), prefetch_size):
                batch = samples_chunk[start : start + prefetch_size]
                rows = _build_candidates_batch(
                    samples=batch,
                    embedder=embedder,
                    top_k=run_summary["top_k"],
                    alpha_dense=run_summary["alpha_dense"],
                    alpha_lexical=run_summary["alpha_lexical"],
                    alpha_bm25=run_summary["alpha_bm25"],
                    mmr_lambda=run_summary["mmr_lambda"],
                    chunking_strategy=chunking_strategy,
                    cpu_workers=cpu_workers,
                )
                for row in rows:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            for sample in samples_chunk:
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


def run_build(cfg: dict[str, Any], *, output_dir: str | Path | None = None, split: str | None = None) -> BuildResult:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger = init_logger(__name__)
    data_cfg = cfg["data"]
    retrieval_cfg = cfg["retrieval"]
    target_dir = Path(output_dir or cfg.get("output_dir", "outputs/cache/build/manual"))
    target_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = int(retrieval_cfg.get("num_gpus", 1))
    run_summary = {
        "output_dir": str(target_dir),
        "embedder_model": retrieval_cfg["embedder_model"],
        "device": retrieval_cfg.get("device", "cuda"),
        "top_k": int(retrieval_cfg.get("top_k", 24)),
        "batch_size": int(retrieval_cfg.get("batch_size", 64)),
        "max_length": int(retrieval_cfg.get("max_length", 256)),
        "precision": retrieval_cfg.get("precision", "fp32"),
        "alpha_dense": float(retrieval_cfg.get("alpha_dense", 0.70)),
        "alpha_lexical": float(retrieval_cfg.get("alpha_lexical", 0.20)),
        "alpha_bm25": float(retrieval_cfg.get("alpha_bm25", 0.10)),
        "mmr_lambda": float(retrieval_cfg.get("mmr_lambda", 0.70)),
        "prefetch_size": int(retrieval_cfg.get("prefetch_size", 1)),
        "cpu_workers": int(retrieval_cfg.get("cpu_workers", 1)),
        "num_gpus": num_gpus,
    }

    split_names = [split] if split else ["train", "val", "test"]
    split_paths: dict[str, Path] = {}

    if num_gpus > 1:
        # Multi-GPU path: fork workers, no CUDA init in parent.
        # Each worker loads the split independently and processes its slice,
        # avoiding pickle serialization overhead for large sample lists.
        run_summary["cuda_available"] = True  # assumed in multi-GPU mode
        logger.info("Build run summary: %s", run_summary)
        ctx = multiprocessing.get_context("fork")
        for split_name in split_names:
            output_path = target_dir / f"build_{split_name}.jsonl"
            chunk_paths: list[Path] = []
            workers: list[multiprocessing.Process] = []

            for gpu_id in range(num_gpus):
                chunk_path = target_dir / f"build_{split_name}_gpu{gpu_id}.jsonl"
                chunk_paths.append(chunk_path)
                p = ctx.Process(
                    target=_build_worker,
                    args=(gpu_id, run_summary, retrieval_cfg, data_cfg, split_name, chunk_path),
                )
                p.start()
                workers.append(p)

            for p in workers:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"Worker GPU {gpu_id} failed with exit code {p.exitcode}")

            # Concatenate chunk files in order
            with output_path.open("w", encoding="utf-8") as writer:
                for chunk_path in chunk_paths:
                    with chunk_path.open("r", encoding="utf-8") as reader:
                        writer.write(reader.read())
                    chunk_path.unlink()  # clean up temp file

            split_paths[split_name] = output_path
            logger.info("Wrote build file: %s", output_path)
    else:
        # Single-GPU path (original behaviour)
        run_summary["cuda_available"] = torch.cuda.is_available()
        logger.info("Build run summary: %s", run_summary)

        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=run_summary["embedder_model"],
                device=run_summary["device"],
                max_length=run_summary["max_length"],
                batch_size=run_summary["batch_size"],
                precision=run_summary["precision"],
            )
        )

        chunking_cfg = retrieval_cfg.get("chunking")
        chunking_strategy = build_chunking_strategy(chunking_cfg)
        logger.info("Chunking strategy: %s", type(chunking_strategy).__name__)
        prefetch_size = run_summary["prefetch_size"]

        for split_name in split_names:
            input_path = data_cfg[f"{split_name}_path"]
            samples = load_split(input_path)
            output_path = target_dir / f"build_{split_name}.jsonl"
            with output_path.open("w", encoding="utf-8") as writer:
                if prefetch_size > 1:
                    for start in tqdm(range(0, len(samples), prefetch_size), desc=f"Build [{split_name}]"):
                        chunk = samples[start : start + prefetch_size]
                        rows = _build_candidates_batch(
                            samples=chunk,
                            embedder=embedder,
                            top_k=run_summary["top_k"],
                            alpha_dense=run_summary["alpha_dense"],
                            alpha_lexical=run_summary["alpha_lexical"],
                            alpha_bm25=run_summary["alpha_bm25"],
                            mmr_lambda=run_summary["mmr_lambda"],
                            chunking_strategy=chunking_strategy,
                            cpu_workers=run_summary["cpu_workers"],
                        )
                        for row in rows:
                            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
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
