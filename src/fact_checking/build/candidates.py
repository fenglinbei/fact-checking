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

from fact_checking.build.cache import (
    BUILD_LOGIC_VERSION,
    CHUNK_MMR_CACHE_VERSION,
    can_reuse_chunk_embeddings,
    chunk_embeddings_by_content,
    chunk_mmr_config_fingerprint,
    chunk_mmr_worker,
    compute_chunk_mmr_split,
    compute_pre_mmr_batch,
    compute_pre_mmr_split,
    dict_to_sentence,
    load_pickle,
    normalize_model_name,
    premmr_config_fingerprint,
    premmr_worker,
    save_pickle_atomic,
    sentence_reader_config,
    sentence_reader_fingerprint_payload,
    sentence_to_dict,
    visible_gpu_for_worker,
)
from fact_checking.build.chunking import ChunkingStrategy, SentenceChunking, build_chunking_strategy
from fact_checking.build.prompts import (
    OPTIONAL_BUILD_ROW_KEYS,
    auto_truncate_evidence,
    build_chat_prompt,
    build_system_message,
    build_target,
    build_training_row,
    build_user_content,
    copy_optional_build_row_metadata,
    count_target_tokens,
    count_tokens,
    decode_token_prefix,
    format_evidence_block,
    label_definitions_text,
    load_prompt_tokenizer,
    render_prompt,
    truncate_single_evidence_to_budget,
)
from fact_checking.build.stats import generate_prompt_stats, rows_to_prepared_samples
from fact_checking.config import load_yaml
from fact_checking.data.constants import LABEL_DEFINITIONS, LABEL_LETTERS, LABEL2ID
from fact_checking.data.io import iter_sentences, load_jsonl, load_split
from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)
from fact_checking.utils.logging import init_logger
from fact_checking.utils.text import robust_sentence_split
from sft.data.labels import normalize_gold_label
from sft.data.types import PreparedSample
from sft.prompting.stats import (
    build_prompt_snapshots,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prebuilt_prompts,
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    split_paths: dict[str, Path]


@dataclass
class PreMMRSample:
    """Cached intermediate: embeddings + metadata for one sample.

    Everything needed to resume from the MMR step without re-running the GPU embedder.
    """
    event_id: str
    claim: str
    label: str
    explain: str
    sentences: list[dict]   # serialized SentenceRecord dicts
    sent_emb: np.ndarray    # [N, D] float32
    claim_emb: np.ndarray   # [D] float32


@dataclass
class ChunkMMRSample:
    """Cached chunk-level retrieval state used by the MMR phase."""

    event_id: str
    claim: str
    label: str
    explain: str
    candidates: list[dict[str, Any]]
    chunk_emb: np.ndarray   # [N_chunks, D] float32
    claim_emb: np.ndarray   # [D] float32


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
    reuse_chunk_embeddings: bool = False,
    sentence_source: str = "content",
    sentence_min_char_len: int = 10,
) -> dict[str, Any]:
    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    pre_sample = compute_pre_mmr_batch(
        [sample],
        embedder,
        sentence_source=sentence_source,
        sentence_min_char_len=sentence_min_char_len,
    )[0]
    chunk_sample = _compute_chunk_mmr_batch(
        [pre_sample],
        embedder=embedder,
        strategy=strategy,
        reuse_chunk_embeddings=reuse_chunk_embeddings,
    )[0]
    return _select_candidates_from_chunk_sample(
        chunk_sample,
        top_k=top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        mmr_lambda=mmr_lambda,
    )


def _source_report(sent) -> dict[str, Any]:
    return {
        "report_id": sent.report_id,
        "link": sent.link,
        "domain": sent.domain,
    }


def _raw_sentence_candidate_metadata(sent) -> dict[str, Any]:
    raw = sent.raw if isinstance(sent.raw, dict) else {}
    metadata: dict[str, Any] = {}
    if "raw_is_evidence" in raw:
        metadata["raw_is_evidence"] = bool(raw.get("raw_is_evidence"))
    if "raw_evidence_label" in raw:
        metadata["raw_evidence_label"] = raw.get("raw_evidence_label")
    if "raw_report_order" in raw:
        metadata["raw_report_order"] = int(raw.get("raw_report_order", 0))
    if "raw_sent_order" in raw:
        metadata["raw_sent_order"] = int(raw.get("raw_sent_order", sent.sent_idx))
    if "raw_sentence_source" in raw:
        metadata["raw_sentence_source"] = str(raw.get("raw_sentence_source") or "")
    return metadata


def _build_chunk_candidate_rows(
    pre: PreMMRSample,
    strategy: ChunkingStrategy,
    reuse_chunk_embeddings: bool = False,
) -> list[dict[str, Any]]:
    sents = [dict_to_sentence(d, event_id_fallback=pre.event_id) for d in pre.sentences]
    if not sents:
        return []

    content_embeddings = chunk_embeddings_by_content(sents, pre.sent_emb) if reuse_chunk_embeddings else {}
    grouped: dict[tuple[str, str], list] = {}
    no_content_rows: list = []
    for sent in sents:
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if content:
            grouped.setdefault((str(sent.report_id), str(content)), []).append(sent)
        else:
            no_content_rows.append(sent)

    candidates: list[dict[str, Any]] = []
    for (_report_id, content), report_sents in grouped.items():
        report_sents.sort(key=lambda s: int(s.sent_idx))
        split_sents = robust_sentence_split(content)
        if not split_sents:
            continue
        embeddings_by_sent_idx = content_embeddings.get(content)
        if embeddings_by_sent_idx is None:
            chunk_records = strategy.chunks_from_presplit(split_sents)
        else:
            chunk_records = strategy.chunks_from_presplit_with_embeddings(split_sents, embeddings_by_sent_idx)

        available = {int(sent.sent_idx): sent for sent in report_sents}
        for chunk in chunk_records:
            anchors = [available[idx] for idx in chunk.sent_indices if idx in available]
            if not anchors:
                continue
            anchor = anchors[0]
            text = str(chunk.text).strip()
            if not text:
                continue
            candidates.append(
                {
                    "report_id": anchor.report_id,
                    "sent_idx": anchor.sent_idx,
                    "chunk_sent_indices": list(chunk.sent_indices),
                    "text": text,
                    "source_report": _source_report(anchor),
                    **_raw_sentence_candidate_metadata(anchor),
                }
            )

    for sent in no_content_rows:
        text = str(sent.text).strip()
        if not text:
            continue
        candidates.append(
            {
                "report_id": sent.report_id,
                "sent_idx": sent.sent_idx,
                "chunk_sent_indices": [int(sent.sent_idx)],
                "text": text,
                "source_report": _source_report(sent),
                **_raw_sentence_candidate_metadata(sent),
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = canonicalize_sentence(str(candidate.get("text", "")))
        if key and key not in deduped:
            deduped[key] = candidate
    return list(deduped.values())


def _compute_chunk_mmr_batch(
    pre_samples: list[PreMMRSample],
    embedder: TextEmbedder,
    strategy: ChunkingStrategy,
    reuse_chunk_embeddings: bool = False,
) -> list[ChunkMMRSample]:
    per_sample_candidates: list[list[dict[str, Any]]] = []
    all_chunk_texts: list[str] = []
    boundaries: list[tuple[int, int]] = []

    for pre in pre_samples:
        candidates = _build_chunk_candidate_rows(pre, strategy, reuse_chunk_embeddings)
        start = len(all_chunk_texts)
        all_chunk_texts.extend(str(candidate["text"]) for candidate in candidates)
        end = len(all_chunk_texts)
        boundaries.append((start, end))
        per_sample_candidates.append(candidates)

    if all_chunk_texts:
        all_chunk_emb = embedder.encode(all_chunk_texts, is_query=False)
    else:
        all_chunk_emb = np.zeros((0, 0), dtype=np.float32)

    results: list[ChunkMMRSample] = []
    for idx, pre in enumerate(pre_samples):
        start, end = boundaries[idx]
        if end > start:
            chunk_emb = all_chunk_emb[start:end].copy()
        else:
            dim = int(np.asarray(pre.claim_emb).reshape(-1).shape[0])
            chunk_emb = np.zeros((0, dim), dtype=np.float32)
        results.append(
            ChunkMMRSample(
                event_id=pre.event_id,
                claim=pre.claim,
                label=pre.label,
                explain=pre.explain,
                candidates=per_sample_candidates[idx],
                chunk_emb=chunk_emb,
                claim_emb=pre.claim_emb.copy(),
            )
        )
    return results


def compute_hybrid_scores(
    sample: ChunkMMRSample,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> dict[str, Any]:
    """Compute the hybrid relevance scores used by the MMR phase.

    Returns a dict with ``n``, ``chunk_emb``, ``hybrid_scores``, and the three
    raw component arrays so downstream consumers (e.g. RL-MMR experiments)
    can reuse the exact same scoring recipe.
    """
    n = min(len(sample.candidates), int(sample.chunk_emb.shape[0]))
    if n == 0:
        return {
            "n": 0,
            "chunk_emb": np.zeros((0, 0), dtype=np.float32),
            "dense_scores": np.zeros((0,), dtype=np.float32),
            "lexical_scores": np.zeros((0,), dtype=np.float32),
            "bm25_scores": np.zeros((0,), dtype=np.float32),
            "hybrid_scores": np.zeros((0,), dtype=np.float32),
        }

    chunk_emb = sample.chunk_emb[:n]
    claim_emb = np.asarray(sample.claim_emb, dtype=np.float32).reshape(-1)
    dense_scores = chunk_emb @ claim_emb

    q_ctr, q_len = content_tokens_counter(sample.claim)
    lexical_scores = np.empty(n, dtype=np.float32)
    bm25_scores = np.empty(n, dtype=np.float32)
    for j, candidate in enumerate(sample.candidates[:n]):
        s_ctr, s_len = content_tokens_counter(str(candidate.get("text", "")))
        lexical_scores[j] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[j] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

    dense_scaled = minmax_scale(dense_scores)
    lexical_scaled = minmax_scale(lexical_scores)
    bm25_scaled = minmax_scale(bm25_scores)
    hybrid_scores = alpha_dense * dense_scaled + alpha_lexical * lexical_scaled + alpha_bm25 * bm25_scaled

    return {
        "n": int(n),
        "chunk_emb": chunk_emb,
        "dense_scores": dense_scores.astype(np.float32, copy=False),
        "lexical_scores": lexical_scores,
        "bm25_scores": bm25_scores,
        "hybrid_scores": hybrid_scores.astype(np.float32, copy=False),
    }


def _select_candidates_from_chunk_sample(
    sample: ChunkMMRSample,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
) -> dict[str, Any]:
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    if n == 0:
        return {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }

    chunk_emb = scored["chunk_emb"]
    dense_scores = scored["dense_scores"]
    lexical_scores = scored["lexical_scores"]
    bm25_scores = scored["bm25_scores"]
    hybrid_scores = scored["hybrid_scores"]

    keep_indices = maximal_marginal_relevance(
        query_scores=hybrid_scores,
        sentence_vectors=chunk_emb,
        top_k=min(top_k, n),
        lambda_weight=mmr_lambda,
    )

    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in keep_indices:
        candidate = dict(sample.candidates[int(idx)])
        candidate.update({
            "dense_score": float(dense_scores[idx]),
            "lexical_score": float(lexical_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
        })
        dedup_key = canonicalize_sentence(str(candidate.get("text", "")))
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


def _rank_mmr_candidates_from_chunk_sample(
    sample: ChunkMMRSample,
    candidate_pool_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return unique candidates in MMR selection order, enriched with score fields."""
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    if n == 0:
        return [], 0

    chunk_emb = scored["chunk_emb"]
    dense_scores = scored["dense_scores"]
    lexical_scores = scored["lexical_scores"]
    bm25_scores = scored["bm25_scores"]
    hybrid_scores = scored["hybrid_scores"]

    keep_indices = maximal_marginal_relevance(
        query_scores=hybrid_scores,
        sentence_vectors=chunk_emb,
        top_k=min(max(1, int(candidate_pool_k)), n),
        lambda_weight=mmr_lambda,
    )

    ranked: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for mmr_rank, idx_raw in enumerate(keep_indices, start=1):
        idx = int(idx_raw)
        candidate = dict(sample.candidates[idx])
        dedup_key = canonicalize_sentence(str(candidate.get("text", "")))
        if dedup_key and dedup_key in seen_texts:
            continue
        if dedup_key:
            seen_texts.add(dedup_key)
        candidate.update({
            "dense_score": float(dense_scores[idx]),
            "lexical_score": float(lexical_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
            "source_candidate_index": idx,
            "mmr_rank": mmr_rank,
        })
        ranked.append(candidate)
    return ranked, n


def _resolve_prompt_budget_reference_path(reference: str | Path, split_name: str) -> Path:
    raw = str(reference or "").strip()
    if not raw:
        raise ValueError(
            "build.retrieval.prompt_budget.reference_build_dir is required for "
            "selection_method=mmr_prompt_budget."
        )
    if "{split}" in raw:
        return Path(raw.format(split=split_name))

    path = Path(raw)
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / f"build_{split_name}.jsonl"
        if direct.exists():
            return direct
        nested = path / "build" / f"build_{split_name}.jsonl"
        if nested.exists():
            return nested
    if path.suffix == ".jsonl":
        return path
    return path / f"build_{split_name}.jsonl"


def _load_prompt_budget_targets(
    prompt_budget_cfg: dict[str, Any],
    split_name: str,
) -> tuple[dict[str, int], Path]:
    reference = (
        prompt_budget_cfg.get("reference_build_dir")
        or prompt_budget_cfg.get("reference_path")
        or prompt_budget_cfg.get("reference")
        or ""
    )
    reference_path = _resolve_prompt_budget_reference_path(reference, split_name)
    if not reference_path.exists():
        raise FileNotFoundError(f"Prompt-budget reference build file not found: {reference_path}")

    id_field = str(prompt_budget_cfg.get("id_field", "event_id"))
    target_field = str(prompt_budget_cfg.get("target_field", "prompt_token_count"))
    targets: dict[str, int] = {}
    for row in load_jsonl(reference_path):
        event_id = str(row.get(id_field) or "").strip()
        if not event_id:
            continue
        raw_target = row.get(target_field)
        if raw_target is None:
            raise ValueError(f"Reference row for event_id={event_id!r} is missing {target_field!r}.")
        targets[event_id] = int(raw_target)
    if not targets:
        raise ValueError(f"No prompt-budget targets loaded from {reference_path}.")
    return targets, reference_path


def _trial_prompt_budget_row(
    sample: ChunkMMRSample,
    candidates: list[dict[str, Any]],
    tokenizer,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    return build_training_row(
        {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": candidates,
            "selection_method": "mmr_prompt_budget",
        },
        tokenizer,
        prompt_cfg,
    )


def _select_candidates_prompt_budget_mmr(
    sample: ChunkMMRSample,
    prompt_budget_targets: dict[str, int],
    prompt_budget_cfg: dict[str, Any],
    tokenizer,
    prompt_cfg: dict[str, Any],
    reference_path: Path,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
) -> dict[str, Any]:
    candidate_pool_k = int(prompt_budget_cfg.get("candidate_pool_k", prompt_budget_cfg.get("pool_k", max(top_k, 32))))
    min_k = max(0, int(prompt_budget_cfg.get("min_k", top_k)))
    max_k = max(min_k, int(prompt_budget_cfg.get("max_k", candidate_pool_k)))
    overshoot_tolerance = max(0, int(prompt_budget_cfg.get("overshoot_tolerance_tokens", 32)))
    missing_policy = str(prompt_budget_cfg.get("missing_reference", "error")).strip().lower()

    ranked, raw_candidate_count = _rank_mmr_candidates_from_chunk_sample(
        sample,
        candidate_pool_k=candidate_pool_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        mmr_lambda=mmr_lambda,
    )

    event_id = str(sample.event_id)
    target_tokens = prompt_budget_targets.get(event_id)
    if target_tokens is None:
        if missing_policy == "prompt_max_length":
            target_tokens = int(prompt_cfg.get("max_length", 2048))
        elif missing_policy == "top_k":
            selected = ranked[: min(max(1, top_k), len(ranked))]
            selected_tokens = int(
                _trial_prompt_budget_row(sample, selected, tokenizer, prompt_cfg).get("prompt_token_count", 0)
            ) if selected else 0
            return {
                "event_id": sample.event_id,
                "claim": sample.claim,
                "label": sample.label,
                "explain": sample.explain,
                "candidates": selected,
                "selection_method": "mmr_prompt_budget",
                "prompt_budget_reference_path": str(reference_path),
                "prompt_budget_target_tokens": 0,
                "prompt_budget_selected_tokens": selected_tokens,
                "prompt_budget_delta_tokens": selected_tokens,
                "prompt_budget_min_k": min_k,
                "prompt_budget_max_k": max_k,
                "prompt_budget_candidate_pool_k": candidate_pool_k,
                "prompt_budget_ranked_count": len(ranked),
                "prompt_budget_overshoot_tolerance_tokens": overshoot_tolerance,
                "prompt_budget_missing_policy": missing_policy,
                "prompt_budget_raw_candidate_count": raw_candidate_count,
            }
        else:
            raise KeyError(
                f"Prompt-budget reference {reference_path} has no target for event_id={event_id!r}. "
                "Set build.retrieval.prompt_budget.missing_reference=top_k or prompt_max_length "
                "to allow fallback."
            )

    target_tokens = int(target_tokens)
    max_eval_k = min(max_k, len(ranked))
    effective_min_k = min(min_k, max_eval_k) if max_eval_k > 0 else 0

    trials: list[dict[str, Any]] = []
    if max_eval_k == 0:
        selected: list[dict[str, Any]] = []
        selected_tokens = 0
    else:
        start_k = 0 if effective_min_k == 0 else 1
        for k in range(start_k, max_eval_k + 1):
            trial_candidates = ranked[:k]
            trial_row = _trial_prompt_budget_row(sample, trial_candidates, tokenizer, prompt_cfg)
            trials.append({
                "k": k,
                "prompt_token_count": int(trial_row.get("prompt_token_count", 0)),
                "was_truncated": bool(trial_row.get("was_truncated", False)),
            })

        eligible = [
            trial
            for trial in trials
            if int(trial["k"]) >= effective_min_k
            and int(trial["prompt_token_count"]) <= target_tokens + overshoot_tolerance
        ]
        if not eligible:
            eligible = [trial for trial in trials if int(trial["k"]) >= effective_min_k] or trials

        best = min(
            eligible,
            key=lambda trial: (
                abs(int(trial["prompt_token_count"]) - target_tokens),
                int(trial["prompt_token_count"]) > target_tokens,
                -int(trial["k"]),
            ),
        )
        best_k = int(best["k"])
        selected = ranked[:best_k]
        selected_tokens = int(best["prompt_token_count"])

    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": selected,
        "selection_method": "mmr_prompt_budget",
        "prompt_budget_reference_path": str(reference_path),
        "prompt_budget_target_tokens": target_tokens,
        "prompt_budget_selected_tokens": selected_tokens,
        "prompt_budget_delta_tokens": selected_tokens - target_tokens,
        "prompt_budget_min_k": min_k,
        "prompt_budget_max_k": max_k,
        "prompt_budget_candidate_pool_k": candidate_pool_k,
        "prompt_budget_ranked_count": len(ranked),
        "prompt_budget_overshoot_tolerance_tokens": overshoot_tolerance,
        "prompt_budget_missing_policy": missing_policy,
        "prompt_budget_raw_candidate_count": raw_candidate_count,
    }


def _select_candidates_reranker(
    sample: ChunkMMRSample,
    reranker: CrossEncoderReranker,
    top_k: int,
) -> dict[str, Any]:
    """Select top-K candidates using cross-encoder reranker scores only.

    No lexical/BM25 scores, no MMR diversity mechanism.
    """
    n = min(len(sample.candidates), int(sample.chunk_emb.shape[0]))
    if n == 0:
        return {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }

    candidate_texts = [str(c.get("text", "")) for c in sample.candidates[:n]]
    reranker_scores = reranker.score(sample.claim, candidate_texts)

    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in range(n):
        candidate = dict(sample.candidates[idx])
        score = float(reranker_scores[idx])
        candidate.update({
            "dense_score": 0.0,
            "lexical_score": 0.0,
            "bm25_score": 0.0,
            "hybrid_score": score,
            "reranker_score": score,
        })
        dedup_key = canonicalize_sentence(str(candidate.get("text", "")))
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


def _candidate_raw_order(candidate: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(candidate.get("raw_report_order", 0)),
        int(candidate.get("raw_sent_order", candidate.get("sent_idx", 0))),
        str(candidate.get("report_id", "")),
        int(candidate.get("sent_idx", 0)),
    )


def _select_candidates_raw_top_evidence(
    sample: ChunkMMRSample,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    method: str = "hybrid_topk",
    positive_only: bool = True,
    pad_to_top_k: bool = False,
) -> dict[str, Any]:
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    base = {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "raw_top_method": method,
        "raw_candidate_count": n,
        "raw_positive_count": 0,
        "raw_selected_positive_count": 0,
        "selection_method": "raw_top_evidence",
    }
    if n == 0:
        return {**base, "candidates": []}

    dense_scores = scored["dense_scores"]
    lexical_scores = scored["lexical_scores"]
    bm25_scores = scored["bm25_scores"]
    hybrid_scores = scored["hybrid_scores"]
    method = str(method or "hybrid_topk").strip().lower()
    if method not in {"hybrid_topk", "hybrid", "original_order", "order"}:
        raise ValueError(
            "build.retrieval.raw_top_evidence.method must be one of "
            "hybrid_topk or original_order."
        )

    enriched: list[dict[str, Any]] = []
    for idx, candidate_raw in enumerate(sample.candidates[:n]):
        candidate = dict(candidate_raw)
        candidate.update({
            "dense_score": float(dense_scores[idx]),
            "lexical_score": float(lexical_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
            "source_candidate_index": int(idx),
            "raw_is_evidence": bool(candidate.get("raw_is_evidence", False)),
            "selection_method": "raw_top_evidence",
            "raw_top_method": method,
        })
        enriched.append(candidate)

    positives = [candidate for candidate in enriched if bool(candidate.get("raw_is_evidence", False))]
    pool = positives if positive_only else list(enriched)

    def _sort_key(candidate: dict[str, Any]):
        if method in {"original_order", "order"}:
            return _candidate_raw_order(candidate)
        order = _candidate_raw_order(candidate)
        return (-float(candidate.get("hybrid_score", 0.0)), order[0], order[1], order[2], order[3])

    ranked = sorted(pool, key=_sort_key)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_until_full(candidates: list[dict[str, Any]]) -> None:
        for candidate in candidates:
            if len(selected) >= top_k:
                return
            key = canonicalize_sentence(str(candidate.get("text", "")))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            selected.append(dict(candidate))

    _append_until_full(ranked)

    if pad_to_top_k and len(selected) < top_k:
        selected_source_indices = {int(c.get("source_candidate_index", -1)) for c in selected}
        fillers = [
            candidate
            for candidate in sorted(enriched, key=_sort_key)
            if int(candidate.get("source_candidate_index", -1)) not in selected_source_indices
        ]
        _append_until_full(fillers)

    for raw_rank, candidate in enumerate(selected, start=1):
        candidate["raw_rank"] = raw_rank

    return {
        **base,
        "raw_top_method": method,
        "raw_positive_count": len(positives),
        "raw_selected_positive_count": sum(1 for candidate in selected if candidate.get("raw_is_evidence")),
        "candidates": selected,
    }


def _raw_top_evidence_phase_from_chunk_cache(
    chunk_samples: list[ChunkMMRSample],
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    raw_top_cfg: dict[str, Any],
    tokenizer,
    prompt_cfg: dict[str, Any],
    output_path: Path,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    method = str(raw_top_cfg.get("method", "hybrid_topk")).strip().lower()
    positive_only = bool(raw_top_cfg.get("positive_only", True))
    pad_to_top_k = bool(raw_top_cfg.get("pad_to_top_k", False))

    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(
            chunk_samples,
            desc=progress_desc or f"Raw top evidence [{method}]",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            row = _select_candidates_raw_top_evidence(
                sample,
                top_k=top_k,
                alpha_dense=alpha_dense,
                alpha_lexical=alpha_lexical,
                alpha_bm25=alpha_bm25,
                method=method,
                positive_only=positive_only,
                pad_to_top_k=pad_to_top_k,
            )
            training_row = build_training_row(row, tokenizer, prompt_cfg)
            writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


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
    reuse_chunk_embeddings: bool = False,
    sentence_source: str = "content",
    sentence_min_char_len: int = 10,
) -> list[dict[str, Any]]:
    """Build candidates for multiple samples with batched embeddings."""
    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    pre_samples = compute_pre_mmr_batch(
        samples,
        embedder,
        sentence_source=sentence_source,
        sentence_min_char_len=sentence_min_char_len,
    )
    chunk_samples = _compute_chunk_mmr_batch(
        pre_samples,
        embedder=embedder,
        strategy=strategy,
        reuse_chunk_embeddings=reuse_chunk_embeddings,
    )

    def _task(i: int) -> dict[str, Any]:
        return _select_candidates_from_chunk_sample(
            chunk_samples[i],
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            mmr_lambda=mmr_lambda,
        )

    if cpu_workers > 1:
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            results = list(pool.map(_task, range(len(chunk_samples))))
    else:
        results = [_task(i) for i in range(len(chunk_samples))]

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
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_for_worker(gpu_id, run_summary)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    samples = load_split(
        data_cfg[f"{split_name}_path"],
        dataset=data_cfg.get("dataset"),
        label_schema=data_cfg.get("label_schema"),
    )
    sample_limit = int(data_cfg.get("sample_limit", 0) or 0)
    if sample_limit > 0:
        samples = samples[:sample_limit]
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

    # Load prompt tokenizer independently (multiprocessing-compatible)
    tokenizer = load_prompt_tokenizer(run_summary["prompt_model_name_or_path"])
    prompt_cfg = {
        "auto_length": run_summary["prompt_auto_length"],
        "max_length": run_summary["prompt_max_length"],
        "output_mode": run_summary["prompt_output_mode"],
        "label_format": run_summary["prompt_label_format"],
        "label_schema": run_summary.get("prompt_label_schema"),
        "system_prompt": run_summary.get("prompt_system_prompt"),
    }

    chunking_cfg = retrieval_cfg.get("chunking")
    chunking_strategy = build_chunking_strategy(chunking_cfg, retrieval_cfg)
    reuse_chunk_embeddings = can_reuse_chunk_embeddings(chunking_strategy, retrieval_cfg)
    prefetch_size = run_summary["prefetch_size"]
    cpu_workers = run_summary["cpu_workers"]

    with output_path.open("w", encoding="utf-8") as writer:
        if prefetch_size > 1:
            for start in tqdm(range(0, len(samples_chunk), prefetch_size),
                              desc=f"Build [{split_name}] GPU {gpu_id}",
                              unit="batch"):
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
                    reuse_chunk_embeddings=reuse_chunk_embeddings,
                    sentence_source=str(run_summary.get("sentence_source", "content")),
                    sentence_min_char_len=int(run_summary.get("sentence_min_char_len", 10)),
                )
                for row in rows:
                    training_row = build_training_row(row, tokenizer, prompt_cfg)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
        else:
            for sample in tqdm(samples_chunk,
                               desc=f"Build [{split_name}] GPU {gpu_id}"):
                row = build_candidates_for_sample(
                    sample=sample,
                    embedder=embedder,
                    top_k=run_summary["top_k"],
                    alpha_dense=run_summary["alpha_dense"],
                    alpha_lexical=run_summary["alpha_lexical"],
                    alpha_bm25=run_summary["alpha_bm25"],
                    mmr_lambda=run_summary["mmr_lambda"],
                    chunking_strategy=chunking_strategy,
                    reuse_chunk_embeddings=reuse_chunk_embeddings,
                    sentence_source=str(run_summary.get("sentence_source", "content")),
                    sentence_min_char_len=int(run_summary.get("sentence_min_char_len", 10)),
                )
                training_row = build_training_row(row, tokenizer, prompt_cfg)
                writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


# MMR phase: CPU-only selection from cached chunk embeddings
# ---------------------------------------------------------------------------


def _mmr_phase_from_chunk_cache(
    chunk_samples: list[ChunkMMRSample],
    mmr_lambda: float,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    tokenizer,
    prompt_cfg: dict[str, Any],
    output_path: Path,
    cpu_workers: int = 1,
    lambda_overrides: dict[str, float] | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    """From cached ChunkMMRSamples, run MMR + training-row construction, write JSONL."""

    def _process_one(sample: ChunkMMRSample):
        effective_lambda = mmr_lambda
        if lambda_overrides is not None:
            effective_lambda = lambda_overrides.get(sample.event_id, mmr_lambda)
        return _select_candidates_from_chunk_sample(
            sample,
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            mmr_lambda=effective_lambda,
        )

    with output_path.open("w", encoding="utf-8") as writer:
        desc = progress_desc or f"MMR λ={mmr_lambda:.2f}"
        if cpu_workers > 1:
            with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                rows = pool.map(_process_one, chunk_samples)
                for row in tqdm(
                    rows,
                    total=len(chunk_samples),
                    desc=desc,
                    unit="sample",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ):
                    training_row = build_training_row(row, tokenizer, prompt_cfg)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
        else:
            for sample in tqdm(
                chunk_samples,
                desc=desc,
                unit="sample",
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                row = _process_one(sample)
                training_row = build_training_row(row, tokenizer, prompt_cfg)
                writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


def _prompt_budget_mmr_phase_from_chunk_cache(
    chunk_samples: list[ChunkMMRSample],
    prompt_budget_targets: dict[str, int],
    prompt_budget_cfg: dict[str, Any],
    reference_path: Path,
    mmr_lambda: float,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    tokenizer,
    prompt_cfg: dict[str, Any],
    output_path: Path,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(
            chunk_samples,
            desc=progress_desc or "MMR prompt-budget",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            row = _select_candidates_prompt_budget_mmr(
                sample=sample,
                prompt_budget_targets=prompt_budget_targets,
                prompt_budget_cfg=prompt_budget_cfg,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
                reference_path=reference_path,
                top_k=top_k,
                alpha_dense=alpha_dense,
                alpha_lexical=alpha_lexical,
                alpha_bm25=alpha_bm25,
                mmr_lambda=mmr_lambda,
            )
            training_row = build_training_row(row, tokenizer, prompt_cfg)
            writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


def _reranker_phase_from_chunk_cache(
    chunk_samples: list[ChunkMMRSample],
    reranker: CrossEncoderReranker,
    top_k: int,
    tokenizer,
    prompt_cfg: dict[str, Any],
    output_path: Path,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    """From cached ChunkMMRSamples, run reranker selection + training-row construction, write JSONL."""

    with output_path.open("w", encoding="utf-8") as writer:
        desc = progress_desc or "Reranker"
        for sample in tqdm(
            chunk_samples,
            desc=desc,
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            row = _select_candidates_reranker(sample, reranker, top_k=top_k)
            training_row = build_training_row(row, tokenizer, prompt_cfg)
            writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_build(cfg: dict[str, Any], *, output_dir: str | Path | None = None, split: str | None = None) -> BuildResult:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger = init_logger(__name__)
    data_cfg = cfg["data"]
    retrieval_cfg = cfg["retrieval"]
    prompt_cfg = cfg.get("prompt", {})
    target_dir = Path(output_dir or cfg.get("output_dir", "outputs/cache/build/manual"))
    target_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = int(retrieval_cfg.get("num_gpus", 1))
    sentence_reader_cfg = sentence_reader_config(cfg)
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
        "selection_method": str(retrieval_cfg.get("selection_method", "mmr")).strip().lower(),
        "cuda_visible_devices": str(retrieval_cfg.get("cuda_visible_devices", "") or ""),
        "sentence_source": sentence_reader_cfg["sentence_source"],
        "sentence_min_char_len": sentence_reader_cfg["sentence_min_char_len"],
        # Prompt config for workers
        "prompt_model_name_or_path": str(prompt_cfg.get("model_name_or_path", "")),
        "prompt_auto_length": bool(prompt_cfg.get("auto_length", True)),
        "prompt_max_length": int(prompt_cfg.get("max_length", 2048)),
        "prompt_output_mode": str(prompt_cfg.get("output_mode", "label_only")).strip().lower(),
        "prompt_label_format": str(prompt_cfg.get("label_format", "name")).strip().lower(),
        "prompt_label_schema": str(
            prompt_cfg.get("label_schema")
            or data_cfg.get("label_schema")
            or cfg.get("label_schema")
            or "liar6"
        ).strip().lower(),
        "prompt_system_prompt": prompt_cfg.get("system_prompt") or None,
    }

    split_names = [split] if split else ["train", "val", "test"]
    split_paths: dict[str, Path] = {}

    # ---- Phase 1: pre-MMR cache (GPU embedding, shared across mmr_lambda values) ----
    premmr_fp = premmr_config_fingerprint(cfg)
    cache_root = Path(str(cfg.get("cache_root") or "outputs/cache"))
    premmr_cache_dir = cache_root / "pre_mmr" / premmr_fp
    logger.info("Pre-MMR cache dir: %s (fp=%s)", premmr_cache_dir, premmr_fp)

    premmr_summary = {
        "embedder_model": run_summary["embedder_model"],
        "max_length": run_summary["max_length"],
        "batch_size": run_summary["batch_size"],
        "precision": run_summary["precision"],
        "prefetch_size": run_summary["prefetch_size"],
        "num_gpus": num_gpus,
        "device": run_summary["device"],
        "cuda_visible_devices": run_summary["cuda_visible_devices"],
        "sentence_source": run_summary["sentence_source"],
        "sentence_min_char_len": run_summary["sentence_min_char_len"],
    }

    pre_mmr_split_paths: dict[str, Path] = {}
    for split_name in split_names:
        pre_mmr_split_paths[split_name] = compute_pre_mmr_split(
            split_name=split_name,
            data_cfg=data_cfg,
            retrieval_cfg=retrieval_cfg,
            run_summary=premmr_summary,
            cache_dir=premmr_cache_dir,
            num_gpus=num_gpus,
        )

    # ---- Phase 2: chunk cache (GPU embedding, shared across top_k/mmr_lambda values) ----
    logger.info("Build run summary: %s", run_summary)
    chunk_mmr_fp = chunk_mmr_config_fingerprint(cfg)
    chunk_mmr_cache_dir = cache_root / "chunk_mmr" / chunk_mmr_fp
    logger.info("Chunk-MMR cache dir: %s (fp=%s)", chunk_mmr_cache_dir, chunk_mmr_fp)
    chunk_mmr_summary = {
        "embedder_model": run_summary["embedder_model"],
        "max_length": run_summary["max_length"],
        "batch_size": run_summary["batch_size"],
        "precision": run_summary["precision"],
        "prefetch_size": run_summary["prefetch_size"],
        "num_gpus": num_gpus,
        "device": run_summary["device"],
        "cuda_visible_devices": run_summary["cuda_visible_devices"],
    }
    chunk_mmr_split_paths: dict[str, Path] = {}
    for split_name in split_names:
        chunk_mmr_split_paths[split_name] = compute_chunk_mmr_split(
            split_name=split_name,
            retrieval_cfg=retrieval_cfg,
            run_summary=chunk_mmr_summary,
            pre_mmr_path=pre_mmr_split_paths[split_name],
            cache_dir=chunk_mmr_cache_dir,
            num_gpus=num_gpus,
        )

    # ---- Phase 3: candidate selection + training row construction ----
    torch.cuda.empty_cache()

    selection_method = run_summary["selection_method"]
    logger.info("Selection method: %s", selection_method)

    tokenizer = load_prompt_tokenizer(run_summary["prompt_model_name_or_path"])
    prompt_cfg_local = {
        "auto_length": run_summary["prompt_auto_length"],
        "max_length": run_summary["prompt_max_length"],
        "output_mode": run_summary["prompt_output_mode"],
        "label_format": run_summary["prompt_label_format"],
        "label_schema": run_summary.get("prompt_label_schema"),
        "system_prompt": run_summary.get("prompt_system_prompt"),
    }

    # ---- Load reranker if needed ----
    reranker = None
    if selection_method == "reranker":
        reranker_cfg = retrieval_cfg.get("reranker", {}) or {}
        logger.info("Loading reranker: %s", reranker_cfg.get("model_name", "BAAI/bge-reranker-base"))
        reranker = CrossEncoderReranker(RerankerConfig(
            model_name=str(reranker_cfg.get("model_name", "BAAI/bge-reranker-base")),
            device=str(reranker_cfg.get("device", "cuda")),
            max_length=int(reranker_cfg.get("max_length", 512)),
            batch_size=int(reranker_cfg.get("batch_size", 32)),
            normalize=bool(reranker_cfg.get("normalize", True)),
        ))

    # ---- Optional: pointwise oracle-supervised selector ----
    pointwise_model = None
    pointwise_cfg = dict(retrieval_cfg.get("pointwise_oracle", {}) or {})
    if selection_method in {"pointwise_oracle", "oracle_pointwise", "pointwise"}:
        from fact_checking.oracle_pointwise import load_pointwise_selector_model

        pointwise_model_path = str(
            pointwise_cfg.get("model_dir")
            or pointwise_cfg.get("model_path")
            or ""
        ).strip()
        if not pointwise_model_path:
            raise ValueError(
                "build.retrieval.pointwise_oracle.model_dir is required when "
                "build.retrieval.selection_method=pointwise_oracle."
            )
        pointwise_model = load_pointwise_selector_model(
            pointwise_model_path,
            expected_chunk_mmr_fingerprint=chunk_mmr_fp,
            strict_fingerprint=bool(pointwise_cfg.get("strict_fingerprint", True)),
        )
        logger.info(
            "Loaded pointwise oracle selector: model=%s features=%d chunk_mmr_fp=%s",
            pointwise_model.path,
            len(pointwise_model.feature_names),
            pointwise_model.metadata.get("chunk_mmr_fingerprint", ""),
        )

    # ---- Optional: Stage2 cross-encoder selector ----
    cross_encoder_selector = None
    cross_encoder_cfg = dict(retrieval_cfg.get("cross_encoder_selector", {}) or {})
    if selection_method in {"cross_encoder_selector", "cross_encoder_oracle", "cross_encoder_pairwise"}:
        from fact_checking.selectors.cross_encoder import (
            CrossEncoderSelector,
            CrossEncoderSelectorConfig,
        )

        cross_encoder_model_dir = str(
            cross_encoder_cfg.get("model_dir")
            or cross_encoder_cfg.get("model_path")
            or ""
        ).strip()
        if not cross_encoder_model_dir:
            raise ValueError(
                "build.retrieval.cross_encoder_selector.model_dir is required when "
                "build.retrieval.selection_method=cross_encoder_selector."
            )
        cross_encoder_selector = CrossEncoderSelector(
            CrossEncoderSelectorConfig(
                model_dir=cross_encoder_model_dir,
                device=str(cross_encoder_cfg.get("device", retrieval_cfg.get("device", "cuda"))),
                max_length=int(cross_encoder_cfg.get("max_length", 384)),
                batch_size=int(cross_encoder_cfg.get("batch_size", 32)),
                strict_fingerprint=bool(cross_encoder_cfg.get("strict_fingerprint", True)),
                expected_chunk_mmr_fingerprint=chunk_mmr_fp,
            )
        )
        logger.info(
            "Loaded cross-encoder selector: model=%s chunk_mmr_fp=%s",
            cross_encoder_selector.model_dir,
            cross_encoder_selector.metadata.get("chunk_mmr_fingerprint", ""),
        )

    # ---- Optional: Stage2 set-aware listwise selector ----
    listwise_selector = None
    listwise_cfg = dict(retrieval_cfg.get("listwise_selector", {}) or {})
    if selection_method in {"listwise_selector", "set_aware_listwise", "listwise"}:
        from fact_checking.selectors.listwise import (
            ListwiseSelector,
            ListwiseSelectorConfig,
        )

        listwise_model_dir = str(
            listwise_cfg.get("model_dir")
            or listwise_cfg.get("model_path")
            or ""
        ).strip()
        if not listwise_model_dir:
            raise ValueError(
                "build.retrieval.listwise_selector.model_dir is required when "
                "build.retrieval.selection_method=listwise_selector."
            )
        listwise_selector = ListwiseSelector(
            ListwiseSelectorConfig(
                model_dir=listwise_model_dir,
                device=str(listwise_cfg.get("device", retrieval_cfg.get("device", "cuda"))),
                max_length=int(listwise_cfg.get("max_length", 384)),
                batch_size=int(listwise_cfg.get("batch_size", 8)),
                strict_fingerprint=bool(listwise_cfg.get("strict_fingerprint", True)),
                expected_chunk_mmr_fingerprint=chunk_mmr_fp,
            )
        )
        logger.info(
            "Loaded listwise selector: model=%s chunk_mmr_fp=%s",
            listwise_selector.model_dir,
            listwise_selector.metadata.get("chunk_mmr_fingerprint", ""),
        )

    # ---- Optional: Stage2 sequential pointer selector ----
    sequential_selector = None
    sequential_cfg = dict(retrieval_cfg.get("sequential_selector", {}) or {})
    if selection_method in {"sequential_selector", "sequential_pointer", "pointer_selector"}:
        from fact_checking.selectors.sequential import (
            SequentialSelector,
            SequentialSelectorConfig,
        )

        sequential_model_dir = str(
            sequential_cfg.get("model_dir")
            or sequential_cfg.get("model_path")
            or ""
        ).strip()
        if not sequential_model_dir:
            raise ValueError(
                "build.retrieval.sequential_selector.model_dir is required when "
                "build.retrieval.selection_method=sequential_selector."
            )
        sequential_selector = SequentialSelector(
            SequentialSelectorConfig(
                model_dir=sequential_model_dir,
                device=str(sequential_cfg.get("device", retrieval_cfg.get("device", "cuda"))),
                max_length=int(sequential_cfg.get("max_length", 384)),
                batch_size=int(sequential_cfg.get("batch_size", 8)),
                strict_fingerprint=bool(sequential_cfg.get("strict_fingerprint", True)),
                expected_chunk_mmr_fingerprint=chunk_mmr_fp,
            )
        )
        logger.info(
            "Loaded sequential selector: model=%s chunk_mmr_fp=%s",
            sequential_selector.model_dir,
            sequential_selector.metadata.get("chunk_mmr_fingerprint", ""),
        )

    # ---- Optional: learned λ predictor (MMR path only) ----
    learned_lambda_cfg = retrieval_cfg.get("learned_lambda", {}) or {}
    use_learned_lambda = bool(learned_lambda_cfg.get("enabled", False))
    learned_lambda_mode = str(learned_lambda_cfg.get("mode", "predictor")).strip().lower()
    raw_top_cfg = dict(retrieval_cfg.get("raw_top_evidence", {}) or {})
    prompt_budget_cfg = dict(retrieval_cfg.get("prompt_budget", {}) or {})

    for split_name in split_names:
        chunk_samples = load_pickle(chunk_mmr_split_paths[split_name])
        output_path = target_dir / f"build_{split_name}.jsonl"

        if selection_method == "reranker":
            _reranker_phase_from_chunk_cache(
                chunk_samples=chunk_samples,
                reranker=reranker,
                top_k=run_summary["top_k"],
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg_local,
                output_path=output_path,
                show_progress=True,
                progress_desc=f"Reranker [{split_name}]",
            )
        elif selection_method in {"raw_top_evidence", "raw_label_topk", "raw_evidence"}:
            if run_summary["sentence_source"] not in {"tokenized", "raw_tokenized", "raw"}:
                raise ValueError(
                    "build.retrieval.selection_method=raw_top_evidence requires "
                    "build.data.sentence_source=tokenized."
                )
            _raw_top_evidence_phase_from_chunk_cache(
                chunk_samples=chunk_samples,
                top_k=run_summary["top_k"],
                alpha_dense=run_summary["alpha_dense"],
                alpha_lexical=run_summary["alpha_lexical"],
                alpha_bm25=run_summary["alpha_bm25"],
                raw_top_cfg=raw_top_cfg,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg_local,
                output_path=output_path,
                show_progress=True,
                progress_desc=f"Raw top evidence [{split_name}]",
            )
        elif selection_method in {"mmr_prompt_budget", "prompt_budget_mmr", "adaptive_budget_mmr"}:
            prompt_budget_targets, reference_path = _load_prompt_budget_targets(prompt_budget_cfg, split_name)
            logger.info(
                "Prompt-budget MMR [%s]: targets=%d reference=%s candidate_pool_k=%s min_k=%s max_k=%s",
                split_name,
                len(prompt_budget_targets),
                reference_path,
                prompt_budget_cfg.get("candidate_pool_k", prompt_budget_cfg.get("pool_k", max(run_summary["top_k"], 32))),
                prompt_budget_cfg.get("min_k", run_summary["top_k"]),
                prompt_budget_cfg.get("max_k", prompt_budget_cfg.get("candidate_pool_k", 32)),
            )
            _prompt_budget_mmr_phase_from_chunk_cache(
                chunk_samples=chunk_samples,
                prompt_budget_targets=prompt_budget_targets,
                prompt_budget_cfg=prompt_budget_cfg,
                reference_path=reference_path,
                mmr_lambda=run_summary["mmr_lambda"],
                top_k=run_summary["top_k"],
                alpha_dense=run_summary["alpha_dense"],
                alpha_lexical=run_summary["alpha_lexical"],
                alpha_bm25=run_summary["alpha_bm25"],
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg_local,
                output_path=output_path,
                show_progress=True,
                progress_desc=f"MMR prompt-budget [{split_name}]",
            )
        elif selection_method in {"pointwise_oracle", "oracle_pointwise", "pointwise"}:
            from fact_checking.oracle_pointwise import select_candidates_pointwise_oracle

            if pointwise_model is None:
                raise RuntimeError("Pointwise selector model was not loaded.")

            raw_pool_size = pointwise_cfg.get("candidate_pool_size")
            candidate_pool_size = None
            if raw_pool_size not in (None, "", 0, "0"):
                candidate_pool_size = int(raw_pool_size)
            candidate_pool_multiplier = int(pointwise_cfg.get("candidate_pool_multiplier", 3))
            dump_trace = bool(pointwise_cfg.get("dump_trace", True))
            trace_rows: list[dict[str, Any]] = []

            with output_path.open("w", encoding="utf-8") as writer:
                pointwise_pbar = tqdm(
                    chunk_samples,
                    desc=f"Pointwise oracle selector [{split_name}]",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for sample in pointwise_pbar:
                    row, trace = select_candidates_pointwise_oracle(
                        sample,
                        pointwise_model,
                        top_k=run_summary["top_k"],
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        candidate_pool_size=candidate_pool_size,
                        candidate_pool_multiplier=candidate_pool_multiplier,
                    )
                    if dump_trace:
                        trace_rows.append(trace)
                    training_row = build_training_row(row, tokenizer, prompt_cfg_local)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                pointwise_pbar.close()

            if dump_trace:
                trace_path = target_dir / f"pointwise_oracle_trace_{split_name}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace_writer:
                    for trace in trace_rows:
                        trace_writer.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Wrote pointwise oracle trace: %s", trace_path)
        elif selection_method in {"cross_encoder_selector", "cross_encoder_oracle", "cross_encoder_pairwise"}:
            from fact_checking.selectors.cross_encoder import select_candidates_cross_encoder

            if cross_encoder_selector is None:
                raise RuntimeError("Cross-encoder selector model was not loaded.")

            raw_pool_size = cross_encoder_cfg.get("candidate_pool_size", 15)
            candidate_pool_size = None
            if raw_pool_size not in (None, "", 0, "0"):
                candidate_pool_size = int(raw_pool_size)
            dump_trace = bool(cross_encoder_cfg.get("dump_trace", True))
            trace_rows: list[dict[str, Any]] = []

            with output_path.open("w", encoding="utf-8") as writer:
                cross_encoder_pbar = tqdm(
                    chunk_samples,
                    desc=f"Cross-encoder selector [{split_name}]",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for sample in cross_encoder_pbar:
                    row, trace = select_candidates_cross_encoder(
                        sample,
                        cross_encoder_selector,
                        top_k=run_summary["top_k"],
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        candidate_pool_size=candidate_pool_size,
                    )
                    if dump_trace:
                        trace_rows.append(trace)
                    training_row = build_training_row(row, tokenizer, prompt_cfg_local)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                cross_encoder_pbar.close()

            if dump_trace:
                trace_path = target_dir / f"cross_encoder_selector_trace_{split_name}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace_writer:
                    for trace in trace_rows:
                        trace_writer.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Wrote cross-encoder selector trace: %s", trace_path)
        elif selection_method in {"listwise_selector", "set_aware_listwise", "listwise"}:
            from fact_checking.selectors.listwise import select_candidates_listwise

            if listwise_selector is None:
                raise RuntimeError("Listwise selector model was not loaded.")

            raw_pool_size = listwise_cfg.get("candidate_pool_size", 15)
            candidate_pool_size = None
            if raw_pool_size not in (None, "", 0, "0"):
                candidate_pool_size = int(raw_pool_size)
            dump_trace = bool(listwise_cfg.get("dump_trace", True))
            trace_rows: list[dict[str, Any]] = []

            with output_path.open("w", encoding="utf-8") as writer:
                listwise_pbar = tqdm(
                    chunk_samples,
                    desc=f"Listwise selector [{split_name}]",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for sample in listwise_pbar:
                    row, trace = select_candidates_listwise(
                        sample,
                        listwise_selector,
                        top_k=run_summary["top_k"],
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        candidate_pool_size=candidate_pool_size,
                    )
                    if dump_trace:
                        trace_rows.append(trace)
                    training_row = build_training_row(row, tokenizer, prompt_cfg_local)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                listwise_pbar.close()

            if dump_trace:
                trace_path = target_dir / f"listwise_selector_trace_{split_name}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace_writer:
                    for trace in trace_rows:
                        trace_writer.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Wrote listwise selector trace: %s", trace_path)
        elif selection_method in {"sequential_selector", "sequential_pointer", "pointer_selector"}:
            from fact_checking.selectors.sequential import select_candidates_sequential

            if sequential_selector is None:
                raise RuntimeError("Sequential selector model was not loaded.")

            raw_pool_size = sequential_cfg.get("candidate_pool_size", 15)
            candidate_pool_size = None
            if raw_pool_size not in (None, "", 0, "0"):
                candidate_pool_size = int(raw_pool_size)
            dump_trace = bool(sequential_cfg.get("dump_trace", True))
            trace_rows: list[dict[str, Any]] = []

            with output_path.open("w", encoding="utf-8") as writer:
                sequential_pbar = tqdm(
                    chunk_samples,
                    desc=f"Sequential selector [{split_name}]",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for sample in sequential_pbar:
                    row, trace = select_candidates_sequential(
                        sample,
                        sequential_selector,
                        top_k=run_summary["top_k"],
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        candidate_pool_size=candidate_pool_size,
                    )
                    if dump_trace:
                        trace_rows.append(trace)
                    training_row = build_training_row(row, tokenizer, prompt_cfg_local)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                sequential_pbar.close()

            if dump_trace:
                trace_path = target_dir / f"sequential_selector_trace_{split_name}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace_writer:
                    for trace in trace_rows:
                        trace_writer.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Wrote sequential selector trace: %s", trace_path)
        else:
            lambda_overrides: dict[str, float] | None = None
            if use_learned_lambda:
                if learned_lambda_mode == "heuristic":
                    a = float(learned_lambda_cfg.get("heuristic_a", -0.0732))
                    b = float(learned_lambda_cfg.get("heuristic_b", 0.6127))
                    lambda_overrides = {}
                    for sample in chunk_samples:
                        n = min(len(sample.candidates), int(sample.chunk_emb.shape[0]))
                        raw = a * np.log(max(n, 1)) + b
                        lambda_overrides[sample.event_id] = float(max(0.0, min(1.0, raw)))
                elif learned_lambda_mode == "sensitivity_gated":
                    from fact_checking.rl_mmr.gated_selector import (
                        build_lambda_overrides_from_sensitivity,
                        dump_trace_rows,
                    )

                    lambda_overrides, trace_rows, sens_summary = build_lambda_overrides_from_sensitivity(
                        chunk_samples,
                        learned_lambda_cfg=learned_lambda_cfg,
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        top_k=run_summary["top_k"],
                    )
                    logger.info(
                        "Sensitivity-gated: gates=%s, chosen_lambda mean=%.3f std=%.3f, sens_mean=%.3f, pool_red_mean=%.3f",
                        sens_summary["gate_counts"],
                        sens_summary["chosen_lambda_mean"],
                        sens_summary["chosen_lambda_std"],
                        sens_summary["sens_low_base_mean"],
                        sens_summary["pool_redundancy_mean"],
                    )
                    if sens_summary["config"].get("dump_trace", True):
                        trace_path = target_dir / f"sensitivity_trace_{split_name}.jsonl"
                        dump_trace_rows(trace_rows, trace_path)
                        logger.info("Wrote sensitivity trace: %s", trace_path)
                elif learned_lambda_mode == "soft_label":
                    from fact_checking.rl_mmr.soft_label_selector import (
                        build_lambda_overrides_from_soft_label,
                        dump_trace_rows,
                    )

                    lambda_overrides, trace_rows, soft_summary = build_lambda_overrides_from_soft_label(
                        chunk_samples,
                        learned_lambda_cfg=learned_lambda_cfg,
                        alpha_dense=run_summary["alpha_dense"],
                        alpha_lexical=run_summary["alpha_lexical"],
                        alpha_bm25=run_summary["alpha_bm25"],
                        top_k=run_summary["top_k"],
                    )
                    logger.info(
                        "Soft-label lambda: model=%s type=%s mode=%s overrides=%d, chosen_lambda mean=%.3f std=%.3f, entropy_mean=%.3f, argmax_counts=%s",
                        soft_summary.get("model_path"),
                        soft_summary.get("model_type"),
                        soft_summary.get("inference_mode"),
                        len(lambda_overrides),
                        soft_summary.get("chosen_lambda_mean", 0.0),
                        soft_summary.get("chosen_lambda_std", 0.0),
                        soft_summary.get("prediction_entropy_mean", 0.0),
                        soft_summary.get("argmax_counts", {}),
                    )
                    if soft_summary.get("config", {}).get("dump_trace", True):
                        trace_path = target_dir / f"soft_label_trace_{split_name}.jsonl"
                        dump_trace_rows(trace_rows, trace_path)
                        logger.info("Wrote soft-label trace: %s", trace_path)
                elif learned_lambda_mode == "dpo_stepwise":
                    from fact_checking.rl_mmr.dpo_selector import (
                        load_dpo_step_policy,
                        select_candidates_dpo_stepwise,
                        dump_trace_rows as dump_dpo_trace,
                    )

                    dpo_cfg = dict(learned_lambda_cfg.get("dpo_stepwise", {}) or {})
                    dpo_model_path = str(dpo_cfg.get("model_path", "outputs/rl_mmr/dpo_stepwise/checkpoints"))
                    lambda_grid = np.array(
                        dpo_cfg.get("lambda_grid", [0.1, 0.3, 0.5, 0.7, 0.9]), dtype=np.float32,
                    )
                    dpo_inference_mode = str(dpo_cfg.get("inference_mode", "argmax")).strip().lower()
                    dpo_sample_temp = float(dpo_cfg.get("sample_temperature", 0.5))

                    policy, feature_stats = load_dpo_step_policy(dpo_model_path)
                    logger.info(
                        "Loaded DPO step-wise policy from %s, inference_mode=%s",
                        dpo_model_path, dpo_inference_mode,
                    )

                    dpo_trace_rows: list[dict[str, Any]] = []
                    with output_path.open("w", encoding="utf-8") as writer:
                        dpo_pbar = tqdm(
                            chunk_samples,
                            desc=f"DPO stepwise [{split_name}]",
                            unit="sample",
                            dynamic_ncols=True,
                        )
                        for sample in dpo_pbar:
                            row = select_candidates_dpo_stepwise(
                                sample, policy, feature_stats, lambda_grid,
                                top_k=run_summary["top_k"],
                                alpha_dense=run_summary["alpha_dense"],
                                alpha_lexical=run_summary["alpha_lexical"],
                                alpha_bm25=run_summary["alpha_bm25"],
                                inference_mode=dpo_inference_mode,
                                sample_temperature=dpo_sample_temp,
                            )
                            chosen = row.pop("_dpo_chosen_lambdas", [])
                            dpo_trace_rows.append({
                                "event_id": sample.event_id,
                                "claim": sample.claim,
                                "label": sample.label,
                                "chosen_lambdas": chosen,
                                "n_candidates": len(sample.candidates),
                                "inference_mode": dpo_inference_mode,
                            })
                            training_row = build_training_row(row, tokenizer, prompt_cfg_local)
                            writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                        dpo_pbar.close()

                    chosen_flat = [v for tr in dpo_trace_rows for v in tr["chosen_lambdas"]]
                    if chosen_flat:
                        logger.info(
                            "DPO stepwise (%s): %d samples, λ mean=%.3f std=%.3f",
                            dpo_inference_mode,
                            len(dpo_trace_rows),
                            float(np.mean(chosen_flat)),
                            float(np.std(chosen_flat)),
                        )
                    if dpo_cfg.get("dump_trace", True):
                        dpo_trace_path = target_dir / f"dpo_stepwise_trace_{split_name}.jsonl"
                        dump_dpo_trace(dpo_trace_rows, dpo_trace_path)
                        logger.info("Wrote DPO stepwise trace: %s", dpo_trace_path)

                    # DPO stepwise writes output directly; skip _mmr_phase_from_chunk_cache
                    dpo_handled = True
                else:
                    from fact_checking.learned_lambda.predictor import load_predictor, predict_lambdas_for_samples
                    model_path = str(learned_lambda_cfg["model_path"])
                    stats_path = str(learned_lambda_cfg["feature_stats_path"])
                    predictor, stats = load_predictor(model_path, stats_path)
                    feature_mode = str(stats.get("feature_mode") or "handcrafted").strip().lower()
                    if feature_mode == "chunk_embedding":
                        lambda_overrides = predict_lambdas_for_samples(chunk_samples, predictor, stats, retrieval_cfg)
                    else:
                        pre_samples = load_pickle(pre_mmr_split_paths[split_name])
                        lambda_overrides = predict_lambdas_for_samples(pre_samples, predictor, stats, retrieval_cfg)

            # Only call standard MMR phase if DPO didn't handle it directly
            dpo_handled = use_learned_lambda and learned_lambda_mode == "dpo_stepwise"
            if not dpo_handled:
                if use_learned_lambda:
                    vals = list(lambda_overrides.values()) if lambda_overrides else []
                    if vals:
                        logger.info(
                            "Learned lambda (%s): %d overrides, mean=%.3f, std=%.3f",
                            learned_lambda_mode, len(vals), float(np.mean(vals)), float(np.std(vals)),
                        )

                _mmr_phase_from_chunk_cache(
                    chunk_samples=chunk_samples,
                    mmr_lambda=run_summary["mmr_lambda"],
                    top_k=run_summary["top_k"],
                    alpha_dense=run_summary["alpha_dense"],
                    alpha_lexical=run_summary["alpha_lexical"],
                    alpha_bm25=run_summary["alpha_bm25"],
                    tokenizer=tokenizer,
                    prompt_cfg=prompt_cfg_local,
                    output_path=output_path,
                    cpu_workers=run_summary["cpu_workers"],
                    lambda_overrides=lambda_overrides,
                )

        split_paths[split_name] = output_path
        logger.info("Wrote build file: %s", output_path)

    if reranker is not None:
        del reranker
        torch.cuda.empty_cache()

    if "train" in split_paths and "val" in split_paths:
        generate_prompt_stats(
            train_path=split_paths["train"],
            val_path=split_paths["val"],
            output_dir=target_dir,
            max_length=run_summary["prompt_max_length"],
            logger=logger,
        )

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


# ---------------------------------------------------------------------------
# Backward compatibility aliases for external importers
# These will be removed in P2-2 when all importers are updated.
# ---------------------------------------------------------------------------
_sentence_reader_config = sentence_reader_config
_sentence_reader_fingerprint_payload = sentence_reader_fingerprint_payload
_premmr_config_fingerprint = premmr_config_fingerprint
_chunk_mmr_config_fingerprint = chunk_mmr_config_fingerprint
_save_pickle_atomic = save_pickle_atomic
_load_pickle = load_pickle
_sentence_to_dict = sentence_to_dict
_dict_to_sentence = dict_to_sentence
_normalize_model_name = normalize_model_name
_can_reuse_chunk_embeddings = can_reuse_chunk_embeddings
_visible_gpu_for_worker = visible_gpu_for_worker
_chunk_embeddings_by_content = chunk_embeddings_by_content
_compute_pre_mmr_batch = compute_pre_mmr_batch
_premmr_worker = premmr_worker
_compute_pre_mmr_split = compute_pre_mmr_split
_chunk_mmr_worker = chunk_mmr_worker
_compute_chunk_mmr_split = compute_chunk_mmr_split
_load_prompt_tokenizer = load_prompt_tokenizer
_count_tokens = count_tokens
_count_target_tokens = count_target_tokens
_build_system_message = build_system_message
_format_evidence_block = format_evidence_block
_label_definitions_text = label_definitions_text
_build_user_content = build_user_content
_build_chat_prompt = build_chat_prompt
_render_prompt = render_prompt
_decode_token_prefix = decode_token_prefix
_truncate_single_evidence_to_budget = truncate_single_evidence_to_budget
_build_target = build_target
_copy_optional_build_row_metadata = copy_optional_build_row_metadata
_auto_truncate_evidence = auto_truncate_evidence
_build_training_row = build_training_row
_OPTIONAL_BUILD_ROW_KEYS = OPTIONAL_BUILD_ROW_KEYS
_generate_prompt_stats = generate_prompt_stats
_rows_to_prepared_samples = rows_to_prepared_samples
