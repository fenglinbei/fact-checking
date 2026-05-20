from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from fact_checking.build.chunking import ChunkingStrategy, SentenceChunking, build_chunking_strategy
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


BUILD_LOGIC_VERSION = "chunk-first-mmr-v1"
CHUNK_MMR_CACHE_VERSION = "chunk-text-embedding-v1"


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


# ---------------------------------------------------------------------------
# Pre-MMR cache helpers
# ---------------------------------------------------------------------------


def _premmr_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint only settings that affect cached sentence/claim embeddings."""
    from fact_checking.pipeline.artifacts import fingerprint

    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {
        "data": cfg.get("data", {}),
        "retrieval": retrieval,
    }
    return fingerprint(payload)


def _chunk_mmr_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint settings that affect chunk texts or chunk embeddings."""
    from fact_checking.pipeline.artifacts import fingerprint

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
    return fingerprint(payload)


def _save_pickle_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _sentence_to_dict(sent) -> dict[str, Any]:
    return {
        "event_id": sent.event_id,
        "report_id": sent.report_id,
        "sent_idx": sent.sent_idx,
        "text": sent.text,
        "link": sent.link,
        "domain": sent.domain,
        "raw": sent.raw,
    }


def _dict_to_sentence(d: dict[str, Any], *, event_id_fallback: str | None = None):
    from fact_checking.data.io import SentenceRecord

    return SentenceRecord(
        event_id=d.get("event_id") or event_id_fallback or "",
        report_id=d["report_id"],
        sent_idx=d["sent_idx"],
        text=d["text"],
        link=d.get("link"),
        domain=d.get("domain"),
        raw=d.get("raw", {}),
    )


def _normalize_model_name(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _can_reuse_chunk_embeddings(strategy: ChunkingStrategy, retrieval_cfg: dict[str, Any]) -> bool:
    embedder_cfg = getattr(strategy, "_embedder_cfg", None)
    if embedder_cfg is None:
        return False
    retrieval_model = retrieval_cfg.get("embedder_model")
    if not retrieval_model:
        return False
    return (
        _normalize_model_name(getattr(embedder_cfg, "model_name", ""))
        == _normalize_model_name(retrieval_model)
        and int(getattr(embedder_cfg, "max_length", 256)) == int(retrieval_cfg.get("max_length", 256))
    )


def _chunk_embeddings_by_content(
    sents: list,
    sent_emb: np.ndarray,
) -> dict[str, dict[int, np.ndarray]]:
    if sent_emb.ndim != 2 or len(sent_emb) < len(sents):
        return {}

    by_content: dict[str, dict[int, np.ndarray]] = {}
    for row_idx, sent in enumerate(sents):
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if not content:
            continue
        by_content.setdefault(str(content), {})[int(sent.sent_idx)] = sent_emb[row_idx]
    return by_content


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
) -> dict[str, Any]:
    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    pre_sample = _compute_pre_mmr_batch([sample], embedder)[0]
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


def _build_chunk_candidate_rows(
    pre: PreMMRSample,
    strategy: ChunkingStrategy,
    reuse_chunk_embeddings: bool = False,
) -> list[dict[str, Any]]:
    sents = [_dict_to_sentence(d, event_id_fallback=pre.event_id) for d in pre.sentences]
    if not sents:
        return []

    content_embeddings = _chunk_embeddings_by_content(sents, pre.sent_emb) if reuse_chunk_embeddings else {}
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
) -> list[dict[str, Any]]:
    """Build candidates for multiple samples with batched embeddings."""
    strategy = chunking_strategy if chunking_strategy is not None else SentenceChunking()
    pre_samples = _compute_pre_mmr_batch(samples, embedder)
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

    # Load prompt tokenizer independently (multiprocessing-compatible)
    tokenizer = _load_prompt_tokenizer(run_summary["prompt_model_name_or_path"])
    prompt_cfg = {
        "auto_length": run_summary["prompt_auto_length"],
        "max_length": run_summary["prompt_max_length"],
        "output_mode": run_summary["prompt_output_mode"],
        "label_format": run_summary["prompt_label_format"],
        "system_prompt": run_summary.get("prompt_system_prompt"),
    }

    chunking_cfg = retrieval_cfg.get("chunking")
    chunking_strategy = build_chunking_strategy(chunking_cfg, retrieval_cfg)
    reuse_chunk_embeddings = _can_reuse_chunk_embeddings(chunking_strategy, retrieval_cfg)
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
                )
                for row in rows:
                    training_row = _build_training_row(row, tokenizer, prompt_cfg)
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
                )
                training_row = _build_training_row(row, tokenizer, prompt_cfg)
                writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant for LIAR-RAW claims. "
    "Classify claims using only the claim and retrieved evidence supplied by the user."
)


def _load_prompt_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _count_tokens(text: str, tokenizer: AutoTokenizer, *, add_special_tokens: bool = False) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=add_special_tokens)["input_ids"])


def _count_target_tokens(target: str, tokenizer: AutoTokenizer) -> int:
    ids = tokenizer(target.strip(), truncation=False, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return len(ids)


def _build_system_message(system_prompt: str | None) -> str:
    if system_prompt and str(system_prompt).strip():
        return str(system_prompt).strip()
    return _DEFAULT_SYSTEM_PROMPT


def _format_evidence_block(evidence_texts: list[str]) -> str:
    lines = [f"[{i}] {text}" for i, text in enumerate(evidence_texts, start=1)]
    return "\n".join(lines)


def _label_definitions_text(label_format: str = "name") -> str:
    if label_format == "letter":
        return "\n".join(
            f"- {LABEL_LETTERS[label]} ({label}): {LABEL_DEFINITIONS[label]}"
            for label in LABEL_DEFINITIONS
        )
    return "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABEL_DEFINITIONS)


def _build_user_content(
    claim: str, evidence_texts: list[str], output_mode: str, label_format: str = "name"
) -> str:
    evidence_block = _format_evidence_block(evidence_texts)
    evidence_display = evidence_block.strip() if evidence_block.strip() else "(no evidence available)"

    label_placeholder = "<a single letter from A-F>" if label_format == "letter" else "<label>"

    if output_mode == "explanation_label":
        return (
            "Classify the claim into exactly one LIAR-RAW label and provide a concise evidence-grounded explanation.\n\n"
            "Labels:\n"
            f"{_label_definitions_text(label_format)}\n\n"
            "Rules:\n"
            "- Use the retrieved evidence as the primary source.\n"
            "- Do not invent facts not supported by the evidence.\n"
            "- Keep the explanation brief and evidence-grounded.\n"
            "- Respond with exactly two lines in this format:\n"
            "Explanation: <brief explanation>\n"
            f"Label: {label_placeholder}\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{evidence_display}"
        )
    # label_only (default)
    return (
        "Classify the claim into exactly one LIAR-RAW label.\n\n"
        "Labels:\n"
        f"{_label_definitions_text(label_format)}\n\n"
        "Rules:\n"
        "- Use the retrieved evidence as the primary source.\n"
        "- Do not invent facts not supported by the evidence.\n"
        f"- Respond with exactly one line: Label: {label_placeholder}\n\n"
        f"Claim:\n{claim.strip()}\n\n"
        f"Evidence:\n{evidence_display}"
    )


def _build_chat_prompt(tokenizer: AutoTokenizer, system_msg: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _render_prompt(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
) -> tuple[str, int]:
    user_content = _build_user_content(claim, evidence_texts, output_mode, label_format)
    prompt = _build_chat_prompt(tokenizer, system_msg, user_content)
    return prompt, _count_tokens(prompt, tokenizer, add_special_tokens=False)


def _decode_token_prefix(tokenizer: AutoTokenizer, token_ids: list[int], length: int) -> str:
    if length <= 0:
        return ""
    return tokenizer.decode(token_ids[:length], skip_special_tokens=True).strip()


def _truncate_single_evidence_to_budget(
    *,
    claim: str,
    evidence_text: str,
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
    budget: int,
) -> tuple[list[str], str, int, bool]:
    """Shorten one evidence item until the full chat prompt fits the prompt budget."""
    token_ids = tokenizer(
        evidence_text,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

    best_text: str | None = None
    best_prompt: str | None = None
    best_tokens: int | None = None
    left = 0
    right = len(token_ids)
    while left <= right:
        mid = (left + right) // 2
        candidate_text = _decode_token_prefix(tokenizer, token_ids, mid)
        prompt, prompt_tokens = _render_prompt(
            claim=claim,
            evidence_texts=[candidate_text],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )
        if prompt_tokens <= budget:
            best_text = candidate_text
            best_prompt = prompt
            best_tokens = prompt_tokens
            left = mid + 1
        else:
            right = mid - 1

    if best_text is not None and best_prompt is not None and best_tokens is not None:
        return [best_text], best_prompt, best_tokens, best_text.strip() != evidence_text.strip()

    no_evidence_prompt, no_evidence_tokens = _render_prompt(
        claim=claim,
        evidence_texts=[],
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
    )
    return [], no_evidence_prompt, no_evidence_tokens, True


def _build_target(row: dict, gold_label: str, output_mode: str, label_format: str = "name") -> str:
    target_label = LABEL_LETTERS[gold_label] if label_format == "letter" else gold_label
    if output_mode == "explanation_label":
        explanation = str(row.get("explain", "")).strip() or "The available evidence supports this label."
        return f"Explanation: {explanation}\nLabel: {target_label}"
    return f"Label: {target_label}"


def _auto_truncate_evidence(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    output_mode: str,
    system_prompt: str | None,
    row: dict,
    gold_label: str,
    label_format: str = "name",
) -> dict:
    """Remove evidence items from the tail until the prompt fits within max_length."""
    system_msg = _build_system_message(system_prompt)
    target = _build_target(row, gold_label, output_mode, label_format)
    target_token_count = _count_target_tokens(target, tokenizer)
    budget = max(0, int(max_length) - target_token_count)

    evidence_count_before = len(evidence_texts)
    kept = list(evidence_texts)

    prompt, prompt_tokens = _render_prompt(
        claim=claim,
        evidence_texts=kept,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
    )

    was_truncated = False
    evidence_text_truncated = False
    while prompt_tokens > budget and len(kept) > 1:
        kept.pop()  # Remove last (lowest-score) evidence item
        was_truncated = True
        prompt, prompt_tokens = _render_prompt(
            claim=claim,
            evidence_texts=kept,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )

    if prompt_tokens > budget and len(kept) == 1:
        kept, prompt, prompt_tokens, evidence_text_truncated = _truncate_single_evidence_to_budget(
            claim=claim,
            evidence_text=kept[0],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
        )
        was_truncated = True

    return {
        "prompt": prompt,
        "target": target,
        "prompt_token_count": prompt_tokens,
        "target_token_count": target_token_count,
        "evidence_count": len(kept),
        "evidence_count_before": evidence_count_before,
        "was_truncated": was_truncated,
        "evidence_text_truncated": evidence_text_truncated,
        "overflow_after": prompt_tokens > budget,
    }


def _build_training_row(
    retrieval_result: dict,
    tokenizer: AutoTokenizer,
    prompt_cfg: dict,
) -> dict:
    row = retrieval_result
    gold_label = normalize_gold_label(row)
    if not gold_label:
        return {**row, "gold_label": "", "gold_id": -1, "gold_explain": "",
                "prompt": "", "target": "", "prompt_add_special_tokens": False,
                "preserve_prompt_prefix": True, "prompt_token_count": 0,
                "target_token_count": 0, "evidence_count": 0, "was_truncated": False}

    candidates = row.get("candidates", [])
    evidence_texts = [str(c.get("text", "")).strip() for c in candidates if isinstance(c, dict)]

    auto_length = bool(prompt_cfg.get("auto_length", True))
    max_length = int(prompt_cfg.get("max_length", 2048))
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "name")).strip().lower()
    system_prompt = prompt_cfg.get("system_prompt") or None

    if auto_length and evidence_texts:
        result = _auto_truncate_evidence(
            claim=str(row.get("claim", "")),
            evidence_texts=evidence_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            output_mode=output_mode,
            system_prompt=system_prompt,
            row=row,
            gold_label=gold_label,
            label_format=label_format,
        )
        return {
            "event_id": row.get("event_id", ""),
            "claim": row.get("claim", ""),
            "label": row.get("label", ""),
            "explain": row.get("explain", ""),
            "candidates": candidates,
            "prompt": result["prompt"],
            "target": result["target"],
            "gold_label": gold_label,
            "gold_id": LABEL2ID.get(gold_label, -1),
            "gold_explain": str(row.get("explain", "")).strip(),
            "prompt_add_special_tokens": False,
            "preserve_prompt_prefix": True,
            "prompt_token_count": result["prompt_token_count"],
            "target_token_count": result["target_token_count"],
            "evidence_count": result["evidence_count"],
            "evidence_count_before": result["evidence_count_before"],
            "was_truncated": result["was_truncated"],
            "evidence_text_truncated": result["evidence_text_truncated"],
        }

    # auto_length disabled: build prompt without truncation
    system_msg = _build_system_message(system_prompt)
    target = _build_target(row, gold_label, output_mode, label_format)
    target_token_count = _count_target_tokens(target, tokenizer)
    user_content = _build_user_content(str(row.get("claim", "")), evidence_texts, output_mode, label_format)
    prompt = _build_chat_prompt(tokenizer, system_msg, user_content)
    prompt_token_count = _count_tokens(prompt, tokenizer, add_special_tokens=False)
    no_evidence = len(evidence_texts) == 0

    return {
        "event_id": row.get("event_id", ""),
        "claim": row.get("claim", ""),
        "label": row.get("label", ""),
        "explain": row.get("explain", ""),
        "candidates": candidates,
        "prompt": prompt,
        "target": target,
        "gold_label": gold_label,
        "gold_id": LABEL2ID.get(gold_label, -1),
        "gold_explain": str(row.get("explain", "")).strip(),
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "evidence_count": len(evidence_texts),
        "was_truncated": False,
    }


# ---------------------------------------------------------------------------
# Pre-MMR phase: embedding + caching
# ---------------------------------------------------------------------------


def _compute_pre_mmr_batch(
    samples: list,
    embedder: TextEmbedder,
) -> list[PreMMRSample]:
    """Batch-embed all sentences and claims; return PreMMRSample list.

    Mirrors the batching logic in _build_candidates_batch() but stops after embedding.
    """
    all_sent_texts: list[str] = []
    sample_boundaries: list[tuple[int, int]] = []
    claims: list[str] = []
    per_sample: list[tuple[list, list[str]]] = []

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

    results: list[PreMMRSample] = []
    for i in range(len(samples)):
        sample = samples[i]
        sents, _sent_texts = per_sample[i]
        start, end = sample_boundaries[i]
        sent_emb = all_sent_emb[start:end].copy() if end > start else np.zeros((0,), dtype=np.float32)
        claim_emb = all_claim_emb[i].copy()
        results.append(PreMMRSample(
            event_id=sample.event_id,
            claim=sample.claim,
            label=sample.label,
            explain=sample.explain,
            sentences=[_sentence_to_dict(s) for s in sents],
            sent_emb=sent_emb,
            claim_emb=claim_emb,
        ))
    return results


def _premmr_worker(
    gpu_id: int,
    run_summary: dict[str, Any],
    data_cfg: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    """GPU worker: embed its chunk of the split, write PreMMRSample pickle."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    samples = load_split(data_cfg[f"{split_name}_path"])
    num_gpus = run_summary["num_gpus"]
    chunk_size = (len(samples) + num_gpus - 1) // num_gpus
    samples_chunk = samples[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
    if not samples_chunk:
        _save_pickle_atomic(output_path, [])
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

    prefetch_size = run_summary["prefetch_size"]
    results: list[PreMMRSample] = []
    if prefetch_size > 1:
        for start in tqdm(range(0, len(samples_chunk), prefetch_size),
                          desc=f"PreMMR [{split_name}] GPU {gpu_id}",
                          unit="batch"):
            batch = samples_chunk[start : start + prefetch_size]
            results.extend(_compute_pre_mmr_batch(batch, embedder))
    else:
        for sample in tqdm(samples_chunk,
                           desc=f"PreMMR [{split_name}] GPU {gpu_id}"):
            results.extend(_compute_pre_mmr_batch([sample], embedder))

    _save_pickle_atomic(output_path, results)


def _compute_pre_mmr_split(
    split_name: str,
    data_cfg: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    run_summary: dict[str, Any],
    cache_dir: Path,
    num_gpus: int,
) -> Path:
    """Ensure pre-MMR cache exists for one split; return path to cached pickle."""
    cache_path = cache_dir / f"{split_name}.pkl"
    if cache_path.exists():
        return cache_path

    if num_gpus > 1:
        ctx = multiprocessing.get_context("fork")
        chunk_paths: list[Path] = []
        workers: list[multiprocessing.Process] = []
        for gpu_id in range(num_gpus):
            chunk_path = cache_dir / f"{split_name}_gpu{gpu_id}.pkl"
            chunk_paths.append(chunk_path)
            p = ctx.Process(
                target=_premmr_worker,
                args=(gpu_id, run_summary, data_cfg, split_name, chunk_path),
            )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"PreMMR worker failed with exit code {p.exitcode}")

        # Merge worker chunks
        all_results: list[PreMMRSample] = []
        for chunk_path in chunk_paths:
            all_results.extend(_load_pickle(chunk_path))
            chunk_path.unlink()
        _save_pickle_atomic(cache_path, all_results)
    else:
        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=run_summary["embedder_model"],
                device=run_summary.get("device", "cuda"),
                max_length=run_summary["max_length"],
                batch_size=run_summary["batch_size"],
                precision=run_summary["precision"],
            )
        )
        samples = load_split(data_cfg[f"{split_name}_path"])
        prefetch_size = run_summary["prefetch_size"]
        results: list[PreMMRSample] = []
        if prefetch_size > 1:
            for start in tqdm(range(0, len(samples), prefetch_size),
                              desc=f"PreMMR [{split_name}]",
                              unit="batch"):
                batch = samples[start : start + prefetch_size]
                results.extend(_compute_pre_mmr_batch(batch, embedder))
        else:
            for sample in tqdm(samples, desc=f"PreMMR [{split_name}]"):
                results.extend(_compute_pre_mmr_batch([sample], embedder))
        _save_pickle_atomic(cache_path, results)

    return cache_path


# ---------------------------------------------------------------------------
# Chunk-MMR cache phase: chunk construction + chunk-text embeddings
# ---------------------------------------------------------------------------


def _chunk_mmr_worker(
    gpu_id: int,
    run_summary: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    pre_samples: list[PreMMRSample],
    output_path: Path,
) -> None:
    """GPU worker: build chunk candidates, re-embed chunk texts, write ChunkMMRSamples."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if not pre_samples:
        _save_pickle_atomic(output_path, [])
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
    strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
    reuse_chunk_embeddings = _can_reuse_chunk_embeddings(strategy, retrieval_cfg)
    prefetch_size = run_summary["prefetch_size"]

    results: list[ChunkMMRSample] = []
    if prefetch_size > 1:
        for start in tqdm(range(0, len(pre_samples), prefetch_size),
                          desc=f"ChunkMMR GPU {gpu_id}",
                          unit="batch"):
            batch = pre_samples[start : start + prefetch_size]
            results.extend(_compute_chunk_mmr_batch(batch, embedder, strategy, reuse_chunk_embeddings))
    else:
        for pre in tqdm(pre_samples, desc=f"ChunkMMR GPU {gpu_id}"):
            results.extend(_compute_chunk_mmr_batch([pre], embedder, strategy, reuse_chunk_embeddings))

    _save_pickle_atomic(output_path, results)


def _compute_chunk_mmr_split(
    split_name: str,
    retrieval_cfg: dict[str, Any],
    run_summary: dict[str, Any],
    pre_mmr_path: Path,
    cache_dir: Path,
    num_gpus: int,
) -> Path:
    """Ensure chunk-level MMR cache exists for one split; return cached pickle path."""
    cache_path = cache_dir / f"{split_name}.pkl"
    if cache_path.exists():
        return cache_path

    pre_samples = _load_pickle(pre_mmr_path)
    if num_gpus > 1:
        ctx = multiprocessing.get_context("fork")
        chunk_size = (len(pre_samples) + num_gpus - 1) // num_gpus
        chunk_paths: list[Path] = []
        workers: list[multiprocessing.Process] = []
        for gpu_id in range(num_gpus):
            chunk_path = cache_dir / f"{split_name}_gpu{gpu_id}.pkl"
            chunk_paths.append(chunk_path)
            pre_chunk = pre_samples[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
            p = ctx.Process(
                target=_chunk_mmr_worker,
                args=(gpu_id, run_summary, retrieval_cfg, pre_chunk, chunk_path),
            )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"ChunkMMR worker failed with exit code {p.exitcode}")

        all_results: list[ChunkMMRSample] = []
        for chunk_path in chunk_paths:
            all_results.extend(_load_pickle(chunk_path))
            chunk_path.unlink()
        _save_pickle_atomic(cache_path, all_results)
    else:
        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=run_summary["embedder_model"],
                device=run_summary.get("device", "cuda"),
                max_length=run_summary["max_length"],
                batch_size=run_summary["batch_size"],
                precision=run_summary["precision"],
            )
        )
        strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
        reuse_chunk_embeddings = _can_reuse_chunk_embeddings(strategy, retrieval_cfg)
        prefetch_size = run_summary["prefetch_size"]
        results: list[ChunkMMRSample] = []
        if prefetch_size > 1:
            for start in tqdm(range(0, len(pre_samples), prefetch_size),
                              desc=f"ChunkMMR [{split_name}]",
                              unit="batch"):
                batch = pre_samples[start : start + prefetch_size]
                results.extend(_compute_chunk_mmr_batch(batch, embedder, strategy, reuse_chunk_embeddings))
        else:
            for pre in tqdm(pre_samples, desc=f"ChunkMMR [{split_name}]"):
                results.extend(_compute_chunk_mmr_batch([pre], embedder, strategy, reuse_chunk_embeddings))
        _save_pickle_atomic(cache_path, results)

    return cache_path


# ---------------------------------------------------------------------------
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
                    training_row = _build_training_row(row, tokenizer, prompt_cfg)
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
                training_row = _build_training_row(row, tokenizer, prompt_cfg)
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
            training_row = _build_training_row(row, tokenizer, prompt_cfg)
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
        # Prompt config for workers
        "prompt_model_name_or_path": str(prompt_cfg.get("model_name_or_path", "")),
        "prompt_auto_length": bool(prompt_cfg.get("auto_length", True)),
        "prompt_max_length": int(prompt_cfg.get("max_length", 2048)),
        "prompt_output_mode": str(prompt_cfg.get("output_mode", "label_only")).strip().lower(),
        "prompt_label_format": str(prompt_cfg.get("label_format", "name")).strip().lower(),
        "prompt_system_prompt": prompt_cfg.get("system_prompt") or None,
    }

    split_names = [split] if split else ["train", "val", "test"]
    split_paths: dict[str, Path] = {}

    # ---- Phase 1: pre-MMR cache (GPU embedding, shared across mmr_lambda values) ----
    premmr_fp = _premmr_config_fingerprint(cfg)
    premmr_cache_dir = Path("outputs/cache/pre_mmr") / premmr_fp
    logger.info("Pre-MMR cache dir: %s (fp=%s)", premmr_cache_dir, premmr_fp)

    premmr_summary = {
        "embedder_model": run_summary["embedder_model"],
        "max_length": run_summary["max_length"],
        "batch_size": run_summary["batch_size"],
        "precision": run_summary["precision"],
        "prefetch_size": run_summary["prefetch_size"],
        "num_gpus": num_gpus,
        "device": run_summary["device"],
    }

    pre_mmr_split_paths: dict[str, Path] = {}
    for split_name in split_names:
        pre_mmr_split_paths[split_name] = _compute_pre_mmr_split(
            split_name=split_name,
            data_cfg=data_cfg,
            retrieval_cfg=retrieval_cfg,
            run_summary=premmr_summary,
            cache_dir=premmr_cache_dir,
            num_gpus=num_gpus,
        )

    # ---- Phase 2: chunk cache (GPU embedding, shared across top_k/mmr_lambda values) ----
    logger.info("Build run summary: %s", run_summary)
    chunk_mmr_fp = _chunk_mmr_config_fingerprint(cfg)
    chunk_mmr_cache_dir = Path("outputs/cache/chunk_mmr") / chunk_mmr_fp
    logger.info("Chunk-MMR cache dir: %s (fp=%s)", chunk_mmr_cache_dir, chunk_mmr_fp)
    chunk_mmr_summary = {
        "embedder_model": run_summary["embedder_model"],
        "max_length": run_summary["max_length"],
        "batch_size": run_summary["batch_size"],
        "precision": run_summary["precision"],
        "prefetch_size": run_summary["prefetch_size"],
        "num_gpus": num_gpus,
        "device": run_summary["device"],
    }
    chunk_mmr_split_paths: dict[str, Path] = {}
    for split_name in split_names:
        chunk_mmr_split_paths[split_name] = _compute_chunk_mmr_split(
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

    tokenizer = _load_prompt_tokenizer(run_summary["prompt_model_name_or_path"])
    prompt_cfg_local = {
        "auto_length": run_summary["prompt_auto_length"],
        "max_length": run_summary["prompt_max_length"],
        "output_mode": run_summary["prompt_output_mode"],
        "label_format": run_summary["prompt_label_format"],
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

    # ---- Optional: learned λ predictor (MMR path only) ----
    learned_lambda_cfg = retrieval_cfg.get("learned_lambda", {}) or {}
    use_learned_lambda = bool(learned_lambda_cfg.get("enabled", False))
    learned_lambda_mode = str(learned_lambda_cfg.get("mode", "predictor")).strip().lower()

    for split_name in split_names:
        chunk_samples = _load_pickle(chunk_mmr_split_paths[split_name])
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
                    training_row = _build_training_row(row, tokenizer, prompt_cfg_local)
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
                    training_row = _build_training_row(row, tokenizer, prompt_cfg_local)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                cross_encoder_pbar.close()

            if dump_trace:
                trace_path = target_dir / f"cross_encoder_selector_trace_{split_name}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace_writer:
                    for trace in trace_rows:
                        trace_writer.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Wrote cross-encoder selector trace: %s", trace_path)
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
                            training_row = _build_training_row(row, tokenizer, prompt_cfg_local)
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
                        pre_samples = _load_pickle(pre_mmr_split_paths[split_name])
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
        _generate_prompt_stats(
            train_path=split_paths["train"],
            val_path=split_paths["val"],
            output_dir=target_dir,
            max_length=run_summary["prompt_max_length"],
            logger=logger,
        )

    return BuildResult(output_dir=target_dir, split_paths=split_paths)


def _generate_prompt_stats(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    max_length: int,
    logger: Any,
) -> None:
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    train_samples = _rows_to_prepared_samples(train_rows)
    val_samples = _rows_to_prepared_samples(val_rows)

    train_summary = summarize_prebuilt_prompts(train_samples, max_length=max_length, split="train")
    val_summary = summarize_prebuilt_prompts(val_samples, max_length=max_length, split="val")

    train_snapshots = build_prompt_snapshots(train_samples, split="train")
    val_snapshots = build_prompt_snapshots(val_samples, split="val")

    log_prompt_summary(train_summary, logger)
    log_prompt_summary(val_summary, logger)

    save_prompt_statistics(
        output_dir,
        train_summary=train_summary,
        val_summary=val_summary,
        train_snapshots=train_snapshots,
        val_snapshots=val_snapshots,
    )
    logger.info("Saved prompt statistics to %s/prompt_stats/", output_dir)


def _rows_to_prepared_samples(rows: list[dict]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for row in rows:
        gold_label = str(row.get("gold_label", ""))
        if not gold_label:
            continue
        samples.append(
            PreparedSample(
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
                preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
                gold_id=int(row.get("gold_id", -1)),
                gold_label=gold_label,
                gold_explain=str(row.get("gold_explain", "")),
                prompt_token_count=int(row.get("prompt_token_count", 0)),
                target_token_count=int(row.get("target_token_count", 0)),
                evidence_count=int(row.get("evidence_count", 0)),
                was_truncated=bool(row.get("was_truncated", False)),
                claim=str(row.get("claim", "")),
                no_evidence=int(row.get("evidence_count", 0)) == 0,
                long_claim=len(str(row.get("claim", "")).split()) > 64,
            )
        )
    return samples


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
