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
from sft.data.labels import normalize_gold_label


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


class _ClaimProxy:
    __slots__ = ("claim", "event_id", "label", "explain")

    def __init__(self, pre: PreMMRSample):
        self.claim = pre.claim
        self.event_id = pre.event_id
        self.label = pre.label
        self.explain = pre.explain


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
    reuse_chunk_embeddings: bool = False,
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
    content_embeddings = _chunk_embeddings_by_content(sents, sent_emb) if reuse_chunk_embeddings else {}
    deduped_by_text: dict[str, dict[str, Any]] = {}
    for idx in keep_indices:
        sent = sents[idx]
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if content:
            if content not in content_splits:
                content_splits[content] = robust_sentence_split(content)
            embeddings_by_sent_idx = content_embeddings.get(content)
            if embeddings_by_sent_idx is None:
                evidence_text = strategy.chunk_from_presplit(content_splits[content], sent.sent_idx)
            else:
                evidence_text = strategy.chunk_from_presplit_with_embeddings(
                    content_splits[content],
                    sent.sent_idx,
                    embeddings_by_sent_idx,
                )
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
    chunking_strategy = build_chunking_strategy(chunking_cfg)
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
# MMR phase: CPU-only candidate construction from cached embeddings
# ---------------------------------------------------------------------------


def _mmr_phase_from_premmr(
    pre_samples: list[PreMMRSample],
    mmr_lambda: float,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    strategy,
    tokenizer,
    prompt_cfg: dict[str, Any],
    output_path: Path,
    cpu_workers: int = 1,
    reuse_chunk_embeddings: bool = False,
    lambda_overrides: dict[str, float] | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    """From cached PreMMRSamples, run MMR + candidates + training rows, write JSONL."""

    def _process_one(pre: PreMMRSample):
        sents = [_dict_to_sentence(d, event_id_fallback=pre.event_id) for d in pre.sentences]
        sent_texts = [d["text"] for d in pre.sentences]
        if not sents:
            return {
                "event_id": pre.event_id,
                "claim": pre.claim,
                "label": pre.label,
                "explain": pre.explain,
                "candidates": [],
            }
        effective_lambda = mmr_lambda
        if lambda_overrides is not None:
            effective_lambda = lambda_overrides.get(pre.event_id, mmr_lambda)
        return _process_sample_post_embed(
            sample=_ClaimProxy(pre),
            sents=sents,
            sent_texts=sent_texts,
            sent_emb=pre.sent_emb,
            claim_emb=pre.claim_emb,
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            mmr_lambda=effective_lambda,
            strategy=strategy,
            reuse_chunk_embeddings=reuse_chunk_embeddings,
        )

    with output_path.open("w", encoding="utf-8") as writer:
        desc = progress_desc or f"MMR λ={mmr_lambda:.2f}"
        if cpu_workers > 1:
            with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                rows = pool.map(_process_one, pre_samples)
                for row in tqdm(
                    rows,
                    total=len(pre_samples),
                    desc=desc,
                    unit="sample",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ):
                    training_row = _build_training_row(row, tokenizer, prompt_cfg)
                    writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
        else:
            for pre in tqdm(
                pre_samples,
                desc=desc,
                unit="sample",
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                row = _process_one(pre)
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

    # ---- Phase 2: MMR + candidate construction (CPU-only, fast) ----
    logger.info("Build run summary: %s", run_summary)
    tokenizer = _load_prompt_tokenizer(run_summary["prompt_model_name_or_path"])
    prompt_cfg_local = {
        "auto_length": run_summary["prompt_auto_length"],
        "max_length": run_summary["prompt_max_length"],
        "output_mode": run_summary["prompt_output_mode"],
        "label_format": run_summary["prompt_label_format"],
        "system_prompt": run_summary.get("prompt_system_prompt"),
    }
    chunking_strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
    reuse_chunk_embeddings = _can_reuse_chunk_embeddings(chunking_strategy, retrieval_cfg)
    logger.info("Chunking strategy: %s", type(chunking_strategy).__name__)
    logger.info("Reuse pre-MMR embeddings for chunking: %s", reuse_chunk_embeddings)

    # ---- Optional: learned λ predictor ----
    learned_lambda_cfg = retrieval_cfg.get("learned_lambda", {}) or {}
    use_learned_lambda = bool(learned_lambda_cfg.get("enabled", False))

    for split_name in split_names:
        pre_samples = _load_pickle(pre_mmr_split_paths[split_name])

        lambda_overrides: dict[str, float] | None = None
        if use_learned_lambda:
            from fact_checking.learned_lambda.predictor import load_predictor, predict_lambdas_for_samples
            model_path = str(learned_lambda_cfg["model_path"])
            stats_path = str(learned_lambda_cfg["feature_stats_path"])
            predictor, stats = load_predictor(model_path, stats_path)
            lambda_overrides = predict_lambdas_for_samples(pre_samples, predictor, stats, retrieval_cfg)
            vals = list(lambda_overrides.values())
            logger.info(
                "Learned lambda: %d overrides, mean=%.3f, std=%.3f",
                len(vals), np.mean(vals), np.std(vals),
            )

        output_path = target_dir / f"build_{split_name}.jsonl"
        _mmr_phase_from_premmr(
            pre_samples=pre_samples,
            mmr_lambda=run_summary["mmr_lambda"],
            top_k=run_summary["top_k"],
            alpha_dense=run_summary["alpha_dense"],
            alpha_lexical=run_summary["alpha_lexical"],
            alpha_bm25=run_summary["alpha_bm25"],
            strategy=chunking_strategy,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg_local,
            output_path=output_path,
            cpu_workers=run_summary["cpu_workers"],
            reuse_chunk_embeddings=reuse_chunk_embeddings,
            lambda_overrides=lambda_overrides,
        )
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
