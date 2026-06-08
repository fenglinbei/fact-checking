#!/usr/bin/env python
"""Evaluate retrieval-signal ablations against full-pool oracle evidence.

This script measures whether a retrieval recipe places oracle evidence into a
fixed-size candidate pool.  It is intended for pre-selector retrieval analysis,
not for evaluating a downstream evidence-map selector.

Example:
    PYTHONPATH=src python scripts/phase3_oracle_evidence/evaluate_candidate_pool_recall.py \
        --oracle-results outputs/oracle_evidence/rawfc_qwen3_4b_2507_fullpool_margin/oracle_results_val.jsonl \
        --top-n 20 32 \
        --selection-mode mmr \
        --output outputs/analysis/retrieval_signal_ablation/qwen3_val_mmr.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.retrieval.text_utils import content_tokens


@dataclass(frozen=True)
class Variant:
    name: str
    weights: tuple[float, float, float] | None
    description: str


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant("dense_only", (1.0, 0.0, 0.0), "Dense similarity only"),
    Variant("lexical_only", (0.0, 1.0, 0.0), "Lexical overlap only"),
    Variant("bm25_like_only", (0.0, 0.0, 1.0), "BM25-like score only"),
    Variant(
        "dense_lexical",
        (0.70 / (0.70 + 0.20), 0.20 / (0.70 + 0.20), 0.0),
        "Dense + lexical, renormalized from the full recipe",
    ),
    Variant(
        "dense_bm25_like",
        (0.70 / (0.70 + 0.10), 0.0, 0.10 / (0.70 + 0.10)),
        "Dense + BM25-like, renormalized from the full recipe",
    ),
    Variant(
        "lexical_bm25_like",
        (0.0, 0.20 / (0.20 + 0.10), 0.10 / (0.20 + 0.10)),
        "Lexical + BM25-like, renormalized from the full recipe",
    ),
    Variant("equal_3way", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), "Equal weights"),
    Variant("full_hybrid_0_70_0_20_0_10", (0.70, 0.20, 0.10), "Main hybrid recipe"),
    Variant("stored_hybrid_score", None, "Hybrid score stored in the oracle file"),
)


_CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+(?:[A-Z][A-Za-z0-9&.'-]*))*\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def _minmax(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _score_array(candidate_scores: list[dict[str, Any]], variant: Variant) -> np.ndarray:
    if variant.weights is None:
        return np.asarray(
            [_safe_float(row.get("hybrid_score"), 0.0) for row in candidate_scores],
            dtype=np.float32,
        )

    dense = _minmax([_safe_float(row.get("dense_score"), 0.0) for row in candidate_scores])
    lexical = _minmax([_safe_float(row.get("lexical_score"), 0.0) for row in candidate_scores])
    bm25 = _minmax([_safe_float(row.get("bm25_score"), 0.0) for row in candidate_scores])
    w_dense, w_lexical, w_bm25 = variant.weights
    return (
        float(w_dense) * dense
        + float(w_lexical) * lexical
        + float(w_bm25) * bm25
    ).astype(np.float32, copy=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _candidate_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text") or "")


def _selected_indices_by_score(
    scores: np.ndarray,
    *,
    top_n: int,
) -> list[int]:
    if scores.size == 0 or top_n <= 0:
        return []
    # Stable tie-break by lower candidate index.
    order = sorted(range(int(scores.size)), key=lambda idx: (-float(scores[idx]), idx))
    return [int(idx) for idx in order[: min(int(top_n), len(order))]]


def _selected_indices_by_mmr(
    scores: np.ndarray,
    vectors: np.ndarray,
    *,
    top_n: int,
    mmr_lambda: float,
) -> list[int]:
    if scores.size == 0 or top_n <= 0:
        return []
    if vectors.shape[0] != scores.size:
        raise ValueError(
            f"MMR vector/score size mismatch: vectors={vectors.shape[0]} scores={scores.size}"
        )
    return maximal_marginal_relevance(
        query_scores=scores,
        sentence_vectors=vectors,
        top_k=min(int(top_n), int(scores.size)),
        lambda_weight=float(mmr_lambda),
    )


def _load_chunk_samples(records: list[dict[str, Any]], chunk_cache: Path | None) -> dict[str, Any]:
    cache_path = chunk_cache
    if cache_path is None:
        for record in records:
            metadata = record.get("candidate_pool_metadata") or {}
            path_value = metadata.get("chunk_mmr_cache_path")
            if path_value:
                cache_path = Path(path_value)
                break
    if cache_path is None:
        raise ValueError(
            "Chunk cache path is required for MMR/redundancy metrics. "
            "Pass --chunk-cache or use oracle rows with candidate_pool_metadata.chunk_mmr_cache_path."
        )
    if not cache_path.exists():
        raise FileNotFoundError(f"Chunk cache not found: {cache_path}")
    with cache_path.open("rb") as fh:
        samples = pickle.load(fh)
    return {str(sample.event_id): sample for sample in samples}


def _vectors_for_record(record: dict[str, Any], sample_by_event: dict[str, Any]) -> np.ndarray:
    event_id = str(record.get("event_id") or "")
    sample = sample_by_event.get(event_id)
    if sample is None:
        raise KeyError(f"Event {event_id!r} not found in chunk cache")
    pool = record.get("candidate_pool") or []
    source_indices = [_safe_int(candidate.get("source_index"), -1) for candidate in pool]
    if any(idx < 0 for idx in source_indices):
        raise ValueError(f"Record {event_id} has candidate without valid source_index")
    vectors = np.asarray(sample.chunk_emb[source_indices], dtype=np.float32)
    return vectors


def _pairwise_redundancy(vectors: np.ndarray, selected_indices: list[int]) -> float:
    if len(selected_indices) < 2:
        return 0.0
    selected = np.asarray(vectors[selected_indices], dtype=np.float32)
    sim = selected @ selected.T
    n = int(sim.shape[0])
    tri = np.triu_indices(n, k=1)
    return float(np.mean(sim[tri])) if tri[0].size else 0.0


def _content_token_coverage(claim: str, selected_texts: list[str]) -> float:
    query_tokens = set(content_tokens(claim))
    if not query_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for text in selected_texts:
        evidence_tokens.update(content_tokens(text))
    return float(len(query_tokens & evidence_tokens) / max(len(query_tokens), 1))


def _number_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if any(ch.isdigit() for ch in token)}


def _number_coverage(claim: str, selected_texts: list[str]) -> float:
    claim_numbers = _number_tokens(claim)
    if not claim_numbers:
        return 1.0
    evidence_numbers: set[str] = set()
    for text in selected_texts:
        evidence_numbers.update(_number_tokens(text))
    return float(len(claim_numbers & evidence_numbers) / max(len(claim_numbers), 1))


def _entity_like_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _CAPITALIZED_RE.finditer(text):
        normalized = " ".join(match.group(0).lower().split())
        if normalized and normalized not in {"i"}:
            terms.add(normalized)
    # Include all-caps or acronym-ish tokens that the capitalized span regex may
    # split awkwardly in punctuation-heavy political claims.
    for token in _TOKEN_RE.findall(text):
        if len(token) > 1 and token.upper() == token and any(ch.isalpha() for ch in token):
            terms.add(token.lower())
    return terms


def _entity_coverage(claim: str, selected_texts: list[str]) -> float:
    claim_entities = _entity_like_terms(claim)
    if not claim_entities:
        return 1.0
    joined_evidence = "\n".join(selected_texts)
    evidence_lower = joined_evidence.lower()
    covered = {entity for entity in claim_entities if entity in evidence_lower}
    return float(len(covered) / max(len(claim_entities), 1))


def _row_metrics(
    record: dict[str, Any],
    selected_indices: list[int],
    vectors: np.ndarray,
) -> dict[str, float]:
    oracle = {
        int(idx)
        for idx in (record.get("selected_indices") or [])
        if 0 <= _safe_int(idx, -1) < len(record.get("candidate_pool") or [])
    }
    selected = {int(idx) for idx in selected_indices}
    overlap = oracle & selected
    pool = record.get("candidate_pool") or []
    selected_texts = [_candidate_text(pool[idx]) for idx in selected_indices if 0 <= idx < len(pool)]
    selected_size = max(len(selected), 1)

    return {
        "oracle_count": float(len(oracle)),
        "selected_count": float(len(selected)),
        "overlap_count": float(len(overlap)),
        "oracle_recall": float(len(overlap) / max(len(oracle), 1)),
        "oracle_precision": float(len(overlap) / selected_size),
        "oracle_hit": float(bool(overlap)),
        "oracle_full_covered": float(bool(oracle) and oracle <= selected),
        "oracle_jaccard": float(len(overlap) / max(len(oracle | selected), 1)),
        "pairwise_redundancy": _pairwise_redundancy(vectors, selected_indices),
        "claim_content_token_coverage": _content_token_coverage(
            str(record.get("claim") or ""),
            selected_texts,
        ),
        "claim_number_coverage": _number_coverage(str(record.get("claim") or ""), selected_texts),
        "claim_entity_coverage": _entity_coverage(str(record.get("claim") or ""), selected_texts),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))


def evaluate(
    records: list[dict[str, Any]],
    *,
    top_ns: list[int],
    selection_mode: str,
    mmr_lambda: float,
    sample_by_event: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for variant in DEFAULT_VARIANTS:
        for top_n in top_ns:
            per_rows: list[dict[str, float]] = []
            for record in records:
                candidate_scores = record.get("candidate_scores") or []
                if not candidate_scores:
                    continue
                scores = _score_array(candidate_scores, variant)
                vectors = _vectors_for_record(record, sample_by_event)
                if selection_mode == "mmr":
                    selected = _selected_indices_by_mmr(
                        scores,
                        vectors,
                        top_n=top_n,
                        mmr_lambda=mmr_lambda,
                    )
                elif selection_mode == "rank":
                    selected = _selected_indices_by_score(scores, top_n=top_n)
                else:
                    raise ValueError(f"Unsupported selection mode: {selection_mode}")
                per_rows.append(_row_metrics(record, selected, vectors))

            summaries.append({
                "variant": variant.name,
                "description": variant.description,
                "top_n": int(top_n),
                "selection_mode": selection_mode,
                "mmr_lambda": float(mmr_lambda) if selection_mode == "mmr" else None,
                "n_claims": int(len(per_rows)),
                "oracle_count_mean": _mean(per_rows, "oracle_count"),
                "selected_count_mean": _mean(per_rows, "selected_count"),
                "overlap_count_mean": _mean(per_rows, "overlap_count"),
                f"oracle_recall@{top_n}": _mean(per_rows, "oracle_recall"),
                f"oracle_precision@{top_n}": _mean(per_rows, "oracle_precision"),
                f"oracle_hit@{top_n}": _mean(per_rows, "oracle_hit"),
                f"oracle_full_covered@{top_n}": _mean(per_rows, "oracle_full_covered"),
                f"oracle_jaccard@{top_n}": _mean(per_rows, "oracle_jaccard"),
                f"pairwise_redundancy@{top_n}": _mean(per_rows, "pairwise_redundancy"),
                f"claim_content_token_coverage@{top_n}": _mean(
                    per_rows,
                    "claim_content_token_coverage",
                ),
                f"claim_number_coverage@{top_n}": _mean(per_rows, "claim_number_coverage"),
                f"claim_entity_coverage@{top_n}": _mean(per_rows, "claim_entity_coverage"),
            })
    return summaries


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _print_compact_table(rows: list[dict[str, Any]], top_ns: list[int]) -> None:
    for top_n in top_ns:
        subset = [row for row in rows if int(row["top_n"]) == int(top_n)]
        recall_key = f"oracle_recall@{top_n}"
        hit_key = f"oracle_hit@{top_n}"
        redundancy_key = f"pairwise_redundancy@{top_n}"
        content_key = f"claim_content_token_coverage@{top_n}"
        number_key = f"claim_number_coverage@{top_n}"
        entity_key = f"claim_entity_coverage@{top_n}"
        print(f"\nCandidate pool recall @ {top_n}")
        print(
            "| variant | recall | hit | redundancy | content_cov | number_cov | entity_cov |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            print(
                "| {variant} | {recall} | {hit} | {redundancy} | {content} | {number} | {entity} |".format(
                    variant=row["variant"],
                    recall=_format_float(row.get(recall_key, 0.0)),
                    hit=_format_float(row.get(hit_key, 0.0)),
                    redundancy=_format_float(row.get(redundancy_key, 0.0)),
                    content=_format_float(row.get(content_key, 0.0)),
                    number=_format_float(row.get(number_key, 0.0)),
                    entity=_format_float(row.get(entity_key, 0.0)),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-pool recall for retrieval signal ablations.",
    )
    parser.add_argument("--oracle-results", required=True, type=Path)
    parser.add_argument(
        "--top-n",
        type=int,
        nargs="+",
        default=[20, 32],
        help="Candidate-pool sizes to evaluate.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["mmr", "rank"],
        default="mmr",
        help="How each variant constructs its candidate pool.",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.70)
    parser.add_argument(
        "--chunk-cache",
        type=Path,
        default=None,
        help="Override chunk-MMR cache path. By default it is read from oracle metadata.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON summary.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Write flat CSV summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.oracle_results)
    if not records:
        raise ValueError(f"No oracle records found: {args.oracle_results}")
    top_ns = sorted({int(value) for value in args.top_n if int(value) > 0})
    if not top_ns:
        raise ValueError("--top-n must contain at least one positive value")

    sample_by_event = _load_chunk_samples(records, args.chunk_cache)
    rows = evaluate(
        records,
        top_ns=top_ns,
        selection_mode=str(args.selection_mode),
        mmr_lambda=float(args.mmr_lambda),
        sample_by_event=sample_by_event,
    )

    payload = {
        "oracle_results": str(args.oracle_results),
        "selection_mode": args.selection_mode,
        "mmr_lambda": float(args.mmr_lambda) if args.selection_mode == "mmr" else None,
        "top_n": top_ns,
        "n_records": len(records),
        "rows": rows,
    }
    if args.output:
        _write_json(args.output, payload)
    if args.csv_output:
        _write_csv(args.csv_output, rows)

    _print_compact_table(rows, top_ns)
    if args.output:
        print(f"\nJSON written to: {args.output}")
    if args.csv_output:
        print(f"CSV written to: {args.csv_output}")


if __name__ == "__main__":
    main()
