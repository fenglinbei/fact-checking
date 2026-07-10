#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SciFact open-corpus retrieval without changing candidate pools.")
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--atom-pool-jsonl", required=True)
    parser.add_argument("--atom-union-jsonl", required=True)
    parser.add_argument("--abc-coverage-jsonl", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_rows = _read_jsonl(Path(args.raw_path))
    raw_by_id = {_text_id(row.get("id")): row for row in raw_rows}
    baseline_rows = _read_jsonl(Path(args.baseline_jsonl))
    atom_rows = _read_jsonl(Path(args.atom_pool_jsonl))
    union_rows = _read_jsonl(Path(args.atom_union_jsonl))
    _validate_event_ids(raw_rows, baseline_rows, atom_rows, union_rows)

    metrics: dict[str, Any] = {
        "claim_route": _audit_candidate_rows(baseline_rows, raw_by_id, top_k=int(args.top_k)),
        "atom_route": _audit_candidate_rows(atom_rows, raw_by_id, top_k=int(args.top_k)),
        "atom_union": _audit_candidate_rows(union_rows, raw_by_id, top_k=int(args.top_k)),
    }
    if args.abc_coverage_jsonl:
        metrics["abc_retrieval_universe"] = _audit_abc_coverage(
            _read_jsonl(Path(args.abc_coverage_jsonl))
        )
    rows_with_gold = int(metrics["atom_union"]["rows_with_gold"])
    report = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": str(args.split),
        "top_k": int(args.top_k),
        "gold_available": rows_with_gold > 0,
        "gold_usage": "offline_audit_only",
        "candidate_construction_uses_gold": False,
        "paths": {
            "raw": str(args.raw_path),
            "baseline": str(args.baseline_jsonl),
            "atom_pool": str(args.atom_pool_jsonl),
            "atom_union": str(args.atom_union_jsonl),
            "abc_coverage": str(args.abc_coverage_jsonl or ""),
        },
        "metrics": metrics,
        "definitions": {
            "micro_gold_doc_recall": "gold evidence documents present in the candidate pool",
            "micro_gold_sentence_recall": "gold rationale sentences present in candidate chunk spans",
            "gold_doc_complete_rationale_rate": "gold documents with at least one complete official rationale set",
            "claim_full_complete_rationale_rate": "verifiable claims with a complete rationale set for every gold document",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote SciFact retrieval audit: {output}")
    return 0


def _audit_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    raw_by_id: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    pool_sizes: list[int] = []
    distinct_doc_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    duplicate_uid_rows = 0
    duplicate_text_rows = 0
    rows_with_atom_only = 0
    rows_with_gold = 0
    total_gold_docs = hit_gold_docs = 0
    total_gold_sentences = hit_gold_sentences = 0
    complete_gold_docs = 0
    claim_any_doc = claim_full_doc = 0
    claim_any_complete = claim_full_complete = 0

    for row in rows:
        candidates = list(row.get("candidates") or [])[: int(top_k)]
        pool_sizes.append(len(candidates))
        candidate_docs, candidate_sentences = _candidate_keys(candidates)
        distinct_doc_counts.append(len(candidate_docs))
        duplicate_uid_rows += int(_has_duplicates(candidates, "candidate_uid"))
        duplicate_text_rows += int(_has_duplicates(candidates, "canonical_text", fallback="text"))
        has_atom_only = False
        for candidate in candidates:
            from_baseline = bool(candidate.get("from_baseline"))
            from_atom = bool(candidate.get("from_atom_route"))
            source = "baseline+atom" if from_baseline and from_atom else "baseline" if from_baseline else "atom" if from_atom else "unknown"
            source_counts[source] += 1
            has_atom_only = has_atom_only or (from_atom and not from_baseline)
        rows_with_atom_only += int(has_atom_only)

        raw = raw_by_id.get(_text_id(row.get("event_id")))
        gold_docs = _gold_rationale_sets(raw or {})
        if not gold_docs:
            continue
        rows_with_gold += 1
        gold_doc_ids = set(gold_docs)
        hit_docs = gold_doc_ids & candidate_docs
        total_gold_docs += len(gold_doc_ids)
        hit_gold_docs += len(hit_docs)
        claim_any_doc += int(bool(hit_docs))
        claim_full_doc += int(hit_docs == gold_doc_ids)

        complete_by_doc: list[bool] = []
        for doc_id, rationale_sets in gold_docs.items():
            available = {sent_idx for candidate_doc, sent_idx in candidate_sentences if candidate_doc == doc_id}
            complete = any(rationale <= available for rationale in rationale_sets)
            complete_by_doc.append(complete)
            complete_gold_docs += int(complete)
        claim_any_complete += int(any(complete_by_doc))
        claim_full_complete += int(all(complete_by_doc))

        gold_sentence_keys = {
            (doc_id, sent_idx)
            for doc_id, rationale_sets in gold_docs.items()
            for rationale in rationale_sets
            for sent_idx in rationale
        }
        total_gold_sentences += len(gold_sentence_keys)
        hit_gold_sentences += len(gold_sentence_keys & candidate_sentences)

    return {
        "row_count": len(rows),
        "rows_with_gold": rows_with_gold,
        "candidate_count": _summary(pool_sizes),
        "distinct_doc_count": _summary(distinct_doc_counts),
        "micro_gold_doc_recall": _ratio(hit_gold_docs, total_gold_docs),
        "micro_gold_sentence_recall": _ratio(hit_gold_sentences, total_gold_sentences),
        "gold_doc_complete_rationale_rate": _ratio(complete_gold_docs, total_gold_docs),
        "claim_any_doc_rate": _ratio(claim_any_doc, rows_with_gold),
        "claim_full_doc_rate": _ratio(claim_full_doc, rows_with_gold),
        "claim_any_complete_rationale_rate": _ratio(claim_any_complete, rows_with_gold),
        "claim_full_complete_rationale_rate": _ratio(claim_full_complete, rows_with_gold),
        "source_counts": dict(sorted(source_counts.items())),
        "rows_with_atom_only": rows_with_atom_only,
        "duplicate_uid_rows": duplicate_uid_rows,
        "duplicate_text_rows": duplicate_text_rows,
    }


def _audit_abc_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_docs = sum(int(row.get("gold_doc_count") or 0) for row in rows)
    hit_docs = sum(int(row.get("hit_gold_doc_count") or 0) for row in rows)
    total_sentences = sum(int(row.get("gold_sentence_count") or 0) for row in rows)
    hit_sentences = sum(int(row.get("hit_gold_sentence_count") or 0) for row in rows)
    rows_with_gold = sum(int(row.get("gold_doc_count") or 0) > 0 for row in rows)
    return {
        "row_count": len(rows),
        "rows_with_gold": rows_with_gold,
        "micro_gold_doc_recall": _ratio(hit_docs, total_docs),
        "micro_gold_sentence_recall": _ratio(hit_sentences, total_sentences),
        "claim_full_doc_rate": _ratio(
            sum(
                int(row.get("gold_doc_count") or 0) > 0
                and int(row.get("hit_gold_doc_count") or 0) == int(row.get("gold_doc_count") or 0)
                for row in rows
            ),
            rows_with_gold,
        ),
        "candidate_count": _summary([int(row.get("n_candidates") or 0) for row in rows]),
    }


def _gold_rationale_sets(row: Mapping[str, Any]) -> dict[str, list[frozenset[int]]]:
    out: dict[str, list[frozenset[int]]] = {}
    for doc_id, rationales in (row.get("evidence") or {}).items():
        sets = [
            frozenset(int(sent_idx) for sent_idx in rationale.get("sentences") or [])
            for rationale in rationales or []
            if isinstance(rationale, Mapping) and rationale.get("sentences")
        ]
        if sets:
            out[str(doc_id)] = sets
    return out


def _candidate_keys(candidates: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[tuple[str, int]]]:
    docs: set[str] = set()
    sentences: set[tuple[str, int]] = set()
    for candidate in candidates:
        doc_id = str(candidate.get("scifact_doc_id") or candidate.get("doc_id") or candidate.get("report_id") or "")
        if not doc_id:
            continue
        docs.add(doc_id)
        for sent_idx in candidate.get("scifact_sentence_ids") or candidate.get("chunk_sent_indices") or []:
            sentences.add((doc_id, int(sent_idx)))
    return docs, sentences


def _has_duplicates(candidates: Sequence[Mapping[str, Any]], key: str, *, fallback: str | None = None) -> bool:
    values = [str(candidate.get(key) or (candidate.get(fallback) if fallback else "") or "").strip().lower() for candidate in candidates]
    values = [value for value in values if value]
    return len(values) != len(set(values))


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_event_ids(raw_rows: Sequence[Mapping[str, Any]], *candidate_sets: Sequence[Mapping[str, Any]]) -> None:
    raw_ids = {_text_id(row.get("id")) for row in raw_rows}
    for rows in candidate_sets:
        event_ids = [_text_id(row.get("event_id")) for row in rows]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Duplicate event_id in retrieval audit input.")
        missing = [event_id for event_id in event_ids if event_id not in raw_ids]
        if missing:
            raise ValueError(f"Candidate events missing from raw split: {missing[:5]}")


def _text_id(value: Any) -> str:
    return "" if value is None else str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
