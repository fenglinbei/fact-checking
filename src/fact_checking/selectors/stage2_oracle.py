from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT = "432dfc970e75"
DEFAULT_CANDIDATE_POOL_SIZE = 15
DEFAULT_SELECTOR_TOP_K = 5


@dataclass(frozen=True)
class Stage2OracleExample:
    event_id: str
    claim: str
    gold_label: str
    candidates: list[dict[str, Any]]
    candidate_scores: list[dict[str, Any]]
    selected_indices: list[int]
    fingerprint: str
    margin: float
    is_correct: bool
    raw: dict[str, Any]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_stage2_oracle_examples(
    path: str | Path,
    *,
    expected_fingerprint: str = EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
    filter_policy: str = "all",
    min_margin: float = 0.0,
    sample_limit: int | None = None,
) -> list[Stage2OracleExample]:
    records = read_jsonl(path)
    if sample_limit is not None:
        records = records[: int(sample_limit)]

    examples: list[Stage2OracleExample] = []
    for record in records:
        example = audit_stage2_oracle_record(
            record,
            expected_fingerprint=expected_fingerprint,
            max_candidates=max_candidates,
            top_k=top_k,
        )
        if _filter_passes(example, filter_policy, min_margin=min_margin):
            examples.append(example)
    return examples


def audit_stage2_oracle_record(
    record: dict[str, Any],
    *,
    expected_fingerprint: str = EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
) -> Stage2OracleExample:
    event_id = str(record.get("event_id") or "")
    if not event_id:
        raise ValueError("Oracle record is missing event_id.")

    candidates = record.get("candidate_pool")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Oracle record {event_id} has no candidate_pool.")
    if len(candidates) > int(max_candidates):
        raise ValueError(
            f"Oracle record {event_id} candidate_pool is too large: "
            f"{len(candidates)} > {max_candidates}."
        )

    selected_indices = _parse_selected_indices(record)
    if not selected_indices:
        raise ValueError(f"Oracle record {event_id} has no selected_indices.")
    if len(selected_indices) > int(top_k):
        raise ValueError(
            f"Oracle record {event_id} selected_indices is too long: "
            f"{len(selected_indices)} > {top_k}."
        )
    out_of_range = [idx for idx in selected_indices if idx < 0 or idx >= len(candidates)]
    if out_of_range:
        raise ValueError(
            f"Oracle record {event_id} selected_indices outside candidate_pool: {out_of_range}."
        )

    metadata = record.get("candidate_pool_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Oracle record {event_id} has no candidate_pool_metadata.")
    fingerprint = str(metadata.get("chunk_mmr_fingerprint") or "")
    if expected_fingerprint:
        if not fingerprint:
            raise ValueError(
                f"Oracle record {event_id} has no candidate_pool_metadata.chunk_mmr_fingerprint."
            )
        if fingerprint != expected_fingerprint:
            raise ValueError(
                f"Oracle record {event_id} fingerprint mismatch: "
                f"expected {expected_fingerprint}, got {fingerprint}."
            )

    objective = str(record.get("search_objective") or "")
    if objective and objective != "margin":
        raise ValueError(
            f"Oracle record {event_id} search_objective must be margin, got {objective}."
        )

    candidate_scores = _aligned_candidate_scores(record, len(candidates))
    return Stage2OracleExample(
        event_id=event_id,
        claim=str(record.get("claim") or ""),
        gold_label=str(record.get("gold_label") or ""),
        candidates=[dict(item) for item in candidates],
        candidate_scores=candidate_scores,
        selected_indices=selected_indices,
        fingerprint=fingerprint,
        margin=_safe_float(record.get("margin"), 0.0),
        is_correct=bool(record.get("is_correct")),
        raw=record,
    )


def candidate_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text") or "").strip()


def _parse_selected_indices(record: dict[str, Any]) -> list[int]:
    selected: list[int] = []
    for raw in record.get("selected_indices") or []:
        try:
            selected.append(int(raw))
        except (TypeError, ValueError):
            continue
    return selected


def _aligned_candidate_scores(record: dict[str, Any], n_candidates: int) -> list[dict[str, Any]]:
    score_by_idx: dict[int, dict[str, Any]] = {}
    for raw in record.get("candidate_scores") or []:
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("candidate_idx"))
        except (TypeError, ValueError):
            continue
        score_by_idx[idx] = dict(raw)

    rows: list[dict[str, Any]] = []
    for idx in range(n_candidates):
        row = dict(score_by_idx.get(idx) or {})
        row.setdefault("candidate_idx", idx)
        row.setdefault("hybrid_rank", idx)
        rows.append(row)
    return rows


def _filter_passes(example: Stage2OracleExample, filter_policy: str, *, min_margin: float) -> bool:
    policy = str(filter_policy).strip().lower()
    if policy == "all":
        return True
    if policy == "is_correct":
        return bool(example.is_correct)
    if policy == "margin_positive":
        return example.margin > 0.0
    if policy == "high_margin":
        return example.margin >= float(min_margin)
    raise ValueError(
        "Unknown filter_policy: "
        f"{filter_policy}. Expected one of all, is_correct, margin_positive, high_margin."
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

