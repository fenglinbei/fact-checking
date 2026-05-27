from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from fact_checking.build.candidates import canonicalize_sentence
from fact_checking.retrieval.text_utils import content_tokens, lexical_overlap_f1


ORIGINAL_POOL = "original_stage2_pool"
QD_UNION_POOL = "qd_union_pool"

_PREDICATE_HINTS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "claim",
    "did",
    "do",
    "does",
    "found",
    "had",
    "has",
    "have",
    "is",
    "made",
    "said",
    "say",
    "says",
    "show",
    "shows",
    "was",
    "were",
    "will",
}
_FRAGMENT_PATTERNS = (
    re.compile(r"skip to (content|navigation)", re.I),
    re.compile(r"update your browser", re.I),
    re.compile(r"^\W*\d+\W*$"),
    re.compile(r"^(home|menu|share|subscribe|advertisement)\W*$", re.I),
)


def canonical_candidate_key(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("canonical_text") or candidate.get("text") or "")
    return canonicalize_sentence(text)


def dedup_key_for_candidate(
    event_id: str,
    candidate: dict[str, Any],
    *,
    source_pool: str,
) -> str:
    chunk_idx = chunk_index_for_candidate(candidate, source_pool=source_pool)
    if chunk_idx is not None:
        return f"{event_id}|chunk:{chunk_idx}"
    key = canonical_candidate_key(candidate)
    return f"{event_id}|text:{key}"


def chunk_index_for_candidate(candidate: dict[str, Any], *, source_pool: str) -> int | None:
    if source_pool == ORIGINAL_POOL:
        return _nullable_int(candidate.get("source_index"))
    if source_pool == QD_UNION_POOL:
        return _first_int(candidate.get("original_candidate_idx"), candidate.get("source_index"))
    return _first_int(candidate.get("source_index"), candidate.get("original_candidate_idx"))


def stable_candidate_uid(event_id: str, dedup_key: str) -> str:
    payload = f"{event_id}|{dedup_key}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def source_domain(candidate: dict[str, Any]) -> str:
    source_report = candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {}
    raw = str(
        candidate.get("source_domain")
        or source_report.get("domain")
        or source_report.get("link")
        or candidate.get("source_link")
        or ""
    )
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.netloc or parsed.path or raw).lower().strip()


def source_group(candidate: dict[str, Any]) -> str:
    existing = str(candidate.get("source_group") or "").strip()
    if existing:
        return existing
    report_id = candidate.get("report_id")
    if report_id is not None and str(report_id) != "":
        return f"report:{report_id}"
    domain = source_domain(candidate)
    if domain:
        return f"domain:{domain}"
    uid = str(candidate.get("candidate_uid") or candidate.get("candidate_key") or "")
    return f"candidate:{uid}"


def question_route_count(candidate: dict[str, Any]) -> int:
    routes = list(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
    ids = {
        str(route.get("question_id") or "")
        for route in routes
        if isinstance(route, dict) and str(route.get("question_id") or "")
    }
    if ids:
        return len(ids)
    value = _nullable_int(candidate.get("qd_question_hit_count") or candidate.get("question_hit_count"))
    if value is not None and value > 0:
        return int(value)
    if bool(candidate.get("from_baseline")) and not bool(candidate.get("from_qd")):
        return 1
    return 0


def question_route_weight(candidate: dict[str, Any]) -> float:
    return min(1.0, 0.5 + 0.25 * float(question_route_count(candidate)))


def question_coverage_score(candidate: dict[str, Any]) -> float:
    route_count = question_route_count(candidate)
    return min(1.0, float(route_count) / 3.0)


def retrieval_score(candidate: dict[str, Any]) -> float:
    qd_rrf = _clip01(_safe_float(candidate.get("qd_rrf_score") or candidate.get("rrf_score"), 0.0) * 20.0)
    values = [
        candidate.get("baseline_hybrid_score"),
        candidate.get("hybrid_score"),
        candidate.get("qd_max_question_hybrid"),
        candidate.get("max_question_hybrid"),
        qd_rrf,
    ]
    return _clip01(max(_safe_float(value, 0.0) for value in values))


def relevance_gate_score(candidate: dict[str, Any], *, claim: str) -> float:
    return _clip01(0.7 * retrieval_score(candidate) + 0.3 * lexical_relevance(candidate, claim=claim))


def lexical_relevance(candidate: dict[str, Any], *, claim: str) -> float:
    return _clip01(lexical_overlap_f1(str(claim or ""), str(candidate.get("text") or "")))


def text_fragment_flags(text: str, *, claim: str = "") -> dict[str, bool]:
    text = str(text or "").strip()
    tokens = content_tokens(text)
    raw_tokens = re.findall(r"[A-Za-z0-9_'-]+", text)
    length_ok = 6 <= len(raw_tokens) <= 96 and len(text) >= 32
    token_counts = Counter(tok.lower() for tok in raw_tokens)
    has_predicate = bool(_PREDICATE_HINTS & set(token_counts))
    if not has_predicate:
        has_predicate = any(tok.endswith(("ed", "ing")) for tok in token_counts)
    sentence_boundary_ok = bool(text and (text[-1] in ".?!\"'" or len(raw_tokens) >= 10))
    fragment_pattern = any(pattern.search(text) for pattern in _FRAGMENT_PATTERNS)
    not_fragment = bool(length_ok and not fragment_pattern and len(tokens) >= 4)
    entity_or_keyword_overlap = lexical_overlap_f1(str(claim or ""), text) > 0.0
    lexical_or_embedding_relevance = entity_or_keyword_overlap
    return {
        "length_ok": bool(length_ok),
        "has_predicate": bool(has_predicate),
        "sentence_boundary_ok": bool(sentence_boundary_ok),
        "not_fragment": bool(not_fragment),
        "entity_or_keyword_overlap": bool(entity_or_keyword_overlap),
        "lexical_or_embedding_relevance": bool(lexical_or_embedding_relevance),
        "fragment_pattern": bool(fragment_pattern),
    }


def semantic_completeness_heuristic(candidate: dict[str, Any], *, claim: str) -> float:
    flags = text_fragment_flags(str(candidate.get("text") or ""), claim=claim)
    score = (
        0.25 * float(flags["length_ok"])
        + 0.20 * float(flags["has_predicate"])
        + 0.15 * float(flags["sentence_boundary_ok"])
        + 0.15 * float(flags["not_fragment"])
        + 0.15 * float(flags["entity_or_keyword_overlap"])
        + 0.10 * float(flags["lexical_or_embedding_relevance"])
    )
    return _clip01(score)


def enrich_quality_fields(
    candidate: dict[str, Any],
    *,
    claim: str,
    semantic_completeness_score: float | None = None,
    annotation_missing: bool = False,
) -> dict[str, Any]:
    item = dict(candidate)
    heuristic_score = semantic_completeness_heuristic(item, claim=claim)
    completeness = heuristic_score if semantic_completeness_score is None else _clip01(semantic_completeness_score)
    item["semantic_completeness_score"] = float(completeness)
    item["semantic_completeness_heuristic"] = float(heuristic_score)
    item["annotation_missing"] = bool(annotation_missing)
    item["retrieval_score"] = float(retrieval_score(item))
    item["claim_lexical_f1"] = float(lexical_relevance(item, claim=claim))
    item["relevance_gate_score"] = float(relevance_gate_score(item, claim=claim))
    item["text_fragment_flags"] = text_fragment_flags(str(item.get("text") or ""), claim=claim)
    item["question_route_count"] = int(question_route_count(item))
    item["question_route_weight"] = float(question_route_weight(item))
    item["question_coverage_score"] = float(question_coverage_score(item))
    item["source_domain"] = source_domain(item)
    item["source_group"] = source_group(item)
    return item


def quality_gate(candidate: dict[str, Any], *, tau_c: float = 0.50, tau_r: float = 0.15) -> bool:
    completeness = _safe_float(candidate.get("semantic_completeness_score"), 0.0)
    relevance = _safe_float(candidate.get("relevance_gate_score"), 0.0)
    return completeness >= float(tau_c) and relevance >= float(tau_r)


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _nullable_int(value)
        if parsed is not None:
            return parsed
    return None


def _nullable_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
