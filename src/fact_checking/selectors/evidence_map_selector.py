from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    COUNT_AMPLIFIED_SELECTOR,
    CountAmplifiedParams,
    build_selector_trace,
    select_count_amplified_topk,
    select_order_control,
    selection_quality_metrics,
    summarize_selector_traces,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.direct_evidence_cross_encoder import DIRECT_CE_TEXT_ONLY_SELECTOR
from fact_checking.selectors.direct_evidence_fusion_selector import FUSION_REFIT_SELECTOR
from fact_checking.selectors.evidence_quality import retrieval_score, source_group
from fact_checking.selectors.oracle_likelihood_constrained_selector import ORACLE_LIKELIHOOD_SELECTOR


EVIDENCE_MAP_SELECTOR = "v0_5a_evidence_map_top5"
EVIDENCE_MAP_BASE_ONLY_SELECTOR = "v0_5a_base_only_top5"
EVIDENCE_MAP_COVERAGE_ONLY_SELECTOR = "v0_5a_coverage_only_top5"
PROMPT_VERSION = "evidence_map_v0_5a"
COMPACT_PROMPT_VERSION = "evidence_map_v0_6b"
ATOM_FACTS_PROMPT_VERSION = "evidence_map_v0_7_atom_facts"
DEFAULT_MAX_EVIDENCE_CHARS = 700

ALLOWED_RELATIONS = {"support", "refute", "qualify", "mixed", "background", "irrelevant"}
ALLOWED_DIRECTNESS = {"direct", "partial", "context", "none"}
ALLOWED_ROLES = {
    "primary_support",
    "primary_refute",
    "partial_support",
    "partial_refute",
    "qualifying_context",
    "background_context",
    "duplicate",
    "irrelevant",
}
FORBIDDEN_PROMPT_FIELDS = {
    "gold_label",
    "oracle_selected",
    "oracle_step",
    "event_id",
    "candidate_key",
    "candidate_uid",
    "baseline_rank",
    "qd_pool_rank",
    "union_pool_rank",
    "source_group",
    "source_report",
    "retrieval_score",
    "oracle_likelihood_score",
    "direct_ce_score",
    "fusion_refit_score",
}


@dataclass(frozen=True)
class EvidenceMapParams:
    top_k: int = 5
    base_weight: float = 1.0
    atom_coverage_weight: float = 0.35
    directness_weight: float = 0.20
    polar_relation_weight: float = 0.10
    duplicate_penalty: float = 0.15
    source_penalty: float = 0.08
    background_penalty: float = 0.20


class EvidenceMapSchemaError(ValueError):
    pass


def evidence_map_annotation_key(
    *,
    event_id: str,
    prompt_version: str,
    model: str,
    evidence_fingerprint: str = "",
) -> str:
    payload = f"{event_id}|{prompt_version}|{model}"
    if evidence_fingerprint:
        payload = f"{payload}|{evidence_fingerprint}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def evidence_items_fingerprint(evidence_items: Sequence[dict[str, Any]]) -> str:
    payload = [
        [
            str(item.get("evidence_id") or ""),
            str(item.get("text") or ""),
        ]
        for item in evidence_items
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def prepare_evidence_map_candidate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    candidate_top_n: int,
    candidate_source: str = "fusion",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source = str(candidate_source or "fusion")
        candidates = [
            _normalize_candidate_for_map(candidate, event_id=str(row.get("event_id") or ""), fallback_idx=idx, candidate_source=source)
            for idx, candidate in enumerate(row.get("candidates") or [], start=1)
        ]
        candidates = _top_annotation_candidates(candidates, candidate_top_n=candidate_top_n, candidate_source=source)
        evidence_items: list[dict[str, Any]] = []
        trimmed_candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates, start=1):
            evidence_id = f"E{idx:02d}"
            item = dict(candidate)
            item["evidence_id"] = evidence_id
            trimmed_candidates.append(item)
            evidence_items.append(
                {
                    "evidence_id": evidence_id,
                    "candidate_uid": str(candidate.get("candidate_uid") or ""),
                    "candidate_key": str(candidate.get("candidate_key") or ""),
                    "text": str(candidate.get("text") or ""),
                }
            )
        out.append(
            {
                "event_id": str(row.get("event_id") or ""),
                "claim": str(row.get("claim") or ""),
                "label": str(row.get("label") or ""),
                "gold_label": str(row.get("gold_label") or row.get("label") or ""),
                "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
                "oracle_selected_count": int(row.get("oracle_selected_count") or len(row.get("oracle_ordered_keys") or [])),
                "candidate_top_n": int(candidate_top_n),
                "evidence_map_candidate_source": source,
                "evidence_items_fingerprint": evidence_items_fingerprint(evidence_items),
                "evidence_items": evidence_items,
                "candidates": trimmed_candidates,
                "n_stance_buckets": int(row.get("n_stance_buckets") or 7),
                "stance_bucket_names": list(row.get("stance_bucket_names") or []),
            }
        )
    return out


def build_teacher_messages(
    row: dict[str, Any],
    *,
    prompt_version: str = PROMPT_VERSION,
    max_evidence_chars: int | None = None,
) -> tuple[str, str]:
    prompt_version = str(prompt_version or PROMPT_VERSION)
    if prompt_version == COMPACT_PROMPT_VERSION:
        system_prompt, user_prompt = _build_compact_teacher_messages(
            row,
            max_evidence_chars=max_evidence_chars,
        )
        audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)
        return system_prompt, user_prompt
    if prompt_version == ATOM_FACTS_PROMPT_VERSION:
        system_prompt, user_prompt = _build_atom_facts_teacher_messages(
            row,
            max_evidence_chars=max_evidence_chars,
        )
        audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)
        return system_prompt, user_prompt

    system_prompt = (
        "You are a careful fact-checking evidence analyst. Build a compact evidence map "
        "from the claim and evidence passages only. Do not use outside knowledge. "
        "Return strictly valid JSON."
    )
    lines = [
        "Task: decompose the claim into atomic verifiable facts, then map each evidence passage to those atoms.",
        "",
        "Return JSON with this schema:",
        '{"claim_atoms":[{"atom_id":"A1","text":"...","type":"entity|quantity|date|comparison|cause|outcome|other","importance":1.0}],',
        '"candidate_alignments":[{"evidence_id":"E01","covered_atom_ids":["A1"],"relation":"support|refute|qualify|mixed|background|irrelevant","directness":"direct|partial|context|none","evidence_role":"primary_support|primary_refute|partial_support|partial_refute|qualifying_context|background_context|duplicate|irrelevant","key_spans":["short quote"],"duplicate_group":"G1","confidence":0.0}]}',
        "",
        "Guidelines:",
        "- Prefer atoms for key entities, quantities, dates, comparisons, causes, and outcomes.",
        "- Mark background-only passages as relation=background and directness=context or none.",
        "- Mark near duplicates with the same duplicate_group.",
        "- key_spans must be short substrings copied from the evidence passage.",
        "- Use only the evidence IDs shown below.",
        "",
        f"Claim:\n{str(row.get('claim') or '').strip()}",
        "",
        "Evidence passages:",
    ]
    for item in row.get("evidence_items") or []:
        evidence_id = str(item.get("evidence_id") or "")
        text = _compact_evidence_text(str(item.get("text") or ""), max_evidence_chars=max_evidence_chars)
        lines.append(f"{evidence_id}: {text}")
    user_prompt = "\n".join(lines)
    audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)
    return system_prompt, user_prompt


def _build_compact_teacher_messages(
    row: dict[str, Any],
    *,
    max_evidence_chars: int | None,
) -> tuple[str, str]:
    system_prompt = (
        "You are a fact-checking evidence analyst. Use only the claim and the "
        "provided evidence passages to map evidence to atomic claim facts. "
        "Return valid JSON only."
    )
    schema = (
        '{"claim_atoms":[{"atom_id":"A1","text":"...","type":"entity|quantity|date|comparison|cause|outcome|other","importance":1.0}],'
        '"candidate_alignments":[{"evidence_id":"E01","covered_atom_ids":["A1"],"relation":"support|refute|qualify|mixed|background|irrelevant",'
        '"directness":"direct|partial|context|none","evidence_role":"primary_support|primary_refute|partial_support|partial_refute|qualifying_context|background_context|duplicate|irrelevant",'
        '"key_spans":["short quote"],"duplicate_group":"G1","confidence":0.0}]}'
    )
    lines = [
        "Task: read the claim and the evidence passages, then return one JSON object.",
        "The JSON object must contain exactly two arrays: claim_atoms and candidate_alignments.",
        "Use this schema and do not add other fields:",
        schema,
        "",
        "Field guide:",
        "- claim_atoms.atom_id: A1, A2, ... in claim order.",
        "- claim_atoms.text: one atomic fact from the claim.",
        "- claim_atoms.type: entity, quantity, date, comparison, cause, outcome, or other.",
        "- claim_atoms.importance: 0.0 to 1.0; higher means more central to checking the claim.",
        "- candidate_alignments.evidence_id: one of the evidence IDs below.",
        "- candidate_alignments.covered_atom_ids: claim atoms addressed by the passage.",
        "- candidate_alignments.relation: support, refute, qualify, mixed, background, or irrelevant.",
        "- candidate_alignments.directness: direct, partial, context, or none.",
        "- candidate_alignments.evidence_role: primary_support, primary_refute, partial_support, partial_refute, qualifying_context, background_context, duplicate, or irrelevant.",
        "- candidate_alignments.key_spans: short substrings copied from the passage.",
        "- candidate_alignments.duplicate_group: same group ID for near-duplicate passages; use an empty string if none.",
        "- candidate_alignments.confidence: 0.0 to 1.0 for how certain the alignment is.",
        "",
        "Guidelines:",
        "- Keep atoms small and ordered by their appearance in the claim.",
        "- Mark weak background as relation=background and directness=context or none.",
        "- Use only the evidence IDs shown below.",
        "",
        "Claim:",
        str(row.get("claim") or "").strip(),
        "",
        "Evidence passages:",
    ]
    for item in row.get("evidence_items") or []:
        evidence_id = str(item.get("evidence_id") or "")
        text = _compact_evidence_text(str(item.get("text") or ""), max_evidence_chars=max_evidence_chars)
        lines.append(f"{evidence_id}: {text}")
    return system_prompt, "\n".join(lines)


def _build_atom_facts_teacher_messages(
    row: dict[str, Any],
    *,
    max_evidence_chars: int | None,
) -> tuple[str, str]:
    system_prompt = (
        "You are a fact-checking evidence analyst. Use only the claim and the "
        "provided evidence passages to map evidence to complete proposition atoms. "
        "Return valid JSON only."
    )
    schema = (
        '{"claim_atoms":[{"atom_id":"A1","text":"...","type":"entity|quantity|date|comparison|cause|outcome|other","importance":1.0}],'
        '"candidate_alignments":[{"evidence_id":"E01","covered_atom_ids":["A1"],"relation":"support|refute|qualify|mixed|background|irrelevant",'
        '"directness":"direct|partial|context|none","evidence_role":"primary_support|primary_refute|partial_support|partial_refute|qualifying_context|background_context|duplicate|irrelevant",'
        '"key_spans":["short quote"],"duplicate_group":"G1","confidence":0.0}]}'
    )
    lines = [
        "Task: read the claim and the evidence passages, then return one JSON object.",
        "The JSON object must contain exactly two arrays: claim_atoms and candidate_alignments.",
        "Use this schema and do not add other fields:",
        schema,
        "",
        "Atom policy:",
        "- Each claim_atoms.text must be a complete proposition: include the subject, predicate, object, and necessary qualifiers.",
        "- Do not create standalone entity, date, or quantity atoms.",
        "- Attach dates, quantities, comparison targets, offices, locations, and attribution to the proposition they qualify.",
        "- Split only when the claim contains multiple separately verifiable propositions.",
        "- A single-sentence claim making one factual assertion should usually one atom.",
        "- Preserve the claim's meaning; do not add facts not present in the claim.",
        "",
        "Field guide:",
        "- claim_atoms.atom_id: A1, A2, ... in claim order.",
        "- claim_atoms.text: one independently verifiable claim proposition.",
        "- claim_atoms.type: entity, quantity, date, comparison, cause, outcome, or other.",
        "- claim_atoms.importance: 0.0 to 1.0; higher means more central to checking the claim.",
        "- candidate_alignments.evidence_id: one of the evidence IDs below.",
        "- candidate_alignments.covered_atom_ids: complete proposition atoms addressed by the passage.",
        "- candidate_alignments.relation: support, refute, qualify, mixed, background, or irrelevant.",
        "- candidate_alignments.directness: direct, partial, context, or none.",
        "- candidate_alignments.evidence_role: primary_support, primary_refute, partial_support, partial_refute, qualifying_context, background_context, duplicate, or irrelevant.",
        "- candidate_alignments.key_spans: short substrings copied from the passage.",
        "- candidate_alignments.duplicate_group: same group ID for near-duplicate passages; use an empty string if none.",
        "- candidate_alignments.confidence: 0.0 to 1.0 for how certain the alignment is.",
        "",
        "Claim:",
        str(row.get("claim") or "").strip(),
        "",
        "Evidence passages:",
    ]
    for item in row.get("evidence_items") or []:
        evidence_id = str(item.get("evidence_id") or "")
        text = _compact_evidence_text(str(item.get("text") or ""), max_evidence_chars=max_evidence_chars)
        lines.append(f"{evidence_id}: {text}")
    return system_prompt, "\n".join(lines)


def audit_teacher_prompt(row: dict[str, Any], *, system_prompt: str, user_prompt: str) -> None:
    prompt = f"{system_prompt}\n{user_prompt}"
    lowered = prompt.lower()
    leaked_field_names = sorted(field for field in FORBIDDEN_PROMPT_FIELDS if field.lower() in lowered)
    if leaked_field_names:
        raise ValueError(f"Teacher prompt contains forbidden field names: {leaked_field_names}")
    forbidden_values = [str(row.get("event_id") or "")]
    for candidate in row.get("candidates") or []:
        forbidden_values.append(str(candidate.get("candidate_uid") or ""))
    leaked_values = [value for value in forbidden_values if value and value in prompt]
    if leaked_values:
        raise ValueError(f"Teacher prompt contains forbidden metadata values: {leaked_values[:5]}")


def parse_evidence_map_content(content: str, *, valid_evidence_ids: Iterable[str]) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise EvidenceMapSchemaError(f"invalid JSON: {exc}") from exc
    return validate_evidence_map_payload(payload, valid_evidence_ids=valid_evidence_ids)


def validate_evidence_map_payload(payload: Any, *, valid_evidence_ids: Iterable[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvidenceMapSchemaError("payload must be a JSON object")
    valid_evidence = {str(item) for item in valid_evidence_ids}
    atoms = _validate_atoms(payload.get("claim_atoms"))
    atom_ids = {atom["atom_id"] for atom in atoms}
    alignments = _validate_alignments(
        payload.get("candidate_alignments"),
        valid_evidence_ids=valid_evidence,
        valid_atom_ids=atom_ids,
    )
    return {"claim_atoms": atoms, "candidate_alignments": alignments}


def atom_quality_diagnostics(atoms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    issues_by_atom: dict[str, list[str]] = {}
    fragment_atom_ids: list[str] = []
    issue_counts: Counter[str] = Counter()
    for idx, atom in enumerate(atoms, start=1):
        atom_id = str(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        issues = _atom_fragment_issues(atom)
        if issues:
            issues_by_atom[atom_id] = issues
            fragment_atom_ids.append(atom_id)
            issue_counts.update(issues)
    atom_count = len([atom for atom in atoms if isinstance(atom, dict)])
    return {
        "atom_count": atom_count,
        "fragment_atom_count": len(fragment_atom_ids),
        "fragment_atom_rate": float(len(fragment_atom_ids) / atom_count) if atom_count else 0.0,
        "fragment_atom_ids": fragment_atom_ids,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues_by_atom": issues_by_atom,
    }


def summarize_atom_quality_rows(rows: Sequence[dict[str, Any]], *, max_examples: int = 20) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    total_atoms = 0
    fragment_atom_count = 0
    rows_with_fragments = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        evidence_map = row.get("evidence_map") if isinstance(row.get("evidence_map"), dict) else {}
        atoms = list((evidence_map or {}).get("claim_atoms") or row.get("claim_atoms") or [])
        diagnostics = atom_quality_diagnostics(atoms)
        total_atoms += int(diagnostics.get("atom_count") or 0)
        row_fragment_count = int(diagnostics.get("fragment_atom_count") or 0)
        fragment_atom_count += row_fragment_count
        issue_counts.update(diagnostics.get("issue_counts") or {})
        if row_fragment_count:
            rows_with_fragments += 1
            if len(examples) < int(max_examples):
                examples.append(
                    {
                        "event_id": str(row.get("event_id") or ""),
                        "fragment_atom_count": row_fragment_count,
                        "fragment_atom_ids": list(diagnostics.get("fragment_atom_ids") or []),
                        "issues_by_atom": dict(diagnostics.get("issues_by_atom") or {}),
                    }
                )
    n_rows = len(rows)
    return {
        "n_rows": n_rows,
        "total_atoms": total_atoms,
        "fragment_atom_count": fragment_atom_count,
        "fragment_atom_rate": float(fragment_atom_count / total_atoms) if total_atoms else 0.0,
        "rows_with_fragment_atoms": rows_with_fragments,
        "row_fragment_rate": float(rows_with_fragments / n_rows) if n_rows else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
        "examples": examples,
    }


def mock_evidence_map_for_row(row: dict[str, Any]) -> dict[str, Any]:
    claim = str(row.get("claim") or "")
    atoms = _mock_atoms(claim)
    atom_tokens = {atom["atom_id"]: set(_content_tokens(atom["text"])) for atom in atoms}
    alignments = []
    seen_sigs: dict[str, str] = {}
    for item in row.get("evidence_items") or []:
        evidence_id = str(item.get("evidence_id") or "")
        text = str(item.get("text") or "")
        tokens = set(_content_tokens(text))
        covered: list[str] = []
        for atom in atoms:
            overlap = len(tokens & atom_tokens[atom["atom_id"]])
            if overlap >= max(1, min(3, len(atom_tokens[atom["atom_id"]]) // 3)):
                covered.append(atom["atom_id"])
        claim_overlap = _jaccard(tokens, set(_content_tokens(claim)))
        if covered and claim_overlap >= 0.18:
            relation = "support"
            directness = "direct" if claim_overlap >= 0.32 else "partial"
            role = "primary_support" if directness == "direct" else "partial_support"
        elif covered:
            relation = "qualify"
            directness = "partial"
            role = "qualifying_context"
        elif claim_overlap >= 0.12:
            relation = "background"
            directness = "context"
            role = "background_context"
        else:
            relation = "irrelevant"
            directness = "none"
            role = "irrelevant"
        sig = " ".join(sorted(tokens)[:12])
        duplicate_group = seen_sigs.setdefault(sig, f"G{len(seen_sigs) + 1}")
        alignments.append(
            {
                "evidence_id": evidence_id,
                "covered_atom_ids": covered,
                "relation": relation,
                "directness": directness,
                "evidence_role": role,
                "key_spans": [_short_span(text)] if directness in {"direct", "partial"} else [],
                "duplicate_group": duplicate_group,
                "confidence": min(1.0, max(0.1, claim_overlap + 0.35)),
            }
        )
    return validate_evidence_map_payload(
        {"claim_atoms": atoms, "candidate_alignments": alignments},
        valid_evidence_ids=[str(item.get("evidence_id") or "") for item in row.get("evidence_items") or []],
    )


def attach_evidence_map_annotations(
    rows: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        if row.get("event_id"):
            by_event[str(row.get("event_id") or "")].append(row)
    out: list[dict[str, Any]] = []
    for row in rows:
        annotation = _matching_annotation(row, by_event.get(str(row.get("event_id") or ""), []))
        item = dict(row)
        evidence_ids = [str(ev.get("evidence_id") or "") for ev in row.get("evidence_items") or []]
        if annotation and isinstance(annotation.get("evidence_map"), dict):
            try:
                evidence_map = validate_evidence_map_payload(
                    annotation["evidence_map"],
                    valid_evidence_ids=evidence_ids,
                )
                parse_status = "ok"
            except EvidenceMapSchemaError:
                evidence_map = _fallback_evidence_map(evidence_ids)
                parse_status = "fallback_invalid_annotation"
        else:
            evidence_map = _fallback_evidence_map(evidence_ids)
            parse_status = "fallback_missing_annotation"
        atom_weights = _atom_weights(evidence_map.get("claim_atoms") or [])
        align_by_id = {
            str(align.get("evidence_id") or ""): align
            for align in evidence_map.get("candidate_alignments") or []
        }
        candidates: list[dict[str, Any]] = []
        for candidate in row.get("candidates") or []:
            c = dict(candidate)
            align = align_by_id.get(str(c.get("evidence_id") or ""), _fallback_alignment(str(c.get("evidence_id") or "")))
            c.update(candidate_evidence_map_features(align, atom_weights=atom_weights))
            candidates.append(c)
        item["evidence_map"] = evidence_map
        item["evidence_map_parse_status"] = parse_status
        item["candidates"] = candidates
        out.append(item)
    attach_event_base_scores(out)
    return out


def _matching_annotation(row: dict[str, Any], annotations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    expected_fingerprint = str(row.get("evidence_items_fingerprint") or "")
    if expected_fingerprint:
        for annotation in reversed(list(annotations)):
            if str(annotation.get("evidence_items_fingerprint") or "") == expected_fingerprint:
                return annotation
    return dict(annotations[-1]) if annotations else None


def candidate_evidence_map_features(
    alignment: dict[str, Any],
    *,
    atom_weights: dict[str, float],
) -> dict[str, Any]:
    covered = [atom_id for atom_id in alignment.get("covered_atom_ids") or [] if atom_id in atom_weights]
    total_weight = sum(atom_weights.values()) or 1.0
    covered_weight = sum(atom_weights.get(atom_id, 0.0) for atom_id in covered)
    relation = str(alignment.get("relation") or "irrelevant")
    directness = str(alignment.get("directness") or "none")
    role = str(alignment.get("evidence_role") or "irrelevant")
    spans = [str(span).strip() for span in alignment.get("key_spans") or [] if str(span).strip()]
    background = background_penalty_score(relation=relation, directness=directness, role=role)
    confidence = _clamp01(_safe_float(alignment.get("confidence"), 0.0))
    direct_score = directness_score(directness)
    polar_score = polar_relation_score(relation)
    atom_coverage = float(covered_weight / total_weight)
    quality = float(
        np.clip(
            0.35 * atom_coverage
            + 0.25 * direct_score
            + 0.20 * polar_score
            + 0.10 * confidence
            + 0.10 * (1.0 if spans else 0.0)
            - 0.20 * background,
            0.0,
            1.0,
        )
    )
    return {
        "covered_atom_ids": covered,
        "covered_atom_weight": float(covered_weight),
        "atom_coverage_score": atom_coverage,
        "map_relation": relation,
        "map_directness": directness,
        "map_evidence_role": role,
        "key_spans": spans,
        "duplicate_group": str(alignment.get("duplicate_group") or ""),
        "map_confidence": confidence,
        "directness_score": direct_score,
        "polar_relation_score": polar_score,
        "background_penalty_score": background,
        "evidence_map_quality_score": quality,
        "mean_selected_span_count": float(len(spans)),
    }


def attach_event_base_scores(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        candidates = [candidate for candidate in row.get("candidates") or []]
        source = str(row.get("evidence_map_candidate_source") or "fusion")
        if source == "qd_union":
            for candidate in candidates:
                candidate["evidence_map_base_score"] = _qd_union_rank_prior(candidate)
                candidate["evidence_map_base_score_source"] = "qd_union_rank"
            continue
        fusion = _event_minmax([_safe_float(c.get("fusion_refit_score"), 0.0) for c in candidates])
        oracle = _event_minmax([_safe_float(c.get("oracle_likelihood_score"), 0.0) for c in candidates])
        direct = _event_minmax([_safe_float(c.get("direct_ce_score"), 0.0) for c in candidates])
        for idx, candidate in enumerate(candidates):
            candidate["evidence_map_base_score"] = float(0.70 * fusion[idx] + 0.20 * oracle[idx] + 0.10 * direct[idx])
            candidate["evidence_map_base_score_source"] = "fusion_oracle_direct"


def select_evidence_map_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    params: EvidenceMapParams,
    selector_name: str = EVIDENCE_MAP_SELECTOR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(candidate) for candidate in candidates]
    selected: list[dict[str, Any]] = []
    slot_trace: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_atoms: set[str] = set()
    selected_sources: Counter[str] = Counter()
    selected_duplicates: Counter[str] = Counter()
    for slot in range(1, int(params.top_k) + 1):
        eligible: list[dict[str, Any]] = []
        for candidate in rows:
            key = _selection_key(candidate)
            if not key or key in selected_keys:
                continue
            scored = dict(candidate)
            score_payload = slot_score(scored, selected_atoms=selected_atoms, selected_sources=selected_sources, selected_duplicates=selected_duplicates, params=params)
            scored.update(score_payload)
            eligible.append(scored)
        if not eligible:
            break
        best = max(eligible, key=_evidence_map_tie_key)
        key = _selection_key(best)
        selected_keys.add(key)
        selected_atoms.update(str(atom_id) for atom_id in best.get("covered_atom_ids") or [])
        selected_sources[source_group(best)] += 1
        dup = str(best.get("duplicate_group") or "")
        if dup:
            selected_duplicates[dup] += 1
        item = dict(best)
        item["selector_name"] = selector_name
        item["selection_rank"] = slot
        selected.append(item)
        slot_trace.append(
            {
                "slot": slot,
                "candidate_uid": str(item.get("candidate_uid") or ""),
                "candidate_key": str(item.get("candidate_key") or ""),
                "slot_score": _safe_float(item.get("slot_score"), 0.0),
                "base_score": _safe_float(item.get("score_base_component"), 0.0),
                "new_weighted_atom_coverage": _safe_float(item.get("new_weighted_atom_coverage"), 0.0),
                "covered_atom_ids": list(item.get("covered_atom_ids") or []),
                "map_relation": str(item.get("map_relation") or ""),
                "map_directness": str(item.get("map_directness") or ""),
                "duplicate_group": str(item.get("duplicate_group") or ""),
                "source_group": source_group(item),
                "oracle_selected": bool(item.get("oracle_selected")),
            }
        )
    return selected, slot_trace


def slot_score(
    candidate: dict[str, Any],
    *,
    selected_atoms: set[str],
    selected_sources: Counter[str],
    selected_duplicates: Counter[str],
    params: EvidenceMapParams,
) -> dict[str, float]:
    covered_atoms = [str(atom_id) for atom_id in candidate.get("covered_atom_ids") or []]
    new_atoms = [atom_id for atom_id in covered_atoms if atom_id not in selected_atoms]
    covered_weight = _safe_float(candidate.get("covered_atom_weight"), 0.0)
    atom_count = max(len(covered_atoms), 1)
    per_atom_weight = covered_weight / float(atom_count)
    new_coverage = per_atom_weight * float(len(new_atoms))
    dup = str(candidate.get("duplicate_group") or "")
    same_duplicate = float(selected_duplicates.get(dup, 0)) if dup else 0.0
    same_source = float(selected_sources.get(source_group(candidate), 0))
    base = _safe_float(candidate.get("evidence_map_base_score"), 0.0)
    direct = _safe_float(candidate.get("directness_score"), 0.0)
    polar = _safe_float(candidate.get("polar_relation_score"), 0.0)
    background = _safe_float(candidate.get("background_penalty_score"), 0.0)
    score = (
        float(params.base_weight) * base
        + float(params.atom_coverage_weight) * new_coverage
        + float(params.directness_weight) * direct
        + float(params.polar_relation_weight) * polar
        - float(params.duplicate_penalty) * same_duplicate
        - float(params.source_penalty) * same_source
        - float(params.background_penalty) * background
    )
    return {
        "slot_score": float(score),
        "score_base_component": float(base),
        "new_weighted_atom_coverage": float(new_coverage),
        "same_duplicate_group_selected": float(same_duplicate),
        "same_source_selected_count": float(same_source),
    }


def build_evidence_map_trace(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    selector_name: str,
    top_k: int,
    slot_trace: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = list(selected)
    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": selector_name,
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_candidates": [_candidate_output(candidate) for candidate in selected],
        "slot_trace": list(slot_trace or []),
        "claim_atoms": list((row.get("evidence_map") or {}).get("claim_atoms") or []),
    }
    trace.update(text_ordered_selection_metrics(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected))
    trace.update(evidence_map_selection_metrics(row, selected))
    return trace


def build_all_evidence_map_traces(rows: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        params = EvidenceMapParams(top_k=top_k)
        selected, slots = select_evidence_map_topk(candidates, params=params, selector_name=EVIDENCE_MAP_SELECTOR)
        traces.append(build_evidence_map_trace(row, selected, selector_name=EVIDENCE_MAP_SELECTOR, top_k=top_k, slot_trace=slots))
        base_selected = select_by_score(candidates, score_field="evidence_map_base_score", top_k=top_k, selector_name=EVIDENCE_MAP_BASE_ONLY_SELECTOR)
        traces.append(build_evidence_map_trace(row, base_selected, selector_name=EVIDENCE_MAP_BASE_ONLY_SELECTOR, top_k=top_k))
        coverage_selected, coverage_slots = select_evidence_map_topk(
            candidates,
            params=EvidenceMapParams(top_k=top_k, base_weight=0.0, atom_coverage_weight=1.0, directness_weight=0.0, polar_relation_weight=0.0, duplicate_penalty=0.0, source_penalty=0.0, background_penalty=0.0),
            selector_name=EVIDENCE_MAP_COVERAGE_ONLY_SELECTOR,
        )
        traces.append(build_evidence_map_trace(row, coverage_selected, selector_name=EVIDENCE_MAP_COVERAGE_ONLY_SELECTOR, top_k=top_k, slot_trace=coverage_slots))
        for selector_name, score_field in (
            (FUSION_REFIT_SELECTOR, "fusion_refit_score"),
            (ORACLE_LIKELIHOOD_SELECTOR, "oracle_likelihood_score"),
            (DIRECT_CE_TEXT_ONLY_SELECTOR, "direct_ce_score"),
        ):
            if not any(score_field in candidate for candidate in candidates):
                continue
            selected_control = select_by_score(candidates, score_field=score_field, top_k=top_k, selector_name=selector_name)
            traces.append(build_evidence_map_trace(row, selected_control, selector_name=selector_name, top_k=top_k))
        for mode in ("original_pool_order_top5", "qd_union_source_score_top5"):
            try:
                selected_control = select_order_control(candidates, mode=mode, top_k=top_k)
                traces.append(build_evidence_map_trace(row, selected_control, selector_name=mode, top_k=top_k))
            except ValueError:
                continue
        try:
            count_selected, count_slots, _ = select_count_amplified_topk(
                candidates,
                params=CountAmplifiedParams(top_k=top_k, n_stance_buckets=int(row.get("n_stance_buckets") or 7), use_directness_scoring=True),
                selector_name=COUNT_AMPLIFIED_SELECTOR,
            )
            traces.append(build_evidence_map_trace(row, count_selected, selector_name=COUNT_AMPLIFIED_SELECTOR, top_k=top_k, slot_trace=count_slots))
        except Exception:
            pass
    return traces


def select_by_score(
    candidates: Sequence[dict[str, Any]],
    *,
    score_field: str,
    top_k: int,
    selector_name: str,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get(score_field), 0.0),
            -_safe_float(row.get("evidence_map_quality_score"), 0.0),
            -_safe_float(row.get("evidence_map_base_score"), 0.0),
            int(row.get("union_pool_rank") or 10**9),
            str(row.get("candidate_key") or ""),
        )
    )
    selected = []
    seen: set[str] = set()
    for row in rows:
        key = _selection_key(row)
        if not key or key in seen:
            continue
        item = dict(row)
        item["selector_name"] = selector_name
        item["selection_rank"] = len(selected) + 1
        item["slot_score"] = _safe_float(item.get(score_field), 0.0)
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def summarize_evidence_map_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_selector_traces(traces)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("selector_name") or "")].append(dict(trace))
    for selector, rows in grouped.items():
        selected = [candidate for trace in rows for candidate in trace.get("selected_candidates") or []]
        item = summary.setdefault(selector, {"n_claims": len(rows)})
        item.update(_summarize_explainability_from_candidates(selected))
        item["mean_atom_coverage@5"] = _mean(_safe_float(trace.get("atom_coverage@5"), 0.0) for trace in rows)
        item["mean_weighted_atom_coverage@5"] = _mean(_safe_float(trace.get("weighted_atom_coverage@5"), 0.0) for trace in rows)
        item["missing_atom_rate@5"] = _mean(_safe_float(trace.get("missing_atom_rate@5"), 0.0) for trace in rows)
    return summary


def evidence_map_diagnostics(
    rows: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    selector_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline = selector_metrics.get(FUSION_REFIT_SELECTOR, {})
    primary = selector_metrics.get(EVIDENCE_MAP_SELECTOR, {})
    base_only = selector_metrics.get(EVIDENCE_MAP_BASE_ONLY_SELECTOR, {})
    decision = decide_v05a(selector_metrics)
    return {
        "decision": decision,
        "n_events": len(rows),
        "n_candidates": sum(len(row.get("candidates") or []) for row in rows),
        "parse_status_counts": dict(Counter(str(row.get("evidence_map_parse_status") or "") for row in rows)),
        "primary_vs_fusion_refit": _metric_delta(primary, baseline),
        "primary_vs_base_only": _metric_delta(primary, base_only),
        "selector_explainability": {
            selector: {
                key: value
                for key, value in metrics.items()
                if key in {
                    "mean_atom_coverage@5",
                    "mean_weighted_atom_coverage@5",
                    "direct_or_partial_map_rate@5",
                    "background_only_map_rate@5",
                    "duplicate_group_collapse_rate@5",
                    "missing_atom_rate@5",
                    "mean_selected_span_count@5",
                }
            }
            for selector, metrics in selector_metrics.items()
        },
        "go_criteria": _go_criteria(selector_metrics),
    }


def decide_v05a(selector_metrics: dict[str, Any]) -> str:
    criteria = _go_criteria(selector_metrics)
    if all(criteria.values()):
        return "go_to_v0_5b_map_aware_verifier"
    if criteria.get("coverage_lift") and criteria.get("directness_lift") and criteria.get("jaccard_guardrail"):
        return "analysis_promising_explanation_signal_v0_5a"
    return "analysis_only_evidence_map_needs_review_v0_5a"


def render_case_study_markdown(
    traces: Sequence[dict[str, Any]],
    *,
    case_ids: Sequence[str] | None = None,
    top_n: int = 5,
) -> str:
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        by_event[str(trace.get("event_id") or "")][str(trace.get("selector_name") or "")] = dict(trace)
    rows = []
    for event_id, per_selector in by_event.items():
        if EVIDENCE_MAP_SELECTOR not in per_selector or FUSION_REFIT_SELECTOR not in per_selector:
            continue
        primary = per_selector[EVIDENCE_MAP_SELECTOR]
        base = per_selector[FUSION_REFIT_SELECTOR]
        rows.append((event_id, _safe_float(primary.get("jaccard@5"), 0.0) - _safe_float(base.get("jaccard@5"), 0.0), primary, base))
    best = sorted(rows, key=lambda item: item[1], reverse=True)[:top_n]
    worst = sorted(rows, key=lambda item: item[1])[:top_n]
    wanted = []
    for case_id in case_ids or []:
        for row in rows:
            if str(row[0]).split(".")[0] == str(case_id).split(".")[0] or str(row[0]) == str(case_id):
                wanted.append(row)
                break
    lines = ["# Evidence Map Selector v0.5a Case Studies", ""]
    for title, items in (("Best Improvements", best), ("Worst Regressions", worst), ("Requested Cases", wanted)):
        lines.extend([f"## {title}", ""])
        if not items:
            lines.append("(none)")
            lines.append("")
            continue
        for event_id, delta, primary, base in items:
            lines.extend(_case_lines(event_id, delta, primary, base))
    return "\n".join(lines)


def evidence_map_selection_metrics(row: dict[str, Any], selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    atom_weights = _atom_weights((row.get("evidence_map") or {}).get("claim_atoms") or [])
    selected_atoms = set()
    selected_weight = 0.0
    for candidate in selected:
        for atom_id in candidate.get("covered_atom_ids") or []:
            if atom_id not in selected_atoms:
                selected_atoms.add(str(atom_id))
                selected_weight += atom_weights.get(str(atom_id), 0.0)
    total_weight = sum(atom_weights.values()) or 1.0
    return {
        "atom_coverage@5": float(len(selected_atoms) / max(len(atom_weights), 1)),
        "weighted_atom_coverage@5": float(selected_weight / total_weight),
        "missing_atom_rate@5": float(1.0 - (selected_weight / total_weight)),
    }


def directness_score(value: str) -> float:
    return {"direct": 1.0, "partial": 0.65, "context": 0.25, "none": 0.0}.get(str(value), 0.0)


def polar_relation_score(value: str) -> float:
    return {"support": 1.0, "refute": 1.0, "qualify": 0.60, "mixed": 0.60, "background": 0.15, "irrelevant": 0.0}.get(str(value), 0.0)


def background_penalty_score(*, relation: str, directness: str, role: str) -> float:
    role_text = str(role or "")
    if str(relation) == "irrelevant" or role_text == "irrelevant":
        return 1.0
    if str(relation) == "background" or "background" in role_text:
        return 0.9
    if str(directness) == "none":
        return 0.8
    if str(directness) == "context" or "context" in role_text:
        return 0.55
    return 0.0


def _normalize_candidate_for_map(
    candidate: dict[str, Any],
    *,
    event_id: str,
    fallback_idx: int,
    candidate_source: str,
) -> dict[str, Any]:
    item = dict(candidate)
    if candidate_source == "qd_union":
        key = str(item.get("candidate_key") or item.get("canonical_text") or item.get("text") or "").strip()
        if not key:
            key = f"candidate-{fallback_idx}"
        item["candidate_key"] = key
        item["candidate_uid"] = str(item.get("candidate_uid") or hashlib.sha1(f"{event_id}|{key}".encode("utf-8")).hexdigest()[:12])
        item.setdefault("source_group", f"report:{item.get('report_id')}" if item.get("report_id") is not None else "")
        item.setdefault("evidence_map_base_score", _qd_union_rank_prior(item))
        item.setdefault("evidence_map_base_score_source", "qd_union_rank")
        item["evidence_map_candidate_source"] = "qd_union"
    return item


def _top_annotation_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    candidate_top_n: int,
    candidate_source: str = "fusion",
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if str(candidate_source or "fusion") == "qd_union":
        rows.sort(
            key=lambda row: (
                int(row.get("union_pool_rank") or 10**9),
                str(row.get("candidate_key") or row.get("text") or ""),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                -_safe_float(row.get("fusion_refit_score"), 0.0),
                -_safe_float(row.get("oracle_likelihood_score"), 0.0),
                -_safe_float(row.get("direct_ce_score"), 0.0),
                int(row.get("union_pool_rank") or 10**9),
                str(row.get("candidate_key") or ""),
            )
        )
    return rows[: int(candidate_top_n)]


def _validate_atoms(raw_atoms: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise EvidenceMapSchemaError("claim_atoms must be a non-empty list")
    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_atoms, start=1):
        if not isinstance(raw, dict):
            continue
        atom_id = _safe_id(str(raw.get("atom_id") or f"A{idx}"), prefix="A", fallback_idx=idx)
        if atom_id in seen:
            atom_id = f"A{idx}"
        seen.add(atom_id)
        text = str(raw.get("text") or "").strip()
        if not text:
            text = f"Claim atom {idx}"
        importance = _importance_to_unit(raw.get("importance", 1.0))
        atoms.append(
            {
                "atom_id": atom_id,
                "text": text[:500],
                "type": str(raw.get("type") or "other")[:64],
                "importance": importance,
            }
        )
    if not atoms:
        raise EvidenceMapSchemaError("claim_atoms has no valid atom")
    return atoms


def _validate_alignments(raw_alignments: Any, *, valid_evidence_ids: set[str], valid_atom_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_alignments, list):
        raise EvidenceMapSchemaError("candidate_alignments must be a list")
    align_by_eid: dict[str, dict[str, Any]] = {}
    for raw in raw_alignments:
        if not isinstance(raw, dict):
            continue
        evidence_id = str(raw.get("evidence_id") or "").strip()
        if evidence_id not in valid_evidence_ids:
            continue
        relation = str(raw.get("relation") or "irrelevant").strip().lower()
        if relation not in ALLOWED_RELATIONS:
            relation = "irrelevant"
        directness = str(raw.get("directness") or "none").strip().lower()
        if directness not in ALLOWED_DIRECTNESS:
            directness = "none"
        role = str(raw.get("evidence_role") or "irrelevant").strip().lower()
        if role not in ALLOWED_ROLES:
            role = _role_from_relation(relation, directness)
        covered = []
        for atom_id in raw.get("covered_atom_ids") or []:
            atom = str(atom_id)
            if atom in valid_atom_ids and atom not in covered:
                covered.append(atom)
        spans = [str(span).strip()[:240] for span in raw.get("key_spans") or [] if str(span).strip()][:3]
        align_by_eid[evidence_id] = {
            "evidence_id": evidence_id,
            "covered_atom_ids": covered,
            "relation": relation,
            "directness": directness,
            "evidence_role": role,
            "key_spans": spans,
            "duplicate_group": str(raw.get("duplicate_group") or "").strip()[:64],
            "confidence": _clamp01(_safe_float(raw.get("confidence"), 0.0)),
        }
    return [align_by_eid.get(evidence_id, _fallback_alignment(evidence_id)) for evidence_id in sorted(valid_evidence_ids)]


def _candidate_output(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_uid",
        "candidate_key",
        "evidence_id",
        "selection_rank",
        "union_pool_rank",
        "source_pools",
        "from_baseline",
        "from_qd",
        "baseline_rank",
        "qd_pool_rank",
        "retrieval_score",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "fusion_refit_score",
        "oracle_likelihood_score",
        "direct_ce_score",
        "evidence_map_base_score",
        "slot_score",
        "evidence_map_quality_score",
        "atom_coverage_score",
        "covered_atom_ids",
        "map_relation",
        "map_directness",
        "map_evidence_role",
        "key_spans",
        "duplicate_group",
        "directness_score",
        "polar_relation_score",
        "background_penalty_score",
    )
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["text"] = str(candidate.get("text") or "")
    return out


def _summarize_explainability_from_candidates(candidates: Sequence[dict[str, Any]]) -> dict[str, float]:
    rows = list(candidates)
    if not rows:
        return {
            "direct_or_partial_map_rate@5": 0.0,
            "background_only_map_rate@5": 0.0,
            "duplicate_group_collapse_rate@5": 0.0,
            "mean_selected_span_count@5": 0.0,
        }
    duplicate_counts = Counter(str(row.get("duplicate_group") or "") for row in rows if row.get("duplicate_group"))
    return {
        "direct_or_partial_map_rate@5": _mean(1.0 if str(row.get("map_directness") or "") in {"direct", "partial"} else 0.0 for row in rows),
        "background_only_map_rate@5": _mean(1.0 if str(row.get("map_relation") or "") in {"background", "irrelevant"} else 0.0 for row in rows),
        "duplicate_group_collapse_rate@5": _mean(1.0 if duplicate_counts.get(str(row.get("duplicate_group") or ""), 0) > 1 else 0.0 for row in rows),
        "mean_selected_span_count@5": _mean(float(len(row.get("key_spans") or [])) for row in rows),
    }


def _go_criteria(selector_metrics: dict[str, Any]) -> dict[str, bool]:
    primary = selector_metrics.get(EVIDENCE_MAP_SELECTOR, {})
    base = selector_metrics.get(EVIDENCE_MAP_BASE_ONLY_SELECTOR, {})
    fusion = selector_metrics.get(FUSION_REFIT_SELECTOR, {})
    return {
        "directness_lift": _safe_float(primary.get("direct_or_partial_map_rate@5"), 0.0) - _safe_float(fusion.get("direct_or_partial_map_rate@5"), 0.0) >= 0.05,
        "background_reduction": _safe_float(primary.get("background_only_map_rate@5"), 0.0) - _safe_float(fusion.get("background_only_map_rate@5"), 0.0) <= -0.03,
        "coverage_lift": _safe_float(primary.get("mean_weighted_atom_coverage@5"), 0.0) > _safe_float(base.get("mean_weighted_atom_coverage@5"), 0.0),
        "jaccard_guardrail": _safe_float(primary.get("jaccard@5"), 0.0) >= _safe_float(fusion.get("jaccard@5"), 0.0) - 0.010,
    }


def _metric_delta(primary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "recall@5",
        "jaccard@5",
        "top1_match",
        "oracle_rank_ndcg@5",
        "mean_weighted_atom_coverage@5",
        "direct_or_partial_map_rate@5",
        "background_only_map_rate@5",
    )
    return {key: _safe_float(primary.get(key), 0.0) - _safe_float(baseline.get(key), 0.0) for key in keys}


def _case_lines(event_id: str, delta: float, primary: dict[str, Any], base: dict[str, Any]) -> list[str]:
    lines = [
        f"### {event_id} delta_jaccard={delta:.4f}",
        "",
        f"Claim: {primary.get('claim', '')}",
        "",
        f"- evidence_map selected: {primary.get('selected_keys', [])}",
        f"- fusion_refit selected: {base.get('selected_keys', [])}",
        f"- map coverage: {float(primary.get('weighted_atom_coverage@5', 0.0)):.4f}",
        f"- map direct/background: {float(primary.get('direct_or_partial_map_rate@5', 0.0)):.4f} / {float(primary.get('background_only_map_rate@5', 0.0)):.4f}",
        "",
    ]
    for candidate in primary.get("selected_candidates") or []:
        spans = candidate.get("key_spans") or []
        span_text = "; ".join(spans[:2]) if spans else ""
        lines.append(
            "- {rank}. {relation}/{directness} atoms={atoms} spans={spans} text={text}".format(
                rank=candidate.get("selection_rank", ""),
                relation=candidate.get("map_relation", ""),
                directness=candidate.get("map_directness", ""),
                atoms=candidate.get("covered_atom_ids", []),
                spans=span_text,
                text=str(candidate.get("text") or "")[:220],
            )
        )
    lines.append("")
    return lines


def _fallback_evidence_map(evidence_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "claim_atoms": [{"atom_id": "A1", "text": "Full claim", "type": "other", "importance": 1.0}],
        "candidate_alignments": [_fallback_alignment(evidence_id) for evidence_id in evidence_ids],
    }


def _fallback_alignment(evidence_id: str) -> dict[str, Any]:
    return {
        "evidence_id": str(evidence_id),
        "covered_atom_ids": [],
        "relation": "irrelevant",
        "directness": "none",
        "evidence_role": "irrelevant",
        "key_spans": [],
        "duplicate_group": "",
        "confidence": 0.0,
    }


def _role_from_relation(relation: str, directness: str) -> str:
    if relation == "support":
        return "primary_support" if directness == "direct" else "partial_support"
    if relation == "refute":
        return "primary_refute" if directness == "direct" else "partial_refute"
    if relation in {"qualify", "mixed"}:
        return "qualifying_context"
    if relation == "background":
        return "background_context"
    return "irrelevant"


def _atom_weights(atoms: Sequence[dict[str, Any]]) -> dict[str, float]:
    out = {str(atom.get("atom_id") or ""): _importance_to_unit(atom.get("importance", 1.0)) for atom in atoms}
    return {key: value for key, value in out.items() if key}


def _importance_to_unit(value: Any) -> float:
    raw = _safe_float(value, 1.0)
    if raw > 1.0:
        raw = raw / 5.0
    return float(np.clip(raw, 0.05, 1.0))


def _event_minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [float((value - vmin) / (vmax - vmin)) for value in arr]


def _qd_union_rank_prior(candidate: dict[str, Any]) -> float:
    rank = _safe_int(candidate.get("union_pool_rank"))
    if rank is None or rank <= 0:
        rank = _safe_int(candidate.get("qd_pool_rank"))
    if rank is None or rank <= 0:
        rank = _safe_int(candidate.get("baseline_rank"))
    if rank is None or rank <= 0:
        return 0.0
    return float(1.0 / float(rank))


def _evidence_map_tie_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        _safe_float(candidate.get("slot_score"), 0.0),
        _safe_float(candidate.get("evidence_map_quality_score"), 0.0),
        _safe_float(candidate.get("evidence_map_base_score"), 0.0),
        -int(candidate.get("union_pool_rank") or 10**9),
        str(candidate.get("candidate_key") or ""),
    )


def _selection_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_uid") or candidate.get("candidate_key") or "")


def _safe_id(value: str, *, prefix: str, fallback_idx: int) -> str:
    match = re.search(r"(\d+)", value)
    if match:
        return f"{prefix}{int(match.group(1))}"
    if value.startswith(prefix) and value[1:].isalnum():
        return value
    return f"{prefix}{fallback_idx}"


def _mock_atoms(claim: str) -> list[dict[str, Any]]:
    chunks = [chunk.strip(" .") for chunk in re.split(r"\s*(?:;|,\s+(?:and|but|because|while)\b)\s*", claim) if chunk.strip()]
    if len(chunks) < 2:
        chunks = [claim]
    atoms = []
    for idx, chunk in enumerate(chunks[:5], start=1):
        atoms.append({"atom_id": f"A{idx}", "text": chunk[:300], "type": _mock_atom_type(chunk), "importance": 1.0})
    return atoms or [{"atom_id": "A1", "text": claim[:300] or "Full claim", "type": "other", "importance": 1.0}]


def _mock_atom_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\d|percent|million|billion|half", lowered):
        return "quantity"
    if any(word in lowered for word in ("before", "after", "in 20", "year")):
        return "date"
    if any(word in lowered for word in ("more", "less", "than", "half")):
        return "comparison"
    if any(word in lowered for word in ("because", "caused", "due to")):
        return "cause"
    return "other"


_ATOM_PREPOSITION_STARTS = {
    "about",
    "after",
    "at",
    "before",
    "by",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "of",
    "on",
    "over",
    "than",
    "to",
    "under",
    "with",
    "within",
}
_ATOM_PREDICATE_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "did",
    "does",
    "do",
    "says",
    "said",
    "say",
    "claim",
    "claims",
    "claimed",
    "miss",
    "missed",
    "missing",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "raise",
    "raised",
    "lower",
    "lowered",
    "support",
    "supported",
    "oppose",
    "opposed",
    "vote",
    "voted",
    "spend",
    "spent",
    "cost",
    "costs",
    "make",
    "made",
    "create",
    "created",
    "cut",
    "reduced",
    "reduce",
}


def _atom_fragment_issues(atom: dict[str, Any]) -> list[str]:
    text = " ".join(str(atom.get("text") or "").split())
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9']+", lowered)
    atom_type = str(atom.get("type") or atom.get("atom_type") or "").strip().lower()
    issues: list[str] = []
    if len(tokens) < 4:
        issues.append("too_short")
    if tokens and tokens[0] in _ATOM_PREPOSITION_STARTS:
        issues.append("preposition_start")
    has_predicate = _atom_has_predicate(tokens)
    if not has_predicate:
        issues.append("missing_predicate")
    slot_type = atom_type in {"entity", "date", "quantity", "comparison"}
    if slot_type and (not has_predicate or len(tokens) <= 4):
        issues.append("standalone_entity_or_modifier")
    return issues


def _atom_has_predicate(tokens: Sequence[str]) -> bool:
    token_set = set(tokens)
    if token_set & _ATOM_PREDICATE_HINTS:
        return True
    return any(
        token.endswith(("ed", "ing"))
        and token not in {"according", "during", "thing", "something", "anything", "nothing"}
        for token in tokens
    )


def _content_tokens(text: str) -> list[str]:
    stop = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "has", "have", "is", "are", "was", "were", "says", "said"}
    return [tok for tok in re.findall(r"[a-z0-9]+", str(text).lower()) if len(tok) > 2 and tok not in stop]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _short_span(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:180]


def _clean_for_prompt(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split())


def _compact_evidence_text(text: str, *, max_evidence_chars: int | None) -> str:
    clean = _clean_for_prompt(text)
    if max_evidence_chars is None or int(max_evidence_chars) <= 0:
        return clean
    limit = int(max_evidence_chars)
    if len(clean) <= limit:
        return clean
    marker = " ... "
    if limit <= len(marker) + 40:
        return clean[:limit]
    tail_chars = min(max(120, limit // 4), max(40, limit // 2 - len(marker)))
    head_chars = max(1, limit - tail_chars - len(marker))
    return f"{clean[:head_chars].rstrip()}{marker}{clean[-tail_chars:].lstrip()}"


def _strip_json_fence(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return float(parsed)


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0
