from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


GRAPH_VERSION = "atom_anchored_qec_v1"
ADAPTIVE_POLICY = "aa_qec_view"
CONSTRAINED_ADAPTIVE_POLICY = "aa_qec_constrained_atom_facts_abc"
FULL_ADAPTIVE_POLICY = "aa_qec_full_atom_facts_abc"
DEFAULT_SOURCE_SELECTOR_NAME = "v0_7_budgeted_marginal_chain_adaptive5_10"
FALLBACK_CUE = "Verify the main factual claim."
STAGE1_SELECTION_POLICIES = {"keep_all_reorder", "primary_secondary_order", "shuffled"}
CONSTRAINED_SELECTION_POLICIES = {
    "primary_only",
    "primary_secondary",
    "primary_secondary_fallback_min5",
    "primary_fallback_min5_no_secondary",
}


@dataclass(frozen=True)
class AtomAnchoredQECParams:
    candidate_top_n: int = 20
    min_chain_steps: int = 5
    max_chain_steps: int = 10
    max_secondary_per_atom: int = 1
    min_secondary_confidence: float = 0.4
    cue_policy: str = "qd_prefer"
    candidate_scope: str = "selected"
    selection_policy: str = "keep_all_reorder"
    source_selector_name: str = DEFAULT_SOURCE_SELECTOR_NAME
    random_seed: int = 0


def atom_anchored_qec_selector_name(params: AtomAnchoredQECParams) -> str:
    candidate_scope = str(params.candidate_scope)
    policy = str(params.selection_policy)
    if candidate_scope == "top20":
        return _full_atom_facts_abc_selector_name(params)
    if candidate_scope != "selected":
        raise ValueError(f"unsupported AA-QEC candidate_scope={candidate_scope!r}")
    if policy == "keep_all_reorder":
        policy_slug = "keep_all"
    elif policy == "primary_secondary_order":
        policy_slug = "primary_secondary_order"
    elif policy == "shuffled":
        policy_slug = "shuffled"
    else:
        return _constrained_atom_facts_abc_selector_name(params)
    return (
        f"aa_qec_view_{policy_slug}_{params.cue_policy}_"
        f"{params.candidate_scope}_min{int(params.min_chain_steps)}_{int(params.max_chain_steps)}"
    )


def atom_anchored_qec_adaptive_policy(params: AtomAnchoredQECParams) -> str:
    candidate_scope = str(params.candidate_scope)
    policy = str(params.selection_policy)
    if policy in STAGE1_SELECTION_POLICIES:
        if candidate_scope != "selected":
            raise ValueError(f"AA-QEC view only supports candidate_scope=selected, got {candidate_scope!r}")
        return ADAPTIVE_POLICY
    if policy in CONSTRAINED_SELECTION_POLICIES:
        if candidate_scope == "selected":
            return CONSTRAINED_ADAPTIVE_POLICY
        if candidate_scope == "top20":
            return FULL_ADAPTIVE_POLICY
        raise ValueError(f"unsupported AA-QEC candidate_scope={candidate_scope!r}")
    raise ValueError(f"unsupported AA-QEC selection_policy={policy!r}")


def build_atom_anchored_qec_trace_row(
    row: dict[str, Any],
    *,
    params: AtomAnchoredQECParams | None = None,
) -> dict[str, Any]:
    params = params or AtomAnchoredQECParams()

    candidate_pool = [dict(candidate) for candidate in row.get("candidate_pool") or []]
    if not candidate_pool:
        raise ValueError("AA-QEC input trace has no candidate_pool.")

    source_selected_indices = _dedupe_in_range(_ordered_trace_indices(row), len(candidate_pool))
    candidate_indices = _candidate_scope_indices(
        source_selected_indices,
        candidate_pool_len=len(candidate_pool),
        params=params,
    )
    if not candidate_indices:
        raise ValueError("AA-QEC input trace has no valid selected indices.")

    claim_atoms = _claim_atoms(row)
    atom_by_id = {_atom_id(atom): atom for atom in claim_atoms if _atom_id(atom)}
    atom_order = {_atom_id(atom): idx for idx, atom in enumerate(claim_atoms) if _atom_id(atom)}
    if str(params.selection_policy) in CONSTRAINED_SELECTION_POLICIES:
        ordered_indices, role_by_idx, anchor_atom_by_idx = _stage2_order_and_roles(
            candidate_indices,
            candidate_pool=candidate_pool,
            atom_order=atom_order,
            params=params,
        )
    else:
        role_by_idx, anchor_atom_by_idx = _assign_stage1_roles(
            candidate_indices,
            candidate_pool=candidate_pool,
            atom_order=atom_order,
        )
        ordered_indices = _stage1_order(
            candidate_indices,
            candidate_pool=candidate_pool,
            role_by_idx=role_by_idx,
            anchor_atom_by_idx=anchor_atom_by_idx,
            atom_order=atom_order,
            params=params,
        )

    selector_name = atom_anchored_qec_selector_name(params)
    adaptive_policy = atom_anchored_qec_adaptive_policy(params)
    fingerprint = _trace_fingerprint(row)
    chain_steps = _chain_steps(
        ordered_indices,
        candidate_pool=candidate_pool,
        role_by_idx=role_by_idx,
        anchor_atom_by_idx=anchor_atom_by_idx,
        atom_by_id=atom_by_id,
        atom_order=atom_order,
    )
    selected_candidates = []
    for rank, idx in enumerate(ordered_indices):
        candidate = dict(candidate_pool[idx])
        candidate.setdefault("candidate_idx", idx)
        candidate["selector_trace_rank"] = rank
        candidate["selector_candidate_idx"] = idx
        selected_candidates.append(candidate)

    metadata = dict(row.get("candidate_pool_metadata") or {})
    metadata.update(
        {
            "chunk_mmr_fingerprint": fingerprint,
            "selector_name": selector_name,
            "graph_version": GRAPH_VERSION,
            "adaptive_policy": adaptive_policy,
            "source_selector_name": str(row.get("selector_name") or params.source_selector_name),
            "candidate_scope": str(params.candidate_scope),
            "input_candidate_count": len(candidate_indices),
            "source_selected_count": len(source_selected_indices),
        }
    )

    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": selector_name,
        "graph_version": GRAPH_VERSION,
        "fingerprint": fingerprint,
        "candidate_pool_metadata": metadata,
        "candidate_pool": candidate_pool,
        "candidate_scores": list(row.get("candidate_scores") or []),
        "selector_ordered_indices": ordered_indices,
        "selected_indices": ordered_indices,
        "oracle_ordered_indices": _int_list(row.get("oracle_ordered_indices") or []),
        "selected_candidates": selected_candidates,
        "selected_evidence_ids": [str(candidate.get("evidence_id") or "") for candidate in selected_candidates],
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected_candidates],
        "claim_atoms": claim_atoms,
        "chain_steps": chain_steps,
        "chain_diagnostics": _chain_diagnostics(
            chain_steps,
            selected_indices=candidate_indices,
            source_selected_indices=source_selected_indices,
            candidate_scope=str(params.candidate_scope),
            claim_atoms=claim_atoms,
        ),
        "params": asdict(params),
        "adaptive_policy": adaptive_policy,
        "source_selector_name": str(row.get("selector_name") or params.source_selector_name),
    }
    return trace


def summarize_atom_anchored_qec_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selector_names: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    cue_source_counts: Counter[str] = Counter()
    chain_lengths: list[int] = []
    qd_rates: list[float] = []
    atom_rates: list[float] = []
    fallback_rates: list[float] = []
    duplicate_rates: list[float] = []
    fallback_fill_rates: list[float] = []
    atom_coverage_rates: list[float] = []
    uncovered_atom_rates: list[float] = []
    secondary_step_rates: list[float] = []
    multi_atom_evidence_rates: list[float] = []

    for row in rows:
        selector_names[str(row.get("selector_name") or "")] += 1
        steps = [step for step in row.get("chain_steps") or [] if isinstance(step, dict)]
        chain_lengths.append(len(steps))
        for step in steps:
            role_counts[str(step.get("role") or "")] += 1
            cue_source_counts[str(step.get("cue_source") or "")] += 1
        diagnostics = row.get("chain_diagnostics") or {}
        if isinstance(diagnostics, dict):
            qd_rates.append(float(diagnostics.get("qd_cue_rate") or 0.0))
            atom_rates.append(float(diagnostics.get("atom_cue_rate") or 0.0))
            fallback_rates.append(float(diagnostics.get("claim_fallback_rate") or 0.0))
            duplicate_rates.append(float(diagnostics.get("duplicate_evidence_rate") or 0.0))
            fallback_fill_rates.append(float(diagnostics.get("fallback_fill_rate") or 0.0))
            atom_coverage_rates.append(float(diagnostics.get("atom_coverage_rate") or 0.0))
            uncovered_atom_rates.append(float(diagnostics.get("uncovered_atom_rate") or 0.0))
            secondary_step_rates.append(float(diagnostics.get("secondary_step_rate") or 0.0))
            multi_atom_evidence_rates.append(float(diagnostics.get("multi_atom_evidence_rate") or 0.0))

    return {
        "n_rows": len(rows),
        "selector_names": dict(selector_names),
        "chain_steps": _numeric_summary(chain_lengths),
        "role_counts": {key: value for key, value in dict(role_counts).items() if key},
        "cue_source_counts": {key: value for key, value in dict(cue_source_counts).items() if key},
        "qd_cue_rate": _numeric_summary(qd_rates),
        "atom_cue_rate": _numeric_summary(atom_rates),
        "claim_fallback_rate": _numeric_summary(fallback_rates),
        "duplicate_evidence_rate": _numeric_summary(duplicate_rates),
        "fallback_fill_rate": _numeric_summary(fallback_fill_rates),
        "atom_coverage_rate": _numeric_summary(atom_coverage_rates),
        "uncovered_atom_rate": _numeric_summary(uncovered_atom_rates),
        "secondary_step_rate": _numeric_summary(secondary_step_rates),
        "multi_atom_evidence_rate": _numeric_summary(multi_atom_evidence_rates),
    }


def _constrained_atom_facts_abc_selector_name(params: AtomAnchoredQECParams) -> str:
    policy = str(params.selection_policy)
    prefix = "aa_qec_constrained_atom_facts_abc"
    cue_scope = f"{params.cue_policy}_{params.candidate_scope}"
    max_steps = int(params.max_chain_steps)
    min_steps = int(params.min_chain_steps)
    if policy == "primary_only":
        return f"{prefix}_primary_only_{cue_scope}_max{max_steps}"
    if policy == "primary_secondary":
        return f"{prefix}_primary_secondary_{cue_scope}_max{max_steps}"
    if policy == "primary_secondary_fallback_min5":
        return f"{prefix}_primary_secondary_fallback_{cue_scope}_min{min_steps}_{max_steps}"
    if policy == "primary_fallback_min5_no_secondary":
        return f"{prefix}_primary_fallback_no_secondary_{cue_scope}_min{min_steps}_{max_steps}"
    raise ValueError(f"unsupported AA-QEC selection_policy={policy!r}")


def _full_atom_facts_abc_selector_name(params: AtomAnchoredQECParams) -> str:
    policy = str(params.selection_policy)
    if str(params.candidate_scope) != "top20":
        raise ValueError(f"AA-QEC full scope requires candidate_scope=top20, got {params.candidate_scope!r}")
    prefix = "aa_qec_full_atom_facts_abc"
    cue_scope = f"{params.cue_policy}_{params.candidate_scope}"
    max_steps = int(params.max_chain_steps)
    min_steps = int(params.min_chain_steps)
    if policy == "primary_only":
        return f"{prefix}_primary_only_{cue_scope}_max{max_steps}"
    if policy == "primary_secondary":
        if min_steps == 0 and max_steps == 0:
            return f"{prefix}_primary_secondary_dynamic_{cue_scope}"
        return f"{prefix}_primary_secondary_{cue_scope}_max{max_steps}"
    if policy == "primary_secondary_fallback_min5":
        return f"{prefix}_primary_secondary_fallback_{cue_scope}_min{min_steps}_{max_steps}"
    if policy == "primary_fallback_min5_no_secondary":
        return f"{prefix}_primary_fallback_no_secondary_{cue_scope}_min{min_steps}_{max_steps}"
    raise ValueError(f"unsupported AA-QEC selection_policy={policy!r} for candidate_scope=top20")


def _candidate_scope_indices(
    source_selected_indices: list[int],
    *,
    candidate_pool_len: int,
    params: AtomAnchoredQECParams,
) -> list[int]:
    candidate_scope = str(params.candidate_scope)
    if candidate_scope == "selected":
        return list(source_selected_indices)
    if candidate_scope == "top20":
        top_n = int(params.candidate_top_n)
        limit = candidate_pool_len if top_n <= 0 else min(candidate_pool_len, top_n)
        return list(range(limit))
    raise ValueError(f"unsupported AA-QEC candidate_scope={candidate_scope!r}")


def _stage2_order_and_roles(
    selected_indices: list[int],
    *,
    candidate_pool: list[dict[str, Any]],
    atom_order: dict[str, int],
    params: AtomAnchoredQECParams,
) -> tuple[list[int], dict[int, str], dict[int, str]]:
    selected_indices = _dedupe_in_range(selected_indices, len(candidate_pool))
    if params.candidate_top_n > 0:
        selected_indices = selected_indices[: int(params.candidate_top_n)]
    selected_rank = {idx: rank for rank, idx in enumerate(selected_indices)}
    policy = str(params.selection_policy)
    include_secondary = policy in {"primary_secondary", "primary_secondary_fallback_min5"}
    fill_to_min = policy in {"primary_secondary_fallback_min5", "primary_fallback_min5_no_secondary"}
    max_steps = max(int(params.max_chain_steps), 0)
    min_steps = max(int(params.min_chain_steps), 0) if fill_to_min else 0

    role_by_idx: dict[int, str] = {}
    anchor_atom_by_idx: dict[int, str] = {}
    ordered: list[int] = []
    used_indices: set[int] = set()

    atom_ids = [atom_id for atom_id, _ in sorted(atom_order.items(), key=lambda item: item[1])]
    for atom_id in atom_ids:
        atom_candidates = [
            idx
            for idx in selected_indices
            if idx not in used_indices and atom_id in _covered_atom_ids(candidate_pool[idx].get("covered_atom_ids"))
        ]
        if not atom_candidates:
            continue
        primary_idx = min(
            atom_candidates,
            key=lambda idx: _primary_sort_key(candidate_pool[idx], atom_id=atom_id, selected_rank=selected_rank, idx=idx),
        )
        ordered.append(primary_idx)
        used_indices.add(primary_idx)
        role_by_idx[primary_idx] = "primary"
        anchor_atom_by_idx[primary_idx] = atom_id

        if include_secondary:
            secondary_candidates = [
                idx
                for idx in selected_indices
                if idx not in used_indices and atom_id in _covered_atom_ids(candidate_pool[idx].get("covered_atom_ids"))
            ]
            if secondary_candidates:
                primary_relation = _relation_group(candidate_pool[primary_idx].get("map_relation"))
                secondary_idx = min(
                    secondary_candidates,
                    key=lambda idx: _secondary_sort_key(
                        candidate_pool[idx],
                        primary_relation=primary_relation,
                        selected_rank=selected_rank,
                        idx=idx,
                        min_confidence=float(params.min_secondary_confidence),
                    ),
                )
                ordered.append(secondary_idx)
                used_indices.add(secondary_idx)
                role_by_idx[secondary_idx] = "secondary"
                anchor_atom_by_idx[secondary_idx] = atom_id

        if max_steps and len(ordered) >= max_steps:
            ordered = ordered[:max_steps]
            break

    if not ordered and selected_indices:
        fallback_idx = selected_indices[0]
        ordered.append(fallback_idx)
        used_indices.add(fallback_idx)
        role_by_idx[fallback_idx] = "fallback"
        anchor_atom = _best_candidate_atom(candidate_pool[fallback_idx], atom_order)
        if anchor_atom:
            anchor_atom_by_idx[fallback_idx] = anchor_atom

    if fill_to_min and len(ordered) < min_steps:
        for idx in selected_indices:
            if idx in used_indices:
                continue
            ordered.append(idx)
            used_indices.add(idx)
            role_by_idx[idx] = "fallback"
            anchor_atom = _best_candidate_atom(candidate_pool[idx], atom_order)
            if anchor_atom:
                anchor_atom_by_idx[idx] = anchor_atom
            if len(ordered) >= min_steps:
                break

    if max_steps:
        ordered = ordered[:max_steps]
    for idx in ordered:
        role_by_idx.setdefault(idx, "fallback")
        if idx not in anchor_atom_by_idx:
            anchor_atom = _best_candidate_atom(candidate_pool[idx], atom_order)
            if anchor_atom:
                anchor_atom_by_idx[idx] = anchor_atom
    return ordered, role_by_idx, anchor_atom_by_idx


def _stage1_order(
    selected_indices: list[int],
    *,
    candidate_pool: list[dict[str, Any]],
    role_by_idx: dict[int, str],
    anchor_atom_by_idx: dict[int, str],
    atom_order: dict[str, int],
    params: AtomAnchoredQECParams,
) -> list[int]:
    selected_rank = {idx: rank for rank, idx in enumerate(selected_indices)}
    if params.selection_policy == "shuffled":
        out = list(selected_indices)
        random.Random(int(params.random_seed)).shuffle(out)
        return out
    if params.selection_policy == "keep_all_reorder":
        return sorted(
            selected_indices,
            key=lambda idx: (
                _atom_sort_value(anchor_atom_by_idx.get(idx), atom_order),
                selected_rank[idx],
                _candidate_rank(candidate_pool[idx], idx),
            ),
        )
    if params.selection_policy == "primary_secondary_order":
        return sorted(
            selected_indices,
            key=lambda idx: (
                _role_sort_value(role_by_idx.get(idx, "fallback")),
                _atom_sort_value(anchor_atom_by_idx.get(idx), atom_order),
                selected_rank[idx],
                _candidate_rank(candidate_pool[idx], idx),
            ),
        )
    raise ValueError(f"unsupported AA-QEC Stage 1 selection_policy={params.selection_policy!r}")


def _assign_stage1_roles(
    selected_indices: list[int],
    *,
    candidate_pool: list[dict[str, Any]],
    atom_order: dict[str, int],
) -> tuple[dict[int, str], dict[int, str]]:
    role_by_idx: dict[int, str] = {}
    anchor_atom_by_idx: dict[int, str] = {}
    primary_atoms: set[str] = set()
    for idx in selected_indices:
        candidate = candidate_pool[idx]
        anchor_atom = _best_candidate_atom(candidate, atom_order)
        if not anchor_atom:
            role_by_idx[idx] = "fallback"
            continue
        anchor_atom_by_idx[idx] = anchor_atom
        if anchor_atom not in primary_atoms:
            role_by_idx[idx] = "primary"
            primary_atoms.add(anchor_atom)
        else:
            role_by_idx[idx] = "secondary"
    return role_by_idx, anchor_atom_by_idx


def _chain_steps(
    ordered_indices: list[int],
    *,
    candidate_pool: list[dict[str, Any]],
    role_by_idx: dict[int, str],
    anchor_atom_by_idx: dict[int, str],
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    used_evidence_ids: dict[str, int] = {}
    for step_idx, idx in enumerate(ordered_indices, start=1):
        candidate = candidate_pool[idx]
        atom_id = anchor_atom_by_idx.get(idx, "")
        atom = atom_by_id.get(atom_id, {})
        cue = _choose_cue(candidate, atom=atom, atom_by_id=atom_by_id, atom_order=atom_order)
        evidence_id = str(candidate.get("evidence_id") or candidate.get("candidate_uid") or idx)
        anchor_step = used_evidence_ids.get(evidence_id, 0)
        if evidence_id:
            used_evidence_ids.setdefault(evidence_id, step_idx)
        steps.append(
            {
                "step": step_idx,
                "atom_id": atom_id,
                "atom_text": _compact(atom.get("text") or ""),
                "cue_text": str(cue["cue_text"]),
                "cue_source": str(cue["cue_source"]),
                "candidate_idx": int(candidate.get("candidate_idx", idx)),
                "selector_candidate_idx": int(idx),
                "evidence_id": evidence_id,
                "evidence_text": str(candidate.get("text") or ""),
                "role": role_by_idx.get(idx, "fallback"),
                "relation": _compact(candidate.get("map_relation") or "") or "unknown",
                "directness": _compact(candidate.get("map_directness") or "") or "unknown",
                "map_confidence": _float_or_none(candidate.get("map_confidence")),
                "evidence_map_quality_score": _float_or_none(candidate.get("evidence_map_quality_score")),
                "from_qd": bool(candidate.get("from_qd") or candidate.get("qd_question_routes")),
                "qd_question_id": str(cue.get("question_id") or ""),
                "qd_question_rank": cue.get("question_rank"),
                "qd_question_hybrid_score": cue.get("question_hybrid_score"),
                "covered_atom_ids": _covered_atom_ids(candidate.get("covered_atom_ids")),
                "covered_by_previous_step": bool(anchor_step),
                "anchor_step": int(anchor_step),
            }
        )
    return steps


def _choose_cue(
    candidate: dict[str, Any],
    *,
    atom: dict[str, Any],
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> dict[str, Any]:
    route = _best_qd_route(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
    if route is not None:
        return {
            "cue_source": "qd_question",
            "cue_text": _compact(route.get("question") or ""),
            "question_id": _compact(route.get("question_id") or ""),
            "question_rank": _int_or_none(route.get("rank")),
            "question_hybrid_score": _float_or_none(route.get("hybrid_score")),
        }
    if _compact(atom.get("text") or ""):
        return {"cue_source": "claim_atom", "cue_text": _compact(atom.get("text") or "")}
    fallback_atom = _best_covered_atom(candidate.get("covered_atom_ids"), atom_by_id=atom_by_id, atom_order=atom_order)
    if fallback_atom is not None:
        return {"cue_source": "claim_atom", "cue_text": _compact(fallback_atom.get("text") or "")}
    return {"cue_source": "fallback", "cue_text": FALLBACK_CUE}


def _best_qd_route(routes: Any) -> dict[str, Any] | None:
    if not isinstance(routes, list):
        return None
    usable = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        question = _compact(route.get("question") or "")
        if not question or len(question.split()) < 4:
            continue
        if question.lower() in {"is this claim true?", "is the claim true?"}:
            continue
        usable.append(route)
    if not usable:
        return None
    return min(
        usable,
        key=lambda route: (
            _rank_sort_value(route.get("rank")),
            -_float_or_default(route.get("hybrid_score"), 0.0),
            _compact(route.get("question_id") or ""),
        ),
    )


def _best_covered_atom(
    covered_atom_ids: Any,
    *,
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> dict[str, Any] | None:
    atom_ids = [atom_id for atom_id in _covered_atom_ids(covered_atom_ids) if atom_id in atom_by_id]
    if not atom_ids:
        return None
    atom_ids.sort(key=lambda atom_id: atom_order.get(atom_id, 10**9))
    return atom_by_id[atom_ids[0]]


def _chain_diagnostics(
    chain_steps: list[dict[str, Any]],
    *,
    selected_indices: list[int],
    source_selected_indices: list[int] | None = None,
    candidate_scope: str = "selected",
    claim_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    role_counts = Counter(str(step.get("role") or "") for step in chain_steps)
    cue_counts = Counter(str(step.get("cue_source") or "") for step in chain_steps)
    evidence_ids = [str(step.get("evidence_id") or "") for step in chain_steps]
    duplicate_count = len(evidence_ids) - len(set(evidence_ids))
    total = max(len(chain_steps), 1)
    covered_atoms = {
        str(atom_id)
        for step in chain_steps
        for atom_id in step.get("covered_atom_ids") or []
        if str(atom_id)
    }
    claim_atom_ids = {_atom_id(atom) for atom in claim_atoms if _atom_id(atom)}
    covered_claim_atoms = covered_atoms & claim_atom_ids
    n_claim_atoms = max(len(claim_atom_ids), 1)
    multi_atom_count = sum(1 for step in chain_steps if len(step.get("covered_atom_ids") or []) > 1)
    fallback_steps = int(role_counts.get("fallback", 0))
    secondary_steps = int(role_counts.get("secondary", 0))
    return {
        "chain_steps": len(chain_steps),
        "input_selected_count": len(selected_indices),
        "input_candidate_count": len(selected_indices),
        "source_selected_count": len(source_selected_indices or selected_indices),
        "candidate_scope": str(candidate_scope),
        "role_counts": dict(role_counts),
        "cue_source_counts": dict(cue_counts),
        "qd_cue_rate": float(cue_counts.get("qd_question", 0) / total),
        "atom_cue_rate": float(cue_counts.get("claim_atom", 0) / total),
        "claim_fallback_rate": float(cue_counts.get("fallback", 0) / total),
        "primary_step_count": int(role_counts.get("primary", 0)),
        "secondary_step_count": secondary_steps,
        "fallback_step_count": fallback_steps,
        "secondary_step_rate": float(secondary_steps / total),
        "fallback_fill_rate": float(fallback_steps / total),
        "duplicate_evidence_count": int(duplicate_count),
        "duplicate_evidence_rate": float(duplicate_count / total),
        "covered_atom_ids": sorted(covered_atoms),
        "covered_claim_atom_ids": sorted(covered_claim_atoms),
        "atom_coverage_rate": float(len(covered_claim_atoms) / n_claim_atoms),
        "uncovered_atom_rate": float((len(claim_atom_ids) - len(covered_claim_atoms)) / n_claim_atoms),
        "multi_atom_evidence_count": int(multi_atom_count),
        "multi_atom_evidence_rate": float(multi_atom_count / total),
    }


def _numeric_summary(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    return {
        "count": float(count),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
    }


def _claim_atoms(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("claim_atoms")
    if not raw and isinstance(row.get("evidence_map"), dict):
        raw = (row.get("evidence_map") or {}).get("claim_atoms")
    return [dict(atom) for atom in raw or [] if isinstance(atom, dict)]


def _ordered_trace_indices(row: dict[str, Any]) -> list[int]:
    return _int_list(row.get("selector_ordered_indices") or row.get("selected_indices") or [])


def _int_list(values: Any) -> list[int]:
    out: list[int] = []
    for value in values if isinstance(values, list) else []:
        parsed = _int_or_none(value)
        if parsed is not None:
            out.append(parsed)
    return out


def _dedupe_in_range(indices: list[int], n: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _best_candidate_atom(candidate: dict[str, Any], atom_order: dict[str, int]) -> str:
    atoms = [atom_id for atom_id in _covered_atom_ids(candidate.get("covered_atom_ids")) if atom_id in atom_order]
    if not atoms:
        return ""
    atoms.sort(key=lambda atom_id: atom_order.get(atom_id, 10**9))
    return atoms[0]


def _primary_sort_key(
    candidate: dict[str, Any],
    *,
    atom_id: str,
    selected_rank: dict[int, int],
    idx: int,
) -> tuple[Any, ...]:
    relation = _relation_group(candidate.get("map_relation"))
    return (
        _primary_relation_priority(relation),
        _directness_priority(candidate.get("map_directness")),
        -_float_or_default(candidate.get("map_confidence"), 0.0),
        -_candidate_quality_score(candidate),
        -_candidate_base_score(candidate),
        -_candidate_qd_score(candidate),
        len(_covered_atom_ids(candidate.get("covered_atom_ids"))),
        selected_rank.get(idx, 10**9),
        _candidate_rank(candidate, idx),
        atom_id,
        idx,
    )


def _secondary_sort_key(
    candidate: dict[str, Any],
    *,
    primary_relation: str,
    selected_rank: dict[int, int],
    idx: int,
    min_confidence: float,
) -> tuple[Any, ...]:
    relation = _relation_group(candidate.get("map_relation"))
    confidence = _float_or_default(candidate.get("map_confidence"), 0.0)
    return (
        _secondary_relation_priority(primary_relation, relation),
        0 if confidence >= min_confidence else 1,
        _directness_priority(candidate.get("map_directness")),
        -confidence,
        -_candidate_quality_score(candidate),
        -_candidate_base_score(candidate),
        -_candidate_qd_score(candidate),
        selected_rank.get(idx, 10**9),
        _candidate_rank(candidate, idx),
        idx,
    )


def _primary_relation_priority(relation: str) -> int:
    if relation == "support":
        return 0
    if relation == "refute":
        return 1
    if relation in {"qualify", "mixed"}:
        return 2
    if relation == "background":
        return 3
    return 4


def _secondary_relation_priority(primary_relation: str, relation: str) -> int:
    if primary_relation == "support":
        order = {"refute": 0, "qualify": 1, "mixed": 2, "support": 8}
    elif primary_relation == "refute":
        order = {"support": 0, "qualify": 1, "mixed": 2, "refute": 8}
    else:
        order = {"support": 0, "refute": 1, "qualify": 4, "mixed": 4}
    return order.get(relation, 5)


def _relation_group(value: Any) -> str:
    relation = _compact(value or "").lower()
    if relation in {"support", "supports", "supported_by", "entails", "consistent"}:
        return "support"
    if relation in {"refute", "refutes", "contradict", "contradicts", "counter", "conflict"}:
        return "refute"
    if relation in {"qualify", "qualifies", "qualified", "condition", "context", "hedge"}:
        return "qualify"
    if relation in {"mixed", "partially_supports", "partial"}:
        return "mixed"
    if relation in {"background", "unknown", "none", ""}:
        return "background"
    return relation


def _directness_priority(value: Any) -> int:
    directness = _compact(value or "").lower()
    if directness == "direct":
        return 0
    if directness == "partial":
        return 1
    if directness in {"indirect", "context"}:
        return 2
    if directness in {"background", "unknown", ""}:
        return 3
    return 4


def _candidate_quality_score(candidate: dict[str, Any]) -> float:
    return _float_or_default(candidate.get("evidence_map_quality_score"), 0.0)


def _candidate_base_score(candidate: dict[str, Any]) -> float:
    for key in ("evidence_map_base_score", "hybrid_score", "selection_score"):
        value = _float_or_none(candidate.get(key))
        if value is not None:
            return value
    return 0.0


def _candidate_qd_score(candidate: dict[str, Any]) -> float:
    value = _float_or_none(candidate.get("qd_max_question_hybrid"))
    if value is not None:
        return value
    routes = candidate.get("qd_question_routes") or candidate.get("question_routes") or []
    if not isinstance(routes, list):
        return 0.0
    scores = [_float_or_none(route.get("hybrid_score")) for route in routes if isinstance(route, dict)]
    usable = [score for score in scores if score is not None]
    if not usable:
        return 0.0
    return max(usable)


def _covered_atom_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple):
        raw_values = list(value)
    else:
        raw_values = []
    return [_compact(item) for item in raw_values if _compact(item)]


def _atom_id(atom: dict[str, Any]) -> str:
    return _compact(atom.get("atom_id") or atom.get("node_id") or "")


def _atom_sort_value(atom_id: str | None, atom_order: dict[str, int]) -> int:
    if not atom_id:
        return 10**9
    return atom_order.get(str(atom_id), 10**9)


def _role_sort_value(role: str) -> int:
    if role == "primary":
        return 0
    if role == "secondary":
        return 1
    return 2


def _candidate_rank(candidate: dict[str, Any], fallback: int) -> int:
    for key in ("union_pool_rank", "baseline_rank", "qd_pool_rank", "candidate_idx"):
        value = _int_or_none(candidate.get(key))
        if value is not None:
            return value
    return fallback


def _trace_fingerprint(row: dict[str, Any]) -> str:
    if row.get("fingerprint"):
        return str(row.get("fingerprint"))
    metadata = row.get("candidate_pool_metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("chunk_mmr_fingerprint") or metadata.get("fingerprint") or "")
    return ""


def _rank_sort_value(value: Any) -> int:
    parsed = _int_or_none(value)
    if parsed is None:
        return 10**9
    return parsed


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        return float(default)
    return parsed


def _compact(value: Any) -> str:
    return " ".join(str(value).split())
