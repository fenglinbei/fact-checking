from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


GRAPH_VERSION = "atom_anchored_qec_v1"
ADAPTIVE_POLICY = "aa_qec_view"
DEFAULT_SOURCE_SELECTOR_NAME = "v0_7_budgeted_marginal_chain_adaptive5_10"
FALLBACK_CUE = "Verify the main factual claim."


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
    if params.candidate_scope != "selected":
        raise ValueError(f"Stage 1 only supports candidate_scope=selected, got {params.candidate_scope!r}")
    policy = str(params.selection_policy)
    if policy == "keep_all_reorder":
        policy_slug = "keep_all"
    elif policy == "primary_secondary_order":
        policy_slug = "primary_secondary_order"
    elif policy == "shuffled":
        policy_slug = "shuffled"
    else:
        raise ValueError(f"unsupported AA-QEC Stage 1 selection_policy={policy!r}")
    return (
        f"aa_qec_view_{policy_slug}_{params.cue_policy}_"
        f"{params.candidate_scope}_min{int(params.min_chain_steps)}_{int(params.max_chain_steps)}"
    )


def build_atom_anchored_qec_trace_row(
    row: dict[str, Any],
    *,
    params: AtomAnchoredQECParams | None = None,
) -> dict[str, Any]:
    params = params or AtomAnchoredQECParams()
    if params.candidate_scope != "selected":
        raise ValueError("AA-QEC Stage 1 only supports candidate_scope=selected.")

    candidate_pool = [dict(candidate) for candidate in row.get("candidate_pool") or []]
    if not candidate_pool:
        raise ValueError("AA-QEC input trace has no candidate_pool.")

    selected_indices = _dedupe_in_range(_ordered_trace_indices(row), len(candidate_pool))
    if not selected_indices:
        raise ValueError("AA-QEC input trace has no valid selected indices.")

    claim_atoms = _claim_atoms(row)
    atom_by_id = {_atom_id(atom): atom for atom in claim_atoms if _atom_id(atom)}
    atom_order = {_atom_id(atom): idx for idx, atom in enumerate(claim_atoms) if _atom_id(atom)}
    role_by_idx, anchor_atom_by_idx = _assign_stage1_roles(
        selected_indices,
        candidate_pool=candidate_pool,
        atom_order=atom_order,
    )
    ordered_indices = _stage1_order(
        selected_indices,
        candidate_pool=candidate_pool,
        role_by_idx=role_by_idx,
        anchor_atom_by_idx=anchor_atom_by_idx,
        atom_order=atom_order,
        params=params,
    )

    selector_name = atom_anchored_qec_selector_name(params)
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
            "adaptive_policy": ADAPTIVE_POLICY,
            "source_selector_name": str(row.get("selector_name") or params.source_selector_name),
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
        "chain_diagnostics": _chain_diagnostics(chain_steps, selected_indices=selected_indices),
        "params": asdict(params),
        "adaptive_policy": ADAPTIVE_POLICY,
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

    return {
        "n_rows": len(rows),
        "selector_names": dict(selector_names),
        "chain_steps": _numeric_summary(chain_lengths),
        "role_counts": {key: value for key, value in dict(role_counts).items() if key},
        "cue_source_counts": {key: value for key, value in dict(cue_source_counts).items() if key},
        "qd_cue_rate": _numeric_summary(qd_rates),
        "atom_cue_rate": _numeric_summary(atom_rates),
        "claim_fallback_rate": _numeric_summary(fallback_rates),
    }


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


def _chain_diagnostics(chain_steps: list[dict[str, Any]], *, selected_indices: list[int]) -> dict[str, Any]:
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
    return {
        "chain_steps": len(chain_steps),
        "input_selected_count": len(selected_indices),
        "role_counts": dict(role_counts),
        "cue_source_counts": dict(cue_counts),
        "qd_cue_rate": float(cue_counts.get("qd_question", 0) / total),
        "atom_cue_rate": float(cue_counts.get("claim_atom", 0) / total),
        "claim_fallback_rate": float(cue_counts.get("fallback", 0) / total),
        "primary_step_count": int(role_counts.get("primary", 0)),
        "secondary_step_count": int(role_counts.get("secondary", 0)),
        "fallback_step_count": int(role_counts.get("fallback", 0)),
        "duplicate_evidence_count": int(duplicate_count),
        "duplicate_evidence_rate": float(duplicate_count / total),
        "covered_atom_ids": sorted(covered_atoms),
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
