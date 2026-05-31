from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    selection_quality_metrics,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.evidence_map_selector import evidence_map_selection_metrics


GRAPH_VERSION = "evidence_chain_graph_v0_6b"
CHAIN_SELECTOR = "v0_6b_chain_graph_top5"
RULE_STEP_GRAPH_VERSION = "evidence_chain_graph_v0_6c"
RULE_STEP_CHAIN_SELECTOR = "v0_6c_rule_step_adaptive5_10"
DEFAULT_CHUNK_MMR_FINGERPRINT = "432dfc970e75"

POSITIVE_CHAIN_EDGE_TYPES = {"complements", "corroborates", "tension", "bridge_context"}
RULE_STEP_STRONG_EDGE_TYPES = {"complements", "corroborates", "tension"}
POST_ORDER_EDGE_REWARDS = {
    "complements": 0.35,
    "corroborates": 0.25,
    "tension": 0.20,
    "bridge_context": 0.12,
    "same_source_context": 0.04,
}
BACKGROUND_RELATIONS = {"background", "irrelevant"}
DIRECTNESS_VALUES = {"direct", "partial"}
POLAR_RELATIONS = {"support", "refute", "qualify", "mixed"}


@dataclass(frozen=True)
class EvidenceChainParams:
    candidate_top_n: int = 20
    top_k: int = 5
    beam_size: int = 12
    chunk_mmr_fingerprint: str = DEFAULT_CHUNK_MMR_FINGERPRINT


@dataclass(frozen=True)
class RuleStepEvidenceChainParams:
    candidate_top_n: int = 20
    min_top_k: int = 5
    max_top_k: int = 10
    chunk_mmr_fingerprint: str = DEFAULT_CHUNK_MMR_FINGERPRINT


def build_evidence_chain_graph_row(row: dict[str, Any], *, params: EvidenceChainParams | None = None) -> dict[str, Any]:
    params = params or EvidenceChainParams()
    candidates = _sorted_candidates(row.get("candidates") or [], candidate_top_n=params.candidate_top_n)
    atom_nodes = _atom_nodes(row)
    evidence_nodes = [_evidence_node(candidate, idx=idx) for idx, candidate in enumerate(candidates, start=1)]
    atom_by_id = {str(atom.get("node_id") or ""): atom for atom in atom_nodes}
    evidence_by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    edges = _build_edges(atom_nodes, evidence_nodes)
    edge_index = _edge_index(edges)
    chains = _build_chains(evidence_nodes, atom_by_id=atom_by_id, edge_index=edge_index, params=params)
    selected_chain = chains[0] if chains else _empty_chain()
    selected_ids = list(selected_chain.get("evidence_ids") or [])
    selected_candidates = [_candidate_for_evidence_id(evidence_by_id[eid]) for eid in selected_ids if eid in evidence_by_id]
    graph_row = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "graph_version": GRAPH_VERSION,
        "selector_name": CHAIN_SELECTOR,
        "params": {
            "candidate_top_n": int(params.candidate_top_n),
            "top_k": int(params.top_k),
            "beam_size": int(params.beam_size),
            "chunk_mmr_fingerprint": str(params.chunk_mmr_fingerprint or ""),
        },
        "fingerprint": str(params.chunk_mmr_fingerprint or ""),
        "candidate_pool_metadata": {
            "chunk_mmr_fingerprint": str(params.chunk_mmr_fingerprint or ""),
        },
        "claim_node": {"node_id": "C0", "type": "claim", "text": str(row.get("claim") or "")},
        "atom_nodes": atom_nodes,
        "evidence_nodes": evidence_nodes,
        "edges": edges,
        "chains": chains,
        "selected_chain_id": str(selected_chain.get("chain_id") or ""),
        "selected_evidence_ids": selected_ids,
        "selected_candidates": selected_candidates,
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "diagnostics": _graph_diagnostics(atom_nodes, evidence_nodes, edges, selected_chain),
    }
    graph_row["selection_trace"] = build_evidence_chain_trace(row, graph_row, top_k=params.top_k)
    return graph_row


def build_rule_step_evidence_chain_graph_row(
    row: dict[str, Any],
    *,
    params: RuleStepEvidenceChainParams | None = None,
) -> dict[str, Any]:
    params = params or RuleStepEvidenceChainParams()
    min_top_k = max(1, int(params.min_top_k))
    max_top_k = max(min_top_k, int(params.max_top_k))
    candidates = _sorted_candidates(row.get("candidates") or [], candidate_top_n=params.candidate_top_n)
    atom_nodes = _atom_nodes(row)
    evidence_nodes = [_evidence_node(candidate, idx=idx) for idx, candidate in enumerate(candidates, start=1)]
    atom_by_id = {str(atom.get("node_id") or ""): atom for atom in atom_nodes}
    evidence_by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    edges = _build_edges(atom_nodes, evidence_nodes)
    edge_index = _edge_index(edges)
    rule_result = _select_rule_step_evidence_ids(
        evidence_nodes,
        atom_by_id=atom_by_id,
        edge_index=edge_index,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
    )
    selected_ids = list(rule_result.get("evidence_ids") or [])
    selected_candidates = [_candidate_for_evidence_id(evidence_by_id[eid]) for eid in selected_ids if eid in evidence_by_id]
    selected_chain = _rule_step_chain_summary(
        selected_ids,
        evidence_nodes=evidence_nodes,
        atom_by_id=atom_by_id,
        edge_index=edge_index,
        rule_result=rule_result,
    )
    graph_row = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "graph_version": RULE_STEP_GRAPH_VERSION,
        "selector_name": RULE_STEP_CHAIN_SELECTOR,
        "params": {
            "candidate_top_n": int(params.candidate_top_n),
            "min_top_k": min_top_k,
            "max_top_k": max_top_k,
            "chunk_mmr_fingerprint": str(params.chunk_mmr_fingerprint or ""),
        },
        "fingerprint": str(params.chunk_mmr_fingerprint or ""),
        "candidate_pool_metadata": {
            "chunk_mmr_fingerprint": str(params.chunk_mmr_fingerprint or ""),
        },
        "claim_node": {"node_id": "C0", "type": "claim", "text": str(row.get("claim") or "")},
        "atom_nodes": atom_nodes,
        "evidence_nodes": evidence_nodes,
        "edges": edges,
        "chains": [selected_chain] if selected_chain.get("evidence_ids") else [],
        "selected_chain_id": str(selected_chain.get("chain_id") or ""),
        "selected_evidence_ids": selected_ids,
        "selected_candidates": selected_candidates,
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "adaptive_policy": "rule_step_v0_6c",
        "min_top_k": min_top_k,
        "max_top_k": max_top_k,
        "adaptive_evidence_count": len(selected_ids),
        "adaptive_stop_reason": str(rule_result.get("stop_reason") or ""),
        "selection_steps": list(rule_result.get("selection_steps") or []),
        "adaptive_additions": [
            dict(step)
            for step in rule_result.get("selection_steps") or []
            if int(step.get("step", 0)) > min_top_k or bool(step.get("fallback_used"))
        ],
        "diagnostics": _graph_diagnostics(atom_nodes, evidence_nodes, edges, selected_chain),
    }
    graph_row["selection_trace"] = build_evidence_chain_trace(row, graph_row, top_k=max_top_k)
    return graph_row


def build_evidence_chain_trace(row: dict[str, Any], graph_row: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    selected = list(graph_row.get("selected_candidates") or [])
    candidate_pool = _pipeline_candidate_pool(graph_row)
    candidate_scores = _pipeline_candidate_scores(graph_row, candidate_pool, top_k=top_k)
    selector_ordered_indices = _selector_ordered_pool_indices(graph_row, candidate_pool)
    oracle_ordered_indices = _oracle_ordered_pool_indices(
        list(row.get("oracle_ordered_keys") or graph_row.get("oracle_ordered_keys") or []),
        candidate_pool,
    )
    fingerprint = str(graph_row.get("fingerprint") or (graph_row.get("candidate_pool_metadata") or {}).get("chunk_mmr_fingerprint") or "")
    selector_name = str(graph_row.get("selector_name") or CHAIN_SELECTOR)
    graph_version = str(graph_row.get("graph_version") or GRAPH_VERSION)
    trace = {
        "event_id": str(row.get("event_id") or graph_row.get("event_id") or ""),
        "claim": str(row.get("claim") or graph_row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or graph_row.get("gold_label") or ""),
        "selector_name": selector_name,
        "graph_version": graph_version,
        "fingerprint": fingerprint,
        "candidate_pool_metadata": {
            "chunk_mmr_fingerprint": fingerprint,
            "graph_version": graph_version,
            "selector_name": selector_name,
        },
        "candidate_pool": candidate_pool,
        "candidate_scores": candidate_scores,
        "selector_ordered_indices": selector_ordered_indices,
        "selected_indices": selector_ordered_indices,
        "oracle_ordered_indices": oracle_ordered_indices,
        "selected_chain_id": str(graph_row.get("selected_chain_id") or ""),
        "selected_evidence_ids": list(graph_row.get("selected_evidence_ids") or []),
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or graph_row.get("oracle_ordered_keys") or []),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_candidates": [_candidate_trace_output(candidate) for candidate in selected],
        "chain_summary": _selected_chain_summary(graph_row),
        "claim_atoms": [dict(atom) for atom in graph_row.get("atom_nodes") or []],
    }
    for key in (
        "adaptive_policy",
        "min_top_k",
        "max_top_k",
        "adaptive_evidence_count",
        "adaptive_stop_reason",
        "selection_steps",
        "adaptive_additions",
    ):
        if key in graph_row:
            trace[key] = graph_row[key]
    trace.update(_text_ordered_selection_metrics_multi(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected[: min(5, int(top_k), len(selected))]))
    trace.update(_evidence_map_selection_metrics_multi(graph_row, selected, top_k=top_k))
    return trace


def summarize_chain_graph_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    traces = [row.get("selection_trace") or {} for row in rows]
    diagnostics = [row.get("diagnostics") or {} for row in rows]
    edge_counts: Counter[str] = Counter()
    for row in rows:
        edge_counts.update(str(edge.get("edge_type") or "") for edge in row.get("edges") or [])
    return {
        "graph_version": GRAPH_VERSION,
        "selector_name": CHAIN_SELECTOR,
        "n_events": len(rows),
        "n_edges_by_type": dict(sorted(edge_counts.items())),
        "mean_atom_isolate_rate": _mean(_safe_float(item.get("atom_isolate_rate"), 0.0) for item in diagnostics),
        "mean_evidence_isolate_rate": _mean(_safe_float(item.get("evidence_isolate_rate"), 0.0) for item in diagnostics),
        "mean_oracle_evidence_connected_rate": _mean(_safe_float(item.get("oracle_evidence_connected_rate"), 0.0) for item in diagnostics),
        "mean_oracle_pair_edge_rate": _mean(_safe_float(item.get("oracle_pair_edge_rate"), 0.0) for item in diagnostics),
        "mean_max_chain_atom_coverage": _mean(_safe_float(item.get("max_chain_atom_coverage"), 0.0) for item in diagnostics),
        "mean_selected_chain_score": _mean(_safe_float((row.get("chains") or [{}])[0].get("chain_score"), 0.0) for row in rows),
        "selection_metrics": _summarize_traces(traces),
    }


def summarize_rule_step_chain_graph_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    traces = [row.get("selection_trace") or {} for row in rows]
    diagnostics = [row.get("diagnostics") or {} for row in rows]
    edge_counts: Counter[str] = Counter()
    selected_lengths: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    step_rules: Counter[str] = Counter()
    fallback_rows = 0
    p3_rows = 0
    background_additions = 0
    total_additions = 0
    for row in rows:
        edge_counts.update(str(edge.get("edge_type") or "") for edge in row.get("edges") or [])
        selected_lengths[str(len(row.get("selected_evidence_ids") or []))] += 1
        stop_reasons[str(row.get("adaptive_stop_reason") or "")] += 1
        row_has_fallback = False
        row_has_p3 = False
        for step in row.get("selection_steps") or []:
            rule = str(step.get("rule") or "")
            step_rules[rule] += 1
            if bool(step.get("fallback_used")):
                row_has_fallback = True
            if rule == "P3_bridge_context":
                row_has_p3 = True
            if int(step.get("step", 0)) > int(row.get("min_top_k") or 5):
                total_additions += 1
                if str(step.get("relation") or "") in BACKGROUND_RELATIONS:
                    background_additions += 1
        if row_has_fallback:
            fallback_rows += 1
        if row_has_p3:
            p3_rows += 1
    return {
        "graph_version": RULE_STEP_GRAPH_VERSION,
        "selector_name": RULE_STEP_CHAIN_SELECTOR,
        "n_events": len(rows),
        "n_edges_by_type": dict(sorted(edge_counts.items())),
        "selected_lengths": dict(sorted(selected_lengths.items())),
        "adaptive_stop_reasons": dict(sorted(stop_reasons.items())),
        "selection_rules": dict(sorted(step_rules.items())),
        "fallback_row_rate": float(fallback_rows / max(len(rows), 1)),
        "p3_row_rate": float(p3_rows / max(len(rows), 1)),
        "post_min_background_addition_rate": float(background_additions / max(total_additions, 1)),
        "mean_atom_isolate_rate": _mean(_safe_float(item.get("atom_isolate_rate"), 0.0) for item in diagnostics),
        "mean_evidence_isolate_rate": _mean(_safe_float(item.get("evidence_isolate_rate"), 0.0) for item in diagnostics),
        "mean_oracle_evidence_connected_rate": _mean(_safe_float(item.get("oracle_evidence_connected_rate"), 0.0) for item in diagnostics),
        "mean_oracle_pair_edge_rate": _mean(_safe_float(item.get("oracle_pair_edge_rate"), 0.0) for item in diagnostics),
        "mean_max_chain_atom_coverage": _mean(_safe_float(item.get("max_chain_atom_coverage"), 0.0) for item in diagnostics),
        "selection_metrics": _summarize_traces(traces),
    }


def render_case_studies(rows: Sequence[dict[str, Any]], *, top_n: int = 5) -> str:
    ranked = sorted(rows, key=lambda row: _safe_float((row.get("chains") or [{}])[0].get("chain_score"), 0.0), reverse=True)
    oracle_ranked = sorted(rows, key=lambda row: _safe_float((row.get("selection_trace") or {}).get("jaccard@5"), 0.0), reverse=True)
    lines = ["# Evidence Chain Graph v0.6b Case Studies", ""]
    for title, items in (("Highest Chain Scores", ranked[:top_n]), ("Highest Oracle Overlap", oracle_ranked[:top_n])):
        lines.extend([f"## {title}", ""])
        if not items:
            lines.extend(["(none)", ""])
            continue
        for row in items:
            trace = row.get("selection_trace") or {}
            chain = (row.get("chains") or [{}])[0]
            lines.extend(
                [
                    f"### {row.get('event_id')} chain={chain.get('chain_id')} score={_safe_float(chain.get('chain_score'), 0.0):.4f}",
                    "",
                    f"Claim: {row.get('claim', '')}",
                    "",
                    f"- jaccard@5: {_safe_float(trace.get('jaccard@5'), 0.0):.4f}",
                    f"- recall@5: {_safe_float(trace.get('recall@5'), 0.0):.4f}",
                    f"- weighted atom coverage: {_safe_float(chain.get('weighted_atom_coverage'), 0.0):.4f}",
                    f"- evidence ids: {chain.get('evidence_ids', [])}",
                    "",
                ]
            )
    return "\n".join(lines)


def render_rule_step_case_studies(rows: Sequence[dict[str, Any]], *, top_n: int = 5) -> str:
    ranked = sorted(rows, key=lambda row: len(row.get("selected_evidence_ids") or []), reverse=True)
    fallback_ranked = sorted(rows, key=lambda row: sum(1 for step in row.get("selection_steps") or [] if bool(step.get("fallback_used"))), reverse=True)
    lines = ["# Evidence Chain Graph v0.6c Rule-Step Case Studies", ""]
    for title, items in (("Longest Adaptive Selections", ranked[:top_n]), ("Most Fallback Steps", fallback_ranked[:top_n])):
        lines.extend([f"## {title}", ""])
        if not items:
            lines.extend(["(none)", ""])
            continue
        for row in items:
            trace = row.get("selection_trace") or {}
            lines.extend(
                [
                    f"### {row.get('event_id')} selected={len(row.get('selected_evidence_ids') or [])} stop={row.get('adaptive_stop_reason', '')}",
                    "",
                    f"Claim: {row.get('claim', '')}",
                    "",
                    f"- jaccard@5: {_safe_float(trace.get('jaccard@5'), 0.0):.4f}",
                    f"- jaccard@10: {_safe_float(trace.get('jaccard@10'), 0.0):.4f}",
                    f"- evidence ids: {row.get('selected_evidence_ids', [])}",
                    f"- rules: {[step.get('rule') for step in row.get('selection_steps') or []]}",
                    "",
                ]
            )
    return "\n".join(lines)


def _atom_nodes(row: dict[str, Any]) -> list[dict[str, Any]]:
    atoms = list((row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or [])
    out: list[dict[str, Any]] = []
    for idx, atom in enumerate(atoms, start=1):
        atom_id = str(atom.get("atom_id") or f"A{idx}")
        out.append(
            {
                "node_id": atom_id,
                "type": "atom",
                "atom_id": atom_id,
                "text": str(atom.get("text") or ""),
                "atom_type": str(atom.get("type") or "other"),
                "importance": _importance_to_unit(atom.get("importance", 1.0)),
            }
        )
    if not out:
        out.append({"node_id": "A1", "type": "atom", "atom_id": "A1", "text": "Full claim", "atom_type": "other", "importance": 1.0})
    return out


def _evidence_node(candidate: dict[str, Any], *, idx: int) -> dict[str, Any]:
    evidence_id = str(candidate.get("evidence_id") or f"E{idx:02d}")
    relation = str(candidate.get("map_relation") or "irrelevant")
    directness = str(candidate.get("map_directness") or "none")
    source = _source_group(candidate)
    oracle_step = _safe_int(candidate.get("oracle_step"))
    node = {
        "node_id": evidence_id,
        "type": "evidence",
        "evidence_id": evidence_id,
        "candidate_uid": str(candidate.get("candidate_uid") or ""),
        "candidate_key": str(candidate.get("candidate_key") or ""),
        "text": str(candidate.get("text") or ""),
        "covered_atom_ids": [str(atom_id) for atom_id in candidate.get("covered_atom_ids") or []],
        "relation": relation,
        "directness": directness,
        "evidence_role": str(candidate.get("map_evidence_role") or ""),
        "key_spans": [str(span) for span in candidate.get("key_spans") or []],
        "duplicate_group": str(candidate.get("duplicate_group") or ""),
        "source_group": source,
        "source_domain": str(candidate.get("source_domain") or ""),
        "report_id": candidate.get("report_id"),
        "sent_idx": candidate.get("sent_idx"),
        "base_score": _base_score(candidate),
        "evidence_map_quality_score": _safe_float(candidate.get("evidence_map_quality_score"), 0.0),
        "union_pool_rank": candidate.get("union_pool_rank"),
        "oracle_selected": bool(candidate.get("oracle_selected")),
        "oracle_step": oracle_step if oracle_step is not None else -1,
        "candidate": dict(candidate),
        "is_background": relation in BACKGROUND_RELATIONS or directness in {"context", "none"},
    }
    if "fusion_refit_score" in candidate:
        node["fusion_refit_score"] = _safe_float(candidate.get("fusion_refit_score"), 0.0)
    return node


def _build_edges(atom_nodes: Sequence[dict[str, Any]], evidence_nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    atom_ids = {str(atom.get("node_id") or "") for atom in atom_nodes}
    for atom in atom_nodes:
        edges.append(_edge("claim_has_atom", "C0", str(atom.get("node_id") or ""), weight=_safe_float(atom.get("importance"), 1.0)))
    for evidence in evidence_nodes:
        for atom_id in evidence.get("covered_atom_ids") or []:
            if atom_id not in atom_ids:
                continue
            edges.append(
                _edge(
                    "evidence_covers_atom",
                    str(evidence.get("node_id") or ""),
                    atom_id,
                    weight=max(0.1, _safe_float(evidence.get("evidence_map_quality_score"), 0.0)),
                    atom_ids=[atom_id],
                    relation=evidence.get("relation"),
                    directness=evidence.get("directness"),
                )
            )
    for i, left in enumerate(evidence_nodes):
        for right in evidence_nodes[i + 1 :]:
            edges.extend(_pair_edges(left, right))
    for idx, edge in enumerate(edges, start=1):
        edge["edge_id"] = f"EG{idx:04d}"
    return edges


def _pair_edges(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    left_id = str(left.get("node_id") or "")
    right_id = str(right.get("node_id") or "")
    left_atoms = set(str(atom_id) for atom_id in left.get("covered_atom_ids") or [])
    right_atoms = set(str(atom_id) for atom_id in right.get("covered_atom_ids") or [])
    shared_atoms = sorted(left_atoms & right_atoms)
    duplicate = _is_duplicate_pair(left, right)
    if duplicate:
        out.append(_edge("duplicate", left_id, right_id, weight=1.0, reason="same duplicate_group or normalized text"))
    if _same_source_context(left, right):
        out.append(_edge("same_source_context", left_id, right_id, weight=0.45, reason="same source and nearby sentence"))
    if not duplicate and left_atoms and right_atoms and left_atoms != right_atoms and (left_atoms | right_atoms):
        if left.get("directness") in DIRECTNESS_VALUES or right.get("directness") in DIRECTNESS_VALUES:
            new_ratio = len(left_atoms ^ right_atoms) / max(len(left_atoms | right_atoms), 1)
            out.append(_edge("complements", left_id, right_id, weight=0.45 + 0.55 * new_ratio, atom_ids=sorted(left_atoms | right_atoms)))
    if shared_atoms and _source_group(left) != _source_group(right) and _same_direction(left, right):
        out.append(_edge("corroborates", left_id, right_id, weight=0.75, atom_ids=shared_atoms, relation=left.get("relation")))
    if shared_atoms and _has_tension(left, right):
        out.append(_edge("tension", left_id, right_id, weight=0.85, atom_ids=shared_atoms, reason="shared atom with conflicting relation"))
    if _bridge_context(left, right, shared_atoms):
        out.append(_edge("bridge_context", left_id, right_id, weight=0.55, atom_ids=shared_atoms, reason="context evidence linked to direct evidence"))
    return out


def _build_chains(
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    params: EvidenceChainParams,
) -> list[dict[str, Any]]:
    if not evidence_nodes:
        return []
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    start_ids = [str(node.get("node_id") or "") for node in sorted(evidence_nodes, key=_start_key, reverse=True)[: max(params.beam_size, 1)]]
    beams: list[list[str]] = [[node_id] for node_id in start_ids if node_id]
    for _slot in range(2, min(int(params.top_k), len(evidence_nodes)) + 1):
        next_beams: dict[tuple[str, ...], tuple[float, list[str]]] = {}
        for chain in beams:
            extension_ids = _extension_ids(chain, evidence_nodes, edge_index=edge_index, atom_by_id=atom_by_id, limit=max(params.beam_size, 4))
            for evidence_id in extension_ids:
                if evidence_id in chain:
                    continue
                candidate_chain = chain + [evidence_id]
                score = _chain_score(candidate_chain, by_id=by_id, atom_by_id=atom_by_id, edge_index=edge_index)["chain_score"]
                key = tuple(candidate_chain)
                previous = next_beams.get(key)
                if previous is None or score > previous[0]:
                    next_beams[key] = (score, candidate_chain)
        if not next_beams:
            break
        beams = [chain for _, chain in sorted(next_beams.values(), key=lambda item: item[0], reverse=True)[: max(params.beam_size, 1)]]
    chains = []
    seen: set[tuple[str, ...]] = set()
    for chain_ids in beams:
        key = tuple(sorted(chain_ids))
        if key in seen:
            continue
        seen.add(key)
        payload = _chain_score(chain_ids, by_id=by_id, atom_by_id=atom_by_id, edge_index=edge_index)
        search_order = list(payload.get("evidence_ids") or [])
        ordered_ids, post_order = _post_order_evidence_ids(
            search_order,
            by_id=by_id,
            atom_by_id=atom_by_id,
            edge_index=edge_index,
        )
        payload["search_order_evidence_ids"] = search_order
        payload["evidence_ids"] = ordered_ids
        payload["post_order"] = post_order
        chains.append(payload)
    chains.sort(key=lambda chain: (_safe_float(chain.get("chain_score"), 0.0), _safe_float(chain.get("weighted_atom_coverage"), 0.0)), reverse=True)
    for idx, chain in enumerate(chains, start=1):
        chain["chain_id"] = f"CH{idx:02d}"
        chain["rank"] = idx
    return chains


def _select_rule_step_evidence_ids(
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    min_top_k: int,
    max_top_k: int,
) -> dict[str, Any]:
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    selected: list[str] = []
    steps: list[dict[str, Any]] = []
    stop_reason = ""

    while len(selected) < min(max_top_k, len(evidence_nodes)):
        pick = _best_rule_step_candidate(
            selected,
            evidence_nodes,
            atom_by_id=atom_by_id,
            edge_index=edge_index,
        )
        fallback_used = False
        if pick is None:
            if len(selected) >= int(min_top_k):
                stop_reason = "reached_min_top_k_no_rule_candidate"
                break
            pick = _best_rule_step_fallback(
                selected,
                evidence_nodes,
                atom_by_id=atom_by_id,
                edge_index=edge_index,
            )
            fallback_used = True
        if pick is None:
            stop_reason = "pool_exhausted_before_min_top_k" if len(selected) < int(min_top_k) else "pool_exhausted"
            break

        evidence_id = str(pick["evidence_id"])
        if evidence_id in selected or evidence_id not in by_id:
            stop_reason = "duplicate_pick_guard"
            break
        selected.append(evidence_id)
        steps.append(
            {
                "step": len(selected),
                "evidence_id": evidence_id,
                "rule": str(pick.get("rule") or ""),
                "covered_new_atom_ids": list(pick.get("covered_new_atom_ids") or []),
                "anchor_evidence_ids": list(pick.get("anchor_evidence_ids") or []),
                "fallback_used": bool(fallback_used or pick.get("fallback_used")),
                "directness": str(by_id[evidence_id].get("directness") or ""),
                "relation": str(by_id[evidence_id].get("relation") or ""),
            }
        )

    if not stop_reason:
        stop_reason = "reached_max_top_k" if len(selected) >= int(max_top_k) else "pool_exhausted"
    return {
        "evidence_ids": selected,
        "selection_steps": steps,
        "stop_reason": stop_reason,
    }


def _best_rule_step_candidate(
    selected: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    selected_set = {str(eid) for eid in selected}
    selected_atoms = _covered_atoms_for_ids(selected, evidence_nodes)
    if not selected:
        rows = []
        for node in evidence_nodes:
            evidence_id = str(node.get("node_id") or "")
            covered = _covered_atoms(node, atom_by_id=atom_by_id)
            if _is_rule_core_node(node) and covered:
                rows.append(
                    {
                        "evidence_id": evidence_id,
                        "rule": "anchor_core",
                        "covered_new_atom_ids": covered,
                        "anchor_evidence_ids": [],
                        "sort_key": _rule_step_sort_key(node, covered, [], atom_by_id=atom_by_id, selected=evidence_nodes),
                    }
                )
        return _min_pick(rows)

    for rule_name, builder in (
        ("P1_new_atom_core", _rule_step_p1_candidates),
        ("P2_strong_edge_core", _rule_step_p2_candidates),
        ("P3_bridge_context", _rule_step_p3_candidates),
    ):
        rows = builder(
            selected,
            evidence_nodes,
            atom_by_id=atom_by_id,
            edge_index=edge_index,
            selected_set=selected_set,
            selected_atoms=selected_atoms,
            rule_name=rule_name,
        )
        pick = _min_pick(rows)
        if pick is not None:
            return pick
    return None


def _rule_step_p1_candidates(
    selected: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    selected_set: set[str],
    selected_atoms: set[str],
    rule_name: str,
) -> list[dict[str, Any]]:
    del selected, edge_index
    rows: list[dict[str, Any]] = []
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        if evidence_id in selected_set or not _is_rule_core_node(node):
            continue
        new_atoms = [atom_id for atom_id in _covered_atoms(node, atom_by_id=atom_by_id) if atom_id not in selected_atoms]
        if not new_atoms:
            continue
        rows.append(
            {
                "evidence_id": evidence_id,
                "rule": rule_name,
                "covered_new_atom_ids": new_atoms,
                "anchor_evidence_ids": [],
                "sort_key": _rule_step_sort_key(node, new_atoms, [], atom_by_id=atom_by_id, selected=evidence_nodes),
            }
        )
    return rows


def _rule_step_p2_candidates(
    selected: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    selected_set: set[str],
    selected_atoms: set[str],
    rule_name: str,
) -> list[dict[str, Any]]:
    del selected_atoms
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    core_anchor_ids = [str(eid) for eid in selected if _is_rule_core_node(by_id.get(str(eid), {}))]
    rows: list[dict[str, Any]] = []
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        if evidence_id in selected_set or not _is_rule_core_node(node):
            continue
        anchor_ids, edge_rank = _rule_step_anchor_ids(
            evidence_id,
            core_anchor_ids,
            edge_index=edge_index,
            allowed_edges=RULE_STEP_STRONG_EDGE_TYPES,
        )
        if not anchor_ids:
            continue
        covered = _covered_atoms(node, atom_by_id=atom_by_id)
        rows.append(
            {
                "evidence_id": evidence_id,
                "rule": rule_name,
                "covered_new_atom_ids": [],
                "anchor_evidence_ids": anchor_ids,
                "sort_key": _rule_step_sort_key(
                    node,
                    covered,
                    anchor_ids,
                    atom_by_id=atom_by_id,
                    selected=evidence_nodes,
                    edge_rank=edge_rank,
                ),
            }
        )
    return rows


def _rule_step_p3_candidates(
    selected: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    selected_set: set[str],
    selected_atoms: set[str],
    rule_name: str,
) -> list[dict[str, Any]]:
    del selected_atoms
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    core_anchor_ids = [str(eid) for eid in selected if _is_rule_core_node(by_id.get(str(eid), {}))]
    rows: list[dict[str, Any]] = []
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        if evidence_id in selected_set:
            continue
        if str(node.get("relation") or "") == "irrelevant" or str(node.get("directness") or "") == "none":
            continue
        anchor_ids, edge_rank = _rule_step_anchor_ids(
            evidence_id,
            core_anchor_ids,
            edge_index=edge_index,
            allowed_edges={"bridge_context"},
        )
        if not anchor_ids:
            continue
        covered = _covered_atoms(node, atom_by_id=atom_by_id)
        rows.append(
            {
                "evidence_id": evidence_id,
                "rule": rule_name,
                "covered_new_atom_ids": [],
                "anchor_evidence_ids": anchor_ids,
                "sort_key": _rule_step_sort_key(
                    node,
                    covered,
                    anchor_ids,
                    atom_by_id=atom_by_id,
                    selected=evidence_nodes,
                    edge_rank=edge_rank,
                    context_rule=True,
                ),
            }
        )
    return rows


def _best_rule_step_fallback(
    selected: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    del edge_index
    selected_set = {str(eid) for eid in selected}
    selected_atoms = _covered_atoms_for_ids(selected, evidence_nodes)
    rows: list[dict[str, Any]] = []
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        if evidence_id in selected_set:
            continue
        covered = _covered_atoms(node, atom_by_id=atom_by_id)
        new_atoms = [atom_id for atom_id in covered if atom_id not in selected_atoms]
        rows.append(
            {
                "evidence_id": evidence_id,
                "rule": "fallback_core_first",
                "covered_new_atom_ids": new_atoms,
                "anchor_evidence_ids": [],
                "fallback_used": True,
                "sort_key": _fallback_sort_key(node, new_atoms, atom_by_id=atom_by_id, selected=evidence_nodes),
            }
        )
    return _min_pick(rows)


def _min_pick(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return dict(min(rows, key=lambda row: row.get("sort_key") or ()))


def _rule_step_chain_summary(
    evidence_ids: Sequence[str],
    *,
    evidence_nodes: Sequence[dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    rule_result: dict[str, Any],
) -> dict[str, Any]:
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    ids = [str(eid) for eid in evidence_ids if str(eid) in by_id]
    nodes = [by_id[eid] for eid in ids]
    total_weight = sum(_safe_float(atom.get("importance"), 1.0) for atom in atom_by_id.values()) or 1.0
    covered_atoms = set(atom_id for node in nodes for atom_id in node.get("covered_atom_ids") or [] if atom_id in atom_by_id)
    covered_weight = sum(_safe_float(atom_by_id[atom_id].get("importance"), 1.0) for atom_id in covered_atoms)
    pair_counts = _pair_edge_counts(ids, edge_index=edge_index)
    duplicate_groups = [str(node.get("duplicate_group") or "") for node in nodes if node.get("duplicate_group")]
    source_groups = [str(node.get("source_group") or "") for node in nodes if node.get("source_group")]
    return {
        "chain_id": "CH01" if ids else "",
        "rank": 1 if ids else 0,
        "evidence_ids": ids,
        "search_order_evidence_ids": ids,
        "selection_steps": list(rule_result.get("selection_steps") or []),
        "adaptive_stop_reason": str(rule_result.get("stop_reason") or ""),
        "chain_score": 0.0,
        "rule_step_score": float(len(ids)),
        "mean_base_score": _mean(_safe_float(node.get("base_score"), 0.0) for node in nodes),
        "weighted_atom_coverage": float(covered_weight / total_weight),
        "covered_atom_ids": sorted(covered_atoms),
        "direct_or_partial_rate": _mean(1.0 if node.get("directness") in DIRECTNESS_VALUES else 0.0 for node in nodes),
        "background_rate": _mean(1.0 if bool(node.get("is_background")) else 0.0 for node in nodes),
        "edge_counts": dict(sorted(pair_counts.items())),
        "positive_pair_edge_density": float(
            sum(pair_counts.get(edge_type, 0) for edge_type in POSITIVE_CHAIN_EDGE_TYPES)
            / max(len(ids) * (len(ids) - 1) / 2.0, 1.0)
        ),
        "duplicate_repeat_count": int(sum(max(count - 1, 0) for count in Counter(duplicate_groups).values())),
        "same_source_excess_count": int(sum(max(count - 2, 0) for count in Counter(source_groups).values())),
    }


def _is_rule_core_node(node: dict[str, Any]) -> bool:
    return (
        str(node.get("directness") or "") in DIRECTNESS_VALUES
        and str(node.get("relation") or "") in POLAR_RELATIONS
        and bool(node.get("covered_atom_ids") or [])
    )


def _covered_atoms(node: dict[str, Any], *, atom_by_id: dict[str, dict[str, Any]]) -> list[str]:
    atom_order = {str(atom_id): idx for idx, atom_id in enumerate(atom_by_id)}
    atoms = [str(atom_id) for atom_id in node.get("covered_atom_ids") or [] if str(atom_id) in atom_by_id]
    return sorted(set(atoms), key=lambda atom_id: atom_order.get(atom_id, 10**9))


def _rule_step_anchor_ids(
    evidence_id: str,
    selected_core_ids: Sequence[str],
    *,
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    allowed_edges: set[str],
) -> tuple[list[str], int]:
    edge_priority = {"complements": 0, "corroborates": 1, "tension": 2, "bridge_context": 3}
    rows: list[tuple[int, int, str]] = []
    for anchor_pos, anchor_id in enumerate(selected_core_ids):
        edge_types = {
            str(edge.get("edge_type") or "")
            for edge in edge_index.get(_pair_key(str(evidence_id), str(anchor_id)), [])
        }
        matched = edge_types & allowed_edges
        if matched:
            rows.append((min(edge_priority.get(edge_type, 99) for edge_type in matched), int(anchor_pos), str(anchor_id)))
    rows.sort()
    return [anchor_id for _edge_rank, _anchor_pos, anchor_id in rows], rows[0][0] if rows else 99


def _rule_step_sort_key(
    node: dict[str, Any],
    atoms: Sequence[str],
    anchor_ids: Sequence[str],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    edge_rank: int = 99,
    context_rule: bool = False,
) -> tuple[Any, ...]:
    del selected
    atom_order = {str(atom_id): idx for idx, atom_id in enumerate(atom_by_id)}
    directness_rank = {"direct": 0, "partial": 1, "context": 2, "none": 3}.get(str(node.get("directness") or ""), 3)
    relation_rank = 1 if str(node.get("relation") or "") in BACKGROUND_RELATIONS else 0
    return (
        _min_atom_order(atoms, atom_order),
        int(edge_rank),
        _min_evidence_id_order(anchor_ids),
        relation_rank if context_rule else 0,
        directness_rank,
        -_safe_float(node.get("evidence_map_quality_score"), 0.0),
        -_safe_float(node.get("base_score"), 0.0),
        int(bool(node.get("duplicate_group"))),
        _evidence_id_sort_value(str(node.get("node_id") or "")),
    )


def _fallback_sort_key(
    node: dict[str, Any],
    atoms: Sequence[str],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    selected: Sequence[dict[str, Any]],
) -> tuple[Any, ...]:
    del selected
    atom_order = {str(atom_id): idx for idx, atom_id in enumerate(atom_by_id)}
    directness_rank = {"direct": 0, "partial": 1, "context": 2, "none": 3}.get(str(node.get("directness") or ""), 3)
    return (
        0 if _is_rule_core_node(node) else 1,
        int(_is_background_node(node) or str(node.get("relation") or "") == "irrelevant"),
        int(bool(node.get("duplicate_group"))),
        _min_atom_order(atoms, atom_order),
        directness_rank,
        -_safe_float(node.get("evidence_map_quality_score"), 0.0),
        -_safe_float(node.get("base_score"), 0.0),
        _evidence_id_sort_value(str(node.get("node_id") or "")),
    )


def _min_atom_order(atoms: Sequence[str], atom_order: dict[str, int]) -> int:
    if not atoms:
        return 10**9
    return min(atom_order.get(str(atom_id), 10**9) for atom_id in atoms)


def _min_evidence_id_order(evidence_ids: Sequence[str]) -> tuple[str, int, str]:
    if not evidence_ids:
        return ("", 10**9, "")
    return min(_evidence_id_sort_value(str(evidence_id)) for evidence_id in evidence_ids)


def _extension_ids(
    chain: Sequence[str],
    evidence_nodes: Sequence[dict[str, Any]],
    *,
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    atom_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    selected = set(chain)
    all_ids = [str(node.get("node_id") or "") for node in evidence_nodes]
    neighbor_ids: set[str] = set()
    for selected_id in selected:
        for evidence_id in all_ids:
            if evidence_id in selected:
                continue
            if any(edge.get("edge_type") in POSITIVE_CHAIN_EDGE_TYPES for edge in edge_index.get(_pair_key(selected_id, evidence_id), [])):
                neighbor_ids.add(evidence_id)
    pool = list(neighbor_ids) if neighbor_ids else [evidence_id for evidence_id in all_ids if evidence_id not in selected]
    if neighbor_ids:
        selected_atoms = _covered_atoms_for_ids(chain, evidence_nodes)
        missing_atoms = set(atom_by_id) - selected_atoms
        for evidence_id in all_ids:
            if evidence_id in selected or evidence_id in neighbor_ids:
                continue
            node = _node_by_id(evidence_nodes, evidence_id)
            if missing_atoms & set(node.get("covered_atom_ids") or []):
                pool.append(evidence_id)
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    pool = sorted(set(pool), key=lambda evidence_id: _extension_key(evidence_id, chain=chain, by_id=by_id, edge_index=edge_index), reverse=True)
    return pool[: max(limit, 1)]


def _chain_score(
    evidence_ids: Sequence[str],
    *,
    by_id: dict[str, dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    ids = [str(eid) for eid in evidence_ids if str(eid) in by_id]
    nodes = [by_id[eid] for eid in ids]
    total_weight = sum(_safe_float(atom.get("importance"), 1.0) for atom in atom_by_id.values()) or 1.0
    covered_atoms = set(atom_id for node in nodes for atom_id in node.get("covered_atom_ids") or [] if atom_id in atom_by_id)
    covered_weight = sum(_safe_float(atom_by_id[atom_id].get("importance"), 1.0) for atom_id in covered_atoms)
    pair_counts = _pair_edge_counts(ids, edge_index=edge_index)
    pair_total = max(len(ids) * (len(ids) - 1) / 2.0, 1.0)
    duplicate_groups = [str(node.get("duplicate_group") or "") for node in nodes if node.get("duplicate_group")]
    source_groups = [str(node.get("source_group") or "") for node in nodes if node.get("source_group")]
    duplicate_repeat_count = sum(max(count - 1, 0) for count in Counter(duplicate_groups).values())
    same_source_excess_count = sum(max(count - 2, 0) for count in Counter(source_groups).values())
    direct_rate = _mean(1.0 if node.get("directness") in DIRECTNESS_VALUES else 0.0 for node in nodes)
    background_rate = _mean(1.0 if bool(node.get("is_background")) else 0.0 for node in nodes)
    mean_base = _mean(_safe_float(node.get("base_score"), 0.0) for node in nodes)
    weighted_atom_coverage = float(covered_weight / total_weight)
    complement_density = pair_counts.get("complements", 0) / pair_total
    corroborate_density = pair_counts.get("corroborates", 0) / pair_total
    tension_density = pair_counts.get("tension", 0) / pair_total
    bridge_density = pair_counts.get("bridge_context", 0) / pair_total
    duplicate_rate = duplicate_repeat_count / max(len(ids), 1)
    same_source_excess_rate = same_source_excess_count / max(len(ids), 1)
    score = (
        0.45 * mean_base
        + 0.35 * weighted_atom_coverage
        + 0.15 * direct_rate
        + 0.12 * complement_density
        + 0.08 * corroborate_density
        + 0.08 * tension_density
        + 0.05 * bridge_density
        - 0.25 * duplicate_rate
        - 0.15 * background_rate
        - 0.10 * same_source_excess_rate
    )
    return {
        "chain_id": "",
        "rank": 0,
        "evidence_ids": ids,
        "chain_score": float(score),
        "mean_base_score": float(mean_base),
        "weighted_atom_coverage": weighted_atom_coverage,
        "covered_atom_ids": sorted(covered_atoms),
        "direct_or_partial_rate": float(direct_rate),
        "background_rate": float(background_rate),
        "edge_counts": dict(sorted(pair_counts.items())),
        "positive_pair_edge_density": float(
            sum(pair_counts.get(edge_type, 0) for edge_type in POSITIVE_CHAIN_EDGE_TYPES) / pair_total
        ),
        "duplicate_repeat_count": int(duplicate_repeat_count),
        "same_source_excess_count": int(same_source_excess_count),
    }


def _post_order_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    by_id: dict[str, dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[str], dict[str, Any]]:
    ids = [str(eid) for eid in evidence_ids if str(eid) in by_id]
    if len(ids) <= 1:
        return ids, {
            "strategy": "post_order_v0_6b",
            "order_score": 0.0,
            "early_base_score": _early_base_score(ids, by_id=by_id),
            "search_order_inversions": 0,
        }

    search_rank = {evidence_id: rank for rank, evidence_id in enumerate(ids)}
    best_ids = list(ids)
    best_components: dict[str, Any] = {}
    best_sort: tuple[float, float, int] | None = None
    best_id_key: tuple[tuple[str, int, str], ...] | None = None
    for perm in itertools.permutations(ids):
        components = _post_order_score(
            perm,
            by_id=by_id,
            atom_by_id=atom_by_id,
            edge_index=edge_index,
            search_rank=search_rank,
        )
        inversions = int(components["search_order_inversions"])
        sort_key = (
            _safe_float(components.get("order_score"), 0.0),
            _safe_float(components.get("early_base_score"), 0.0),
            -inversions,
        )
        id_key = tuple(_evidence_id_sort_value(evidence_id) for evidence_id in perm)
        if best_sort is None or sort_key > best_sort or (sort_key == best_sort and (best_id_key is None or id_key < best_id_key)):
            best_sort = sort_key
            best_id_key = id_key
            best_ids = list(perm)
            best_components = dict(components)

    best_components["strategy"] = "post_order_v0_6b"
    return best_ids, best_components


def _post_order_score(
    evidence_ids: Sequence[str],
    *,
    by_id: dict[str, dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    search_rank: dict[str, int],
) -> dict[str, Any]:
    n = max(len(evidence_ids), 1)
    atom_order = {str(atom_id): idx for idx, atom_id in enumerate(atom_by_id)}
    seen_atoms: set[str] = set()
    first_atom_pos: dict[str, int] = {}
    early_new_atom_score = 0.0
    directness_early_score = 0.0
    background_before_core_penalty = 0.0

    for pos, evidence_id in enumerate(evidence_ids):
        node = by_id[str(evidence_id)]
        weight = float((n - pos) / n)
        covered_atoms = [str(atom_id) for atom_id in node.get("covered_atom_ids") or [] if str(atom_id) in atom_by_id]
        for atom_id in covered_atoms:
            first_atom_pos.setdefault(atom_id, pos)
            if atom_id not in seen_atoms:
                early_new_atom_score += weight * _safe_float(atom_by_id[atom_id].get("importance"), 1.0)
                seen_atoms.add(atom_id)
        directness_early_score += weight * _post_order_directness_score(node)
        if _is_background_node(node) and any(_is_core_node(by_id[str(other_id)]) for other_id in evidence_ids[pos + 1 :]):
            background_before_core_penalty += 0.25

    claim_order_score = 0.0
    atom_ids = sorted(first_atom_pos, key=lambda atom_id: atom_order.get(atom_id, 10**9))
    for i, left in enumerate(atom_ids):
        for right in atom_ids[i + 1 :]:
            if first_atom_pos[left] < first_atom_pos[right]:
                claim_order_score += 0.15
            elif first_atom_pos[left] > first_atom_pos[right]:
                claim_order_score -= 0.35

    adjacent_edge_score = 0.0
    duplicate_adjacent_penalty = 0.0
    same_source_unstructured_penalty = 0.0
    for left, right in zip(evidence_ids, evidence_ids[1:]):
        edge_types = [str(edge.get("edge_type") or "") for edge in edge_index.get(_pair_key(str(left), str(right)), [])]
        for edge_type in edge_types:
            adjacent_edge_score += POST_ORDER_EDGE_REWARDS.get(edge_type, 0.0)
        if "duplicate" in edge_types:
            duplicate_adjacent_penalty += 0.20
        if "same_source_context" in edge_types and not any(edge_type in POSITIVE_CHAIN_EDGE_TYPES for edge_type in edge_types):
            same_source_unstructured_penalty += 0.05

    inversions = _search_order_inversions(evidence_ids, search_rank=search_rank)
    early_base = _early_base_score(evidence_ids, by_id=by_id)
    order_score = (
        early_new_atom_score
        + claim_order_score
        + adjacent_edge_score
        + 0.35 * directness_early_score
        - background_before_core_penalty
        - duplicate_adjacent_penalty
        - same_source_unstructured_penalty
    )
    return {
        "order_score": float(order_score),
        "early_new_atom_score": float(early_new_atom_score),
        "claim_order_score": float(claim_order_score),
        "adjacent_edge_score": float(adjacent_edge_score),
        "directness_early_score": float(directness_early_score),
        "background_before_core_penalty": float(background_before_core_penalty),
        "duplicate_adjacent_penalty": float(duplicate_adjacent_penalty),
        "same_source_unstructured_penalty": float(same_source_unstructured_penalty),
        "early_base_score": float(early_base),
        "search_order_inversions": int(inversions),
    }


def _post_order_directness_score(node: dict[str, Any]) -> float:
    directness = str(node.get("directness") or "")
    if directness == "direct":
        return 1.0
    if directness == "partial":
        return 0.65
    if directness == "context":
        return 0.25
    return 0.0


def _is_background_node(node: dict[str, Any]) -> bool:
    return bool(node.get("is_background")) or str(node.get("directness") or "") in {"context", "none"}


def _is_core_node(node: dict[str, Any]) -> bool:
    return not _is_background_node(node) and str(node.get("directness") or "") in DIRECTNESS_VALUES


def _early_base_score(evidence_ids: Sequence[str], *, by_id: dict[str, dict[str, Any]]) -> float:
    n = max(len(evidence_ids), 1)
    return float(
        sum(
            ((n - pos) / n) * _safe_float(by_id[str(evidence_id)].get("base_score"), 0.0)
            for pos, evidence_id in enumerate(evidence_ids)
            if str(evidence_id) in by_id
        )
    )


def _search_order_inversions(evidence_ids: Sequence[str], *, search_rank: dict[str, int]) -> int:
    inversions = 0
    ranks = [int(search_rank.get(str(evidence_id), 10**9)) for evidence_id in evidence_ids]
    for i, left in enumerate(ranks):
        for right in ranks[i + 1 :]:
            if left > right:
                inversions += 1
    return inversions


def _evidence_id_sort_value(evidence_id: str) -> tuple[str, int, str]:
    prefix = evidence_id.rstrip("0123456789")
    suffix = evidence_id[len(prefix) :]
    if not suffix:
        return (str(evidence_id), 10**9, str(evidence_id))
    try:
        return (prefix, int(suffix), str(evidence_id))
    except ValueError:
        return (str(evidence_id), 10**9, str(evidence_id))


def _graph_diagnostics(
    atom_nodes: Sequence[dict[str, Any]],
    evidence_nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    selected_chain: dict[str, Any],
) -> dict[str, Any]:
    edge_index = _edge_index(edges)
    atom_ids = {str(atom.get("node_id") or "") for atom in atom_nodes}
    evidence_ids = {str(node.get("node_id") or "") for node in evidence_nodes}
    atom_connected = {str(edge.get("target") or "") for edge in edges if edge.get("edge_type") == "evidence_covers_atom"}
    evidence_connected = set()
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in evidence_ids or target in evidence_ids:
            evidence_connected.add(source if source in evidence_ids else target)
            if target in evidence_ids:
                evidence_connected.add(target)
    oracle_nodes = [node for node in evidence_nodes if bool(node.get("oracle_selected"))]
    oracle_connected = [node for node in oracle_nodes if str(node.get("node_id") or "") in evidence_connected]
    oracle_pair_total = max(len(oracle_nodes) * (len(oracle_nodes) - 1) / 2.0, 1.0)
    oracle_pair_edges = 0
    for i, left in enumerate(oracle_nodes):
        for right in oracle_nodes[i + 1 :]:
            pair_edges = edge_index.get(_pair_key(str(left.get("node_id") or ""), str(right.get("node_id") or "")), [])
            if any(edge.get("edge_type") in POSITIVE_CHAIN_EDGE_TYPES for edge in pair_edges):
                oracle_pair_edges += 1
    return {
        "atom_count": len(atom_nodes),
        "evidence_count": len(evidence_nodes),
        "edge_count": len(edges),
        "atom_isolate_rate": float(1.0 - len(atom_connected & atom_ids) / max(len(atom_ids), 1)),
        "evidence_isolate_rate": float(1.0 - len(evidence_connected & evidence_ids) / max(len(evidence_ids), 1)),
        "direct_atom_coverage": _direct_atom_coverage(atom_nodes, evidence_nodes),
        "duplicate_component_rate": _duplicate_component_rate(evidence_nodes),
        "oracle_evidence_connected_rate": float(len(oracle_connected) / max(len(oracle_nodes), 1)) if oracle_nodes else 0.0,
        "oracle_pair_edge_rate": float(oracle_pair_edges / oracle_pair_total) if oracle_nodes else 0.0,
        "max_chain_atom_coverage": _safe_float(selected_chain.get("weighted_atom_coverage"), 0.0),
    }


def _summarize_traces(traces: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "recall@5",
        "jaccard@5",
        "top1_match",
        "oracle_rank_ndcg@5",
        "weighted_atom_coverage@5",
        "direct_or_partial_rate@5",
        "recall@10",
        "jaccard@10",
        "oracle_rank_ndcg@10",
        "weighted_atom_coverage@10",
    )
    return {key: _mean(_safe_float(trace.get(key), 0.0) for trace in traces) for key in keys}


def _text_ordered_selection_metrics_multi(
    oracle_ordered_keys: Sequence[Any],
    selected: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    fixed = text_ordered_selection_metrics(oracle_ordered_keys, selected, top_k=min(5, int(top_k)))
    if int(top_k) <= 5:
        return fixed
    out = dict(fixed)
    dynamic = text_ordered_selection_metrics(oracle_ordered_keys, selected, top_k=int(top_k))
    out.update(_rename_at5_metrics(dynamic, suffix=f"@{int(top_k)}"))
    return out


def _evidence_map_selection_metrics_multi(
    graph_row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    row = {"evidence_map": {"claim_atoms": graph_row.get("atom_nodes") or []}}
    fixed = evidence_map_selection_metrics(row, list(selected)[: min(5, int(top_k))])
    if int(top_k) <= 5:
        return fixed
    out = dict(fixed)
    dynamic = evidence_map_selection_metrics(row, list(selected)[: int(top_k)])
    out.update(_rename_at5_metrics(dynamic, suffix=f"@{int(top_k)}"))
    return out


def _rename_at5_metrics(metrics: dict[str, Any], *, suffix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.endswith("@5"):
            out[f"{key[:-2]}{suffix}"] = value
        elif key in {"set_overlap", "overlap_pair_count"}:
            out[f"{key}{suffix}"] = value
    return out


def _edge(edge_type: str, source: str, target: str, *, weight: float, **extra: Any) -> dict[str, Any]:
    payload = {"edge_id": "", "edge_type": edge_type, "source": source, "target": target, "weight": float(weight)}
    payload.update({key: value for key, value in extra.items() if value not in (None, "", [])})
    return payload


def _edge_index(edges: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source.startswith("E") and target.startswith("E"):
            out[_pair_key(source, target)].append(dict(edge))
    return out


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _pair_edge_counts(ids: Sequence[str], *, edge_index: dict[tuple[str, str], list[dict[str, Any]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            counts.update(str(edge.get("edge_type") or "") for edge in edge_index.get(_pair_key(left, right), []))
    return counts


def _candidate_for_evidence_id(node: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(node.get("candidate") or {})
    candidate["evidence_id"] = str(node.get("evidence_id") or node.get("node_id") or "")
    candidate["chain_base_score"] = _safe_float(node.get("base_score"), 0.0)
    return candidate


def _pipeline_candidate_pool(graph_row: dict[str, Any]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for pool_idx, node in enumerate(graph_row.get("evidence_nodes") or []):
        candidate = _candidate_for_evidence_id(node)
        original_idx = _safe_int(candidate.get("original_candidate_idx"))
        if original_idx is None:
            original_idx = _safe_int(candidate.get("candidate_idx"))
        if original_idx is None:
            original_idx = int(pool_idx)
        candidate["candidate_idx"] = int(pool_idx)
        candidate["original_candidate_idx"] = int(original_idx)
        candidate["selector_pool_rank"] = int(pool_idx)
        candidate["selector_pool_evidence_id"] = str(node.get("evidence_id") or node.get("node_id") or "")
        candidate["evidence_id"] = str(node.get("evidence_id") or node.get("node_id") or "")
        candidate["candidate_uid"] = str(candidate.get("candidate_uid") or node.get("candidate_uid") or "")
        candidate["candidate_key"] = str(candidate.get("candidate_key") or node.get("candidate_key") or "")
        candidate["text"] = str(candidate.get("text") or node.get("text") or "")
        candidate["source_group"] = str(candidate.get("source_group") or node.get("source_group") or "")
        candidate["source_domain"] = str(candidate.get("source_domain") or node.get("source_domain") or "")
        candidate["report_id"] = candidate.get("report_id", node.get("report_id"))
        candidate["sent_idx"] = candidate.get("sent_idx", node.get("sent_idx"))
        pool.append(candidate)
    return pool


def _pipeline_candidate_scores(
    graph_row: dict[str, Any],
    candidate_pool: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    selected_rank_by_evidence_id = {
        str(evidence_id): rank
        for rank, evidence_id in enumerate(graph_row.get("selected_evidence_ids") or [])
    }
    rows: list[dict[str, Any]] = []
    for pool_idx, candidate in enumerate(candidate_pool):
        evidence_id = str(candidate.get("evidence_id") or "")
        selected_rank = selected_rank_by_evidence_id.get(evidence_id)
        base_score = _safe_float(candidate.get("chain_base_score", candidate.get("evidence_map_base_score")), 0.0)
        original_idx = _safe_int(candidate.get("original_candidate_idx"))
        hybrid_rank = _safe_int(candidate.get("hybrid_rank"))
        hybrid_score = _first_present_float(
            candidate.get("hybrid_score"),
            candidate.get("baseline_hybrid_score"),
            candidate.get("fusion_refit_score"),
            candidate.get("retrieval_score"),
            base_score,
        )
        map_score = _first_present_float(
            candidate.get("evidence_map_quality_score"),
            candidate.get("evidence_map_base_score"),
            base_score,
        )
        selector_score = float(max(int(top_k) - int(selected_rank), 0)) if selected_rank is not None else 0.0
        row = {
            "candidate_idx": int(pool_idx),
            "original_candidate_idx": int(original_idx) if original_idx is not None else int(pool_idx),
            "candidate_uid": str(candidate.get("candidate_uid") or ""),
            "candidate_key": str(candidate.get("candidate_key") or ""),
            "evidence_id": evidence_id,
            "hybrid_rank": int(hybrid_rank) if hybrid_rank is not None else int(pool_idx),
            "hybrid_score": float(hybrid_score),
            "retrieval_score": _safe_float(candidate.get("retrieval_score"), float(hybrid_score)),
            "dense_score": _safe_float(candidate.get("dense_score"), 0.0),
            "lexical_score": _safe_float(candidate.get("lexical_score"), 0.0),
            "bm25_score": _safe_float(candidate.get("bm25_score"), 0.0),
            "map_score": float(map_score),
            "evidence_map_base_score": _safe_float(candidate.get("evidence_map_base_score"), base_score),
            "evidence_map_quality_score": _safe_float(candidate.get("evidence_map_quality_score"), 0.0),
            "base_score": float(base_score),
            "selector_score": selector_score,
            "selector_selected_step": int(selected_rank) if selected_rank is not None else -1,
        }
        for optional_field in ("fusion_refit_score", "direct_ce_score", "oracle_likelihood_score"):
            if optional_field in candidate:
                row[optional_field] = _safe_float(candidate.get(optional_field), 0.0)
        if "fusion_refit_score" in row:
            row["fusion_score"] = row["fusion_refit_score"]
        rows.append(row)
    return rows


def _selector_ordered_pool_indices(graph_row: dict[str, Any], candidate_pool: Sequence[dict[str, Any]]) -> list[int]:
    idx_by_evidence_id = {str(candidate.get("evidence_id") or ""): idx for idx, candidate in enumerate(candidate_pool)}
    out: list[int] = []
    for evidence_id in graph_row.get("selected_evidence_ids") or []:
        idx = idx_by_evidence_id.get(str(evidence_id))
        if idx is not None:
            out.append(int(idx))
    return out


def _oracle_ordered_pool_indices(oracle_ordered_keys: Sequence[Any], candidate_pool: Sequence[dict[str, Any]]) -> list[int]:
    exact: dict[str, int] = {}
    normalized: dict[str, int] = {}
    for idx, candidate in enumerate(candidate_pool):
        for key in (candidate.get("candidate_key"), candidate.get("text")):
            raw = str(key or "").strip()
            if not raw:
                continue
            exact.setdefault(raw, int(idx))
            normalized.setdefault(_norm_text(raw), int(idx))
    out: list[int] = []
    seen: set[int] = set()
    for raw_key in oracle_ordered_keys:
        key = str(raw_key or "").strip()
        idx = exact.get(key)
        if idx is None:
            idx = normalized.get(_norm_text(key))
        if idx is not None and idx not in seen:
            out.append(int(idx))
            seen.add(int(idx))
    return out


def _candidate_trace_output(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_uid",
        "candidate_key",
        "evidence_id",
        "union_pool_rank",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "fusion_refit_score",
        "oracle_likelihood_score",
        "direct_ce_score",
        "evidence_map_base_score",
        "chain_base_score",
        "evidence_map_quality_score",
        "covered_atom_ids",
        "map_relation",
        "map_directness",
        "map_evidence_role",
        "key_spans",
        "duplicate_group",
    )
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["text"] = str(candidate.get("text") or "")
    return out


def _selected_chain_summary(graph_row: dict[str, Any]) -> dict[str, Any]:
    selected = str(graph_row.get("selected_chain_id") or "")
    for chain in graph_row.get("chains") or []:
        if str(chain.get("chain_id") or "") == selected:
            return dict(chain)
    return {}


def _empty_chain() -> dict[str, Any]:
    return {"chain_id": "", "rank": 0, "evidence_ids": [], "chain_score": 0.0}


def _sorted_candidates(candidates: Sequence[dict[str, Any]], *, candidate_top_n: int) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(
        key=lambda row: (
            -_base_score(row),
            -_safe_float(row.get("evidence_map_quality_score"), 0.0),
            int(row.get("union_pool_rank") or 10**9),
            str(row.get("candidate_key") or ""),
        )
    )
    return rows[: max(int(candidate_top_n), 1)]


def _base_score(candidate: dict[str, Any]) -> float:
    if candidate.get("evidence_map_base_score") is not None:
        return _safe_float(candidate.get("evidence_map_base_score"), 0.0)
    return _safe_float(candidate.get("fusion_refit_score"), 0.0)


def _start_key(node: dict[str, Any]) -> tuple[float, float, float, float]:
    direct = 1.0 if node.get("directness") in DIRECTNESS_VALUES else 0.0
    background = 1.0 if bool(node.get("is_background")) else 0.0
    return (
        _safe_float(node.get("base_score"), 0.0),
        direct,
        float(len(node.get("covered_atom_ids") or [])),
        -background,
    )


def _extension_key(
    evidence_id: str,
    *,
    chain: Sequence[str],
    by_id: dict[str, dict[str, Any]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[float, float, float, float]:
    node = by_id[evidence_id]
    positive_edges = 0
    duplicate_edges = 0
    for selected_id in chain:
        for edge in edge_index.get(_pair_key(evidence_id, selected_id), []):
            if edge.get("edge_type") in POSITIVE_CHAIN_EDGE_TYPES:
                positive_edges += 1
            if edge.get("edge_type") == "duplicate":
                duplicate_edges += 1
    return (
        float(positive_edges),
        _safe_float(node.get("base_score"), 0.0),
        float(len(node.get("covered_atom_ids") or [])),
        -float(duplicate_edges),
    )


def _covered_atoms_for_ids(ids: Sequence[str], evidence_nodes: Sequence[dict[str, Any]]) -> set[str]:
    by_id = {str(node.get("node_id") or ""): node for node in evidence_nodes}
    return {str(atom_id) for evidence_id in ids for atom_id in by_id.get(str(evidence_id), {}).get("covered_atom_ids") or []}


def _node_by_id(nodes: Sequence[dict[str, Any]], node_id: str) -> dict[str, Any]:
    for node in nodes:
        if str(node.get("node_id") or "") == str(node_id):
            return node
    return {}


def _is_duplicate_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_dup = str(left.get("duplicate_group") or "")
    right_dup = str(right.get("duplicate_group") or "")
    if left_dup and right_dup and left_dup == right_dup:
        return True
    return _norm_text(left.get("text")) and _norm_text(left.get("text")) == _norm_text(right.get("text"))


def _same_source_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _source_group(left) != _source_group(right) or not _source_group(left):
        return False
    left_sent = _safe_int(left.get("sent_idx"))
    right_sent = _safe_int(right.get("sent_idx"))
    if left_sent is None or right_sent is None:
        return False
    return abs(left_sent - right_sent) <= 3


def _same_direction(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("relation") or "") == str(right.get("relation") or "") and str(left.get("relation") or "") in POLAR_RELATIONS


def _has_tension(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_relation = str(left.get("relation") or "")
    right_relation = str(right.get("relation") or "")
    return (left_relation, right_relation) in {
        ("support", "refute"),
        ("refute", "support"),
        ("support", "qualify"),
        ("qualify", "support"),
        ("refute", "qualify"),
        ("qualify", "refute"),
        ("support", "mixed"),
        ("mixed", "support"),
        ("refute", "mixed"),
        ("mixed", "refute"),
    }


def _bridge_context(left: dict[str, Any], right: dict[str, Any], shared_atoms: Sequence[str]) -> bool:
    left_context = bool(left.get("is_background"))
    right_context = bool(right.get("is_background"))
    left_direct = left.get("directness") in DIRECTNESS_VALUES
    right_direct = right.get("directness") in DIRECTNESS_VALUES
    if not ((left_context and right_direct) or (right_context and left_direct)):
        return False
    return bool(shared_atoms) or (_source_group(left) and _source_group(left) == _source_group(right))


def _direct_atom_coverage(atom_nodes: Sequence[dict[str, Any]], evidence_nodes: Sequence[dict[str, Any]]) -> float:
    atom_ids = {str(atom.get("node_id") or "") for atom in atom_nodes}
    covered = {
        str(atom_id)
        for node in evidence_nodes
        if node.get("directness") in DIRECTNESS_VALUES
        for atom_id in node.get("covered_atom_ids") or []
    }
    return float(len(covered & atom_ids) / max(len(atom_ids), 1))


def _duplicate_component_rate(evidence_nodes: Sequence[dict[str, Any]]) -> float:
    duplicate_counts = Counter(str(node.get("duplicate_group") or "") for node in evidence_nodes if node.get("duplicate_group"))
    duplicate_nodes = sum(count for count in duplicate_counts.values() if count > 1)
    return float(duplicate_nodes / max(len(evidence_nodes), 1))


def _source_group(value: dict[str, Any]) -> str:
    return str(value.get("source_group") or (f"report:{value.get('report_id')}" if value.get("report_id") is not None else ""))


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _importance_to_unit(value: Any) -> float:
    raw = _safe_float(value, 1.0)
    if raw > 1.0:
        raw = raw / 5.0
    return float(np.clip(raw, 0.05, 1.0))


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


def _first_present_float(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        return _safe_float(value, 0.0)
    return 0.0


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0
