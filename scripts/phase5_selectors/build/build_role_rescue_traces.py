#!/usr/bin/env python3
"""Materialize controlled role-rescue evidence slates from one learned trace.

The role cells share the same learned resolving prefix.  The non-R-only
cells fill to a fixed count without consulting labels or verifier outputs:

* ``learned_fixed5`` is the original learned order truncated directly at K;
* ``random`` uses one event-local stable random order;
* ``retr`` uses the frozen retrieval order;
* ``cor`` / ``opp`` / ``ctx`` promote at most one candidate per atom-role
  slot in the original learned suffix order, then use the common stable-random
  remainder for the remaining capacity;
* ``full`` interleaves all available atom-role slots by original learned rank,
  then uses that same stable-random remainder.

Each output trace contains only its selected, prompt-hydrated candidate pool.
The source-pool coordinates and role decisions remain explicit in
``role_rescue_metadata`` for replay and auditing.  The traces are therefore
directly consumable by ``build_trace_verifier_data.py`` with
``--prompt-evidence-policy selected_set``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "role_rescue_trace_v0_1"
PROJECTION_SCHEMA = "role_rescue_prompt_hydrated_selected_pool_v0_1"
CELLS = (
    "r_only",
    "learned_fixed5",
    "cor",
    "opp",
    "ctx",
    "retr",
    "random",
    "full",
)
ROLE_CELLS = ("cor", "opp", "ctx")
FULL_ROLE_ORDER = ROLE_CELLS

_SUPPORT_RELATIONS = {
    "support",
    "supports",
    "supported_by",
    "entails",
    "consistent",
}
_REFUTE_RELATIONS = {
    "refute",
    "refutes",
    "contradict",
    "contradicts",
    "counter",
    "conflict",
}
_QUALIFY_RELATIONS = {
    "qualify",
    "qualifies",
    "qualified",
    "condition",
    "hedge",
    "mixed",
    "partially_supports",
    "partial",
}
_CONTEXT_RELATIONS = {"background", "context", "insufficient"}
_DIRECTNESS = {"direct", "partial"}

_CANDIDATE_PROJECTION_FIELDS = (
    "candidate_uid",
    "candidate_key",
    "evidence_id",
    "text",
    "anchor_text",
    "canonical_text",
    "covered_atom_ids",
    "matched_atom_ids",
    "candidate_atom_alignments",
    "source_group",
    "source_report",
    "report_id",
    "duplicate_group",
    "map_relation",
    "map_directness",
    "map_evidence_role",
    "map_confidence",
    "key_spans",
    "evidence_map_quality_score",
    "evidence_map_base_score",
    "baseline_hybrid_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
    "lexical_score",
    "union_pool_rank",
    "baseline_rank",
    "mrec_token_cost",
    "num_tokens",
)


class RoleRescueError(ValueError):
    """Raised when a source row cannot define auditable rescue cells."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Learned full-pool selection_trace JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args(argv)
    if int(args.k) <= 0:
        parser.error("--k must be positive")
    if int(args.sample_limit) < 0:
        parser.error("--sample-limit must be non-negative")
    return args


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    summary = materialize_role_rescue_traces(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        split=str(args.split),
        k=int(args.k),
        seed=int(args.seed),
        sample_limit=int(args.sample_limit),
    )
    print(
        f"Wrote {summary['row_count']} rows x {len(CELLS)} role-rescue cells "
        f"to {args.output_dir}"
    )
    return 0


def materialize_role_rescue_traces(
    *,
    input_path: Path,
    output_dir: Path,
    split: str,
    k: int,
    seed: int,
    sample_limit: int = 0,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise RoleRescueError(f"input trace does not exist: {input_path}")
    if k <= 0:
        raise RoleRescueError("k must be positive")
    if sample_limit < 0:
        raise RoleRescueError("sample_limit must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = {
        cell: output_dir / cell / f"selection_trace_{split}.jsonl" for cell in CELLS
    }
    temp_paths = {
        cell: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for cell, path in trace_paths.items()
    }
    for path in trace_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    seen_events: set[str] = set()
    source_sha = hashlib.sha256()
    core_counts: list[int] = []
    core_stop_reasons: Counter[str] = Counter()
    cell_stats = {cell: _new_cell_stats() for cell in CELLS}

    promoted = False
    try:
        with input_path.open(encoding="utf-8") as source, ExitStack() as stack:
            handles = {
                cell: stack.enter_context(path.open("w", encoding="utf-8"))
                for cell, path in temp_paths.items()
            }
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if sample_limit > 0 and row_count >= sample_limit:
                    break
                source_sha.update(line.encode("utf-8"))
                try:
                    source_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RoleRescueError(
                        f"invalid JSON at {input_path}:{line_number}"
                    ) from exc
                if not isinstance(source_row, Mapping):
                    raise RoleRescueError(
                        f"source row is not an object at {input_path}:{line_number}"
                    )
                event_id = _compact(source_row.get("event_id"))
                if not event_id:
                    raise RoleRescueError(f"missing event_id at {input_path}:{line_number}")
                if event_id in seen_events:
                    raise RoleRescueError(f"duplicate event_id {event_id!r}")
                seen_events.add(event_id)

                try:
                    event_rows = build_role_rescue_rows(source_row, k=k, seed=seed)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RoleRescueError(f"{event_id}: {exc}") from exc
                core_meta = event_rows["r_only"]["role_rescue_metadata"]
                core_counts.append(int(core_meta["core_count"]))
                core_stop_reasons[str(core_meta["core_stop_reason"])] += 1
                random_uids = tuple(
                    event_rows["random"]["role_rescue_metadata"]["selected_source_candidate_uids"]
                )
                for cell in CELLS:
                    row = event_rows[cell]
                    handles[cell].write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    _update_cell_stats(
                        cell_stats[cell],
                        row,
                        random_uids=random_uids,
                    )
                row_count += 1

        if row_count == 0:
            raise RoleRescueError(f"input trace has no rows: {input_path}")
        for cell in CELLS:
            os.replace(temp_paths[cell], trace_paths[cell])
        promoted = True
    finally:
        if not promoted:
            for path in temp_paths.values():
                path.unlink(missing_ok=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "projection_schema": PROJECTION_SCHEMA,
        "input": str(input_path),
        "input_sha256": source_sha.hexdigest(),
        "output_dir": str(output_dir),
        "split": split,
        "k": int(k),
        "seed": int(seed),
        "sample_limit": int(sample_limit),
        "row_count": int(row_count),
        "cells": {
            cell: {
                "trace": str(trace_paths[cell]),
                "trace_file": f"{cell}/selection_trace_{split}.jsonl",
                **_finalize_cell_stats(cell_stats[cell], row_count=row_count),
            }
            for cell in CELLS
        },
        "core_count": _numeric_summary(core_counts),
        "core_stop_reasons": dict(sorted(core_stop_reasons.items())),
        "contracts": {
            "core": "learned selector prefix through first target_resolved, capped at K",
            "learned_fixed5": "original learned selector prefix truncated directly at K",
            "random": "sha256(seed,event_id,candidate_uid) stable order",
            "cor": "same atom and support/refute direction, novel non-empty source",
            "opp": "same-atom opposite support/refute, qualify/mixed, or source CONTRAST step",
            "ctx": "same-atom background/context/insufficient; irrelevant is excluded",
            "role_cap": "at most one promoted candidate per atom-role slot",
            "full_role_order": "interleaved by original learned suffix rank",
            "verifier_build": {
                "selection_mode": "trace",
                "prompt_evidence_policy": "selected_set",
                "trace_prompt_style": "mrec_min",
            },
        },
    }
    _write_json(output_dir / "manifest.json", summary)
    return summary


def build_role_rescue_rows(
    source_row: Mapping[str, Any], *, k: int = 5, seed: int = 0
) -> dict[str, dict[str, Any]]:
    """Build all seven role-rescue cells for one source trace row."""

    event_id = _compact(source_row.get("event_id"))
    if not event_id:
        raise RoleRescueError("source row has no event_id")
    raw_pool = source_row.get("candidate_pool")
    if not isinstance(raw_pool, Sequence) or isinstance(raw_pool, (str, bytes)):
        raise RoleRescueError("candidate_pool must be an array")
    pool = [dict(candidate) for candidate in raw_pool if isinstance(candidate, Mapping)]
    if len(pool) != len(raw_pool):
        raise RoleRescueError("candidate_pool must contain only objects")
    if not pool:
        raise RoleRescueError("candidate_pool is empty")
    uids = [_candidate_uid(candidate, idx=idx) for idx, candidate in enumerate(pool)]
    if len(uids) != len(set(uids)):
        raise RoleRescueError("candidate_pool candidate_uid values must be unique")

    ordered = _source_ordered_indices(source_row, pool_size=len(pool))
    step_by_idx = _source_step_by_index(source_row, pool_size=len(pool))
    core, core_stop_reason = _resolving_core(
        ordered,
        step_by_idx=step_by_idx,
        k=min(k, len(pool)),
    )
    core_set = set(core)
    remaining = [idx for idx in range(len(pool)) if idx not in core_set]
    learned_suffix = [idx for idx in ordered if idx not in core_set]
    stable_random = sorted(
        remaining,
        key=lambda idx: _stable_random_key(seed, event_id, uids[idx]),
    )
    retrieval = sorted(remaining, key=lambda idx: _retrieval_key(pool[idx], uids[idx]))
    role_slots = _eligible_role_slots(
        pool=pool,
        core_indices=core,
        step_by_idx=step_by_idx,
        learned_suffix=learned_suffix,
    )
    role_available = {
        role: _flatten_role_slot_indices(role_slots.get(role, {}), learned_suffix)
        for role in ROLE_CELLS
    }

    selections: dict[str, tuple[list[int], dict[str, int], list[int], str]] = {}
    selections["r_only"] = (list(core), {}, [], "resolving_core_only")
    learned_fixed = [int(idx) for idx in ordered[: min(k, len(ordered))]]
    selections["learned_fixed5"] = (
        learned_fixed,
        {},
        [idx for idx in learned_fixed if idx not in core_set],
        "original_learned_prefix_fixed_k",
    )
    selections["random"] = _fill_random(
        core,
        stable_random=stable_random,
        pool=pool,
        k=k,
        promoted={},
        policy="stable_random_fill",
    )
    selections["retr"] = _fill_retrieval(core, retrieval=retrieval, pool=pool, k=k)
    for role in ROLE_CELLS:
        selections[role] = _fill_role_slots_then_random(
            core,
            roles=(role,),
            role_slots=role_slots,
            learned_suffix=learned_suffix,
            stable_random=stable_random,
            pool=pool,
            k=k,
            policy=f"promote_{role}_atom_slots_by_learned_rank_then_common_random",
        )
    selections["full"] = _fill_role_slots_then_random(
        core,
        roles=FULL_ROLE_ORDER,
        role_slots=role_slots,
        learned_suffix=learned_suffix,
        stable_random=stable_random,
        pool=pool,
        k=k,
        policy="interleave_atom_role_slots_by_learned_rank_then_common_random",
    )

    out: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        selected, promoted_by_slot, random_fill, fill_policy = selections[cell]
        out[cell] = _project_trace_row(
            source_row,
            source_pool=pool,
            selected_source_indices=selected,
            core_indices=core,
            core_stop_reason=core_stop_reason,
            role_available=role_available,
            role_slots=role_slots,
            promoted_by_slot=promoted_by_slot,
            random_fill_indices=random_fill,
            stable_random=stable_random,
            cell=cell,
            fill_policy=fill_policy,
            k=k,
            seed=seed,
        )
    return out


def _fill_random(
    core: Sequence[int],
    *,
    stable_random: Sequence[int],
    pool: Sequence[Mapping[str, Any]],
    k: int,
    promoted: Mapping[str, int],
    policy: str,
) -> tuple[list[int], dict[str, int], list[int], str]:
    selected = list(core)
    random_fill: list[int] = []
    for idx in stable_random:
        if len(selected) >= k:
            break
        if idx in selected or _duplicates_any(pool[idx], (pool[item] for item in selected)):
            continue
        selected.append(int(idx))
        random_fill.append(int(idx))
    return selected, dict(promoted), random_fill, policy


def _fill_retrieval(
    core: Sequence[int],
    *,
    retrieval: Sequence[int],
    pool: Sequence[Mapping[str, Any]],
    k: int,
) -> tuple[list[int], dict[str, int], list[int], str]:
    selected = list(core)
    retrieval_fill: list[int] = []
    for idx in retrieval:
        if len(selected) >= k:
            break
        if idx in selected or _duplicates_any(pool[idx], (pool[item] for item in selected)):
            continue
        selected.append(int(idx))
        retrieval_fill.append(int(idx))
    return selected, {}, retrieval_fill, "retrieval_fill"


def _fill_role_slots_then_random(
    core: Sequence[int],
    *,
    roles: Sequence[str],
    role_slots: Mapping[str, Mapping[str, Sequence[int]]],
    learned_suffix: Sequence[int],
    stable_random: Sequence[int],
    pool: Sequence[Mapping[str, Any]],
    k: int,
    policy: str,
) -> tuple[list[int], dict[str, int], list[int], str]:
    selected = list(core)
    promoted: dict[str, int] = {}
    requested_slots = {
        f"{role}:{atom_id}": set(int(idx) for idx in indices)
        for role in roles
        for atom_id, indices in role_slots.get(role, {}).items()
    }
    for idx in learned_suffix:
        if len(selected) >= k:
            break
        open_slots = [
            slot
            for slot, candidates in requested_slots.items()
            if slot not in promoted and int(idx) in candidates
        ]
        if not open_slots:
            continue
        if idx in selected or _duplicates_any(pool[idx], (pool[item] for item in selected)):
            continue
        selected.append(int(idx))
        # A multi-atom candidate may legitimately satisfy several previously
        # open slots.  Each atom-role slot is still activated at most once.
        for slot in open_slots:
            promoted[slot] = int(idx)
    random_fill: list[int] = []
    for idx in stable_random:
        if len(selected) >= k:
            break
        if idx in selected or _duplicates_any(pool[idx], (pool[item] for item in selected)):
            continue
        selected.append(int(idx))
        random_fill.append(int(idx))
    return selected, promoted, random_fill, policy


def _eligible_role_slots(
    *,
    pool: Sequence[Mapping[str, Any]],
    core_indices: Sequence[int],
    step_by_idx: Mapping[int, Mapping[str, Any]],
    learned_suffix: Sequence[int],
) -> dict[str, dict[str, list[int]]]:
    core_pairs = [
        (idx, pair)
        for idx in core_indices
        for pair in _candidate_alignments(pool[idx])
    ]
    core_atoms = {
        _compact(pair.get("atom_id")) for _idx, pair in core_pairs if _compact(pair.get("atom_id"))
    }
    core_directions: dict[str, set[str]] = {}
    core_sources: dict[tuple[str, str], set[str]] = {}
    for idx, pair in core_pairs:
        atom_id = _compact(pair.get("atom_id"))
        direction = _direction(pair.get("relation"))
        if not atom_id or direction not in {"support", "refute"} or not _valid_polar(pair):
            continue
        core_directions.setdefault(atom_id, set()).add(direction)
        source = _source_key(pool[idx])
        if source:
            core_sources.setdefault((atom_id, direction), set()).add(source)

    slots: dict[str, dict[str, list[int]]] = {
        role: {} for role in ROLE_CELLS
    }
    for idx in learned_suffix:
        if _duplicates_any(pool[idx], (pool[item] for item in core_indices)):
            continue
        pairs = _candidate_alignments(pool[idx])
        candidate_source = _source_key(pool[idx])
        for pair in pairs:
            atom_id = _compact(pair.get("atom_id"))
            direction = _direction(pair.get("relation"))
            if not atom_id or atom_id not in core_atoms:
                continue
            if candidate_source and _valid_polar(pair) and direction in {"support", "refute"}:
                anchor_sources = core_sources.get((atom_id, direction), set())
                if anchor_sources and candidate_source not in anchor_sources:
                    _append_slot(slots["cor"], f"{atom_id}|{direction}", idx)
            if _valid_polar(pair):
                anchors = core_directions.get(atom_id, set())
                if (
                    (direction == "qualify" and anchors)
                    or (direction == "support" and "refute" in anchors)
                    or (direction == "refute" and "support" in anchors)
                ):
                    _append_slot(slots["opp"], atom_id, idx)
            if _valid_context(pair):
                _append_slot(slots["ctx"], atom_id, idx)

        step = step_by_idx.get(idx, {})
        step_atom = _compact(step.get("atom_id"))
        if (
            _compact(step.get("operation")).upper() == "CONTRAST"
            and step_atom in core_atoms
            and any(_valid_polar(pair) for pair in pairs)
        ):
            _append_slot(slots["opp"], step_atom, idx)
    return slots


def _append_slot(slots: dict[str, list[int]], atom_id: str, idx: int) -> None:
    values = slots.setdefault(atom_id, [])
    if int(idx) not in values:
        values.append(int(idx))


def _flatten_role_slot_indices(
    slots: Mapping[str, Sequence[int]], learned_suffix: Sequence[int]
) -> list[int]:
    eligible = {int(idx) for values in slots.values() for idx in values}
    return [int(idx) for idx in learned_suffix if int(idx) in eligible]


def _project_trace_row(
    source_row: Mapping[str, Any],
    *,
    source_pool: Sequence[Mapping[str, Any]],
    selected_source_indices: Sequence[int],
    core_indices: Sequence[int],
    core_stop_reason: str,
    role_available: Mapping[str, Sequence[int]],
    role_slots: Mapping[str, Mapping[str, Sequence[int]]],
    promoted_by_slot: Mapping[str, int],
    random_fill_indices: Sequence[int],
    stable_random: Sequence[int],
    cell: str,
    fill_policy: str,
    k: int,
    seed: int,
) -> dict[str, Any]:
    event_id = _compact(source_row.get("event_id"))
    projected_pool = [
        _project_candidate(source_pool[source_idx], source_idx=source_idx, local_idx=local_idx)
        for local_idx, source_idx in enumerate(selected_source_indices)
    ]
    local_indices = list(range(len(projected_pool)))
    claim_atoms = _claim_atoms(source_row)
    mrec_steps = [
        _cue_step(candidate, claim_atoms=claim_atoms, step=position)
        for position, candidate in enumerate(projected_pool, start=1)
    ]
    source_uids = [
        _candidate_uid(source_pool[idx], idx=idx) for idx in selected_source_indices
    ]
    promoted_by_role_all = {
        role: _unique_in_order(
            idx
            for slot, idx in promoted_by_slot.items()
            if slot.startswith(f"{role}:")
        )
        for role in ROLE_CELLS
    }
    promoted_by_role = {
        role: indices for role, indices in promoted_by_role_all.items() if indices
    }
    available_role_uids = {
        role: [_candidate_uid(source_pool[idx], idx=idx) for idx in indices]
        for role, indices in role_available.items()
    }
    selected_set = set(selected_source_indices)
    selected_role_presence = {
        role: [int(idx) for idx in indices if idx in selected_set]
        for role, indices in role_available.items()
    }
    source_metadata = dict(source_row.get("candidate_pool_metadata") or {})
    fingerprint = _compact(source_row.get("fingerprint")) or _compact(
        source_metadata.get("chunk_mmr_fingerprint")
    )
    selected_set_fingerprint = _uid_fingerprint(event_id, source_uids)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "k": int(k),
        "seed": int(seed),
        "core_source_indices": [int(idx) for idx in core_indices],
        "core_source_candidate_uids": [
            _candidate_uid(source_pool[idx], idx=idx) for idx in core_indices
        ],
        "core_count": len(core_indices),
        "core_stop_reason": core_stop_reason,
        "core_target_resolved": core_stop_reason == "target_resolved",
        "fill_policy": fill_policy,
        "stable_random_source_indices": [int(idx) for idx in stable_random],
        "stable_random_order_fingerprint": _uid_fingerprint(
            event_id,
            [_candidate_uid(source_pool[idx], idx=idx) for idx in stable_random],
        ),
        "available_role_source_indices": {
            role: [int(idx) for idx in indices] for role, indices in role_available.items()
        },
        "available_role_candidate_uids": available_role_uids,
        "available_atom_role_slot_source_indices": {
            f"{role}:{atom_id}": [int(idx) for idx in indices]
            for role, atom_slots in role_slots.items()
            for atom_id, indices in atom_slots.items()
        },
        "promoted_role_source_indices": {
            role: [int(idx) for idx in indices]
            for role, indices in promoted_by_role.items()
        },
        "promoted_role_candidate_uids": {
            role: [_candidate_uid(source_pool[idx], idx=idx) for idx in indices]
            for role, indices in promoted_by_role.items()
        },
        "realized_atom_role_slots": {
            slot: {
                "source_candidate_idx": int(idx),
                "candidate_uid": _candidate_uid(source_pool[idx], idx=idx),
            }
            for slot, idx in promoted_by_slot.items()
        },
        "selected_role_presence_source_indices": selected_role_presence,
        "selected_role_presence_candidate_uids": {
            role: [_candidate_uid(source_pool[idx], idx=idx) for idx in indices]
            for role, indices in selected_role_presence.items()
        },
        "random_or_retrieval_fill_source_indices": [int(idx) for idx in random_fill_indices],
        "selected_source_indices": [int(idx) for idx in selected_source_indices],
        "selected_source_candidate_uids": source_uids,
        "selected_count": len(selected_source_indices),
        "underfilled": len(selected_source_indices) < min(k, len(source_pool)),
        "source_pool_count": len(source_pool),
        "selected_set_fingerprint": selected_set_fingerprint,
        "selection_uses_gold_label": False,
        "selection_uses_verifier_output": False,
    }
    candidate_scores = [
        {
            "candidate_idx": idx,
            "candidate_uid": _candidate_uid(candidate, idx=idx),
            "selector_selected_step": idx,
            "selector_score": float(len(projected_pool) - idx),
            "hybrid_score": _safe_float(
                candidate.get("hybrid_score") or candidate.get("baseline_hybrid_score")
            ),
        }
        for idx, candidate in enumerate(projected_pool)
    ]
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": SCHEMA_VERSION,
        "mrec_trace_version": SCHEMA_VERSION,
        "event_id": event_id,
        "claim": str(source_row.get("claim") or ""),
        "gold_label": str(source_row.get("gold_label") or source_row.get("label") or ""),
        "selector_name": f"role_rescue_{cell}_v0_1",
        "mrec_selector_name": f"role_rescue_{cell}_v0_1",
        "selection_policy": f"role_rescue_{cell}",
        "adaptive_policy": fill_policy,
        "fingerprint": fingerprint,
        "candidate_pool_metadata": {
            **source_metadata,
            "schema_version": SCHEMA_VERSION,
            "projection_schema": PROJECTION_SCHEMA,
            "source_selector_name": str(source_row.get("selector_name") or ""),
            "source_pool_count": len(source_pool),
            "selected_set_fingerprint": selected_set_fingerprint,
        },
        "candidate_pool": projected_pool,
        "candidate_scores": candidate_scores,
        "selector_ordered_indices": local_indices,
        "display_ordered_indices": local_indices,
        "selector_available_ordered_indices": local_indices,
        "selector_full_ordered_indices": local_indices,
        "selected_indices": local_indices,
        "selected_candidates": [dict(candidate) for candidate in projected_pool],
        "selected_candidate_uids": source_uids,
        "selected_evidence_ids": [str(candidate.get("evidence_id") or "") for candidate in projected_pool],
        "selected_keys": source_uids,
        "selected_count": len(projected_pool),
        "claim_atoms": claim_atoms,
        "mrec_steps": mrec_steps,
        "role_rescue_metadata": metadata,
        "params": {
            "k": int(k),
            "seed": int(seed),
            "cell": cell,
            "prompt_evidence_policy": "selected_set",
        },
    }
    return row


def _project_candidate(
    source: Mapping[str, Any], *, source_idx: int, local_idx: int
) -> dict[str, Any]:
    candidate = {
        field: deepcopy(source[field])
        for field in _CANDIDATE_PROJECTION_FIELDS
        if field in source
    }
    uid = _candidate_uid(source, idx=source_idx)
    candidate["candidate_uid"] = uid
    candidate.setdefault("candidate_key", uid)
    candidate.setdefault("evidence_id", uid)
    candidate["source_candidate_idx"] = int(source_idx)
    candidate["source_selector_candidate_idx"] = int(
        _int_or_default(source.get("selector_candidate_idx"), source_idx)
    )
    candidate["candidate_idx"] = int(local_idx)
    candidate["selector_candidate_idx"] = int(local_idx)
    candidate["selector_pool_rank"] = int(local_idx)
    candidate["mrec_token_cost"] = _token_cost(source)
    if not candidate.get("covered_atom_ids"):
        candidate["covered_atom_ids"] = sorted(
            {
                _compact(pair.get("atom_id"))
                for pair in _candidate_alignments(source)
                if _compact(pair.get("atom_id"))
            }
        )
    return candidate


def _cue_step(
    candidate: Mapping[str, Any], *, claim_atoms: Sequence[Mapping[str, Any]], step: int
) -> dict[str, Any]:
    atom_lookup = {
        _compact(atom.get("atom_id") or atom.get("node_id")): atom
        for atom in claim_atoms
        if _compact(atom.get("atom_id") or atom.get("node_id"))
    }
    pairs = _candidate_alignments(candidate)
    ranked_pairs = sorted(pairs, key=_cue_pair_key)
    atom_id = next(
        (
            _compact(pair.get("atom_id"))
            for pair in ranked_pairs
            if _compact(pair.get("atom_id")) in atom_lookup
        ),
        next(iter(atom_lookup), "A1"),
    )
    atom = atom_lookup.get(atom_id, {})
    cue_text = _compact(
        atom.get("text")
        or atom.get("proposition")
        or atom.get("query_rendering")
        or atom_id
    )
    covered = sorted(
        {
            _compact(pair.get("atom_id"))
            for pair in pairs
            if _compact(pair.get("atom_id"))
        }
    )
    return {
        "step": int(step),
        "candidate_idx": int(candidate.get("candidate_idx", step - 1)),
        "selector_candidate_idx": int(candidate.get("selector_candidate_idx", step - 1)),
        "candidate_uid": str(candidate.get("candidate_uid") or ""),
        "evidence_id": str(candidate.get("evidence_id") or ""),
        "atom_id": atom_id,
        "cue_text": cue_text,
        "cue_source": "claim_atom",
        "covered_atom_ids": covered,
        "token_cost": _token_cost(candidate),
    }


def _resolving_core(
    ordered: Sequence[int],
    *,
    step_by_idx: Mapping[int, Mapping[str, Any]],
    k: int,
) -> tuple[list[int], str]:
    core: list[int] = []
    for idx in ordered[:k]:
        core.append(int(idx))
        if _step_target_resolved(step_by_idx.get(int(idx), {})):
            return core, "target_resolved"
    if len(core) >= k:
        return core, "k_cap"
    return core, "source_order_exhausted"


def _source_ordered_indices(row: Mapping[str, Any], *, pool_size: int) -> list[int]:
    raw = row.get("selector_ordered_indices")
    if raw is None:
        raw = row.get("display_ordered_indices")
    if raw is None:
        raw = row.get("selected_indices")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RoleRescueError("source trace has no ordered-index array")
    out = [_int(value, "source ordered index") for value in raw]
    if not out:
        raise RoleRescueError("source ordered-index array is empty")
    if len(out) != len(set(out)) or any(idx < 0 or idx >= pool_size for idx in out):
        raise RoleRescueError("source ordered indices are duplicate or out of range")
    return out


def _source_step_by_index(
    row: Mapping[str, Any], *, pool_size: int
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for fallback_idx, raw in enumerate(row.get("mrec_steps") or []):
        if not isinstance(raw, Mapping):
            continue
        idx = _int_or_default(
            raw.get("selector_candidate_idx"),
            _int_or_default(raw.get("candidate_idx"), fallback_idx),
        )
        if 0 <= idx < pool_size:
            out[int(idx)] = dict(raw)
    return out


def _step_target_resolved(step: Mapping[str, Any]) -> bool:
    state = step.get("trace_state")
    if isinstance(state, Mapping) and "target_resolved" in state:
        return bool(state.get("target_resolved"))
    return bool(step.get("target_resolved", False))


def _candidate_alignments(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_eid = _compact(candidate.get("evidence_id"))
    out: list[dict[str, Any]] = []
    for raw in candidate.get("candidate_atom_alignments") or []:
        if not isinstance(raw, Mapping):
            continue
        row_eid = _compact(raw.get("evidence_id"))
        if row_eid and candidate_eid and row_eid != candidate_eid:
            continue
        atom_id = _compact(raw.get("atom_id"))
        if not atom_id:
            continue
        copied = dict(raw)
        copied["atom_id"] = atom_id
        out.append(copied)
    return out


def _valid_polar(pair: Mapping[str, Any]) -> bool:
    return (
        _direction(pair.get("relation")) in {"support", "refute", "qualify"}
        and _compact(pair.get("directness")).lower() in _DIRECTNESS
        and _safe_float(pair.get("confidence")) > 0.0
    )


def _valid_context(pair: Mapping[str, Any]) -> bool:
    relation = _compact(pair.get("relation")).lower()
    directness = _compact(pair.get("directness")).lower()
    evidence_role = _compact(
        pair.get("evidence_role") or pair.get("map_evidence_role") or pair.get("role")
    ).lower()
    if relation in {"irrelevant", "unrelated"} or evidence_role in {
        "irrelevant",
        "unrelated",
    }:
        return False
    if directness == "context" or relation in {"background", "context"}:
        return True
    if evidence_role in {"background_context", "background", "context"}:
        return True
    if relation == "insufficient":
        return bool(
            directness in {"partial", "context"}
            or evidence_role in {"background_context", "background", "context"}
            or pair.get("key_spans")
        )
    return False


def _direction(value: Any) -> str:
    relation = _compact(value).lower()
    if relation in _SUPPORT_RELATIONS:
        return "support"
    if relation in _REFUTE_RELATIONS:
        return "refute"
    if relation in _QUALIFY_RELATIONS:
        return "qualify"
    return relation


def _source_key(candidate: Mapping[str, Any]) -> str:
    direct = _compact(candidate.get("source_group"))
    if direct:
        return direct
    report_id = _compact(candidate.get("report_id"))
    if report_id:
        return f"report:{report_id}"
    report = candidate.get("source_report")
    if isinstance(report, Mapping):
        for field in ("report_id", "domain", "link", "url"):
            value = _compact(report.get(field))
            if value:
                return f"{field}:{value}"
    return ""


def _duplicates_any(
    candidate: Mapping[str, Any], others: Any
) -> bool:
    return any(_is_duplicate(candidate, other) for other in others)


def _is_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_group = _compact(left.get("duplicate_group"))
    right_group = _compact(right.get("duplicate_group"))
    if left_group and right_group and left_group == right_group:
        return True
    left_text = _normalize_text(left.get("text") or left.get("canonical_text") or "")
    right_text = _normalize_text(right.get("text") or right.get("canonical_text") or "")
    return bool(left_text and right_text and left_text == right_text)


def _stable_random_key(seed: int, event_id: str, uid: str) -> tuple[str, str]:
    payload = f"{int(seed)}\0{event_id}\0{uid}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), uid


def _retrieval_key(candidate: Mapping[str, Any], uid: str) -> tuple[float, int, str]:
    return -_retrieval_score(candidate), _rank(candidate.get("union_pool_rank")), uid


def _retrieval_score(candidate: Mapping[str, Any]) -> float:
    values = (
        candidate.get("retrieval_score"),
        candidate.get("baseline_hybrid_score"),
        candidate.get("hybrid_score"),
        candidate.get("qd_max_question_hybrid"),
        candidate.get("max_question_hybrid"),
    )
    return float(min(1.0, max(0.0, max(_safe_float(value) for value in values))))


def _cue_pair_key(pair: Mapping[str, Any]) -> tuple[int, int, str]:
    directness_rank = {"direct": 0, "partial": 1, "context": 2, "none": 3}
    relation_rank = {"support": 0, "refute": 0, "qualify": 1, "mixed": 1, "background": 2, "insufficient": 3, "irrelevant": 4}
    return (
        directness_rank.get(_compact(pair.get("directness")).lower(), 4),
        relation_rank.get(_compact(pair.get("relation")).lower(), 3),
        _compact(pair.get("atom_id")),
    )


def _claim_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("claim_atoms")
    evidence_map = row.get("evidence_map")
    if raw is None and isinstance(evidence_map, Mapping):
        raw = evidence_map.get("claim_atoms")
    return [deepcopy(dict(atom)) for atom in raw or [] if isinstance(atom, Mapping)]


def _candidate_uid(candidate: Mapping[str, Any], *, idx: int) -> str:
    uid = _compact(
        candidate.get("candidate_uid")
        or candidate.get("candidate_key")
        or candidate.get("evidence_id")
    )
    if not uid:
        raise RoleRescueError(f"candidate_pool[{idx}] has no stable UID")
    return uid


def _token_cost(candidate: Mapping[str, Any]) -> int:
    for field in ("mrec_token_cost", "token_cost", "num_tokens"):
        value = candidate.get(field)
        if value is not None:
            return max(0, _int(value, field))
    text = str(candidate.get("text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _uid_fingerprint(event_id: str, uids: Sequence[str]) -> str:
    payload = json.dumps(
        {"event_id": event_id, "candidate_uids": list(uids)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_in_order(values: Any) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _new_cell_stats() -> dict[str, Any]:
    return {
        "selected_counts": [],
        "underfilled": 0,
        "same_as_random": 0,
        "role_available": Counter(),
        "role_promoted": Counter(),
        "role_present": Counter(),
    }


def _update_cell_stats(
    stats: dict[str, Any], row: Mapping[str, Any], *, random_uids: tuple[str, ...]
) -> None:
    metadata = dict(row.get("role_rescue_metadata") or {})
    selected_uids = tuple(metadata.get("selected_source_candidate_uids") or [])
    stats["selected_counts"].append(len(selected_uids))
    stats["underfilled"] += int(bool(metadata.get("underfilled")))
    stats["same_as_random"] += int(selected_uids == random_uids)
    available = metadata.get("available_role_source_indices") or {}
    promoted = metadata.get("promoted_role_source_indices") or {}
    present = metadata.get("selected_role_presence_source_indices") or {}
    for role in ROLE_CELLS:
        stats["role_available"][role] += int(bool(available.get(role)))
        stats["role_promoted"][role] += int(bool(promoted.get(role)))
        stats["role_present"][role] += int(bool(present.get(role)))


def _finalize_cell_stats(stats: Mapping[str, Any], *, row_count: int) -> dict[str, Any]:
    return {
        "selected_count": _numeric_summary(stats["selected_counts"]),
        "underfilled_count": int(stats["underfilled"]),
        "underfilled_rate": float(stats["underfilled"] / max(row_count, 1)),
        "same_as_random_count": int(stats["same_as_random"]),
        "same_as_random_rate": float(stats["same_as_random"] / max(row_count, 1)),
        "role_available_count": dict(sorted(stats["role_available"].items())),
        "role_promoted_count": dict(sorted(stats["role_promoted"].items())),
        "role_present_count": dict(sorted(stats["role_present"].items())),
    }


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "sum": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_text(value: Any) -> str:
    return " ".join(_compact(value).lower().split())


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _rank(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10**9
    return parsed if parsed >= 0 else 10**9


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RoleRescueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RoleRescueError(f"{name} must be an integer") from exc
    return parsed


def _int_or_default(value: Any, default: int) -> int:
    try:
        return _int(value, "integer value")
    except RoleRescueError:
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
