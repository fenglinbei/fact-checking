#!/usr/bin/env python3
"""Audit and freeze the fixed-K structure-mechanism gate matrix.

The script is deliberately selector-agnostic.  It consumes already materialized
full-order traces and verifier builds, proves the cross-cell contracts, projects
the common exact-K/no-truncation support, and writes a directory directly
consumable by ``sft.label_token_matrix_infer``::

    OUTPUT_DIR/<cell_id>/build/build_<split>.jsonl
    OUTPUT_DIR/audit.json
    OUTPUT_DIR/manifest.json

The four mechanism cells are retrieval, hard-structure, one-shot learned, and
stateful learned.  Any additional ``--cell`` entries are treated as strict
post-prefix shuffles of the stateful cell.  Integrity failures are fail-closed;
an inactive S-vs-O mechanism is instead recorded as ``all_ready=false`` so its
no-go conclusion remains auditable without launching inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "structure-mechanism-gate-matrix-v0.1"
AUDIT_SCHEMA_VERSION = "structure-mechanism-gate-audit-v0.1"
DEFAULT_ACTIVITY_THRESHOLD = 0.10


class MechanismGateError(ValueError):
    """Raised when an input violates the mechanism-gate contract."""


@dataclass(frozen=True)
class BuildRowMeta:
    event_id: str
    offset: int
    line_number: int
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    evidence_count: int
    label_contract: dict[str, Any] | None
    candidate_uids: tuple[str, ...] | None
    uid_text_fingerprint: str | None
    candidate_block_fingerprint: str | None
    prompt_token_count: int | None
    prompt_token_multiset_fingerprint: str | None
    prompt_token_sequence_fingerprint: str | None


@dataclass(frozen=True)
class BuildIndex:
    path: Path
    file_sha256: str
    event_order: tuple[str, ...]
    rows: dict[str, BuildRowMeta]
    eligible_event_ids: frozenset[str]
    exclusion_counts: dict[str, int]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="ID=BUILD_JSONL",
        help="Repeat for R/H/O/S and optional strict shuffle cells.",
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="ID=TRACE_JSONL",
        help="Repeat for at least the four mechanism traces.",
    )
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--expected-k", required=True, type=int)
    parser.add_argument("--stateful-id", required=True)
    parser.add_argument("--one-shot-id", required=True)
    parser.add_argument("--hard-id", default="H")
    parser.add_argument("--retrieval-id", default="R")
    parser.add_argument(
        "--expected-weights-fingerprint",
        "--expected-weights-fp",
        dest="expected_weights_fingerprint",
        required=True,
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=DEFAULT_ACTIVITY_THRESHOLD,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.cells = _parse_id_paths(args.cell, context="--cell")
        args.traces = _parse_id_paths(args.trace, context="--trace")
        _validate_top_level_args(
            cell_paths=args.cells,
            trace_paths=args.traces,
            expected_k=args.expected_k,
            stateful_id=args.stateful_id,
            one_shot_id=args.one_shot_id,
            hard_id=args.hard_id,
            retrieval_id=args.retrieval_id,
            expected_weights_fingerprint=args.expected_weights_fingerprint,
            activity_threshold=args.activity_threshold,
        )
    except MechanismGateError as exc:
        parser.error(str(exc))
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_structure_mechanism_gate(
        cell_paths=args.cells,
        trace_paths=args.traces,
        split=str(args.split),
        expected_k=int(args.expected_k),
        stateful_id=str(args.stateful_id),
        one_shot_id=str(args.one_shot_id),
        hard_id=str(args.hard_id),
        retrieval_id=str(args.retrieval_id),
        expected_weights_fingerprint=str(args.expected_weights_fingerprint),
        output_dir=Path(args.output_dir),
        activity_threshold=float(args.activity_threshold),
        force=bool(args.force),
    )
    status = "READY" if manifest["all_ready"] else "NO-GO"
    print(
        f"{status}: froze {manifest['cell_count']} cells x "
        f"{manifest['event_count']} common events at {args.output_dir}"
    )
    return 0


def prepare_structure_mechanism_gate(
    *,
    cell_paths: Mapping[str, Path],
    trace_paths: Mapping[str, Path],
    split: str,
    expected_k: int,
    stateful_id: str,
    one_shot_id: str,
    hard_id: str,
    retrieval_id: str,
    expected_weights_fingerprint: str,
    output_dir: Path,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically audit inputs and materialize the common-support matrix."""

    normalized_cells = {str(key): Path(value) for key, value in cell_paths.items()}
    normalized_traces = {str(key): Path(value) for key, value in trace_paths.items()}
    _validate_top_level_args(
        cell_paths=normalized_cells,
        trace_paths=normalized_traces,
        expected_k=expected_k,
        stateful_id=stateful_id,
        one_shot_id=one_shot_id,
        hard_id=hard_id,
        retrieval_id=retrieval_id,
        expected_weights_fingerprint=expected_weights_fingerprint,
        activity_threshold=activity_threshold,
    )
    output_dir = Path(output_dir)
    if output_dir.exists() and not force:
        raise MechanismGateError(
            f"output directory already exists; pass --force to replace: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise MechanismGateError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    promoted = False
    try:
        manifest = _materialize_gate(
            cell_paths=normalized_cells,
            trace_paths=normalized_traces,
            split=split,
            expected_k=expected_k,
            stateful_id=stateful_id,
            one_shot_id=one_shot_id,
            hard_id=hard_id,
            retrieval_id=retrieval_id,
            expected_weights_fingerprint=expected_weights_fingerprint,
            staging_dir=staging,
            logical_output_dir=output_dir,
            activity_threshold=activity_threshold,
        )
        _promote_directory(staging, output_dir, force=force)
        promoted = True
        return manifest
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def _materialize_gate(
    *,
    cell_paths: Mapping[str, Path],
    trace_paths: Mapping[str, Path],
    split: str,
    expected_k: int,
    stateful_id: str,
    one_shot_id: str,
    hard_id: str,
    retrieval_id: str,
    expected_weights_fingerprint: str,
    staging_dir: Path,
    logical_output_dir: Path,
    activity_threshold: float,
) -> dict[str, Any]:
    base_ids = (retrieval_id, hard_id, one_shot_id, stateful_id)
    shuffle_ids = tuple(cell_id for cell_id in cell_paths if cell_id not in base_ids)
    trace_audit = _audit_traces_streaming(
        trace_paths={cell_id: trace_paths[cell_id] for cell_id in base_ids},
        reference_id=stateful_id,
        one_shot_id=one_shot_id,
        stateful_id=stateful_id,
        expected_weights_fingerprint=expected_weights_fingerprint,
    )

    indexes = {
        cell_id: _index_build(path, expected_k=expected_k, cell_id=cell_id)
        for cell_id, path in cell_paths.items()
    }
    reference = indexes[stateful_id]
    common_set = set(reference.eligible_event_ids)
    for index in indexes.values():
        common_set.intersection_update(index.eligible_event_ids)
    common_event_ids = [
        event_id for event_id in reference.event_order if event_id in common_set
    ]
    if not common_event_ids:
        raise MechanismGateError(
            "the cells have no common exact-K, no-truncation build support"
        )

    label_schema = _validate_common_rows(
        indexes=indexes,
        event_ids=common_event_ids,
        stateful_id=stateful_id,
        expected_k=expected_k,
    )
    comparisons = {
        f"{stateful_id}_vs_{one_shot_id}": _comparison_stats(
            indexes[stateful_id], indexes[one_shot_id], common_event_ids
        ),
        f"{stateful_id}_vs_{hard_id}": _comparison_stats(
            indexes[stateful_id], indexes[hard_id], common_event_ids
        ),
        f"{hard_id}_vs_{retrieval_id}": _comparison_stats(
            indexes[hard_id], indexes[retrieval_id], common_event_ids
        ),
    }
    primary_key = f"{stateful_id}_vs_{one_shot_id}"
    primary_rate = float(comparisons[primary_key]["visible_sequence_difference_rate"])
    activity_gate = {
        "comparison": primary_key,
        "metric": "visible_sequence_difference_rate",
        "threshold": activity_threshold,
        "observed_rate": primary_rate,
        "passed": primary_rate >= activity_threshold,
    }

    shuffle_audits: dict[str, dict[str, Any]] = {}
    for shuffle_id in shuffle_ids:
        shuffle_audits[shuffle_id] = _validate_shuffle(
            stateful=indexes[stateful_id],
            shuffle=indexes[shuffle_id],
            event_ids=common_event_ids,
            expected_k=expected_k,
        )
    shuffle_order_gate = {
        "required_rate": 1.0 if expected_k > 1 else 0.0,
        "passed": all(
            audit["order_change_rate"] == (1.0 if expected_k > 1 else 0.0)
            for audit in shuffle_audits.values()
        ),
        "cell_count": len(shuffle_audits),
    }
    all_ready = bool(activity_gate["passed"] and shuffle_order_gate["passed"])
    event_sha = _event_sequence_sha256(common_event_ids)

    output_cells: list[dict[str, Any]] = []
    for cell_id, index in indexes.items():
        output_path = staging_dir / cell_id / "build" / f"build_{split}.jsonl"
        _copy_indexed_rows(index, common_event_ids, output_path)
        build_sha = _sha256_file(output_path)
        output_cells.append(
            {
                "cell_id": cell_id,
                "ready": True,
                "row_count": len(common_event_ids),
                "event_id_sequence_sha256": event_sha,
                "build_file": str(
                    (logical_output_dir / cell_id / "build" / output_path.name).resolve()
                ),
                "build_sha256": build_sha,
                "build_sha": build_sha,
                "cell_kind": "shuffle" if cell_id in shuffle_ids else "mechanism",
            }
        )

    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "passed": all_ready,
        "split": split,
        "expected_k": expected_k,
        "mechanism_cells": {
            "retrieval": retrieval_id,
            "hard_structure": hard_id,
            "one_shot_learned": one_shot_id,
            "stateful_learned": stateful_id,
        },
        "shuffle_cells": list(shuffle_ids),
        "expected_weights_fingerprint": expected_weights_fingerprint,
        "trace_audit": trace_audit,
        "common_support": {
            "reference_cell": stateful_id,
            "event_count": len(common_event_ids),
            "event_id_sequence_sha256": event_sha,
            "reference_event_count": len(reference.event_order),
            "dropped_reference_event_count": len(reference.event_order)
            - len(common_event_ids),
            "cells": {
                cell_id: {
                    "source_build": str(index.path.resolve()),
                    "source_build_sha256": index.file_sha256,
                    "source_row_count": len(index.event_order),
                    "eligible_event_count": len(index.eligible_event_ids),
                    "exclusion_counts": index.exclusion_counts,
                    "missing_from_common_reference_count": len(
                        set(reference.event_order) - set(index.event_order)
                    ),
                }
                for cell_id, index in indexes.items()
            },
        },
        "label_schema": label_schema,
        "comparisons": comparisons,
        "activity_gate": activity_gate,
        "shuffle_audits": shuffle_audits,
        "shuffle_order_gate": shuffle_order_gate,
        "checks": [
            "four traces have identical event and candidate-pool UID sequences",
            "one-shot and stateful weights fingerprints match the frozen expectation",
            "common support is exact-K with explicit no-truncation flags",
            "claim, target, label schema, and gold metadata agree across cells",
            "strict shuffles preserve visible UID/text/block and prompt-token multisets",
            "stateful-vs-one-shot activity meets the declared threshold",
        ],
    }
    _write_json(staging_dir / "audit.json", audit)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "fixed_k_structure_mechanism_gate",
        "split": split,
        "label_schema": label_schema,
        "expected_k": expected_k,
        "event_count": len(common_event_ids),
        "event_id_sequence_sha256": event_sha,
        "cell_count": len(output_cells),
        "all_ready": all_ready,
        "activity_gate_passed": bool(activity_gate["passed"]),
        "shuffle_order_gate_passed": bool(shuffle_order_gate["passed"]),
        "audit_file": str((logical_output_dir / "audit.json").resolve()),
        "audit_sha256": _sha256_file(staging_dir / "audit.json"),
        "cells": output_cells,
    }
    _write_json(staging_dir / "manifest.json", manifest)
    return manifest


def _audit_traces_streaming(
    *,
    trace_paths: Mapping[str, Path],
    reference_id: str,
    one_shot_id: str,
    stateful_id: str,
    expected_weights_fingerprint: str,
) -> dict[str, Any]:
    for cell_id, path in trace_paths.items():
        if not path.is_file():
            raise MechanismGateError(f"missing trace for {cell_id}: {path}")
    hashers = {cell_id: hashlib.sha256() for cell_id in trace_paths}
    event_hasher = hashlib.sha256()
    pool_hasher = hashlib.sha256()
    weights_by_id: dict[str, set[str]] = {one_shot_id: set(), stateful_id: set()}
    seen_events: set[str] = set()
    row_count = 0
    with ExitStack() as stack:
        handles = {
            cell_id: stack.enter_context(path.open("rb"))
            for cell_id, path in trace_paths.items()
        }
        while True:
            raw_by_id = {cell_id: handle.readline() for cell_id, handle in handles.items()}
            ended = {cell_id for cell_id, raw in raw_by_id.items() if raw == b""}
            if ended:
                if len(ended) != len(raw_by_id):
                    raise MechanismGateError(
                        "trace row counts differ; ended=" + ",".join(sorted(ended))
                    )
                break
            rows: dict[str, dict[str, Any]] = {}
            for cell_id, raw in raw_by_id.items():
                if not raw.strip():
                    raise MechanismGateError(
                        f"blank line in trace {cell_id} at line {row_count + 1}"
                    )
                hashers[cell_id].update(raw)
                rows[cell_id] = _decode_json_object(
                    raw, context=f"trace {cell_id}:{row_count + 1}"
                )
            reference_row = rows[reference_id]
            event_id = _event_id(reference_row, context=f"trace {reference_id}")
            if event_id in seen_events:
                raise MechanismGateError(
                    f"duplicate event_id {event_id!r} in mechanism traces"
                )
            seen_events.add(event_id)
            reference_uids = _trace_candidate_pool_uids(reference_row, event_id=event_id)
            for cell_id, row in rows.items():
                other_event = _event_id(row, context=f"trace {cell_id}")
                if other_event != event_id:
                    raise MechanismGateError(
                        f"trace event mismatch at row {row_count}: "
                        f"{reference_id}={event_id!r}, {cell_id}={other_event!r}"
                    )
                uids = _trace_candidate_pool_uids(row, event_id=event_id)
                if uids != reference_uids:
                    raise MechanismGateError(
                        f"{event_id}: candidate-pool UID sequence differs between "
                        f"{reference_id} and {cell_id}"
                    )
                if cell_id in weights_by_id:
                    fingerprint = _trace_weight_fingerprint(row, cell_id=cell_id)
                    if fingerprint != expected_weights_fingerprint:
                        raise MechanismGateError(
                            f"{event_id}:{cell_id}: weight fingerprint {fingerprint!r}; "
                            f"expected {expected_weights_fingerprint!r}"
                        )
                    weights_by_id[cell_id].add(fingerprint)
            event_hasher.update(event_id.encode("utf-8"))
            event_hasher.update(b"\0")
            pool_hasher.update(_canonical_json_bytes([event_id, list(reference_uids)]))
            pool_hasher.update(b"\0")
            row_count += 1
    if row_count == 0:
        raise MechanismGateError("mechanism traces contain zero rows")
    for cell_id, fingerprints in weights_by_id.items():
        if fingerprints != {expected_weights_fingerprint}:
            raise MechanismGateError(
                f"trace {cell_id} did not prove the expected weights fingerprint"
            )
    return {
        "passed": True,
        "reference_cell": reference_id,
        "row_count": row_count,
        "event_id_sequence_sha256": event_hasher.hexdigest(),
        "candidate_pool_uid_sequence_sha256": pool_hasher.hexdigest(),
        "weights_fingerprints": {
            cell_id: sorted(values) for cell_id, values in weights_by_id.items()
        },
        "trace_files": {
            cell_id: {
                "path": str(trace_paths[cell_id].resolve()),
                "sha256": hashers[cell_id].hexdigest(),
            }
            for cell_id in trace_paths
        },
    }


def _index_build(path: Path, *, expected_k: int, cell_id: str) -> BuildIndex:
    if not path.is_file():
        raise MechanismGateError(f"missing build for {cell_id}: {path}")
    rows: dict[str, BuildRowMeta] = {}
    event_order: list[str] = []
    eligible_ids: set[str] = set()
    exclusions: Counter[str] = Counter()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if raw == b"":
                break
            line_number += 1
            if not raw.strip():
                raise MechanismGateError(f"blank line in build {cell_id}:{line_number}")
            hasher.update(raw)
            row = _decode_json_object(raw, context=f"build {cell_id}:{line_number}")
            event_id = _event_id(row, context=f"build {cell_id}:{line_number}")
            if event_id in rows:
                raise MechanismGateError(
                    f"duplicate event_id {event_id!r} in build {cell_id}"
                )
            evidence_count = _nonnegative_int(
                row.get("evidence_count"), context=f"{cell_id}:{event_id}:evidence_count"
            )
            reasons: list[str] = []
            if evidence_count != expected_k:
                reasons.append("not_exact_k")
            if row.get("was_truncated") is not False:
                reasons.append("was_truncated_or_missing")
            if row.get("evidence_text_truncated") is not False:
                reasons.append("evidence_text_truncated_or_missing")
            eligible = not reasons
            if eligible:
                metadata = _eligible_build_metadata(
                    row, cell_id=cell_id, event_id=event_id, expected_k=expected_k
                )
                eligible_ids.add(event_id)
            else:
                for reason in reasons:
                    exclusions[reason] += 1
                metadata = {
                    "label_contract": None,
                    "candidate_uids": None,
                    "uid_text_fingerprint": None,
                    "candidate_block_fingerprint": None,
                    "prompt_token_count": None,
                    "prompt_token_multiset_fingerprint": None,
                    "prompt_token_sequence_fingerprint": None,
                }
            rows[event_id] = BuildRowMeta(
                event_id=event_id,
                offset=offset,
                line_number=line_number,
                eligible=eligible,
                exclusion_reasons=tuple(reasons),
                evidence_count=evidence_count,
                **metadata,
            )
            event_order.append(event_id)
    if not event_order:
        raise MechanismGateError(f"build {cell_id} contains zero rows")
    return BuildIndex(
        path=path,
        file_sha256=hasher.hexdigest(),
        event_order=tuple(event_order),
        rows=rows,
        eligible_event_ids=frozenset(eligible_ids),
        exclusion_counts=dict(sorted(exclusions.items())),
    )


def _eligible_build_metadata(
    row: Mapping[str, Any], *, cell_id: str, event_id: str, expected_k: int
) -> dict[str, Any]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < expected_k:
        raise MechanismGateError(
            f"{cell_id}:{event_id}: candidates must contain at least K={expected_k} items"
        )
    visible: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates[:expected_k]):
        if not isinstance(candidate, Mapping):
            raise MechanismGateError(
                f"{cell_id}:{event_id}: candidates[{position}] is not an object"
            )
        copied = dict(candidate)
        uid = _candidate_uid(copied, context=f"{cell_id}:{event_id}:{position}")
        if not isinstance(copied.get("text"), str):
            raise MechanismGateError(
                f"{cell_id}:{event_id}:{uid}: candidate text must be a string"
            )
        visible.append(copied)
    uids = tuple(_candidate_uid(candidate, context=event_id) for candidate in visible)
    if len(set(uids)) != len(uids):
        raise MechanismGateError(
            f"{cell_id}:{event_id}: visible candidate UIDs are not unique"
        )
    prompt_ids = _token_ids(
        row.get("prompt_input_ids"), context=f"{cell_id}:{event_id}:prompt_input_ids"
    )
    if not prompt_ids:
        raise MechanismGateError(f"{cell_id}:{event_id}: prompt_input_ids is empty")
    prompt_count = _nonnegative_int(
        row.get("prompt_token_count"), context=f"{cell_id}:{event_id}:prompt_token_count"
    )
    if prompt_count != len(prompt_ids):
        raise MechanismGateError(
            f"{cell_id}:{event_id}: prompt_token_count={prompt_count}, "
            f"len(prompt_input_ids)={len(prompt_ids)}"
        )
    label_contract = _label_contract(row, cell_id=cell_id, event_id=event_id)
    by_uid = {uid: candidate for uid, candidate in zip(uids, visible)}
    sorted_uids = sorted(by_uid)
    return {
        "label_contract": label_contract,
        "candidate_uids": uids,
        "uid_text_fingerprint": _sha256_json(
            [
                {"candidate_uid": uid, "text": by_uid[uid]["text"]}
                for uid in sorted_uids
            ]
        ),
        "candidate_block_fingerprint": _sha256_json(
            [
                {"candidate_uid": uid, "candidate": by_uid[uid]}
                for uid in sorted_uids
            ]
        ),
        "prompt_token_count": prompt_count,
        "prompt_token_multiset_fingerprint": _token_multiset_fingerprint(prompt_ids),
        "prompt_token_sequence_fingerprint": _sha256_json(prompt_ids),
    }


def _validate_common_rows(
    *,
    indexes: Mapping[str, BuildIndex],
    event_ids: Sequence[str],
    stateful_id: str,
    expected_k: int,
) -> str:
    common_schema: str | None = None
    for event_id in event_ids:
        reference = indexes[stateful_id].rows[event_id]
        if reference.label_contract is None:
            raise AssertionError("common support contains an ineligible reference row")
        if reference.evidence_count != expected_k:
            raise AssertionError("common support contains a non-exact-K row")
        for cell_id, index in indexes.items():
            row = index.rows[event_id]
            if row.label_contract != reference.label_contract:
                raise MechanismGateError(
                    f"{event_id}: claim/target/label/gold contract differs between "
                    f"{stateful_id} and {cell_id}"
                )
            if row.evidence_count != reference.evidence_count:
                raise MechanismGateError(
                    f"{event_id}: evidence_count differs between {stateful_id} and {cell_id}"
                )
        schema = str(reference.label_contract["label_schema"])
        if common_schema is None:
            common_schema = schema
        elif schema != common_schema:
            raise MechanismGateError(
                f"common support mixes label schemas {common_schema!r} and {schema!r}"
            )
    if not common_schema:
        raise MechanismGateError("common support has no label schema")
    return common_schema


def _comparison_stats(
    left: BuildIndex, right: BuildIndex, event_ids: Sequence[str]
) -> dict[str, Any]:
    set_difference = 0
    order_difference = 0
    same_set_order_difference = 0
    top1_difference = 0
    jaccard_total = 0.0
    for event_id in event_ids:
        left_uids = left.rows[event_id].candidate_uids
        right_uids = right.rows[event_id].candidate_uids
        if left_uids is None or right_uids is None:
            raise AssertionError("comparison received an ineligible row")
        left_set = set(left_uids)
        right_set = set(right_uids)
        sets_differ = left_set != right_set
        orders_differ = left_uids != right_uids
        set_difference += int(sets_differ)
        order_difference += int(orders_differ)
        same_set_order_difference += int(not sets_differ and orders_differ)
        top1_difference += int(left_uids[0] != right_uids[0])
        union = left_set | right_set
        jaccard_total += len(left_set & right_set) / len(union) if union else 1.0
    count = len(event_ids)
    return {
        "event_count": count,
        "visible_set_difference_count": set_difference,
        "visible_set_difference_rate": set_difference / count,
        "visible_sequence_difference_count": order_difference,
        "visible_sequence_difference_rate": order_difference / count,
        "same_set_order_difference_count": same_set_order_difference,
        "same_set_order_difference_rate": same_set_order_difference / count,
        "top1_difference_count": top1_difference,
        "top1_difference_rate": top1_difference / count,
        "mean_visible_set_jaccard": jaccard_total / count,
    }


def _validate_shuffle(
    *,
    stateful: BuildIndex,
    shuffle: BuildIndex,
    event_ids: Sequence[str],
    expected_k: int,
) -> dict[str, Any]:
    order_changed = 0
    prompt_sequence_changed = 0
    for event_id in event_ids:
        source = stateful.rows[event_id]
        control = shuffle.rows[event_id]
        invariants = (
            ("candidate UID set", set(source.candidate_uids or ()), set(control.candidate_uids or ())),
            ("UID-to-text multiset", source.uid_text_fingerprint, control.uid_text_fingerprint),
            (
                "candidate block multiset",
                source.candidate_block_fingerprint,
                control.candidate_block_fingerprint,
            ),
            ("evidence_count", source.evidence_count, control.evidence_count),
            ("prompt token count", source.prompt_token_count, control.prompt_token_count),
            (
                "prompt token multiset",
                source.prompt_token_multiset_fingerprint,
                control.prompt_token_multiset_fingerprint,
            ),
        )
        for name, source_value, control_value in invariants:
            if source_value != control_value:
                raise MechanismGateError(
                    f"{event_id}: shuffle {shuffle.path} changed {name}"
                )
        if source.evidence_count != expected_k:
            raise AssertionError("shuffle common support is not exact-K")
        order_changed += int(source.candidate_uids != control.candidate_uids)
        prompt_sequence_changed += int(
            source.prompt_token_sequence_fingerprint
            != control.prompt_token_sequence_fingerprint
        )
    count = len(event_ids)
    return {
        "event_count": count,
        "order_change_count": order_changed,
        "order_change_rate": order_changed / count,
        "prompt_token_sequence_change_count": prompt_sequence_changed,
        "prompt_token_sequence_change_rate": prompt_sequence_changed / count,
        "same_uid_set_rate": 1.0,
        "same_uid_text_multiset_rate": 1.0,
        "same_candidate_block_multiset_rate": 1.0,
        "same_prompt_token_multiset_rate": 1.0,
    }


def _copy_indexed_rows(
    index: BuildIndex, event_ids: Sequence[str], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with index.path.open("rb") as source, temp_path.open("wb") as destination:
            for event_id in event_ids:
                meta = index.rows.get(event_id)
                if meta is None or not meta.eligible:
                    raise AssertionError("common support cannot be copied from this cell")
                source.seek(meta.offset)
                raw = source.readline()
                row = _decode_json_object(raw, context=f"copy {index.path}:{event_id}")
                if _event_id(row, context="copied build row") != event_id:
                    raise MechanismGateError(
                        f"source build drifted after indexing: {index.path}:{event_id}"
                    )
                destination.write(raw if raw.endswith((b"\n", b"\r")) else raw + b"\n")
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _label_contract(
    row: Mapping[str, Any], *, cell_id: str, event_id: str
) -> dict[str, Any]:
    claim = str(row.get("claim") or "")
    target = str(row.get("target") or "")
    label_schema = str(row.get("label_schema") or "")
    gold_label = str(row.get("gold_label") or "")
    if not claim or not target or not label_schema or not gold_label:
        raise MechanismGateError(
            f"{cell_id}:{event_id}: claim/target/label_schema/gold_label must be non-empty"
        )
    gold_id = _nonnegative_int(
        row.get("gold_id"), context=f"{cell_id}:{event_id}:gold_id"
    )
    return {
        "claim": claim,
        "target": target,
        "target_token_count": _nonnegative_int(
            row.get("target_token_count", 0),
            context=f"{cell_id}:{event_id}:target_token_count",
        ),
        "label_schema": label_schema,
        "gold_id": gold_id,
        "gold_label": gold_label,
        "label": row.get("label"),
    }


def _trace_candidate_pool_uids(
    row: Mapping[str, Any], *, event_id: str
) -> tuple[str, ...]:
    pool = row.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise MechanismGateError(f"{event_id}: trace candidate_pool must be non-empty")
    uids: list[str] = []
    for position, candidate in enumerate(pool):
        if not isinstance(candidate, Mapping):
            raise MechanismGateError(
                f"{event_id}: candidate_pool[{position}] is not an object"
            )
        uids.append(_candidate_uid(candidate, context=f"{event_id}:pool:{position}"))
    if len(set(uids)) != len(uids):
        raise MechanismGateError(f"{event_id}: candidate_pool UIDs are not unique")
    return tuple(uids)


def _trace_weight_fingerprint(row: Mapping[str, Any], *, cell_id: str) -> str:
    values = {
        str(value).strip()
        for value in (
            row.get("weight_fingerprint"),
            (row.get("params") or {}).get("weight_fingerprint")
            if isinstance(row.get("params"), Mapping)
            else None,
            (row.get("mrec_diagnostics") or {}).get("weight_fingerprint")
            if isinstance(row.get("mrec_diagnostics"), Mapping)
            else None,
        )
        if value is not None and str(value).strip()
    }
    if len(values) != 1:
        raise MechanismGateError(
            f"trace {cell_id} has missing or conflicting weight fingerprints: {sorted(values)}"
        )
    return next(iter(values))


def _parse_id_paths(values: Sequence[str], *, context: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        cell_id, separator, raw_path = str(value).partition("=")
        cell_id = cell_id.strip()
        raw_path = raw_path.strip()
        if not separator or not cell_id or not raw_path:
            raise MechanismGateError(f"{context} requires ID=PATH, got {value!r}")
        if cell_id in parsed:
            raise MechanismGateError(f"duplicate {context} ID {cell_id!r}")
        if "/" in cell_id or "\\" in cell_id or cell_id in {".", ".."}:
            raise MechanismGateError(f"unsafe cell ID {cell_id!r}")
        parsed[cell_id] = Path(raw_path)
    return parsed


def _validate_top_level_args(
    *,
    cell_paths: Mapping[str, Path],
    trace_paths: Mapping[str, Path],
    expected_k: int,
    stateful_id: str,
    one_shot_id: str,
    hard_id: str,
    retrieval_id: str,
    expected_weights_fingerprint: str,
    activity_threshold: float,
) -> None:
    base_ids = (retrieval_id, hard_id, one_shot_id, stateful_id)
    if any(not str(cell_id).strip() for cell_id in base_ids):
        raise MechanismGateError("mechanism cell IDs must be non-empty")
    if len(set(base_ids)) != 4:
        raise MechanismGateError("retrieval/hard/one-shot/stateful IDs must be distinct")
    missing_cells = set(base_ids) - set(cell_paths)
    if missing_cells:
        raise MechanismGateError(f"missing mechanism builds: {sorted(missing_cells)}")
    missing_traces = set(base_ids) - set(trace_paths)
    if missing_traces:
        raise MechanismGateError(f"missing mechanism traces: {sorted(missing_traces)}")
    if expected_k <= 0:
        raise MechanismGateError("expected_k must be positive")
    if not str(expected_weights_fingerprint).strip():
        raise MechanismGateError("expected weights fingerprint must be non-empty")
    if not 0.0 <= activity_threshold <= 1.0:
        raise MechanismGateError("activity threshold must be in [0, 1]")


def _promote_directory(staging: Path, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        raise MechanismGateError(f"refusing to replace existing directory: {target}")
    backup = target.with_name(f".{target.name}.old.{os.getpid()}")
    if backup.exists():
        raise MechanismGateError(f"backup directory already exists: {backup}")
    if target.exists():
        target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _decode_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismGateError(f"invalid JSON object in {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise MechanismGateError(f"expected JSON object in {context}")
    return value


def _event_id(row: Mapping[str, Any], *, context: str) -> str:
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        raise MechanismGateError(f"{context} has no non-empty event_id")
    return event_id


def _candidate_uid(candidate: Mapping[str, Any], *, context: str) -> str:
    uid = str(candidate.get("candidate_uid") or "").strip()
    if not uid:
        raise MechanismGateError(f"{context} has no non-empty candidate_uid")
    return uid


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise MechanismGateError(f"{context} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MechanismGateError(f"{context} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise MechanismGateError(f"{context} must be an integer")
    if parsed < 0:
        raise MechanismGateError(f"{context} must be non-negative")
    return parsed


def _token_ids(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list):
        raise MechanismGateError(f"{context} must be a list")
    output: list[int] = []
    for position, token_id in enumerate(value):
        if isinstance(token_id, bool):
            raise MechanismGateError(f"{context}[{position}] must be int, not bool")
        try:
            output.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise MechanismGateError(f"{context}[{position}] is not an int") from exc
    return output


def _token_multiset_fingerprint(token_ids: Sequence[int]) -> str:
    counts = Counter(int(token_id) for token_id in token_ids)
    return _sha256_json([[token_id, counts[token_id]] for token_id in sorted(counts)])


def _event_sequence_sha256(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for event_id in event_ids:
        digest.update(event_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
