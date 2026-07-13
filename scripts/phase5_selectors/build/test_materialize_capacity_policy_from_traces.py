from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.materialize_capacity_policy_from_traces import (
    CapacityPolicyError,
    materialize_capacity_policy,
)


def _trace_row(
    event_id: str,
    *,
    selector: str,
    selected: list[int],
    full_order: list[int] | None = None,
) -> dict:
    order = full_order or [2, 0, 1]
    pool = [{"candidate_uid": f"u{index}"} for index in range(3)]
    return {
        "event_id": event_id,
        "factor_selector": selector,
        "factor_controller": "ordinal_replay_minmax5_10",
        "factorial_metadata": {
            "selector_level": selector,
            "controller_level": "ordinal_replay_minmax5_10",
        },
        "candidate_pool": pool,
        "selector_full_ordered_indices": order,
        "selector_full_ordered_candidate_uids": [f"u{index}" for index in order],
        "selector_available_ordered_indices": order,
        "selector_available_ordered_candidate_uids": [f"u{index}" for index in order],
        "selected_indices": selected,
        "selected_candidate_uids": [f"u{index}" for index in selected],
        "selected_count": len(selected),
    }


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ordinal_pool_exhausted_row(event_id: str, *, selector: str) -> dict:
    order = [2, 0, 1]
    row = _trace_row(
        event_id,
        selector=selector,
        selected=order,
        full_order=order,
    )
    row["factorial_metadata"].update(
        {
            "controller_contract": (
                "first prefix t>=5 reaching the common exact Kmax=10 ordinal target, "
                "else 10"
            ),
            "k_min": 5,
            "k_max": 10,
            "stored_target_resolved_used": False,
            "common_exact_kmax10_target_state": [99],
            "controller_stop_reason": "pool_exhausted",
        }
    )
    row["baces_display_steps"] = [
        {
            "position": position,
            "candidate_idx": candidate_index,
            "candidate_uid": f"u{candidate_index}",
            "state_after": [position],
        }
        for position, candidate_index in enumerate(order, start=1)
    ]
    row["baces_display"] = {"length": 3, "terminal_state": [3]}
    return row


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_materializes_multi_selector_policy_and_summary(tmp_path: Path) -> None:
    baces = tmp_path / "baces.jsonl"
    learned = tmp_path / "learned.jsonl"
    _write_trace(
        baces,
        [
            _trace_row("e1", selector="baces_exact", selected=[2]),
            _trace_row("e2", selector="baces_exact", selected=[2, 0]),
        ],
    )
    _write_trace(
        learned,
        [
            _trace_row("e1", selector="learned_marginal", selected=[2, 0]),
            _trace_row("e2", selector="learned_marginal", selected=[2, 0, 1]),
        ],
    )
    output = tmp_path / "resolve_stop.jsonl"

    summary = materialize_capacity_policy(
        selector_traces={"baces_exact": baces, "learned_marginal": learned},
        policy_id="resolve_stop",
        output_policy=output,
        expected_controller="ordinal_replay_minmax5_10",
        min_k=1,
        max_k=3,
    )

    rows = _read_jsonl(output)
    assert rows == [
        {
            "event_id": "e1",
            "policy_id": "resolve_stop",
            "selected_k": 1,
            "selector_level": "baces_exact",
        },
        {
            "event_id": "e2",
            "policy_id": "resolve_stop",
            "selected_k": 2,
            "selector_level": "baces_exact",
        },
        {
            "event_id": "e1",
            "policy_id": "resolve_stop",
            "selected_k": 2,
            "selector_level": "learned_marginal",
        },
        {
            "event_id": "e2",
            "policy_id": "resolve_stop",
            "selected_k": 3,
            "selector_level": "learned_marginal",
        },
    ]
    assert summary["selector_count"] == 2
    assert summary["event_count"] == 2
    assert summary["assignment_count"] == 4
    assert summary["k_distribution"] == {"1": 1, "2": 2, "3": 1}
    assert summary["event_id_sequence_sha256"]
    assert summary["output_policy"] == str(output.resolve())
    assert summary["provenance"]["verification_status"] == (
        "controller_provenance_unknown"
    )
    assert summary["provenance"]["uses_gold"] is None
    assert summary["provenance"]["deployable_ex_ante"] is False
    assert all(source["sha256"] for source in summary["sources"])
    sidecar = json.loads(
        (tmp_path / "resolve_stop.summary.json").read_text(encoding="utf-8")
    )
    assert sidecar == summary
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_rejects_cross_selector_event_sequence_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_trace(
        first,
        [
            _trace_row("e1", selector="a", selected=[2]),
            _trace_row("e2", selector="a", selected=[2]),
        ],
    )
    _write_trace(
        second,
        [
            _trace_row("e2", selector="b", selected=[2]),
            _trace_row("e1", selector="b", selected=[2]),
        ],
    )

    with pytest.raises(CapacityPolicyError, match="event sequence differs"):
        materialize_capacity_policy(
            selector_traces={"a": first, "b": second},
            policy_id="p",
            output_policy=tmp_path / "policy.jsonl",
        )
    assert not (tmp_path / "policy.jsonl").exists()


def test_rejects_nonprefix_selection(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [_trace_row("e1", selector="a", selected=[2, 1], full_order=[2, 0, 1])],
    )

    with pytest.raises(CapacityPolicyError, match="not a prefix"):
        materialize_capacity_policy(
            selector_traces={"a": trace},
            policy_id="p",
            output_policy=tmp_path / "policy.jsonl",
        )


def test_requires_overwrite_for_policy_or_sidecar(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, [_trace_row("e1", selector="a", selected=[2])])
    output = tmp_path / "policy.jsonl"
    materialize_capacity_policy(
        selector_traces={"a": trace},
        policy_id="p",
        output_policy=output,
    )

    with pytest.raises(CapacityPolicyError, match="--overwrite"):
        materialize_capacity_policy(
            selector_traces={"a": trace},
            policy_id="p",
            output_policy=output,
        )

    summary = materialize_capacity_policy(
        selector_traces={"a": trace},
        policy_id="p2",
        output_policy=output,
        overwrite=True,
    )
    assert _read_jsonl(output)[0]["policy_id"] == "p2"
    assert summary["policy_id"] == "p2"


def test_rejects_order_index_outside_candidate_pool(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    row = _trace_row("e1", selector="a", selected=[2])
    row["selector_available_ordered_indices"] = [3]
    row["selector_available_ordered_candidate_uids"] = ["u3"]
    _write_trace(trace, [row])

    with pytest.raises(CapacityPolicyError, match="outside candidate_pool"):
        materialize_capacity_policy(
            selector_traces={"a": trace},
            policy_id="p",
            output_policy=tmp_path / "policy.jsonl",
        )


def test_rejects_candidate_uid_binding_mismatch(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    row = _trace_row("e1", selector="a", selected=[2])
    row["selected_candidate_uids"] = ["u0"]
    _write_trace(trace, [row])

    with pytest.raises(CapacityPolicyError, match="does not bind selected_indices"):
        materialize_capacity_policy(
            selector_traces={"a": trace},
            policy_id="p",
            output_policy=tmp_path / "policy.jsonl",
        )


def test_verifies_known_structure_only_controller_against_factorial_manifest(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _ordinal_pool_exhausted_row("e1", selector="baces_exact"),
            _ordinal_pool_exhausted_row("e2", selector="baces_exact"),
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "baces_factorial_trace_v0_1",
                "factorial_version": "test",
                "split": "val",
                "event_count": 2,
                "controller_contracts": {
                    "ordinal_replay_minmax5_10": (
                        "first prefix t>=5 reaching the common exact Kmax=10 ordinal "
                        "target, else 10"
                    )
                },
                "source_contract": {"coverage_and_pool": "features"},
                "cells": [
                    {
                        "cell_id": "baces_exact__ordinal_replay_minmax5_10",
                        "selector_level": "baces_exact",
                        "controller_level": "ordinal_replay_minmax5_10",
                        "trace_file": trace.name,
                        "row_count": 2,
                        "ready": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = materialize_capacity_policy(
        selector_traces={"baces_exact": trace},
        policy_id="ordinal_replay_minmax5_10",
        output_policy=tmp_path / "policy.jsonl",
        expected_controller="ordinal_replay_minmax5_10",
        source_factorial_manifest=manifest,
        min_k=1,
        max_k=3,
        sample_limit=1,
    )

    provenance = summary["provenance"]
    assert provenance["verification_status"] == (
        "verified_known_structure_only_factorial_controller"
    )
    assert provenance["uses_gold"] is False
    assert provenance["uses_verifier_logits"] is False
    assert provenance["deployable_ex_ante"] is True
    assert summary["factorial_manifest"]["sha256"]
    assert summary["factorial_manifest"]["cells"][0]["trace_sha256"]
    assert summary["event_count"] == 1
    assert summary["assignment_count"] == 1
    assert summary["factorial_manifest"]["event_count"] == 2
    assert summary["factorial_manifest"]["policy_event_count"] == 1

    tampered_rows = _read_jsonl(trace)
    tampered_rows[0]["factorial_metadata"]["controller_stop_reason"] = "max10"
    _write_trace(trace, tampered_rows)
    with pytest.raises(CapacityPolicyError, match="first-hit/limit rule"):
        materialize_capacity_policy(
            selector_traces={"baces_exact": trace},
            policy_id="tampered",
            output_policy=tmp_path / "tampered.jsonl",
            expected_controller="ordinal_replay_minmax5_10",
            source_factorial_manifest=manifest,
            min_k=1,
            max_k=3,
            sample_limit=1,
        )
