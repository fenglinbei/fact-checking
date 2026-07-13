from __future__ import annotations

import importlib.util
from pathlib import Path

from fact_checking.selectors.baces_objective import BacesCandidate, BacesProblem


MODULE_PATH = Path(__file__).with_name("evaluate_baces_reference.py")
SPEC = importlib.util.spec_from_file_location("evaluate_baces_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_baces_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_baces_reference)


def test_audit_one_recovers_five_stage_regret_and_same_set_order(tmp_path: Path) -> None:
    trace_path = tmp_path / "selection_trace_val.jsonl"
    feature_row = {
        "event_id": "event-1",
        "claim_atoms": [
            {"atom_id": "A1", "importance": 0.3},
            {"atom_id": "A2", "importance": 0.8},
        ],
        "candidates": [
            _candidate("A", 2, "A1"),
            _candidate("B", 2, "A2"),
            _candidate("C", 1, "A1", "A2"),
            _candidate("N", 0),
        ],
    }
    trace_pool = [
        {**candidate, "mrec_token_cost": cost}
        for candidate, cost in zip(feature_row["candidates"], (2, 2, 1, 1))
    ]
    pool_by_uid = {candidate["candidate_uid"]: candidate for candidate in trace_pool}
    trace_row = {
        "event_id": "event-1",
        "candidate_pool": trace_pool,
        "selected_candidates": [pool_by_uid[key] for key in ("N", "C", "A")],
        "selector_ordered_indices": [3, 2, 0],
        "selected_indices": [3, 2, 0],
        "params": {"max_steps": 3, "token_budget": None},
    }
    build_row = {
        "event_id": "event-1",
        "candidates": [pool_by_uid[key] for key in ("N", "C")],
        "prompt_evidence_selected_count_before_prompt_truncation": 2,
        "evidence_count": 1,
        "evidence_count_before": 2,
        "was_truncated": True,
        "evidence_text_truncated": False,
        "selector_trace": {"source_path": str(trace_path)},
    }

    audit = evaluate_baces_reference._audit_one(
        feature_row=feature_row,
        trace_row=trace_row,
        build_row=build_row,
        build_supplied=True,
        policy_name="fixture",
        trace_path=trace_path,
        k_max_override=None,
    )

    assert audit["status"] == "ok"
    assert audit["exact_keys"] == ["A", "B"]
    assert audit["full_keys"] == ["N", "C", "A"]
    assert audit["pre_keys"] == ["N", "C"]
    assert audit["final_keys"] == ["N"]
    assert (
        audit["U_ideal"],
        audit["U_pool"],
        audit["U_opt"],
        audit["U_full"],
        audit["U_pre"],
        audit["U_final"],
    ) == (4, 4, 4, 3, 2, 0)
    assert (
        audit["loss_pool"],
        audit["loss_budget"],
        audit["loss_selector"],
        audit["loss_stop"],
        audit["loss_realization"],
    ) == (0, 0, 1, 1, 2)
    assert audit["loss_total"] == 4
    assert audit["decomposition_sum"] == 4
    assert audit["conservation_ok"] is True
    assert audit["losses_nonnegative"] is True
    assert audit["selector_order_regret"] == 3
    assert audit["pre_order_regret"] == 2
    assert audit["final_order_regret"] == 0
    assert audit["K_core"] == 2
    assert audit["K_full"] == 3
    assert audit["K_pre"] == 2
    assert audit["K_final"] == 1
    assert audit["final_identity_status"] == "tail_prefix_exact"


def test_summary_excludes_alignment_failures_and_text_truncation() -> None:
    valid = {
        "status": "ok",
        "alignment_status": "ok",
        "m": 1,
        "U_ideal": 2,
        "conservation_scope": "verifier_visible",
        "conservation_ok": True,
        "losses_nonnegative": True,
        "prompt_tail_truncated": False,
        "build_evidence_text_truncated": False,
    }
    invalid = {
        **valid,
        "status": "alignment_error",
        "alignment_status": "error",
        "U_ideal": 999,
    }
    text_truncated = {
        **valid,
        "status": "partial",
        "alignment_status": "warning",
        "U_ideal": 777,
        "build_evidence_text_truncated": True,
    }

    summary = evaluate_baces_reference._summarize_group(
        [valid, invalid, text_truncated]
    )

    assert summary["n"] == 3
    assert summary["valid_metric_n"] == 1
    assert summary["numeric"]["U_ideal"]["n"] == 1
    assert summary["numeric"]["U_ideal"]["mean"] == 2.0


def test_unbounded_fullpool_order_uses_verifier_facing_build_cap() -> None:
    full_keys = [f"C{index}" for index in range(20)]

    unbounded = evaluate_baces_reference._resolve_k_max(
        override=None,
        params={"max_steps": 0},
        feature_row={"candidate_top_n": 20},
        build_row={"prompt_evidence_max_count": 10},
        full_keys=full_keys,
    )
    jointly_bounded = evaluate_baces_reference._resolve_k_max(
        override=None,
        params={"max_steps": 12},
        feature_row={"candidate_top_n": 20},
        build_row={"prompt_evidence_max_count": 10},
        full_keys=full_keys,
    )

    assert unbounded == (10, "build_prompt_max_count")
    assert jointly_bounded == (10, "trace_and_build_min")


def test_token_budget_uses_tightest_trace_build_constraint_and_prefix() -> None:
    budget = evaluate_baces_reference._resolve_token_budget(
        params={"token_budget": 8},
        build_row={"prompt_evidence_token_budget": 5},
        event_id="event-1",
    )
    problem = BacesProblem(
        candidates=(
            BacesCandidate("A", (2,), 3),
            BacesCandidate("B", (1,), 3),
            BacesCandidate("C", (0,), 1),
        ),
        weights=(1,),
        k_max=3,
        token_budget=5,
    )

    assert budget == (5, "trace_and_build_min")
    assert evaluate_baces_reference._token_feasible_prefix(
        ("A", "B", "C"), problem=problem, token_budget=5
    ) == ["A"]


def test_missing_build_event_and_internal_trace_order_drift_are_errors(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    feature_row = {
        "event_id": "event-1",
        "claim_atoms": [{"atom_id": "A1"}],
        "candidates": [
            _candidate("A", 2, "A1"),
            _candidate("B", 1, "A1"),
        ],
    }
    trace_pool = [
        {**candidate, "mrec_token_cost": 1}
        for candidate in feature_row["candidates"]
    ]
    base_trace = {
        "event_id": "event-1",
        "candidate_pool": trace_pool,
        "selected_candidates": [trace_pool[0]],
        "selector_ordered_indices": [0],
        "selected_indices": [0],
        "params": {"max_steps": 1},
    }

    missing_build = evaluate_baces_reference._audit_one(
        feature_row=feature_row,
        trace_row=base_trace,
        build_row=None,
        build_supplied=True,
        policy_name="fixture",
        trace_path=trace_path,
        k_max_override=None,
    )
    drifted_trace = {
        **base_trace,
        "selector_ordered_indices": [1],
        "selected_indices": [1],
    }
    drift = evaluate_baces_reference._audit_one(
        feature_row=feature_row,
        trace_row=drifted_trace,
        build_row=None,
        build_supplied=False,
        policy_name="fixture",
        trace_path=trace_path,
        k_max_override=None,
    )

    assert missing_build["status"] == "alignment_error"
    assert "build_event_missing" in missing_build["alignment_errors"]
    assert drift["status"] == "alignment_error"
    assert (
        "selected_candidates_differs_from_selector_ordered_indices"
        in drift["alignment_errors"]
    )


def test_build_truncation_count_invariants_are_checked(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    candidate = _candidate("A", 2, "A1")
    trace_candidate = {**candidate, "mrec_token_cost": 1}
    feature_row = {
        "event_id": "event-1",
        "claim_atoms": [{"atom_id": "A1"}],
        "candidates": [candidate],
    }
    trace_row = {
        "event_id": "event-1",
        "candidate_pool": [trace_candidate],
        "selected_candidates": [trace_candidate],
        "selector_ordered_indices": [0],
        "selected_indices": [0],
        "params": {"max_steps": 1},
    }
    build_row = {
        "event_id": "event-1",
        "candidates": [trace_candidate],
        "prompt_evidence_selected_count_before_prompt_truncation": 1,
        "evidence_count": 1,
        "evidence_count_before": 2,
        "was_truncated": True,
        "evidence_text_truncated": False,
        "selector_trace": {"source_path": str(trace_path)},
    }

    audit = evaluate_baces_reference._audit_one(
        feature_row=feature_row,
        trace_row=trace_row,
        build_row=build_row,
        build_supplied=True,
        policy_name="fixture",
        trace_path=trace_path,
        k_max_override=None,
    )

    assert audit["status"] == "alignment_error"
    assert (
        "build_evidence_count_before_differs_from_candidates_length"
        in audit["alignment_errors"]
    )
    assert "build_was_truncated_inconsistent_with_counts" in audit["alignment_errors"]


def _candidate(uid: str, quality: int, *atom_ids: str) -> dict:
    directness = "direct" if quality == 2 else "partial"
    alignments = [
        {
            "evidence_id": f"E-{uid}",
            "atom_id": atom_id,
            "relation": "support",
            "directness": directness,
            "confidence": 1.0,
            "key_spans": [f"span-{uid}-{atom_id}"],
        }
        for atom_id in atom_ids
    ]
    return {
        "candidate_uid": uid,
        "candidate_key": f"candidate-{uid}",
        "evidence_id": f"E-{uid}",
        "num_tokens": 1,
        "candidate_atom_alignments": alignments if quality > 0 else None,
    }
