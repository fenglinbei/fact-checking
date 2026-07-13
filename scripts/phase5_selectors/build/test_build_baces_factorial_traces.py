from __future__ import annotations

from argparse import Namespace
import copy
import json
from pathlib import Path

from scripts.phase5_selectors.build.build_baces_factorial_traces import (
    CONTROLLER_LEVELS,
    SELECTOR_LEVELS,
    _exact_order_for_controller,
    build_event_factorial_rows,
    main,
)
from fact_checking.selectors.baces_exact import solve_exact
from fact_checking.selectors.baces_objective import BacesCandidate, BacesProblem


def test_builder_writes_all_eighteen_ready_cells(tmp_path: Path) -> None:
    feature, learned, reference = _artifacts()
    features_path = _write_jsonl(tmp_path / "features.jsonl", [feature])
    learned_path = _write_jsonl(tmp_path / "learned.jsonl", [learned])
    reference_path = _write_jsonl(tmp_path / "reference.jsonl", [reference])
    output_dir = tmp_path / "factorial"

    assert main(
        Namespace(
            features=str(features_path),
            learned_trace=str(learned_path),
            reference_build=str(reference_path),
            split="val",
            output_dir=str(output_dir),
            sample_limit=0,
        )
    ) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cell_count"] == 18
    assert manifest["all_ready"] is True
    assert len(manifest["cells"]) == len(SELECTOR_LEVELS) * len(CONTROLLER_LEVELS)
    assert manifest["verifier_build_contract"]["prompt_evidence_policy"] == "selected_set"
    assert manifest["verifier_build_contract"]["trace_prompt_style"] == "mrec_min"
    assert all(cell["ready"] for cell in manifest["cells"])
    for cell in manifest["cells"]:
        rows = _read_jsonl(output_dir / cell["trace_file"])
        assert len(rows) == 1
        row = rows[0]
        assert row["selected_indices"] == row["selector_ordered_indices"]
        assert row["selected_indices"] == row["display_ordered_indices"]
        assert len(row["selected_candidates"]) == len(row["selected_indices"])
        assert len(row["candidate_pool"]) == 10
        assert len(row["baces_display_steps"]) == len(row["selected_indices"])
        assert len(row["mrec_steps"]) == len(row["selected_indices"])
        by_candidate_idx = {step["candidate_idx"]: step for step in row["mrec_steps"]}
        assert [by_candidate_idx[candidate["candidate_idx"]] for candidate in row["selected_candidates"]] == row["mrec_steps"]
        assert all(step["selector_candidate_idx"] == step["candidate_idx"] for step in row["mrec_steps"])
        assert all(step["cue_source"] == "claim_atom" and step["cue_text"] for step in row["mrec_steps"])
        assert all("operation" not in step and "state_after" not in step for step in row["mrec_steps"])


def test_minmax_replays_each_new_order_and_ignores_stored_target_poison() -> None:
    feature, learned, reference = _artifacts()
    clean = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )
    poisoned = copy.deepcopy(learned)
    poisoned["target_resolved"] = True
    poisoned["mrec_diagnostics"] = {"target_resolved": True, "resolved_atom_rate": 999}
    poisoned["mrec_steps"] = [
        {
            "selector_candidate_idx": idx,
            "target_resolved": not bool(idx % 2),
            "trace_state": {"target_resolved": True, "atom_states_after": {"A1": "R", "A2": "R"}},
        }
        for idx in range(10)
    ]
    poisoned_rows = build_event_factorial_rows(
        feature_row=feature,
        learned_row=poisoned,
        reference_row=reference,
    )

    assert clean == poisoned_rows
    retrieval = clean[("retrieval_source", "ordinal_replay_minmax5_10")]
    # Retrieval order places the only direct A2 evidence (u5) sixth.  Replaying
    # the newly generated order therefore cannot stop at the poisoned step 1.
    assert retrieval["selected_candidate_uids"] == [f"u{idx}" for idx in range(6)]
    assert retrieval["factorial_metadata"]["controller_stop_reason"] == "ordinal_target_reached"
    assert retrieval["baces_display_steps"][-1]["state_after"] == [2, 2]
    assert retrieval["mrec_steps"][-1]["target_resolved"] is True
    assert retrieval["mrec_steps"][-1]["resolved_atom_rate"] == 1.0
    structural = clean[("ordinal_coverage_greedy", "ordinal_replay_minmax5_10")]
    assert len(structural["selected_indices"]) == 5
    assert structural["baces_display_steps"][-1]["state_after"] == [2, 2]


def test_matched_token_cap_is_per_event_prefix_and_exact_is_budget_feasible() -> None:
    feature, learned, reference = _artifacts()
    rows = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )

    for selector in SELECTOR_LEVELS:
        row = rows[(selector, "matched_token_cap")]
        assert row["selected_token_cost"] <= 7
        assert len(row["selected_indices"]) <= 10
        assert row["factorial_metadata"]["matched_token_cap"] == 7
    learned_cell = rows[("learned_marginal", "matched_token_cap")]
    assert learned_cell["selected_candidate_uids"] == ["u5"]
    assert learned_cell["selected_token_cost"] == 4
    exact_cell = rows[("baces_exact", "matched_token_cap")]
    assert exact_cell["baces_exact_core"]["token_cost"] <= 7
    assert exact_cell["baces_exact_core"]["terminal_utility"] >= 2


def test_mrec_min_steps_align_to_selected_candidates_by_candidate_idx() -> None:
    feature, learned, reference = _artifacts()
    rows = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )
    row = rows[("learned_marginal", "fixed5")]
    steps_by_idx = {step["candidate_idx"]: step for step in row["mrec_steps"]}
    aligned = [steps_by_idx[candidate["candidate_idx"]] for candidate in row["selected_candidates"]]

    assert aligned == row["mrec_steps"]
    assert [step["candidate_uid"] for step in aligned] == row["selected_candidate_uids"]
    assert all(step["selector_candidate_idx"] == step["candidate_idx"] for step in aligned)
    assert all(step["cue_source"] == "claim_atom" for step in aligned)
    assert all("operation" not in step and "state_before" not in step for step in aligned)


def test_partial_historical_learned_order_is_not_fabricated_into_a_full_rank() -> None:
    feature, learned, reference = _artifacts()
    # The historical selector may leave duplicate/no-transition candidates
    # unranked even in its nominal full-pool artifact.  Keep that boundary
    # explicit instead of inventing learned preferences for the tail.
    learned["selector_ordered_indices"] = [0, 1, 2, 3, 4]
    learned["selected_indices"] = [0, 1, 2, 3, 4]

    rows = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )
    learned_minmax = rows[("learned_marginal", "ordinal_replay_minmax5_10")]

    assert learned_minmax["selected_indices"] == [0, 1, 2, 3, 4]
    assert learned_minmax["selector_order_is_complete"] is False
    assert learned_minmax["selector_full_ordered_indices"] is None
    assert learned_minmax["selector_available_ordered_indices"] == [0, 1, 2, 3, 4]
    assert learned_minmax["selector_unranked_indices"] == [5, 6, 7, 8, 9]
    assert learned_minmax["factorial_metadata"]["controller_stop_reason"] == (
        "selector_order_exhausted"
    )
    assert learned_minmax["factorial_metadata"]["selector_unranked_count"] == 5


def test_exact_factor_uses_frozen_direct_partial_retrieval_fill_key() -> None:
    problem = BacesProblem(
        candidates=(
            BacesCandidate("core", (2, 2), 1),
            BacesCandidate("zero", (0, 0), 1),
            BacesCandidate("partial", (1, 0), 1),
            BacesCandidate("direct", (2, 0), 1),
        ),
        weights=(1, 1),
        k_max=4,
        token_budget=None,
        atom_ids=("A1", "A2"),
    )
    pool = [
        {"candidate_uid": candidate.key, "mrec_token_cost": 1, "retrieval_score": 0.1}
        for candidate in problem.candidates
    ]
    target = solve_exact(problem)

    order, exact = _exact_order_for_controller(
        problem=problem,
        pool=pool,
        controller="ordinal_replay_minmax5_10",
        matched_cap=99,
        target_eval=target,
    )

    assert exact.keys == ("core",)
    assert [pool[idx]["candidate_uid"] for idx in order] == [
        "core",
        "direct",
        "partial",
        "zero",
    ]


def test_factorial_rows_and_serialized_outputs_are_deterministic(tmp_path: Path) -> None:
    feature, learned, reference = _artifacts()
    first = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )
    second = build_event_factorial_rows(
        feature_row=copy.deepcopy(feature),
        learned_row=copy.deepcopy(learned),
        reference_row=copy.deepcopy(reference),
    )
    assert first == second

    features_path = _write_jsonl(tmp_path / "features.jsonl", [feature])
    learned_path = _write_jsonl(tmp_path / "learned.jsonl", [learned])
    reference_path = _write_jsonl(tmp_path / "reference.jsonl", [reference])
    roots = [tmp_path / "run1", tmp_path / "run2"]
    for output_dir in roots:
        main(
            Namespace(
                features=str(features_path),
                learned_trace=str(learned_path),
                reference_build=str(reference_path),
                split="test",
                output_dir=str(output_dir),
                sample_limit=0,
            )
        )
    relative_files = sorted(path.relative_to(roots[0]) for path in roots[0].rglob("*") if path.is_file())
    assert relative_files
    for relative in relative_files:
        assert (roots[0] / relative).read_bytes() == (roots[1] / relative).read_bytes()


def _artifacts() -> tuple[dict, dict, dict]:
    atoms = [
        {"atom_id": "A1", "text": "first atom", "importance": 0.1},
        {"atom_id": "A2", "text": "second atom", "importance": 99},
    ]
    candidates = []
    for idx in range(10):
        alignments = []
        if idx == 0:
            alignments.append(_alignment("E00", "A1", directness="direct"))
        elif idx == 5:
            alignments.append(_alignment("E05", "A2", directness="direct"))
        elif idx == 7:
            alignments.extend(
                [
                    _alignment("E07", "A1", directness="partial"),
                    _alignment("E07", "A2", directness="partial"),
                ]
            )
        candidates.append(
            {
                "candidate_uid": f"u{idx}",
                "candidate_key": f"key-{idx}",
                "evidence_id": f"E{idx:02d}",
                "text": f"Evidence number {idx}",
                "hybrid_score": 1.0 - idx / 20.0,
                "evidence_map_base_score": 1.0 / (idx + 1),
                "evidence_map_quality_score": 0.9 if idx in {0, 5, 7} else 0.0,
                "union_pool_rank": idx + 1,
                "candidate_atom_alignments": alignments,
            }
        )
    feature = {
        "event_id": "event-1",
        "claim": "a two atom claim",
        "gold_label": "true",
        "claim_atoms": atoms,
        "evidence_map": {"claim_atoms": atoms, "source": "frozen-map"},
        "candidates": candidates,
        # Poison fields belong to neither the frozen pool nor the objective.
        "oracle_ordered_keys": ["u9", "u8"],
    }
    costs = [4, 4, 1, 2, 2, 4, 1, 3, 2, 1]
    learned_pool = []
    for candidate, cost in zip(candidates, costs):
        copied = dict(candidate)
        copied["mrec_token_cost"] = cost
        learned_pool.append(copied)
    learned_order = [5, 0, 7, 1, 2, 3, 4, 6, 8, 9]
    learned = {
        "event_id": "event-1",
        "candidate_pool": learned_pool,
        "selector_ordered_indices": learned_order,
        "selected_indices": learned_order,
        "mrec_steps": [
            {"selector_candidate_idx": idx, "trace_state": {"target_resolved": False}}
            for idx in learned_order
        ],
    }
    reference = {
        "event_id": "event-1",
        "prompt_evidence_selected_token_cost": 7,
    }
    return feature, learned, reference


def _alignment(evidence_id: str, atom_id: str, *, directness: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "atom_id": atom_id,
        "relation": "support",
        "directness": directness,
        "confidence": 0.8,
        "key_spans": ["span"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
