from __future__ import annotations

from itertools import permutations

import pytest

from fact_checking.selectors.baces_exact import (
    solve_exact,
    solve_fixed_set_order,
)
from fact_checking.selectors.baces_objective import (
    BacesCandidate,
    BacesProblem,
    compile_feature_problem,
    evaluate_core,
    evaluate_display,
    padded_auc,
    quality_from_alignment,
)


def test_ordinal_transition_uses_componentwise_max_and_allows_upgrade() -> None:
    problem = _problem(
        _candidate("partial-a", (1,)),
        _candidate("partial-b", (1,)),
        _candidate("direct", (2,)),
        k_max=3,
    )

    upgraded = evaluate_core(problem, ("partial-a", "direct"))

    assert upgraded.state == (2,)
    assert upgraded.utility == 2
    assert upgraded.acquisition_time == 3
    assert [step.delta for step in upgraded.steps] == [1, 1]
    assert [step.state_after for step in upgraded.steps] == [(1,), (2,)]

    # A second partial observation cannot be accumulated into direct coverage.
    with pytest.raises(ValueError, match="positive|gain|zero"):
        evaluate_core(problem, ("partial-a", "partial-b"))

    displayed = evaluate_display(problem, ("partial-a", "partial-b"))
    assert displayed.state == (1,)
    assert displayed.utility == 1
    assert displayed.acquisition_time == 1
    assert [step.delta for step in displayed.steps] == [1, 0]


def test_one_candidate_updates_all_aligned_atoms_synchronously() -> None:
    problem = _problem(
        _candidate("multi", (1, 2)),
        _candidate("upgrade-first", (2, 0)),
        weights=(2, 1),
        k_max=2,
    )

    evaluation = evaluate_core(problem, ("multi", "upgrade-first"))

    assert evaluation.state == (2, 2)
    assert evaluation.utility == 6
    assert evaluation.acquisition_time == 8
    assert [step.delta for step in evaluation.steps] == [4, 2]


def test_exact_solver_obeys_all_five_lexicographic_levels() -> None:
    # 1. Terminal utility dominates every later objective component.
    coverage = solve_exact(
        _problem(
            _candidate("a-partial-cheap", (1,), cost=0),
            _candidate("z-direct-expensive", (2,), cost=100),
            k_max=1,
        )
    )
    assert coverage.keys == ("z-direct-expensive",)

    # 2. With equal terminal utility, earlier acquisition wins.  The first
    # ordering has deltas 3,1 (T=5), while the reverse has 2,2 (T=6).
    acquisition = solve_exact(
        _problem(
            _candidate("a-early", (2, 1)),
            _candidate("b-late", (0, 2)),
            k_max=2,
        )
    )
    assert acquisition.keys == ("a-early", "b-late")
    assert acquisition.acquisition_time == 5

    # 3. A redundant display item never lengthens the positive-gain core.
    length = solve_exact(
        _problem(
            _candidate("a-direct", (2,)),
            _candidate("z-fill", (0,)),
            k_max=2,
        )
    )
    assert length.keys == ("a-direct",)
    assert length.length == 1

    # 4. Equal U/T/L is broken by total token cost.
    cost = solve_exact(
        _problem(
            _candidate("a-expensive", (2,), cost=4),
            _candidate("z-cheap", (2,), cost=1),
            k_max=1,
        )
    )
    assert cost.keys == ("z-cheap",)

    # 5. Equal U/T/L/cost is broken by the stable-key sequence.
    stable_key = solve_exact(
        _problem(
            _candidate("z", (2,), cost=1),
            _candidate("a", (2,), cost=1),
            k_max=1,
        )
    )
    assert stable_key.keys == ("a",)


def test_exact_solver_respects_count_token_and_joint_budgets() -> None:
    count_only = solve_exact(
        _problem(
            _candidate("A", (2, 0), cost=2),
            _candidate("B", (0, 2), cost=2),
            k_max=1,
        )
    )
    assert count_only.utility == 2
    assert count_only.length == 1
    assert count_only.keys == ("A",)

    token_only = solve_exact(
        _problem(
            _candidate("A", (2, 0), cost=3),
            _candidate("B", (0, 2), cost=3),
            _candidate("C", (1, 1), cost=2),
            k_max=3,
            token_budget=3,
        )
    )
    assert token_only.keys == ("C",)
    assert token_only.utility == 2
    assert token_only.token_cost == 2

    joint = solve_exact(
        _problem(
            _candidate("A", (2, 0), cost=2),
            _candidate("B", (0, 2), cost=2),
            _candidate("C", (1, 1), cost=1),
            k_max=2,
            token_budget=3,
        )
    )
    assert joint.keys == ("A", "C")
    assert joint.state == (2, 1)
    assert joint.utility == 3
    assert joint.token_cost == 3


@pytest.mark.parametrize(
    "problem, expected_state, expected_keys",
    [
        (
            BacesProblem(candidates=(), weights=(1, 1), k_max=3),
            (0, 0),
            (),
        ),
        (
            BacesProblem(
                candidates=(BacesCandidate(key="noise", q=(0,), cost=0),),
                weights=(1,),
                k_max=1,
            ),
            (0,),
            (),
        ),
        (
            BacesProblem(
                candidates=(BacesCandidate(key="excluded-by-count", q=(2,), cost=0),),
                weights=(1,),
                k_max=0,
            ),
            (0,),
            (),
        ),
        (
            BacesProblem(
                candidates=(BacesCandidate(key="excluded-by-budget", q=(2,), cost=1),),
                weights=(1,),
                k_max=1,
                token_budget=0,
            ),
            (0,),
            (),
        ),
        (
            BacesProblem(
                candidates=(BacesCandidate(key="free", q=(2,), cost=0),),
                weights=(1,),
                k_max=1,
                token_budget=0,
            ),
            (2,),
            ("free",),
        ),
    ],
)
def test_empty_zero_gain_and_zero_cost_boundaries(
    problem: BacesProblem,
    expected_state: tuple[int, ...],
    expected_keys: tuple[str, ...],
) -> None:
    result = solve_exact(problem)

    assert result.state == expected_state
    assert result.keys == expected_keys
    assert result.utility == sum(weight * level for weight, level in zip(problem.weights, expected_state))


def test_candidate_input_permutation_does_not_change_canonical_solution() -> None:
    candidates = (
        _candidate("A", (2, 0), cost=2),
        _candidate("B", (0, 2), cost=2),
        _candidate("C", (1, 1), cost=1),
    )

    solutions = {
        solve_exact(BacesProblem(candidates=order, weights=(1, 1), k_max=2)).keys
        for order in permutations(candidates)
    }

    assert solutions == {("A", "B")}


def test_padded_auc_matches_acquisition_time_identity() -> None:
    problem = _problem(
        _candidate("partial", (1, 0)),
        _candidate("direct-both", (2, 2)),
        k_max=4,
    )
    evaluation = evaluate_core(problem, ("partial", "direct-both"))

    # Prefix utilities are 1,4,4,4.
    assert padded_auc(evaluation, horizon=4) == 13
    assert padded_auc(evaluation, horizon=4) == (4 + 1) * evaluation.utility - evaluation.acquisition_time


def test_fixed_set_order_keeps_the_set_and_places_fill_after_optimal_core() -> None:
    problem = _problem(
        _candidate("A", (2, 1)),
        _candidate("B", (0, 2)),
        _candidate("Y-fill", (0, 0)),
        _candidate("Z-fill", (0, 0)),
        k_max=4,
    )
    supplied = ("B", "Z-fill", "A", "Y-fill")

    result = solve_fixed_set_order(problem, supplied)

    assert result.keys == ("A", "B", "Y-fill", "Z-fill")
    assert set(result.keys) == set(supplied)
    assert result.length == len(supplied)
    assert result.utility == 4
    assert result.acquisition_time == 5
    assert [step.delta for step in result.steps] == [3, 1, 0, 0]


def test_fixed_set_order_uses_full_sequence_lex_tie_not_core_cost_or_length() -> None:
    problem = _problem(
        _candidate("A", (1, 0, 0), cost=2),
        _candidate("B", (0, 1, 2), cost=4),
        _candidate("C", (1, 0, 2), cost=3),
        _candidate("D", (2, 1, 0), cost=0),
        weights=(1, 3, 1),
        k_max=4,
    )

    result = solve_fixed_set_order(problem, ("A", "B", "C", "D"))

    assert result.keys == ("B", "D", "A", "C")
    assert result.acquisition_time == 9
    assert result.token_cost == 9


def test_feature_compiler_uses_pair_rows_unit_weights_and_cost_override() -> None:
    row = {
        "claim_atoms": [
            {"atom_id": "A1", "importance": 0.3},
            {"atom_id": "A2", "importance": 0.8},
        ],
        "candidates": [
            {
                "candidate_uid": "uid-1",
                "candidate_key": "readable evidence",
                "evidence_id": "E01",
                "num_tokens": 7,
                "candidate_atom_alignments": [
                    {
                        "evidence_id": "E01",
                        "atom_id": "A1",
                        "relation": "supports",
                        "directness": "partial",
                        "confidence": 0.7,
                        "key_spans": ["partial span"],
                    },
                    {
                        "evidence_id": "E01",
                        "atom_id": "A1",
                        "relation": "support",
                        "directness": "direct",
                        "confidence": 0.8,
                        "key_spans": ["direct span"],
                    },
                    {
                        "evidence_id": "E01",
                        "atom_id": "A2",
                        "relation": "refute",
                        "directness": "partial",
                        "confidence": 0.6,
                        "key_spans": ["partial refutation"],
                    },
                ],
            },
            {
                "candidate_uid": "uid-2",
                "candidate_key": "no map row",
                "evidence_id": "E02",
                "num_tokens": 5,
                "candidate_atom_alignments": None,
            },
        ],
    }

    problem = compile_feature_problem(
        row,
        k_max=2,
        cost_overrides={"uid-1": 11},
    )

    assert problem.atom_ids == ("A1", "A2")
    assert problem.weights == (1, 1)  # Fractional artifact importance is not the main weight.
    assert problem.candidates[0] == BacesCandidate(
        key="uid-1",
        q=(2, 1),
        cost=11,
        uid="uid-1",
        display_key="readable evidence",
    )
    assert problem.candidates[1].q == (0, 0)
    assert problem.candidates[1].cost == 5


def test_alignment_valid_gate_and_aliases_are_frozen() -> None:
    valid = {
        "relation": "partially_supports",
        "directness": "partial",
        "confidence": 0.4,
        "key_spans": ["span"],
    }
    assert quality_from_alignment(valid) == 1
    assert quality_from_alignment({**valid, "directness": "direct"}) == 2
    assert quality_from_alignment({**valid, "confidence": 0}) == 0
    assert quality_from_alignment({**valid, "key_spans": []}) == 0
    assert quality_from_alignment({**valid, "relation": "insufficient"}) == 0


def _candidate(key: str, q: tuple[int, ...], *, cost: int = 0) -> BacesCandidate:
    return BacesCandidate(key=key, q=q, cost=cost)


def _problem(
    *candidates: BacesCandidate,
    weights: tuple[int, ...] | None = None,
    k_max: int,
    token_budget: int | None = None,
) -> BacesProblem:
    if weights is None:
        dimensions = len(candidates[0].q) if candidates else 1
        weights = (1,) * dimensions
    return BacesProblem(
        candidates=tuple(candidates),
        weights=weights,
        k_max=k_max,
        token_budget=token_budget,
    )
