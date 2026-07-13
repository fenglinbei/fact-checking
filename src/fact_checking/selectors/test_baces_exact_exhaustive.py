from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random

import pytest

from fact_checking.selectors.baces_exact import (
    solve_bruteforce,
    solve_bruteforce_cells,
    solve_exact,
    solve_exact_cells,
    solve_fixed_set_order,
)
from fact_checking.selectors.baces_objective import (
    BacesCandidate,
    BacesProblem,
    evaluate_display,
)


@dataclass(frozen=True)
class _ReferenceResult:
    keys: tuple[str, ...]
    state: tuple[int, ...]
    utility: int
    acquisition_time: int
    length: int
    token_cost: int

    @property
    def objective(self) -> tuple[object, ...]:
        return (
            -self.utility,
            self.acquisition_time,
            self.length,
            self.token_cost,
            self.keys,
        )


@pytest.mark.parametrize(
    "problem",
    [
        BacesProblem(candidates=(), weights=(1,), k_max=3),
        BacesProblem(
            candidates=(BacesCandidate("zero", (0,), 0),),
            weights=(1,),
            k_max=1,
        ),
        BacesProblem(
            candidates=(
                BacesCandidate("partial-1", (1,), 1),
                BacesCandidate("partial-2", (1,), 1),
                BacesCandidate("direct", (2,), 3),
            ),
            weights=(1,),
            k_max=3,
        ),
        BacesProblem(
            candidates=(
                BacesCandidate("multi", (1, 2), 2),
                BacesCandidate("first", (2, 0), 1),
                BacesCandidate("second", (0, 2), 1),
            ),
            weights=(2, 1),
            k_max=2,
        ),
        BacesProblem(
            candidates=(
                BacesCandidate("A", (2, 0), 2),
                BacesCandidate("B", (0, 2), 2),
                BacesCandidate("C", (1, 1), 1),
            ),
            weights=(1, 1),
            k_max=2,
            token_budget=3,
        ),
        BacesProblem(
            candidates=(BacesCandidate("free", (2,), 0),),
            weights=(3,),
            k_max=1,
            token_budget=0,
        ),
    ],
)
def test_exact_and_library_bruteforce_match_independent_exhaustive_reference(
    problem: BacesProblem,
) -> None:
    expected = _solve_independently(problem)
    exact = solve_exact(problem)
    library_brute = solve_bruteforce(problem)

    assert exact.objective == expected.objective
    assert exact.keys == expected.keys
    assert exact.state == expected.state
    assert library_brute.objective == expected.objective
    assert library_brute.keys == expected.keys
    assert library_brute.state == expected.state
    assert solve_exact_cells(problem) == solve_bruteforce_cells(problem)
    assert _normalized_library_cells(solve_exact_cells(problem)) == _normalized_reference_cells(
        _cells_independently(problem)
    )


def test_exact_matches_independent_exhaustive_reference_on_seeded_random_instances() -> None:
    rng = random.Random(0xBACE5)

    for case_index in range(512):
        atom_count = rng.randint(1, 3)
        candidate_count = rng.randint(0, 6)
        candidates = tuple(
            BacesCandidate(
                key=f"case-{case_index:03d}-candidate-{candidate_index:02d}",
                q=tuple(rng.randint(0, 2) for _ in range(atom_count)),
                cost=rng.randint(0, 5),
            )
            for candidate_index in range(candidate_count)
        )
        problem = BacesProblem(
            candidates=candidates,
            weights=tuple(rng.randint(1, 3) for _ in range(atom_count)),
            k_max=rng.randint(0, 4),
            token_budget=None if rng.random() < 0.5 else rng.randint(0, 10),
        )

        expected = _solve_independently(problem)
        exact = solve_exact(problem)
        library_brute = solve_bruteforce(problem)

        assert exact.objective == expected.objective, f"exact mismatch in random case {case_index}"
        assert exact.keys == expected.keys
        assert exact.state == expected.state
        assert library_brute.objective == expected.objective, f"bruteforce mismatch in random case {case_index}"
        assert solve_exact_cells(problem) == solve_bruteforce_cells(problem), (
            f"retained-cell mismatch in random case {case_index}"
        )
        assert _normalized_library_cells(
            solve_exact_cells(problem)
        ) == _normalized_reference_cells(_cells_independently(problem)), (
            f"independent retained-cell mismatch in random case {case_index}"
        )


def test_exhaustive_reference_rejects_repeated_or_zero_gain_core_steps() -> None:
    problem = BacesProblem(
        candidates=(
            BacesCandidate("partial-a", (1,), 1),
            BacesCandidate("partial-b", (1,), 1),
            BacesCandidate("direct", (2,), 1),
        ),
        weights=(1,),
        k_max=3,
    )

    assert _evaluate_independently(problem, ("partial-a", "partial-b")) is None
    assert _evaluate_independently(problem, ("direct", "partial-a")) is None
    assert _evaluate_independently(problem, ("partial-a", "direct")) is not None


def test_fixed_set_order_matches_all_full_permutations_on_seeded_instances() -> None:
    rng = random.Random(0x51A7E)

    for case_index in range(64):
        atom_count = rng.randint(1, 3)
        candidate_count = rng.randint(0, 6)
        candidates = tuple(
            BacesCandidate(
                key=f"fixed-{case_index:03d}-{candidate_index:02d}",
                q=tuple(rng.randint(0, 2) for _ in range(atom_count)),
                cost=rng.randint(0, 5),
            )
            for candidate_index in range(candidate_count)
        )
        problem = BacesProblem(
            candidates=candidates,
            weights=tuple(rng.randint(1, 3) for _ in range(atom_count)),
            k_max=candidate_count,
            token_budget=sum(candidate.cost for candidate in candidates),
        )
        keys = tuple(candidate.key for candidate in candidates)
        expected = min(
            (evaluate_display(problem, order) for order in permutations(keys)),
            key=lambda evaluation: (evaluation.acquisition_time, evaluation.keys),
        )
        actual = solve_fixed_set_order(problem, reversed(keys))

        assert actual.keys == expected.keys, f"fixed-set key mismatch in case {case_index}"
        assert actual.acquisition_time == expected.acquisition_time


def _solve_independently(problem: BacesProblem) -> _ReferenceResult:
    """Tiny-instance oracle intentionally independent of production replay code."""

    cells = _cells_independently(problem)
    return min(cells.values(), key=lambda result: result.objective)


def _cells_independently(
    problem: BacesProblem,
) -> dict[tuple[int, tuple[int, ...], int | None], _ReferenceResult]:
    """Build the exhaustive retained-cell table without production relax logic."""

    max_length = min(problem.k_max, len(problem.candidates), 2 * len(problem.weights))
    cells: dict[tuple[int, tuple[int, ...], int | None], _ReferenceResult] = {}

    for length in range(max_length + 1):
        for ordered_candidates in permutations(problem.candidates, length):
            result = _evaluate_independently(
                problem,
                tuple(candidate.key for candidate in ordered_candidates),
            )
            if result is None:
                continue
            budget_key = result.token_cost if problem.token_budget is not None else None
            cell_key = (result.length, result.state, budget_key)
            current = cells.get(cell_key)
            if problem.token_budget is not None:
                rank = (result.acquisition_time, result.keys)
                current_rank = (
                    (current.acquisition_time, current.keys) if current is not None else None
                )
            else:
                rank = (result.acquisition_time, result.token_cost, result.keys)
                current_rank = (
                    (current.acquisition_time, current.token_cost, current.keys)
                    if current is not None
                    else None
                )
            if current_rank is None or rank < current_rank:
                cells[cell_key] = result

    assert cells  # The empty sequence is always feasible.
    return cells


def _normalized_library_cells(cells: dict) -> dict:
    return {
        key: (cell.keys, cell.state, cell.acquisition_time, cell.k, cell.token_cost)
        for key, cell in cells.items()
    }


def _normalized_reference_cells(cells: dict) -> dict:
    return {
        key: (
            result.keys,
            result.state,
            result.acquisition_time,
            result.length,
            result.token_cost,
        )
        for key, result in cells.items()
    }


def _evaluate_independently(
    problem: BacesProblem,
    keys: tuple[str, ...],
) -> _ReferenceResult | None:
    """Evaluate from the acquisition-time definition, not the DP recurrence.

    Returning ``None`` means the sequence is not a feasible positive-gain core.
    Computing first-acquisition positions directly keeps this oracle independent
    from the production implementation's ``sum(t * delta_t)`` recurrence.
    """

    if len(keys) > problem.k_max or len(set(keys)) != len(keys):
        return None

    candidate_by_key = {candidate.key: candidate for candidate in problem.candidates}
    if any(key not in candidate_by_key for key in keys):
        return None

    token_cost = sum(candidate_by_key[key].cost for key in keys)
    if problem.token_budget is not None and token_cost > problem.token_budget:
        return None

    state = [0] * len(problem.weights)
    acquisition_positions: list[list[int]] = [[] for _ in problem.weights]

    for position, key in enumerate(keys, start=1):
        candidate = candidate_by_key[key]
        state_before = tuple(state)
        for atom_index, level in enumerate(candidate.q):
            if level > state[atom_index]:
                for _new_unit in range(state[atom_index] + 1, level + 1):
                    acquisition_positions[atom_index].append(position)
                state[atom_index] = level
        if tuple(state) == state_before:
            return None

    utility = sum(weight * level for weight, level in zip(problem.weights, state))
    acquisition_time = sum(
        weight * sum(positions)
        for weight, positions in zip(problem.weights, acquisition_positions)
    )
    return _ReferenceResult(
        keys=keys,
        state=tuple(state),
        utility=utility,
        acquisition_time=acquisition_time,
        length=len(keys),
        token_cost=token_cost,
    )
