from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from .baces_objective import (
    BacesEvaluation,
    BacesProblem,
    advance,
    evaluate_core,
    evaluate_display,
    utility,
    validate_problem,
)


CellKey = tuple[int, tuple[int, ...], int | None]


@dataclass(frozen=True)
class ExactCell:
    k: int
    state: tuple[int, ...]
    token_cost: int
    acquisition_time: int
    keys: tuple[str, ...]


def _canonical_candidates(problem: BacesProblem):
    return tuple(sorted(problem.candidates, key=lambda candidate: candidate.key))


def _cell_key(problem: BacesProblem, cell: ExactCell) -> CellKey:
    budget_key = cell.token_cost if problem.token_budget is not None else None
    return (cell.k, cell.state, budget_key)


def _cell_rank(problem: BacesProblem, cell: ExactCell) -> tuple:
    if problem.token_budget is not None:
        return (cell.acquisition_time, cell.keys)
    return (cell.acquisition_time, cell.token_cost, cell.keys)


def _relax(
    problem: BacesProblem,
    cells: dict[CellKey, ExactCell],
    candidate: ExactCell,
) -> None:
    key = _cell_key(problem, candidate)
    current = cells.get(key)
    if current is None or _cell_rank(problem, candidate) < _cell_rank(problem, current):
        cells[key] = candidate


def solve_exact_cells(problem: BacesProblem) -> dict[CellKey, ExactCell]:
    """Return every retained ordered-state DP cell.

    The sequence is stored directly in each cell for auditability.  This adds an
    O(r) copying/comparison factor relative to a persistent-backpointer
    implementation, but keeps the bounded-atom reference solver transparent.
    """

    validate_problem(problem)
    candidates = _canonical_candidates(problem)
    atom_count = len(problem.weights)
    max_core_length = min(problem.k_max, 2 * atom_count, len(candidates))
    zero_state = tuple(0 for _ in range(atom_count))
    initial = ExactCell(
        k=0,
        state=zero_state,
        token_cost=0,
        acquisition_time=0,
        keys=(),
    )
    all_cells: dict[CellKey, ExactCell] = {_cell_key(problem, initial): initial}
    current_layer: dict[CellKey, ExactCell] = {_cell_key(problem, initial): initial}

    for k in range(max_core_length):
        next_layer: dict[CellKey, ExactCell] = {}
        for cell in current_layer.values():
            before_utility = utility(cell.state, problem.weights)
            for candidate in candidates:
                state_after = advance(cell.state, candidate.q)
                delta = utility(state_after, problem.weights) - before_utility
                if delta <= 0:
                    continue
                token_cost = cell.token_cost + candidate.cost
                if problem.token_budget is not None and token_cost > problem.token_budget:
                    continue
                proposal = ExactCell(
                    k=k + 1,
                    state=state_after,
                    token_cost=token_cost,
                    acquisition_time=cell.acquisition_time + (k + 1) * delta,
                    keys=cell.keys + (candidate.key,),
                )
                _relax(problem, next_layer, proposal)
        if not next_layer:
            break
        all_cells.update(next_layer)
        current_layer = next_layer

    return all_cells


def _evaluation_for_cell(problem: BacesProblem, cell: ExactCell) -> BacesEvaluation:
    evaluation = evaluate_core(problem, cell.keys)
    if (
        evaluation.state != cell.state
        or evaluation.acquisition_time != cell.acquisition_time
        or evaluation.token_cost != cell.token_cost
        or evaluation.length != cell.k
    ):
        raise AssertionError("DP backpointer replay does not match the retained cell")
    return evaluation


def solve_exact(problem: BacesProblem) -> BacesEvaluation:
    """Solve the frozen BACES lexicographic core objective exactly."""

    cells = solve_exact_cells(problem)
    evaluations = (_evaluation_for_cell(problem, cell) for cell in cells.values())
    return min(evaluations, key=lambda evaluation: evaluation.objective)


def _independent_sequence_metrics(
    problem: BacesProblem,
    keys: Iterable[str],
) -> tuple[tuple[int, ...], int, int, int, tuple[str, ...]] | None:
    """Evaluate a core sequence from the acquisition-time definition.

    This intentionally does not call ``advance``, ``utility`` or the replay
    evaluator.  It is the independent oracle used by exhaustive tests.
    """

    sequence = tuple(keys)
    if len(sequence) != len(set(sequence)) or len(sequence) > problem.k_max:
        return None
    by_key = {candidate.key: candidate for candidate in problem.candidates}
    if any(key not in by_key for key in sequence):
        return None

    state = [0 for _ in problem.weights]
    acquisition_positions: list[list[int | None]] = [[None, None] for _ in problem.weights]
    token_cost = 0
    for position, key in enumerate(sequence, start=1):
        candidate = by_key[key]
        before_utility = sum(weight * value for weight, value in zip(problem.weights, state))
        state_after = [max(value, quality) for value, quality in zip(state, candidate.q)]
        after_utility = sum(weight * value for weight, value in zip(problem.weights, state_after))
        if after_utility <= before_utility:
            return None
        for atom_idx, (before, after) in enumerate(zip(state, state_after)):
            for level in range(before + 1, after + 1):
                acquisition_positions[atom_idx][level - 1] = position
        state = state_after
        token_cost += candidate.cost
        if problem.token_budget is not None and token_cost > problem.token_budget:
            return None

    acquisition_time = 0
    for atom_idx, value in enumerate(state):
        for level_idx in range(value):
            position = acquisition_positions[atom_idx][level_idx]
            if position is None:
                raise AssertionError("acquired ordinal unit is missing its acquisition position")
            acquisition_time += problem.weights[atom_idx] * position
    terminal_utility = sum(weight * value for weight, value in zip(problem.weights, state))
    return tuple(state), terminal_utility, acquisition_time, token_cost, sequence


def solve_bruteforce_cells(
    problem: BacesProblem,
    *,
    max_candidates: int = 8,
) -> dict[CellKey, ExactCell]:
    """Enumerate every feasible positive-gain sequence for tiny test problems."""

    validate_problem(problem)
    candidates = _canonical_candidates(problem)
    if len(candidates) > max_candidates:
        raise ValueError(
            f"brute-force oracle is limited to {max_candidates} candidates; got {len(candidates)}"
        )
    max_core_length = min(problem.k_max, 2 * len(problem.weights), len(candidates))
    cells: dict[CellKey, ExactCell] = {}
    candidate_keys = tuple(candidate.key for candidate in candidates)
    for length in range(max_core_length + 1):
        for sequence in permutations(candidate_keys, length):
            metrics = _independent_sequence_metrics(problem, sequence)
            if metrics is None:
                continue
            state, _terminal_utility, acquisition_time, token_cost, keys = metrics
            cell = ExactCell(
                k=length,
                state=state,
                token_cost=token_cost,
                acquisition_time=acquisition_time,
                keys=keys,
            )
            _relax(problem, cells, cell)
    return cells


def solve_bruteforce(
    problem: BacesProblem,
    *,
    max_candidates: int = 8,
) -> BacesEvaluation:
    """Return the exhaustive optimum for a tiny problem.

    Sequence ranking uses an acquisition-time calculation independent of the
    production evaluator.  The winning sequence is replayed only after the
    independent optimum has been selected, and the two definitions are asserted
    to agree.
    """

    cells = solve_bruteforce_cells(problem, max_candidates=max_candidates)
    ranked: list[tuple[tuple, ExactCell]] = []
    for cell in cells.values():
        terminal_utility = sum(
            weight * value for weight, value in zip(problem.weights, cell.state)
        )
        objective = (
            -terminal_utility,
            cell.acquisition_time,
            cell.k,
            cell.token_cost,
            cell.keys,
        )
        ranked.append((objective, cell))
    _objective, winner = min(ranked, key=lambda item: item[0])
    return _evaluation_for_cell(problem, winner)


def solve_fixed_set_order(
    problem: BacesProblem,
    keys: Iterable[str],
) -> BacesEvaluation:
    """Find the canonical minimum-acquisition-time order for a frozen set.

    Full display length and cost are constant once the set is frozen, so the
    comparison is ``(T, full stable-key sequence)``.  This intentionally does
    not reuse the core solver's shorter-core/lower-core-cost tie-breaks.
    """

    frozen_keys = tuple(keys)
    evaluate_display(problem, frozen_keys)
    frozen_set = set(frozen_keys)
    candidates = tuple(
        sorted(
            (candidate for candidate in problem.candidates if candidate.key in frozen_set),
            key=lambda candidate: candidate.key,
        )
    )
    target_state = tuple(
        max((candidate.q[atom_idx] for candidate in candidates), default=0)
        for atom_idx in range(len(problem.weights))
    )
    zero_state = tuple(0 for _ in problem.weights)
    initial = ExactCell(
        k=0,
        state=zero_state,
        token_cost=0,
        acquisition_time=0,
        keys=(),
    )
    current_layer: dict[tuple[int, ...], ExactCell] = {zero_state: initial}
    terminal_orders: list[BacesEvaluation] = []
    max_core_length = min(2 * len(problem.weights), len(candidates))

    for k in range(max_core_length + 1):
        terminal = current_layer.get(target_state)
        if terminal is not None:
            core_set = set(terminal.keys)
            full_keys = terminal.keys + tuple(sorted(frozen_set - core_set))
            terminal_orders.append(evaluate_display(problem, full_keys))
        if k == max_core_length:
            break

        next_layer: dict[tuple[int, ...], ExactCell] = {}
        for cell in current_layer.values():
            before_utility = utility(cell.state, problem.weights)
            for candidate in candidates:
                state_after = advance(cell.state, candidate.q)
                delta = utility(state_after, problem.weights) - before_utility
                if delta <= 0:
                    continue
                proposal = ExactCell(
                    k=k + 1,
                    state=state_after,
                    token_cost=cell.token_cost + candidate.cost,
                    acquisition_time=cell.acquisition_time + (k + 1) * delta,
                    keys=cell.keys + (candidate.key,),
                )
                current = next_layer.get(state_after)
                proposal_rank = (proposal.acquisition_time, proposal.keys)
                current_rank = (
                    (current.acquisition_time, current.keys) if current is not None else None
                )
                if current_rank is None or proposal_rank < current_rank:
                    next_layer[state_after] = proposal
        if not next_layer:
            break
        current_layer = next_layer

    if not terminal_orders:
        raise AssertionError("frozen candidate set has no terminal coverage order")
    return min(
        terminal_orders,
        key=lambda evaluation: (evaluation.acquisition_time, evaluation.keys),
    )
