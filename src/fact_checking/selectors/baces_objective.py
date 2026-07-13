"""Pure objective and artifact-projection utilities for BACES v0.3.

This module deliberately contains no solver, file I/O, prompt construction, or
verifier logic.  It defines the frozen ordinal-coverage objective and provides
one canonical way to replay either a positive-gain coverage core or a rendered
display sequence against that objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_COVERAGE_RELATIONS = frozenset({"support", "refute", "qualify", "mixed"})
CANONICAL_DIRECTNESS_LEVELS = frozenset({"direct", "partial"})

BacesObjective = tuple[int, int, int, int, tuple[str, ...]]


@dataclass(frozen=True)
class BacesCandidate:
    """A canonical solver-visible evidence candidate.

    ``key`` is the stable identity used for sequence decisions and the final
    lexicographic tie-break.  ``uid`` and ``display_key`` are trace conveniences
    only; they never affect the objective.
    """

    key: str
    q: tuple[int, ...]
    cost: int = 0
    uid: str | None = None
    display_key: str | None = None


@dataclass(frozen=True)
class BacesProblem:
    """A frozen BACES ordinal-coverage instance."""

    candidates: tuple[BacesCandidate, ...]
    weights: tuple[int, ...]
    k_max: int
    token_budget: int | None = None
    atom_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BacesStep:
    """One exact replay step, using one-based sequence positions."""

    position: int
    key: str
    state_before: tuple[int, ...]
    state_after: tuple[int, ...]
    delta: int
    cumulative_utility: int
    candidate_cost: int
    cumulative_cost: int
    acquisition_time_so_far: int


@dataclass(frozen=True)
class BacesEvaluation:
    """Exact objective values for a feasible candidate sequence."""

    keys: tuple[str, ...]
    state: tuple[int, ...]
    utility: int
    acquisition_time: int
    length: int
    token_cost: int
    objective: BacesObjective
    steps: tuple[BacesStep, ...]

    @property
    def terminal_state(self) -> tuple[int, ...]:
        return self.state

    @property
    def terminal_utility(self) -> int:
        return self.utility

    @property
    def total_cost(self) -> int:
        return self.token_cost

    @property
    def T(self) -> int:
        return self.acquisition_time


def validate_problem(problem: BacesProblem) -> BacesProblem:
    """Validate the finite, integer BACES v0.3 input contract.

    The function returns ``problem`` unchanged so it can be used at API
    boundaries without an otherwise redundant second statement.
    """

    if not isinstance(problem, BacesProblem):
        raise TypeError(f"problem must be BacesProblem, got {type(problem).__name__}")
    if not isinstance(problem.candidates, tuple):
        raise TypeError("problem.candidates must be a tuple")
    if not isinstance(problem.weights, tuple):
        raise TypeError("problem.weights must be a tuple")
    if not 1 <= len(problem.weights) <= 6:
        raise ValueError("BACES v0.3 requires between one and six atom weights")
    for idx, weight in enumerate(problem.weights):
        _require_integer(weight, name=f"weights[{idx}]", minimum=1)

    _require_integer(problem.k_max, name="k_max", minimum=0)
    if problem.token_budget is not None:
        _require_integer(problem.token_budget, name="token_budget", minimum=0)

    if problem.atom_ids:
        if not isinstance(problem.atom_ids, tuple):
            raise TypeError("problem.atom_ids must be a tuple")
        if len(problem.atom_ids) != len(problem.weights):
            raise ValueError("atom_ids and weights must have the same length")
        if any(not isinstance(atom_id, str) or not atom_id.strip() for atom_id in problem.atom_ids):
            raise ValueError("atom_ids must contain non-empty strings")
        if len(set(problem.atom_ids)) != len(problem.atom_ids):
            raise ValueError("atom_ids must be unique")

    seen_keys: set[str] = set()
    for idx, candidate in enumerate(problem.candidates):
        if not isinstance(candidate, BacesCandidate):
            raise TypeError(f"candidates[{idx}] must be BacesCandidate")
        if not isinstance(candidate.key, str) or not candidate.key.strip():
            raise ValueError(f"candidates[{idx}].key must be a non-empty string")
        if candidate.key in seen_keys:
            raise ValueError(f"duplicate candidate key: {candidate.key!r}")
        seen_keys.add(candidate.key)
        if not isinstance(candidate.q, tuple):
            raise TypeError(f"candidate {candidate.key!r} q must be a tuple")
        if len(candidate.q) != len(problem.weights):
            raise ValueError(
                f"candidate {candidate.key!r} has q length {len(candidate.q)}, "
                f"expected {len(problem.weights)}"
            )
        _validate_ordinal_vector(candidate.q, name=f"candidate {candidate.key!r} q")
        _require_integer(candidate.cost, name=f"candidate {candidate.key!r} cost", minimum=0)
        if candidate.uid is not None and not isinstance(candidate.uid, str):
            raise TypeError(f"candidate {candidate.key!r} uid must be str or None")
        if candidate.display_key is not None and not isinstance(candidate.display_key, str):
            raise TypeError(f"candidate {candidate.key!r} display_key must be str or None")
    return problem


def advance(state: Sequence[int], q: Sequence[int] | BacesCandidate) -> tuple[int, ...]:
    """Apply the componentwise-max ordinal state transition."""

    state_tuple = tuple(state)
    q_tuple = q.q if isinstance(q, BacesCandidate) else tuple(q)
    if len(state_tuple) != len(q_tuple):
        raise ValueError("state and candidate quality vector must have the same length")
    _validate_ordinal_vector(state_tuple, name="state")
    _validate_ordinal_vector(q_tuple, name="q")
    return tuple(max(before, quality) for before, quality in zip(state_tuple, q_tuple))


def utility(state: Sequence[int], weights: Sequence[int]) -> int:
    """Return the unnormalized integer ordinal coverage utility ``U(x)``."""

    state_tuple = tuple(state)
    weight_tuple = tuple(weights)
    if len(state_tuple) != len(weight_tuple):
        raise ValueError("state and weights must have the same length")
    _validate_ordinal_vector(state_tuple, name="state")
    for idx, weight in enumerate(weight_tuple):
        _require_integer(weight, name=f"weights[{idx}]", minimum=1)
    return sum(weight * level for weight, level in zip(weight_tuple, state_tuple))


def evaluate_core(
    problem: BacesProblem,
    sequence: Iterable[str | BacesCandidate],
) -> BacesEvaluation:
    """Replay a feasible positive-gain coverage-core sequence exactly.

    Every selected candidate must be distinct and every step must have strictly
    positive ordinal utility gain.  Count and optional token budgets are checked
    against the supplied problem.
    """

    return _evaluate(problem, sequence, require_positive_gain=True)


def evaluate_display(
    problem: BacesProblem,
    sequence: Iterable[str | BacesCandidate],
) -> BacesEvaluation:
    """Replay a feasible rendered sequence, permitting zero-gain fill steps."""

    return _evaluate(problem, sequence, require_positive_gain=False)


def padded_auc(evaluation: BacesEvaluation, horizon: int) -> int:
    """Return fixed-horizon padded prefix AUC via ``(H + 1) U - T``."""

    if not isinstance(evaluation, BacesEvaluation):
        raise TypeError("evaluation must be BacesEvaluation")
    _require_integer(horizon, name="horizon", minimum=0)
    if horizon < evaluation.length:
        raise ValueError(
            f"horizon {horizon} is shorter than sequence length {evaluation.length}"
        )
    return (horizon + 1) * evaluation.utility - evaluation.acquisition_time


def quality_from_alignment(alignment: Mapping[str, Any]) -> int:
    """Project one pair-level alignment row to canonical quality 0, 1, or 2.

    Relation aliases are normalized once at this artifact-adapter boundary.
    Confidence is a finite-positive validity gate only; its magnitude never
    enters quality or a tie-break.
    """

    if not isinstance(alignment, Mapping):
        raise TypeError("alignment must be a mapping")
    relation = _canonical_relation(alignment.get("relation"))
    directness = _canonical_token(alignment.get("directness"))
    confidence = _finite_float_or_none(alignment.get("confidence"))
    if (
        relation not in CANONICAL_COVERAGE_RELATIONS
        or directness not in CANONICAL_DIRECTNESS_LEVELS
        or confidence is None
        or confidence <= 0.0
        or not _has_key_span(alignment.get("key_spans"))
    ):
        return 0
    return 2 if directness == "direct" else 1


def compile_feature_problem(
    row: Mapping[str, Any],
    *,
    k_max: int,
    token_budget: int | None = None,
    weights: Sequence[int] | Mapping[str, int] | None = None,
    cost_overrides: Mapping[str, int] | None = None,
) -> BacesProblem:
    """Compile one current evidence-map feature row into a BACES problem.

    The main configuration intentionally defaults to unit atom weights instead
    of reading potentially fractional ``claim_atoms[*].importance`` values.
    Callers may pass already-quantized positive integer weights explicitly.

    Stable candidate keys are selected in the order ``candidate_uid``,
    ``candidate_key``, then ``evidence_id``.  This avoids array-index identity;
    the chosen keys remain visible through ``problem.candidates``.
    """

    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    atoms = _claim_atoms(row)
    atom_ids = _atom_ids(atoms)
    weight_tuple = _compile_weights(weights, atom_ids)

    raw_candidates = row.get("candidates")
    if raw_candidates is None:
        raw_candidates = []
    if not isinstance(raw_candidates, (list, tuple)):
        raise TypeError("row['candidates'] must be a list or tuple")
    if cost_overrides is not None and not isinstance(cost_overrides, Mapping):
        raise TypeError("cost_overrides must be a mapping or None")

    candidates: list[BacesCandidate] = []
    atom_index = {atom_id: idx for idx, atom_id in enumerate(atom_ids)}
    for raw_idx, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise TypeError(f"candidates[{raw_idx}] must be a mapping")
        key = _candidate_stable_key(raw_candidate, index=raw_idx)
        uid = _optional_compact(raw_candidate.get("candidate_uid"))
        display_key = (
            _optional_compact(raw_candidate.get("candidate_key"))
            or _optional_compact(raw_candidate.get("evidence_id"))
            or key
        )
        cost = _candidate_cost(raw_candidate, key=key, overrides=cost_overrides)

        levels = [0] * len(atom_ids)
        alignments = raw_candidate.get("candidate_atom_alignments")
        if alignments is None:
            alignments = []
        if not isinstance(alignments, (list, tuple)):
            raise TypeError(
                f"candidate {key!r} candidate_atom_alignments must be a list, tuple, or null"
            )
        candidate_evidence_id = _optional_compact(raw_candidate.get("evidence_id"))
        for alignment_idx, alignment in enumerate(alignments):
            if not isinstance(alignment, Mapping):
                raise TypeError(
                    f"candidate {key!r} alignment {alignment_idx} must be a mapping"
                )
            row_evidence_id = _optional_compact(alignment.get("evidence_id"))
            if row_evidence_id and candidate_evidence_id and row_evidence_id != candidate_evidence_id:
                continue
            atom_id = _optional_compact(alignment.get("atom_id"))
            if not atom_id:
                raise ValueError(f"candidate {key!r} alignment {alignment_idx} has no atom_id")
            if atom_id not in atom_index:
                raise ValueError(
                    f"candidate {key!r} alignment {alignment_idx} references unknown atom {atom_id!r}"
                )
            atom_pos = atom_index[atom_id]
            levels[atom_pos] = max(levels[atom_pos], quality_from_alignment(alignment))

        candidates.append(
            BacesCandidate(
                key=key,
                q=tuple(levels),
                cost=cost,
                uid=uid,
                display_key=display_key,
            )
        )

    problem = BacesProblem(
        candidates=tuple(candidates),
        weights=weight_tuple,
        k_max=k_max,
        token_budget=token_budget,
        atom_ids=atom_ids,
    )
    return validate_problem(problem)


def _evaluate(
    problem: BacesProblem,
    sequence: Iterable[str | BacesCandidate],
    *,
    require_positive_gain: bool,
) -> BacesEvaluation:
    validate_problem(problem)
    try:
        requested = tuple(sequence)
    except TypeError as exc:
        raise TypeError("sequence must be an iterable of candidate keys or candidates") from exc

    if len(requested) > problem.k_max:
        raise ValueError(
            f"sequence length {len(requested)} exceeds k_max={problem.k_max}"
        )
    by_key = {candidate.key: candidate for candidate in problem.candidates}
    resolved: list[BacesCandidate] = []
    for position, item in enumerate(requested, start=1):
        key = item.key if isinstance(item, BacesCandidate) else item
        if not isinstance(key, str):
            raise TypeError(f"sequence item {position} must be a key or BacesCandidate")
        if key not in by_key:
            raise KeyError(f"unknown candidate key at position {position}: {key!r}")
        resolved.append(by_key[key])

    keys = tuple(candidate.key for candidate in resolved)
    if len(set(keys)) != len(keys):
        raise ValueError("sequence candidate keys must be distinct")

    state = (0,) * len(problem.weights)
    current_utility = 0
    total_cost = 0
    acquisition_time = 0
    steps: list[BacesStep] = []
    for position, candidate in enumerate(resolved, start=1):
        state_before = state
        state_after = advance(state_before, candidate.q)
        next_utility = utility(state_after, problem.weights)
        delta = next_utility - current_utility
        if require_positive_gain and delta <= 0:
            raise ValueError(
                f"coverage-core candidate {candidate.key!r} at position {position} "
                "has zero ordinal utility gain"
            )
        total_cost += candidate.cost
        if problem.token_budget is not None and total_cost > problem.token_budget:
            raise ValueError(
                f"sequence token cost {total_cost} exceeds token_budget={problem.token_budget}"
            )
        acquisition_time += position * delta
        steps.append(
            BacesStep(
                position=position,
                key=candidate.key,
                state_before=state_before,
                state_after=state_after,
                delta=delta,
                cumulative_utility=next_utility,
                candidate_cost=candidate.cost,
                cumulative_cost=total_cost,
                acquisition_time_so_far=acquisition_time,
            )
        )
        state = state_after
        current_utility = next_utility

    objective: BacesObjective = (
        -current_utility,
        acquisition_time,
        len(keys),
        total_cost,
        keys,
    )
    return BacesEvaluation(
        keys=keys,
        state=state,
        utility=current_utility,
        acquisition_time=acquisition_time,
        length=len(keys),
        token_cost=total_cost,
        objective=objective,
        steps=tuple(steps),
    )


def _claim_atoms(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    evidence_map = row.get("evidence_map")
    nested_atoms = evidence_map.get("claim_atoms") if isinstance(evidence_map, Mapping) else None
    raw_atoms = nested_atoms or row.get("claim_atoms") or []
    if not isinstance(raw_atoms, (list, tuple)):
        raise TypeError("claim_atoms must be a list or tuple")
    atoms: list[Mapping[str, Any]] = []
    for idx, atom in enumerate(raw_atoms):
        if not isinstance(atom, Mapping):
            raise TypeError(f"claim_atoms[{idx}] must be a mapping")
        atoms.append(atom)
    if not atoms:
        raise ValueError("BACES requires at least one claim atom")
    if len(atoms) > 6:
        raise ValueError("BACES v0.3 supports at most six claim atoms")
    return tuple(atoms)


def _atom_ids(atoms: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    atom_ids: list[str] = []
    for idx, atom in enumerate(atoms, start=1):
        atom_id = _optional_compact(atom.get("atom_id") or atom.get("node_id")) or f"A{idx}"
        atom_ids.append(atom_id)
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("claim atom IDs must be unique")
    return tuple(atom_ids)


def _compile_weights(
    weights: Sequence[int] | Mapping[str, int] | None,
    atom_ids: tuple[str, ...],
) -> tuple[int, ...]:
    if weights is None:
        return (1,) * len(atom_ids)
    if isinstance(weights, Mapping):
        missing = [atom_id for atom_id in atom_ids if atom_id not in weights]
        extra = [key for key in weights if key not in set(atom_ids)]
        if missing or extra:
            raise ValueError(f"weight keys must exactly match atom_ids; missing={missing}, extra={extra}")
        values = tuple(weights[atom_id] for atom_id in atom_ids)
    else:
        if isinstance(weights, (str, bytes)):
            raise TypeError("weights must be an integer sequence or atom-id mapping")
        values = tuple(weights)
        if len(values) != len(atom_ids):
            raise ValueError("weights and claim_atoms must have the same length")
    for idx, value in enumerate(values):
        _require_integer(value, name=f"weights[{idx}]", minimum=1)
    return values


def _candidate_stable_key(candidate: Mapping[str, Any], *, index: int) -> str:
    for field in ("candidate_uid", "candidate_key", "evidence_id"):
        value = _optional_compact(candidate.get(field))
        if value:
            return value
    raise ValueError(
        f"candidates[{index}] has no stable identity; expected candidate_uid, "
        "candidate_key, or evidence_id"
    )


def _candidate_cost(
    candidate: Mapping[str, Any],
    *,
    key: str,
    overrides: Mapping[str, int] | None,
) -> int:
    identities = [
        key,
        _optional_compact(candidate.get("candidate_uid")),
        _optional_compact(candidate.get("candidate_key")),
        _optional_compact(candidate.get("evidence_id")),
    ]
    if overrides is not None:
        for identity in identities:
            if identity is not None and identity in overrides:
                return _coerce_nonnegative_integer(
                    overrides[identity], name=f"cost override for {identity!r}"
                )
    for field in ("mrec_token_cost", "token_cost", "num_tokens"):
        if candidate.get(field) is not None:
            return _coerce_nonnegative_integer(
                candidate.get(field), name=f"candidate {key!r} {field}"
            )
    raise ValueError(
        f"candidate {key!r} has no token cost; expected mrec_token_cost, "
        "token_cost, num_tokens, or cost_overrides"
    )


def _canonical_relation(value: Any) -> str:
    token = _canonical_token(value)
    aliases = {
        "support": "support",
        "supports": "support",
        "supported": "support",
        "supported_by": "support",
        "entails": "support",
        "consistent": "support",
        "refute": "refute",
        "refutes": "refute",
        "refuted": "refute",
        "contradict": "refute",
        "contradicts": "refute",
        "contradiction": "refute",
        "counter": "refute",
        "conflict": "refute",
        "qualify": "qualify",
        "qualifies": "qualify",
        "qualified": "qualify",
        "condition": "qualify",
        "hedge": "qualify",
        "partially_supports": "support",
        "partially_refutes": "refute",
        "mixed": "mixed",
    }
    return aliases.get(token, token)


def _canonical_token(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _has_key_span(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_key_span(item) for item in value)
    return False


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_compact(value: Any) -> str | None:
    text = " ".join(str(value or "").strip().split())
    return text or None


def _validate_ordinal_vector(vector: Sequence[int], *, name: str) -> None:
    for idx, level in enumerate(vector):
        _require_integer(level, name=f"{name}[{idx}]", minimum=0)
        if level > 2:
            raise ValueError(f"{name}[{idx}] must be one of 0, 1, or 2")


def _require_integer(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _coerce_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdigit():
            raise TypeError(f"{name} must be a non-negative integer")
        result = int(text)
    else:
        raise TypeError(f"{name} must be a non-negative integer")
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


__all__ = [
    "BacesCandidate",
    "BacesEvaluation",
    "BacesObjective",
    "BacesProblem",
    "BacesStep",
    "CANONICAL_COVERAGE_RELATIONS",
    "CANONICAL_DIRECTNESS_LEVELS",
    "advance",
    "compile_feature_problem",
    "evaluate_core",
    "evaluate_display",
    "padded_auc",
    "quality_from_alignment",
    "utility",
    "validate_problem",
]
