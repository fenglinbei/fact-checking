from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fact_checking.data.constants import (
    label2id_for_schema,
    letter2label_for_schema,
    letter_order_for_schema,
    normalize_label_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of searching for the optimal evidence set for one claim."""

    event_id: str = ""
    claim: str = ""
    gold_label: str = ""
    gold_id: int = -1
    n_candidates: int = 0
    top_k: int = 0
    selected_indices: list[int] = field(default_factory=list)
    selected_texts: list[str] = field(default_factory=list)
    final_logprob: float = 0.0
    final_objective: float = 0.0
    gold_logprob: float = 0.0
    best_wrong_logprob: float = 0.0
    margin: float = 0.0
    label_logprobs: dict[str, float] = field(default_factory=dict)
    final_prediction: int = -1
    prediction_source: str = ""
    is_correct: bool = False
    search_method: str = ""
    search_objective: str = "gold_logprob"
    search_steps: list[dict] = field(default_factory=list)
    candidate_pool_fingerprint: str = ""
    candidate_pool: list[dict] = field(default_factory=list)
    candidate_scores: list[dict] = field(default_factory=list)
    candidate_pool_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _LabelContext:
    schema: str
    label2id: dict[str, int]
    letter2label: dict[str, str]
    letter_order: list[str]


def _label_context(label_schema: str | None = None) -> _LabelContext:
    schema = normalize_label_schema(label_schema)
    return _LabelContext(
        schema=schema,
        label2id=label2id_for_schema(schema),
        letter2label=letter2label_for_schema(schema),
        letter_order=letter_order_for_schema(schema),
    )


# ---------------------------------------------------------------------------
# Greedy forward selection
# ---------------------------------------------------------------------------


def greedy_search(
    *,
    claim: str,
    candidates: list[dict],
    top_k: int,
    gold_label_letter: str,
    scorer,
    score_batch_size: int = 512,
    record_step_scores: bool = False,
    objective: str = "gold_logprob",
    label_schema: str | None = None,
) -> SearchResult:
    """Greedy forward selection: at each step, pick the candidate that
    maximizes verifier log-prob of the correct label.

    Complexity: O(N * K) verifier calls.
    """
    label_ctx = _label_context(label_schema)
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="greedy",
            search_objective=objective,
        )

    effective_k = min(top_k, n)
    remaining_indices = list(range(n))
    selected_indices: list[int] = []
    selected_texts: list[str] = []
    search_steps: list[dict] = []

    for step in range(effective_k):
        m = len(remaining_indices)
        # Evaluate each remaining candidate when added to current set
        batch_claims = [claim] * m
        batch_current = [selected_texts] * m
        batch_candidates = [
            str(candidates[idx].get("text", "")).strip()
            for idx in remaining_indices
        ]
        batch_gold = [gold_label_letter] * m

        evaluated_indices = list(remaining_indices)
        score_records = _score_incremental_candidates(
            scorer=scorer,
            claims=batch_claims,
            current_sets=batch_current,
            candidate_texts=batch_candidates,
            gold_label_letters=batch_gold,
            objective=objective,
            score_batch_size=score_batch_size,
            label_schema=label_ctx.schema,
        )

        objective_scores = np.asarray([r["objective_score"] for r in score_records], dtype=np.float32)
        best_local = int(objective_scores.argmax())
        best_global_idx = remaining_indices[best_local]
        best_record = score_records[best_local]

        selected_indices.append(best_global_idx)
        selected_texts.append(batch_candidates[best_local])
        remaining_indices.pop(best_local)

        step_record = {
            "step": step,
            "selected_idx": best_global_idx,
            "selected_text": selected_texts[-1][:200],
            "objective_score": float(best_record["objective_score"]),
            "gold_logprob": float(best_record["gold_logprob"]),
            "best_wrong_logprob": float(best_record["best_wrong_logprob"]),
            "margin": float(best_record["margin"]),
            "logprob": float(best_record["gold_logprob"]),
            "n_evaluated": m,
        }
        if record_step_scores:
            step_record["candidate_scores"] = [
                _candidate_score_record(idx, record)
                for idx, record in zip(evaluated_indices, score_records)
            ]
            step_record["candidate_logprobs"] = [
                {"candidate_idx": int(idx), "logprob": float(record["gold_logprob"])}
                for idx, record in zip(evaluated_indices, score_records)
            ]
        search_steps.append(step_record)

    gold_label = label_ctx.letter2label.get(gold_label_letter, "")
    gold_id = label_ctx.label2id.get(gold_label, -1)
    final_record = _score_final_set(
        scorer=scorer,
        claim=claim,
        selected_texts=selected_texts,
        gold_label_letter=gold_label_letter,
        objective=objective,
        label_schema=label_ctx.schema,
    )
    pred_id = int(final_record["pred_id"])

    return SearchResult(
        event_id="",
        claim=claim,
        gold_label=gold_label,
        gold_id=gold_id,
        n_candidates=n,
        top_k=effective_k,
        selected_indices=selected_indices,
        selected_texts=selected_texts,
        final_logprob=float(final_record["gold_logprob"]),
        final_objective=float(final_record["objective_score"]),
        gold_logprob=float(final_record["gold_logprob"]),
        best_wrong_logprob=float(final_record["best_wrong_logprob"]),
        margin=float(final_record["margin"]),
        label_logprobs=dict(final_record["label_logprobs"]),
        final_prediction=pred_id,
        prediction_source="label_logprob_argmax",
        is_correct=(pred_id == gold_id),
        search_method="greedy",
        search_objective=objective,
        search_steps=search_steps,
    )


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------


def exhaustive_search(
    *,
    claim: str,
    candidates: list[dict],
    top_k: int,
    gold_label_letter: str,
    scorer,
    score_batch_size: int = 512,
    record_step_scores: bool = False,
    objective: str = "gold_logprob",
    label_schema: str | None = None,
) -> SearchResult:
    """Enumerate all C(N, K) subsets and pick the best.

    Only use for small N (≤ 15) — C(15,5) = 3003 combinations.
    """
    label_ctx = _label_context(label_schema)
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="exhaustive",
            search_objective=objective,
        )

    effective_k = min(top_k, n)
    indices = list(range(n))
    combos = list(itertools.combinations(indices, effective_k))

    # Build evidence sets for all combinations
    evidence_sets: list[list[str]] = []
    for combo in combos:
        texts = [str(candidates[i].get("text", "")).strip() for i in combo]
        evidence_sets.append(texts)

    # Batch-score all combinations
    n_combos = len(combos)
    all_claims = [claim] * n_combos
    all_gold = [gold_label_letter] * n_combos

    score_records = _score_complete_sets(
        scorer=scorer,
        claims=all_claims,
        evidence_sets=evidence_sets,
        gold_label_letters=all_gold,
        objective=objective,
        score_batch_size=score_batch_size,
        label_schema=label_ctx.schema,
    )

    objective_scores = np.asarray([r["objective_score"] for r in score_records], dtype=np.float32)
    best_combo_idx = int(objective_scores.argmax())
    best_combo = combos[best_combo_idx]
    best_record = score_records[best_combo_idx]

    selected_indices = list(best_combo)
    selected_texts = [
        str(candidates[i].get("text", "")).strip() for i in selected_indices
    ]

    # Sort by hybrid_score descending for consistent presentation
    sorted_pairs = sorted(
        zip(selected_indices, selected_texts),
        key=lambda p: float(candidates[p[0]].get("hybrid_score", 0.0)),
        reverse=True,
    )
    selected_indices = [p[0] for p in sorted_pairs]
    selected_texts = [p[1] for p in sorted_pairs]

    gold_label = label_ctx.letter2label.get(gold_label_letter, "")
    gold_id = label_ctx.label2id.get(gold_label, -1)
    final_record = _score_final_set(
        scorer=scorer,
        claim=claim,
        selected_texts=selected_texts,
        gold_label_letter=gold_label_letter,
        objective=objective,
        label_schema=label_ctx.schema,
    )
    pred_id = int(final_record["pred_id"])

    step_record = {
        "step": 0,
        "n_combinations": n_combos,
        "best_objective_score": float(best_record["objective_score"]),
        "best_logprob": float(best_record["gold_logprob"]),
        "best_wrong_logprob": float(best_record["best_wrong_logprob"]),
        "best_margin": float(best_record["margin"]),
    }
    if record_step_scores:
        step_record["combination_scores"] = [
            {
                "indices": [int(i) for i in combo],
                **_score_record_payload(record),
            }
            for combo, record in zip(combos, score_records)
        ]
        step_record["combination_logprobs"] = [
            {
                "indices": [int(i) for i in combo],
                "logprob": float(record["gold_logprob"]),
            }
            for combo, record in zip(combos, score_records)
        ]

    return SearchResult(
        event_id="",
        claim=claim,
        gold_label=gold_label,
        gold_id=gold_id,
        n_candidates=n,
        top_k=effective_k,
        selected_indices=selected_indices,
        selected_texts=selected_texts,
        final_logprob=float(final_record["gold_logprob"]),
        final_objective=float(final_record["objective_score"]),
        gold_logprob=float(final_record["gold_logprob"]),
        best_wrong_logprob=float(final_record["best_wrong_logprob"]),
        margin=float(final_record["margin"]),
        label_logprobs=dict(final_record["label_logprobs"]),
        final_prediction=pred_id,
        prediction_source="label_logprob_argmax",
        is_correct=(pred_id == gold_id),
        search_method="exhaustive",
        search_objective=objective,
        search_steps=[step_record],
    )


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def beam_search(
    *,
    claim: str,
    candidates: list[dict],
    top_k: int,
    gold_label_letter: str,
    scorer,
    beam_width: int = 3,
    score_batch_size: int = 512,
    record_step_scores: bool = False,
    objective: str = "gold_logprob",
    label_schema: str | None = None,
) -> SearchResult:
    """Beam search: keep top-B partial sets at each step.

    Complexity: O(B * N * K) verifier calls.
    """
    label_ctx = _label_context(label_schema)
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="beam",
            search_objective=objective,
        )

    effective_k = min(top_k, n)
    candidates_text = [
        str(c.get("text", "")).strip() for c in candidates
    ]

    # Each beam entry: (indices_tuple, texts_list, logprob)
    BeamEntry = tuple[tuple[int, ...], list[str], float]

    # Step 0: pick single best starting candidate
    m = n
    batch_claims = [claim] * m
    batch_current = [[] for _ in range(m)]
    batch_gold = [gold_label_letter] * m

    score_records = _score_incremental_candidates(
        scorer=scorer,
        claims=batch_claims,
        current_sets=batch_current,
        candidate_texts=candidates_text,
        gold_label_letters=batch_gold,
        objective=objective,
        score_batch_size=score_batch_size,
        label_schema=label_ctx.schema,
    )
    objective_scores = np.asarray([r["objective_score"] for r in score_records], dtype=np.float32)

    top_indices = objective_scores.argsort()[-beam_width:][::-1]
    beam: list[BeamEntry] = []
    for idx in top_indices:
        i = int(idx)
        beam.append((
            (i,),
            [candidates_text[i]],
            float(objective_scores[i]),
        ))

    step0_record = {
        "step": 0,
        "beam_size": len(beam),
        "n_evaluated": m,
    }
    if record_step_scores:
        step0_record["candidate_scores"] = [
            _candidate_score_record(i, record)
            for i, record in enumerate(score_records)
        ]
        step0_record["candidate_logprobs"] = [
            {"candidate_idx": int(i), "logprob": float(record["gold_logprob"])}
            for i, record in enumerate(score_records)
        ]
    search_steps: list[dict] = [step0_record]

    # Steps 1..K-1
    for step in range(1, effective_k):
        expansions: list[tuple[tuple[int, ...], list[str], float]] = []
        for indices_tuple, texts_list, _prev_logprob in beam:
            remaining = [i for i in range(n) if i not in set(indices_tuple)]
            if not remaining:
                continue

            m_rem = len(remaining)
            batch_claims = [claim] * m_rem
            batch_current = [texts_list] * m_rem
            batch_rem_texts = [candidates_text[i] for i in remaining]
            batch_gold = [gold_label_letter] * m_rem

            rem_records = _score_incremental_candidates(
                scorer=scorer,
                claims=batch_claims,
                current_sets=batch_current,
                candidate_texts=batch_rem_texts,
                gold_label_letters=batch_gold,
                objective=objective,
                score_batch_size=score_batch_size,
                label_schema=label_ctx.schema,
            )

            for j, record in enumerate(rem_records):
                new_idx = remaining[j]
                expansions.append((
                    indices_tuple + (new_idx,),
                    texts_list + [candidates_text[new_idx]],
                    float(record["objective_score"]),
                ))

        if not expansions:
            break

        expansions.sort(key=lambda x: x[2], reverse=True)
        beam = expansions[:beam_width]
        step_record = {
            "step": step,
            "beam_size": len(beam),
            "n_expansions": len(expansions),
        }
        if record_step_scores:
            step_record["top_expansions"] = [
                {
                    "indices": [int(i) for i in indices_tuple],
                    "objective_score": float(logprob),
                }
                for indices_tuple, _texts, logprob in beam
            ]
        search_steps.append(step_record)

    # Best beam entry
    best = beam[0]
    selected_indices = list(best[0])
    selected_texts = best[1]
    gold_label = label_ctx.letter2label.get(gold_label_letter, "")
    gold_id = label_ctx.label2id.get(gold_label, -1)
    final_record = _score_final_set(
        scorer=scorer,
        claim=claim,
        selected_texts=selected_texts,
        gold_label_letter=gold_label_letter,
        objective=objective,
        label_schema=label_ctx.schema,
    )
    pred_id = int(final_record["pred_id"])

    return SearchResult(
        event_id="",
        claim=claim,
        gold_label=gold_label,
        gold_id=gold_id,
        n_candidates=n,
        top_k=effective_k,
        selected_indices=selected_indices,
        selected_texts=selected_texts,
        final_logprob=float(final_record["gold_logprob"]),
        final_objective=float(final_record["objective_score"]),
        gold_logprob=float(final_record["gold_logprob"]),
        best_wrong_logprob=float(final_record["best_wrong_logprob"]),
        margin=float(final_record["margin"]),
        label_logprobs=dict(final_record["label_logprobs"]),
        final_prediction=pred_id,
        prediction_source="label_logprob_argmax",
        is_correct=(pred_id == gold_id),
        search_method="beam",
        search_objective=objective,
        search_steps=search_steps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_objective(objective: str) -> str:
    normalized = str(objective).strip().lower()
    if normalized not in {"gold_logprob", "margin"}:
        raise ValueError(f"Unsupported oracle search objective={objective!r}")
    return normalized


def _records_from_label_logprobs(
    label_logprobs: np.ndarray,
    gold_label_letters: list[str],
    *,
    objective: str,
    label_schema: str | None = None,
) -> list[dict]:
    objective = _validate_objective(objective)
    label_ctx = _label_context(label_schema)
    records: list[dict] = []
    for row, gold_letter in zip(label_logprobs, gold_label_letters):
        if len(row) != len(label_ctx.letter_order):
            raise ValueError(
                f"label_logprobs row has {len(row)} column(s), "
                f"but label_schema={label_ctx.schema!r} expects {len(label_ctx.letter_order)}"
            )
        label_scores = {
            letter: float(row[i])
            for i, letter in enumerate(label_ctx.letter_order)
        }
        gold_logprob = float(label_scores[gold_letter])
        wrong_scores = [
            float(score)
            for letter, score in label_scores.items()
            if letter != gold_letter
        ]
        best_wrong_logprob = max(wrong_scores) if wrong_scores else float("-inf")
        margin = gold_logprob - best_wrong_logprob
        pred_letter = max(label_scores, key=label_scores.get)
        pred_label = label_ctx.letter2label.get(pred_letter, "")
        pred_id = label_ctx.label2id.get(pred_label, -1)
        objective_score = margin if objective == "margin" else gold_logprob
        records.append(
            {
                "objective_score": float(objective_score),
                "gold_logprob": gold_logprob,
                "best_wrong_logprob": float(best_wrong_logprob),
                "margin": float(margin),
                "label_logprobs": label_scores,
                "pred_letter": pred_letter,
                "pred_id": int(pred_id),
            }
        )
    return records


def _records_from_gold_logprobs(
    gold_logprobs: np.ndarray,
    gold_label_letters: list[str],
) -> list[dict]:
    records: list[dict] = []
    for logprob, gold_letter in zip(gold_logprobs, gold_label_letters):
        records.append(
            {
                "objective_score": float(logprob),
                "gold_logprob": float(logprob),
                "best_wrong_logprob": 0.0,
                "margin": 0.0,
                "label_logprobs": {gold_letter: float(logprob)},
                "pred_letter": "",
                "pred_id": -1,
            }
        )
    return records


def _score_incremental_candidates(
    *,
    scorer,
    claims: list[str],
    current_sets: list[list[str]],
    candidate_texts: list[str],
    gold_label_letters: list[str],
    objective: str,
    score_batch_size: int,
    label_schema: str | None = None,
) -> list[dict]:
    objective = _validate_objective(objective)
    total = len(claims)
    if objective == "margin":
        label_logprobs = _batched_call(
            lambda batch_slice: scorer.score_evidence_sets_all_labels(
                claims=[claims[i] for i in batch_slice],
                current_sets=[current_sets[i] for i in batch_slice],
                candidate_texts=[candidate_texts[i] for i in batch_slice],
            ),
            total=total,
            batch_size=score_batch_size,
        )
        return _records_from_label_logprobs(
            label_logprobs,
            gold_label_letters,
            objective=objective,
            label_schema=label_schema,
        )

    gold_logprobs = _batched_call(
        lambda batch_slice: scorer.score_evidence_sets(
            claims=[claims[i] for i in batch_slice],
            current_sets=[current_sets[i] for i in batch_slice],
            candidate_texts=[candidate_texts[i] for i in batch_slice],
            gold_label_letters=[gold_label_letters[i] for i in batch_slice],
        ),
        total=total,
        batch_size=score_batch_size,
    )
    return _records_from_gold_logprobs(gold_logprobs, gold_label_letters)


def _score_complete_sets(
    *,
    scorer,
    claims: list[str],
    evidence_sets: list[list[str]],
    gold_label_letters: list[str],
    objective: str,
    score_batch_size: int,
    label_schema: str | None = None,
) -> list[dict]:
    objective = _validate_objective(objective)
    total = len(claims)
    if objective == "margin":
        label_logprobs = _batched_call(
            lambda batch_slice: scorer.score_complete_sets_all_labels(
                claims=[claims[i] for i in batch_slice],
                evidence_sets=[evidence_sets[i] for i in batch_slice],
            ),
            total=total,
            batch_size=score_batch_size,
        )
        return _records_from_label_logprobs(
            label_logprobs,
            gold_label_letters,
            objective=objective,
            label_schema=label_schema,
        )

    gold_logprobs = _batched_call(
        lambda batch_slice: scorer.score_complete_sets(
            claims=[claims[i] for i in batch_slice],
            evidence_sets=[evidence_sets[i] for i in batch_slice],
            gold_label_letters=[gold_label_letters[i] for i in batch_slice],
        ),
        total=total,
        batch_size=score_batch_size,
    )
    return _records_from_gold_logprobs(gold_logprobs, gold_label_letters)


def _score_final_set(
    *,
    scorer,
    claim: str,
    selected_texts: list[str],
    gold_label_letter: str,
    objective: str,
    label_schema: str | None = None,
) -> dict:
    objective = _validate_objective(objective)
    label_logprobs = scorer.score_complete_sets_all_labels(
        claims=[claim],
        evidence_sets=[selected_texts],
    )
    return _records_from_label_logprobs(
        label_logprobs,
        [gold_label_letter],
        objective=objective,
        label_schema=label_schema,
    )[0]


def _score_record_payload(record: dict) -> dict:
    return {
        "objective_score": float(record["objective_score"]),
        "gold_logprob": float(record["gold_logprob"]),
        "best_wrong_logprob": float(record["best_wrong_logprob"]),
        "margin": float(record["margin"]),
        "pred_letter": str(record.get("pred_letter", "")),
        "label_logprobs": {
            str(letter): float(value)
            for letter, value in dict(record.get("label_logprobs", {})).items()
        },
    }


def _candidate_score_record(candidate_idx: int, record: dict) -> dict:
    return {
        "candidate_idx": int(candidate_idx),
        **_score_record_payload(record),
    }


def _batched_call(
    fn: Callable[[list[int]], "np.ndarray"],
    total: int,
    batch_size: int,
) -> "np.ndarray":
    """Call *fn* on contiguous index slices, concatenating results."""
    import numpy as np

    if total == 0:
        return np.array([], dtype=np.float32)

    results: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        results.append(fn(list(range(start, end))))
    return np.concatenate(results)
