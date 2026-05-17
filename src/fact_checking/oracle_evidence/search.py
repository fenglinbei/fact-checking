from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable

from fact_checking.data.constants import LABEL2ID, LETTER2LABEL

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of searching for the optimal evidence set for one claim."""

    event_id: str
    claim: str
    gold_label: str = ""
    gold_id: int = -1
    n_candidates: int = 0
    top_k: int = 0
    selected_indices: list[int] = field(default_factory=list)
    selected_texts: list[str] = field(default_factory=list)
    final_logprob: float = 0.0
    final_prediction: int = -1
    is_correct: bool = False
    search_method: str = ""
    search_steps: list[dict] = field(default_factory=list)
    candidate_pool_fingerprint: str = ""
    candidate_pool: list[dict] = field(default_factory=list)
    candidate_scores: list[dict] = field(default_factory=list)
    candidate_pool_metadata: dict = field(default_factory=dict)


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
) -> SearchResult:
    """Greedy forward selection: at each step, pick the candidate that
    maximizes verifier log-prob of the correct label.

    Complexity: O(N * K) verifier calls.
    """
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="greedy",
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

        logprobs = _batched_call(
            lambda batch_slice: scorer.score_evidence_sets(
                claims=[batch_claims[i] for i in batch_slice],
                current_sets=[batch_current[i] for i in batch_slice],
                candidate_texts=[batch_candidates[i] for i in batch_slice],
                gold_label_letters=[batch_gold[i] for i in batch_slice],
            ),
            total=m,
            batch_size=score_batch_size,
        )

        best_local = int(logprobs.argmax())
        best_global_idx = remaining_indices[best_local]
        best_logprob = float(logprobs[best_local])

        selected_indices.append(best_global_idx)
        selected_texts.append(batch_candidates[best_local])
        remaining_indices.pop(best_local)

        step_record = {
            "step": step,
            "selected_idx": best_global_idx,
            "selected_text": selected_texts[-1][:200],
            "logprob": best_logprob,
            "n_evaluated": m,
        }
        if record_step_scores:
            step_record["candidate_logprobs"] = [
                {"candidate_idx": int(idx), "logprob": float(logprob)}
                for idx, logprob in zip(remaining_indices, logprobs)
            ]
        search_steps.append(step_record)

    # Final evaluation: get the verifier's actual prediction
    prompt = scorer._build_prompt(claim, selected_texts)
    pred_id = scorer.predict_labels([prompt])[0]

    gold_label = LETTER2LABEL.get(gold_label_letter, "")
    gold_id = LABEL2ID.get(gold_label, -1)

    return SearchResult(
        event_id="",
        claim=claim,
        gold_label=gold_label,
        gold_id=gold_id,
        n_candidates=n,
        top_k=effective_k,
        selected_indices=selected_indices,
        selected_texts=selected_texts,
        final_logprob=search_steps[-1]["logprob"] if search_steps else 0.0,
        final_prediction=pred_id,
        is_correct=(pred_id == gold_id),
        search_method="greedy",
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
) -> SearchResult:
    """Enumerate all C(N, K) subsets and pick the best.

    Only use for small N (≤ 15) — C(15,5) = 3003 combinations.
    """
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="exhaustive",
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

    logprobs = _batched_call(
        lambda batch_slice: scorer.score_complete_sets(
            claims=[all_claims[i] for i in batch_slice],
            evidence_sets=[evidence_sets[i] for i in batch_slice],
            gold_label_letters=[all_gold[i] for i in batch_slice],
        ),
        total=n_combos,
        batch_size=score_batch_size,
    )

    best_combo_idx = int(logprobs.argmax())
    best_combo = combos[best_combo_idx]
    best_logprob = float(logprobs[best_combo_idx])

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

    # Final prediction
    prompt = scorer._build_prompt(claim, selected_texts)
    pred_id = scorer.predict_labels([prompt])[0]

    gold_label = LETTER2LABEL.get(gold_label_letter, "")
    gold_id = LABEL2ID.get(gold_label, -1)

    step_record = {
        "step": 0,
        "n_combinations": n_combos,
        "best_logprob": best_logprob,
    }
    if record_step_scores:
        step_record["combination_logprobs"] = [
            {
                "indices": [int(i) for i in combo],
                "logprob": float(logprob),
            }
            for combo, logprob in zip(combos, logprobs)
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
        final_logprob=best_logprob,
        final_prediction=pred_id,
        is_correct=(pred_id == gold_id),
        search_method="exhaustive",
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
) -> SearchResult:
    """Beam search: keep top-B partial sets at each step.

    Complexity: O(B * N * K) verifier calls.
    """
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return SearchResult(
            claim=claim,
            n_candidates=n,
            top_k=top_k,
            search_method="beam",
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

    logprobs = _batched_call(
        lambda batch_slice: scorer.score_evidence_sets(
            claims=[batch_claims[i] for i in batch_slice],
            current_sets=[batch_current[i] for i in batch_slice],
            candidate_texts=[candidates_text[i] for i in batch_slice],
            gold_label_letters=[batch_gold[i] for i in batch_slice],
        ),
        total=m,
        batch_size=score_batch_size,
    )

    top_indices = logprobs.argsort()[-beam_width:][::-1]
    beam: list[BeamEntry] = []
    for idx in top_indices:
        i = int(idx)
        beam.append((
            (i,),
            [candidates_text[i]],
            float(logprobs[i]),
        ))

    step0_record = {
        "step": 0,
        "beam_size": len(beam),
        "n_evaluated": m,
    }
    if record_step_scores:
        step0_record["candidate_logprobs"] = [
            {"candidate_idx": int(i), "logprob": float(logprob)}
            for i, logprob in enumerate(logprobs)
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

            rem_logprobs = _batched_call(
                lambda batch_slice: scorer.score_evidence_sets(
                    claims=[batch_claims[i] for i in batch_slice],
                    current_sets=[batch_current[i] for i in batch_slice],
                    candidate_texts=[batch_rem_texts[i] for i in batch_slice],
                    gold_label_letters=[batch_gold[i] for i in batch_slice],
                ),
                total=m_rem,
                batch_size=score_batch_size,
            )

            for j, logprob in enumerate(rem_logprobs):
                new_idx = remaining[j]
                expansions.append((
                    indices_tuple + (new_idx,),
                    texts_list + [candidates_text[new_idx]],
                    float(logprob),
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
                    "logprob": float(logprob),
                }
                for indices_tuple, _texts, logprob in beam
            ]
        search_steps.append(step_record)

    # Best beam entry
    best = beam[0]
    selected_indices = list(best[0])
    selected_texts = best[1]
    best_logprob = best[2]

    prompt = scorer._build_prompt(claim, selected_texts)
    pred_id = scorer.predict_labels([prompt])[0]

    gold_label = LETTER2LABEL.get(gold_label_letter, "")
    gold_id = LABEL2ID.get(gold_label, -1)

    return SearchResult(
        event_id="",
        claim=claim,
        gold_label=gold_label,
        gold_id=gold_id,
        n_candidates=n,
        top_k=effective_k,
        selected_indices=selected_indices,
        selected_texts=selected_texts,
        final_logprob=best_logprob,
        final_prediction=pred_id,
        is_correct=(pred_id == gold_id),
        search_method="beam",
        search_steps=search_steps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
