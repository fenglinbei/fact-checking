from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fact_checking.retrieval.text_utils import lexical_overlap_f1
from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    selection_quality_metrics,
    summarize_selector_traces,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.evidence_quality import retrieval_score, source_group


DIRECT_CE_TEXT_ONLY_SELECTOR = "direct_ce_text_only_top5"
DIRECT_CE_SOURCE_DIVERSE_SELECTOR = "direct_ce_light_source_diverse_top5"
V03_REFERENCE_SELECTOR = "v0_3_1_pointwise_all_features_top5"

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
DEFAULT_PROMPT_VERSION = "direct_evidence_ce_v0_4a"
DEFAULT_INSTRUCTION = (
    "Given a fact-checking claim, score whether the evidence directly verifies or refutes the claim. "
    "Prefer passages that state the claim's key entities, quantities, dates, comparisons, causes, or outcomes. "
    "Penalize background-only, loosely related, duplicate, or context-only passages."
)

FORBIDDEN_SCORER_INPUT_FIELDS = {
    "oracle_selected",
    "oracle_step",
    "gold_label",
    "event_id",
    "candidate_key",
    "candidate_uid",
    "baseline_rank",
    "qd_pool_rank",
    "union_pool_rank",
    "from_baseline",
    "from_qd",
    "source_pools",
    "source_group",
    "report_id",
    "source_domain",
}


@dataclass(frozen=True)
class DirectEvidencePair:
    query: str
    passage: str
    prompt_version: str = DEFAULT_PROMPT_VERSION
    source_fields: tuple[str, ...] = ("claim", "text")


class DirectEvidenceCrossEncoderScorer:
    """Thin fail-fast wrapper around sentence_transformers.CrossEncoder."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 1024,
        device: str = "auto",
        instruction: str = DEFAULT_INSTRUCTION,
        cross_encoder_cls: Any | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.device = str(device)
        self.instruction = str(instruction)
        if cross_encoder_cls is None:
            try:
                from sentence_transformers import CrossEncoder as cross_encoder_cls  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on target env
                raise RuntimeError(
                    "sentence_transformers.CrossEncoder is required for v0.4a direct evidence scoring. "
                    "Install sentence-transformers on the target server; no transformers fallback is provided."
                ) from exc
        kwargs: dict[str, Any] = {
            "max_length": int(max_length),
            "prompts": {"direct_evidence": str(instruction)},
            "default_prompt_name": "direct_evidence",
        }
        if str(device) != "auto":
            kwargs["device"] = str(device)
        try:
            self.model = cross_encoder_cls(str(model_name), **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {model_name!r} through sentence_transformers.CrossEncoder. "
                "v0.4a intentionally has no fallback backend."
            ) from exc

    def score_pairs(
        self,
        pairs: Sequence[DirectEvidencePair],
        *,
        batch_size: int,
        show_progress_bar: bool = True,
    ) -> list[float]:
        payload = [(pair.query, pair.passage) for pair in pairs]
        try:
            raw_scores = self.model.predict(
                payload,
                batch_size=int(batch_size),
                show_progress_bar=bool(show_progress_bar),
            )
        except Exception as exc:
            raise RuntimeError(
                f"sentence_transformers.CrossEncoder predict failed for {self.model_name!r}. "
                "v0.4a intentionally has no fallback backend."
            ) from exc
        return normalize_model_scores(raw_scores)


def build_text_only_pair(
    event_row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> DirectEvidencePair:
    del instruction  # The instruction is passed through CrossEncoder prompts, not the pair payload.
    claim = str(event_row.get("claim") or "").strip()
    evidence = str(candidate.get("text") or candidate.get("candidate_text") or candidate.get("canonical_text") or "").strip()
    return DirectEvidencePair(query=claim, passage=evidence, prompt_version=str(prompt_version))


def audit_text_only_pair(pair: DirectEvidencePair) -> None:
    fields = set(pair.source_fields)
    forbidden = fields & FORBIDDEN_SCORER_INPUT_FIELDS
    if forbidden:
        raise ValueError(f"Forbidden fields were used to build scorer input: {sorted(forbidden)}")
    if not pair.query:
        raise ValueError("Direct evidence scorer input is missing claim text.")
    if not pair.passage:
        raise ValueError("Direct evidence scorer input is missing evidence text.")


def flatten_scoring_jobs(
    rows: Sequence[dict[str, Any]],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> tuple[list[DirectEvidencePair], list[tuple[str, str, str]]]:
    pairs: list[DirectEvidencePair] = []
    keys: list[tuple[str, str, str]] = []
    for event_row in rows:
        event_id = str(event_row.get("event_id") or "")
        for candidate in event_row.get("candidates") or []:
            pair = build_text_only_pair(event_row, candidate, instruction=instruction, prompt_version=prompt_version)
            audit_text_only_pair(pair)
            pairs.append(pair)
            keys.append(_candidate_key(event_id, candidate))
    return pairs, keys


def attach_direct_ce_scores(
    rows: Sequence[dict[str, Any]],
    scores_by_key: dict[tuple[str, str, str], float],
    *,
    model_name: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    score_source: str = "cross_encoder",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event_row in rows:
        event_id = str(event_row.get("event_id") or "")
        item = dict(event_row)
        candidates: list[dict[str, Any]] = []
        for candidate in event_row.get("candidates") or []:
            c = dict(candidate)
            key = _candidate_key(event_id, c)
            if key not in scores_by_key:
                raise KeyError(f"Missing direct CE score for candidate key={key}")
            c["direct_ce_score"] = float(scores_by_key[key])
            c["direct_ce_model"] = str(model_name)
            c["direct_ce_prompt_version"] = str(prompt_version)
            c["direct_ce_score_source"] = str(score_source)
            candidates.append(c)
        item["candidates"] = candidates
        out.append(item)
    return out


def mock_scores_for_pairs(pairs: Sequence[DirectEvidencePair]) -> list[float]:
    scores: list[float] = []
    for pair in pairs:
        lexical = lexical_overlap_f1(pair.query, pair.passage)
        digest = hashlib.sha1(f"{pair.query}\n{pair.passage}".encode("utf-8")).hexdigest()
        jitter = int(digest[:6], 16) / float(0xFFFFFF) * 0.01
        scores.append(float(min(1.0, max(0.0, lexical + jitter))))
    return scores


def score_rows_with_scorer(
    rows: Sequence[dict[str, Any]],
    scorer: DirectEvidenceCrossEncoderScorer | None,
    *,
    batch_size: int,
    model_name: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    instruction: str = DEFAULT_INSTRUCTION,
    mock_scores: bool = False,
) -> list[dict[str, Any]]:
    pairs, keys = flatten_scoring_jobs(rows, instruction=instruction, prompt_version=prompt_version)
    if mock_scores:
        scores = mock_scores_for_pairs(pairs)
        source = "mock_lexical_overlap"
    else:
        if scorer is None:
            raise ValueError("A DirectEvidenceCrossEncoderScorer is required when mock_scores=False.")
        scores = scorer.score_pairs(pairs, batch_size=int(batch_size))
        source = "cross_encoder"
    scores_by_key = {key: float(score) for key, score in zip(keys, scores)}
    return attach_direct_ce_scores(
        rows,
        scores_by_key,
        model_name=model_name,
        prompt_version=prompt_version,
        score_source=source,
    )


def select_event_shard(
    rows: Sequence[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    n = max(1, int(num_shards))
    idx = int(shard_index)
    if idx < 0 or idx >= n:
        raise ValueError(f"Invalid shard_index={shard_index}; expected 0 <= shard_index < {n}")
    return [dict(row) for offset, row in enumerate(rows) if offset % n == idx]


def merge_scored_event_rows(
    reference_rows: Sequence[dict[str, Any]],
    scored_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_event = {str(row.get("event_id") or ""): dict(row) for row in scored_rows}
    missing = [str(row.get("event_id") or "") for row in reference_rows if str(row.get("event_id") or "") not in by_event]
    if missing:
        raise ValueError(f"Missing scored events during merge: {missing[:10]}")
    return [by_event[str(row.get("event_id") or "")] for row in reference_rows]


def select_direct_ce_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    top_k: int,
    selector_name: str = DIRECT_CE_TEXT_ONLY_SELECTOR,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(key=_direct_ce_sort_key, reverse=True)
    return _ranked_unique(rows, top_k=top_k, selector_name=selector_name, origin="direct_ce_rank")


def select_source_diverse_direct_ce_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    top_k: int,
    source_penalty: float = 0.05,
    selector_name: str = DIRECT_CE_SOURCE_DIVERSE_SELECTOR,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    pool = [dict(candidate) for candidate in candidates]
    while len(selected) < int(top_k):
        best: dict[str, Any] | None = None
        for candidate in pool:
            key = _selection_key(candidate)
            if not key or key in seen:
                continue
            item = dict(candidate)
            same_source = int(source_counts[source_group(item)])
            adjusted = _safe_float(item.get("direct_ce_score"), 0.0) - float(source_penalty) * float(same_source)
            item["direct_ce_adjusted_score"] = float(adjusted)
            item["selection_origin"] = "direct_ce_source_diverse"
            item["same_source_selected_count"] = same_source
            if best is None or _direct_ce_adjusted_sort_key(item) > _direct_ce_adjusted_sort_key(best):
                best = item
        if best is None:
            break
        best["selector_name"] = selector_name
        best["selection_rank"] = len(selected) + 1
        selected.append(best)
        seen.add(_selection_key(best))
        source_counts[source_group(best)] += 1
    return selected


def build_direct_ce_trace(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    selector_name: str,
    top_k: int,
) -> dict[str, Any]:
    selected = list(selected)
    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": str(selector_name),
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_candidates": [_candidate_output(candidate) for candidate in selected],
        "slot_trace": [
            {
                "slot": int(candidate.get("selection_rank") or idx + 1),
                "candidate_uid": str(candidate.get("candidate_uid") or ""),
                "candidate_key": str(candidate.get("candidate_key") or ""),
                "selection_origin": str(candidate.get("selection_origin") or ""),
                "oracle_selected": bool(candidate.get("oracle_selected")),
                "direct_ce_score": _safe_float(candidate.get("direct_ce_score"), 0.0),
                "direct_ce_adjusted_score": _safe_float(candidate.get("direct_ce_adjusted_score"), 0.0),
                "source_group": source_group(candidate),
            }
            for idx, candidate in enumerate(selected)
        ],
    }
    trace.update(text_ordered_selection_metrics(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected))
    trace["mean_direct_ce_score@5"] = _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in selected)
    return trace


def summarize_direct_ce_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_selector_traces(traces)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("selector_name") or "")].append(dict(trace))
    for selector, rows in grouped.items():
        item = summary.setdefault(selector, {"n_claims": len(rows)})
        selected = [
            candidate
            for trace in rows
            for candidate in trace.get("selected_candidates") or []
        ]
        item["mean_direct_ce_score@5"] = _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in selected)
        item["direct_ce_hit_rate@5"] = _mean(1.0 if bool(candidate.get("oracle_selected")) else 0.0 for candidate in selected)
    return summary


def direct_ce_diagnostics(
    scored_rows: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    *,
    primary_selector: str = DIRECT_CE_TEXT_ONLY_SELECTOR,
) -> dict[str, Any]:
    flat = [dict(candidate) for row in scored_rows for candidate in row.get("candidates") or []]
    labels = np.asarray([1.0 if candidate.get("oracle_selected") else 0.0 for candidate in flat], dtype=np.float32)
    scores = np.asarray([_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in flat], dtype=np.float32)
    selected_scores = [float(score) for label, score in zip(labels, scores) if label > 0.0]
    nonselected_scores = [float(score) for label, score in zip(labels, scores) if label <= 0.0]
    primary_selected = [
        candidate
        for trace in traces
        if str(trace.get("selector_name") or "") == primary_selector
        for candidate in trace.get("selected_candidates") or []
    ]
    return {
        "candidate_level": candidate_level_metrics(labels, scores),
        "oracle_selected_score_mean": _mean(selected_scores),
        "non_oracle_selected_score_mean": _mean(nonselected_scores),
        "oracle_selected_score_lift": _mean(selected_scores) - _mean(nonselected_scores),
        "same_source_hard_negative_pairwise": same_source_hard_negative_pairwise(scored_rows),
        "within_event_pairwise": within_event_pairwise(scored_rows),
        "high_retrieval_non_oracle_false_positive_rate": high_retrieval_non_oracle_false_positive_rate(scored_rows),
        "primary_selected_score_mean": _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in primary_selected),
        "primary_source_composition": _composition(primary_selected, "source_group"),
        "primary_stance_region_composition": _composition(primary_selected, "stance_region"),
    }


def candidate_level_metrics(labels: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    return {
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()) if y.size else 0.0,
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "brier": float(np.mean((s - y) ** 2)) if y.size else 0.0,
        "calibration_bins": calibration_bins(y, s, n_bins=10),
    }


def same_source_hard_negative_pairwise(scored_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _pairwise_acc(scored_rows, group_by_source=True)


def within_event_pairwise(scored_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _pairwise_acc(scored_rows, group_by_source=False)


def high_retrieval_non_oracle_false_positive_rate(scored_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [candidate for row in scored_rows for candidate in row.get("candidates") or []]
    high_retrieval = [candidate for candidate in rows if retrieval_score(candidate) >= 0.75 and not bool(candidate.get("oracle_selected"))]
    if not high_retrieval:
        return {"n_high_retrieval_non_oracle": 0, "rate_score_ge_0_5": 0.0, "mean_score": 0.0}
    return {
        "n_high_retrieval_non_oracle": int(len(high_retrieval)),
        "rate_score_ge_0_5": float(np.mean([1.0 if _safe_float(candidate.get("direct_ce_score"), 0.0) >= 0.5 else 0.0 for candidate in high_retrieval])),
        "mean_score": _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in high_retrieval),
    }


def normalize_model_scores(raw_scores: Any) -> list[float]:
    arr = np.asarray(raw_scores, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr[:, -1]
    if arr.ndim != 1:
        raise ValueError(f"Expected one score per input pair, got shape={arr.shape}")
    if arr.size and (float(arr.min()) < 0.0 or float(arr.max()) > 1.0):
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))
    return [float(_safe_float(value, 0.0)) for value in arr.tolist()]


def roc_auc_score(labels: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    pairs = [(float(score), int(label > 0.0)) for label, score in zip(labels, scores)]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:end])
        idx = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision_score(labels: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    pairs = sorted(((float(score), int(label > 0.0)) for label, score in zip(labels, scores)), reverse=True)
    positives = sum(label for _, label in pairs)
    if positives == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            hits += 1
            total += hits / rank
    return float(total / positives)


def calibration_bins(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for idx in range(int(n_bins)):
        low = idx / float(n_bins)
        high = (idx + 1) / float(n_bins)
        if idx == int(n_bins) - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        count = int(mask.sum())
        bins.append(
            {
                "bin": int(idx),
                "low": float(low),
                "high": float(high),
                "count": count,
                "mean_score": float(scores[mask].mean()) if count else 0.0,
                "positive_rate": float(labels[mask].mean()) if count else 0.0,
            }
        )
    return bins


def _pairwise_acc(scored_rows: Sequence[dict[str, Any]], *, group_by_source: bool) -> dict[str, Any]:
    hits = 0.0
    total = 0
    for row in scored_rows:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in row.get("candidates") or []:
            key = source_group(candidate) if group_by_source else "__event__"
            groups[key].append(candidate)
        for candidates in groups.values():
            positives = [candidate for candidate in candidates if bool(candidate.get("oracle_selected"))]
            negatives = [candidate for candidate in candidates if not bool(candidate.get("oracle_selected"))]
            for pos in positives:
                for neg in negatives:
                    pos_score = _safe_float(pos.get("direct_ce_score"), 0.0)
                    neg_score = _safe_float(neg.get("direct_ce_score"), 0.0)
                    if pos_score > neg_score:
                        hits += 1.0
                    elif pos_score == neg_score:
                        hits += 0.5
                    total += 1
    return {"pairwise_acc": float(hits / total) if total else 0.0, "n_pairs": int(total)}


def _ranked_unique(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    selector_name: str,
    origin: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _selection_key(row)
        if not key or key in seen:
            continue
        item = dict(row)
        item["selector_name"] = selector_name
        item["selection_origin"] = origin
        item["selection_rank"] = len(selected) + 1
        item["direct_ce_adjusted_score"] = _safe_float(item.get("direct_ce_score"), 0.0)
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _direct_ce_sort_key(candidate: dict[str, Any]) -> tuple[float, str]:
    return (_safe_float(candidate.get("direct_ce_score"), 0.0), _reverse_string(str(candidate.get("candidate_key") or "")))


def _direct_ce_adjusted_sort_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    return (
        _safe_float(candidate.get("direct_ce_adjusted_score"), 0.0),
        _safe_float(candidate.get("direct_ce_score"), 0.0),
        _reverse_string(str(candidate.get("candidate_key") or "")),
    )


def _candidate_output(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_uid",
        "candidate_key",
        "selection_rank",
        "selection_origin",
        "union_pool_rank",
        "source_pools",
        "from_baseline",
        "from_qd",
        "baseline_rank",
        "qd_pool_rank",
        "retrieval_score",
        "semantic_completeness_score",
        "relevance_gate_score",
        "stance_bucket_derived",
        "stance_region",
        "stance_entropy",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "direct_ce_score",
        "direct_ce_adjusted_score",
        "direct_ce_model",
        "direct_ce_score_source",
        "same_source_selected_count",
        "direct_evidence_score",
        "claim_specificity_score",
        "background_only_score",
        "key_fact_overlap_score",
        "evidence_role",
        "role_evidence_score",
        "claim_directness_score",
        "text",
    ]
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["source_group"] = source_group(candidate)
    return out


def _composition(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    if key == "source_group":
        values = [source_group(row) for row in rows]
    else:
        values = [str(row.get(key) or "") for row in rows]
    counts = Counter(value for value in values if value)
    total = max(sum(counts.values()), 1)
    return {
        value: {"count": int(count), "fraction": float(count / total)}
        for value, count in sorted(counts.items())
    }


def _candidate_key(event_id: str, candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (str(event_id), str(candidate.get("candidate_uid") or ""), str(candidate.get("candidate_key") or ""))


def _selection_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_key") or candidate.get("candidate_uid") or "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _mean(values: Sequence[float] | Any) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else 0.0


def _reverse_string(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(ch)) for ch in str(value))
