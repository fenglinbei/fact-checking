from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from fact_checking.selectors.stage2_oracle import Stage2OracleExample


def ordered_selection_metrics(
    oracle_ordered_indices: list[int],
    selector_ordered_indices: list[int],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    metrics = _ordered_selection_metrics_at_k(oracle_ordered_indices, selector_ordered_indices, top_k=min(5, int(top_k)), suffix=5)
    if int(top_k) <= 5:
        return metrics
    dynamic = _ordered_selection_metrics_at_k(oracle_ordered_indices, selector_ordered_indices, top_k=int(top_k), suffix=int(top_k))
    metrics.update({key: value for key, value in dynamic.items() if key not in {"top1_match"}})
    metrics["set_overlap"] = dynamic["set_overlap"]
    metrics["overlap_pair_count"] = dynamic["overlap_pair_count"]
    return metrics


def _ordered_selection_metrics_at_k(
    oracle_ordered_indices: list[int],
    selector_ordered_indices: list[int],
    *,
    top_k: int,
    suffix: int,
) -> dict[str, Any]:
    oracle = [int(idx) for idx in oracle_ordered_indices[:top_k]]
    pred = [int(idx) for idx in selector_ordered_indices[:top_k]]
    oracle_set = set(oracle)
    pred_set = set(pred)
    overlap = oracle_set & pred_set
    union = oracle_set | pred_set
    k = max(len(pred), 1)

    prefix = {
        f"prefix_match@{n}": _prefix_match(oracle, pred, n)
        for n in (1, 3, 5)
        if n <= int(suffix)
    }
    if int(suffix) not in {1, 3, 5}:
        prefix[f"prefix_match@{int(suffix)}"] = _prefix_match(oracle, pred, int(suffix))
    pairwise_acc, pair_count = _pairwise_order_acc(oracle, pred)
    metrics: dict[str, Any] = {
        "set_overlap": int(len(overlap)),
        f"recall@{int(suffix)}": float(len(overlap) / max(len(oracle_set), 1)),
        f"precision@{int(suffix)}": float(len(overlap) / k),
        f"jaccard@{int(suffix)}": float(len(overlap) / max(len(union), 1)),
        "top1_match": float(len(oracle) > 0 and len(pred) > 0 and oracle[0] == pred[0]),
        f"ordered_hit@{int(suffix)}": _ordered_hit_at_k(oracle, pred, top_k),
        f"oracle_rank_ndcg@{int(suffix)}": _oracle_rank_ndcg(oracle, pred, top_k),
        f"pairwise_order_acc@{int(suffix)}": pairwise_acc,
        "overlap_pair_count": int(pair_count),
        f"ordered_exact_match@{int(suffix)}": float(oracle == pred[: len(oracle)] and len(pred) == len(oracle)),
    }
    metrics.update(prefix)
    return metrics


def summarize_ordered_selection(
    traces: list[dict[str, Any]],
    *,
    metric_prefix: str = "",
    include_by_label: bool = True,
) -> dict[str, Any]:
    if not traces:
        return {"n_claims": 0}

    metric_keys = [
        "recall@5",
        "precision@5",
        "jaccard@5",
        "top1_match",
        "prefix_match@1",
        "prefix_match@3",
        "prefix_match@5",
        "ordered_hit@5",
        "oracle_rank_ndcg@5",
        "pairwise_order_acc@5",
        "ordered_exact_match@5",
    ]
    extra_metric_keys = sorted(
        {
            key
            for trace in traces
            for key, value in trace.items()
            if (
                isinstance(value, (int, float))
                and (
                    key.endswith("@10")
                    or key.startswith("prefix_match@")
                    or key.startswith("ordered_hit@")
                    or key.startswith("oracle_rank_ndcg@")
                    or key.startswith("pairwise_order_acc@")
                    or key.startswith("ordered_exact_match@")
                )
            )
        }
        - set(metric_keys)
    )
    metric_keys.extend(extra_metric_keys)
    weights_for_pairwise = np.array(
        [float(trace.get("overlap_pair_count", 0)) for trace in traces],
        dtype=np.float64,
    )

    out: dict[str, Any] = {"n_claims": len(traces)}
    for key in metric_keys:
        values = np.array([float(trace.get(key, 0.0)) for trace in traces], dtype=np.float64)
        if key == "pairwise_order_acc@5" and weights_for_pairwise.sum() > 0:
            value = float((values * weights_for_pairwise).sum() / weights_for_pairwise.sum())
        else:
            value = float(values.mean()) if values.size else 0.0
        out[_join_metric_prefix(metric_prefix, key)] = value

    if include_by_label:
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trace in traces:
            by_label[str(trace.get("gold_label", ""))].append(trace)
        out["by_label"] = {
            label: summarize_ordered_selection(items, metric_prefix="", include_by_label=False)
            for label, items in sorted(by_label.items())
        }
    return out


def build_selection_trace(
    example: Stage2OracleExample,
    selector_scores: list[float] | np.ndarray,
    *,
    selector_name: str,
    top_k: int = 5,
) -> dict[str, Any]:
    scores = np.asarray(selector_scores, dtype=np.float32)
    order = np.argsort(-scores)[: min(int(top_k), len(scores))].astype(int).tolist()
    metrics = ordered_selection_metrics(example.selected_indices, order, top_k=top_k)

    candidate_scores: list[dict[str, Any]] = []
    for idx, candidate in enumerate(example.candidates):
        base = dict(example.candidate_scores[idx]) if idx < len(example.candidate_scores) else {}
        base["candidate_idx"] = int(idx)
        base["candidate_uid"] = str(candidate.get("candidate_uid") or base.get("candidate_uid") or "")
        base["selector_score"] = float(scores[idx]) if idx < len(scores) else 0.0
        candidate_scores.append(base)

    trace = {
        "event_id": example.event_id,
        "claim": example.claim,
        "gold_label": example.gold_label,
        "candidate_pool": example.candidates,
        "candidate_scores": candidate_scores,
        "oracle_ordered_indices": [int(idx) for idx in example.selected_indices],
        "selector_ordered_indices": [int(idx) for idx in order],
        "selector_scores": [float(x) for x in scores.tolist()],
        "selector_name": selector_name,
        "fingerprint": example.fingerprint,
    }
    trace.update(metrics)
    return trace


def build_order_control_trace(
    base_trace: dict[str, Any],
    ordered_indices: list[int],
    *,
    selector_name: str,
    top_k: int = 5,
) -> dict[str, Any]:
    trace = {
        key: base_trace[key]
        for key in (
            "event_id",
            "claim",
            "gold_label",
            "candidate_pool",
            "candidate_scores",
            "oracle_ordered_indices",
            "fingerprint",
        )
    }
    trace["selector_ordered_indices"] = [int(idx) for idx in ordered_indices[:top_k]]
    trace["selector_scores"] = []
    trace["selector_name"] = selector_name
    trace.update(
        ordered_selection_metrics(
            [int(idx) for idx in base_trace.get("oracle_ordered_indices", [])],
            trace["selector_ordered_indices"],
            top_k=top_k,
        )
    )
    return trace


def ranked_indices_from_hybrid(example: Stage2OracleExample, *, top_k: int = 5) -> list[int]:
    def _hybrid_score(idx: int) -> float:
        score = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
        try:
            return float(score.get("hybrid_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    order = sorted(range(len(example.candidates)), key=_hybrid_score, reverse=True)
    return [int(idx) for idx in order[:top_k]]


def ranked_indices_from_candidate_pool(example: Stage2OracleExample, *, top_k: int = 5) -> list[int]:
    return list(range(min(int(top_k), len(example.candidates))))


def reorder_predicted_set(
    selected_indices: list[int],
    *,
    example: Stage2OracleExample,
    mode: str,
) -> list[int]:
    selected_set = {int(idx) for idx in selected_indices}
    if mode == "candidate_pool_order":
        return [idx for idx in range(len(example.candidates)) if idx in selected_set]
    if mode == "hybrid_order":
        return [idx for idx in ranked_indices_from_hybrid(example, top_k=len(example.candidates)) if idx in selected_set]
    raise ValueError(f"Unknown reorder mode: {mode}")


def random_order_controls(
    selected_indices: list[int],
    *,
    example: Stage2OracleExample,
    seeds: list[int],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    selected = [int(idx) for idx in selected_indices[:top_k]]
    traces: list[dict[str, Any]] = []
    stub = {
        "event_id": example.event_id,
        "claim": example.claim,
        "gold_label": example.gold_label,
        "candidate_pool": example.candidates,
        "candidate_scores": example.candidate_scores,
        "oracle_ordered_indices": example.selected_indices,
        "fingerprint": example.fingerprint,
    }
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        permuted = list(selected)
        rng.shuffle(permuted)
        traces.append(
            build_order_control_trace(
                stub,
                permuted,
                selector_name=f"same_set_random_seed_{seed}",
                top_k=top_k,
            )
        )
    return traces


def _prefix_match(oracle: list[int], pred: list[int], n: int) -> float:
    if len(oracle) < n or len(pred) < n:
        return 0.0
    return float(all(oracle[i] == pred[i] for i in range(n)))


def _ordered_hit_at_k(oracle: list[int], pred: list[int], top_k: int) -> float:
    hits = 0
    denom = min(int(top_k), len(oracle))
    if denom <= 0:
        return 0.0
    for pos in range(denom):
        if pos < len(pred) and pred[pos] == oracle[pos]:
            hits += 1
    return float(hits / denom)


def _oracle_rank_ndcg(oracle: list[int], pred: list[int], top_k: int) -> float:
    rel_by_idx = {idx: max(int(top_k) - rank, 1) for rank, idx in enumerate(oracle[:top_k])}
    dcg = 0.0
    for rank, idx in enumerate(pred[:top_k], start=1):
        rel = rel_by_idx.get(idx, 0)
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1)
    ideal_rels = sorted(rel_by_idx.values(), reverse=True)[:top_k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _pairwise_order_acc(oracle: list[int], pred: list[int]) -> tuple[float, int]:
    oracle_pos = {idx: pos for pos, idx in enumerate(oracle)}
    pred_pos = {idx: pos for pos, idx in enumerate(pred)}
    overlap = [idx for idx in oracle if idx in pred_pos]
    if len(overlap) < 2:
        return 0.0, 0
    total = 0
    correct = 0
    for i, left in enumerate(overlap):
        for right in overlap[i + 1 :]:
            total += 1
            oracle_order = oracle_pos[left] < oracle_pos[right]
            pred_order = pred_pos[left] < pred_pos[right]
            if oracle_order == pred_order:
                correct += 1
    return float(correct / total) if total else 0.0, total


def _join_metric_prefix(prefix: str, key: str) -> str:
    return f"{prefix}{key}" if prefix else key
