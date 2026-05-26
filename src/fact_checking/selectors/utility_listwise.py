from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fact_checking.selectors.listwise import ListwiseCandidateGroup
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    Stage2OracleExample,
    read_jsonl,
)


DEFAULT_UTILITY_SOFT_TAU = 0.3
DEFAULT_UTILITY_POSITIVE_BEST_MARGIN = 0.05
DEFAULT_UTILITY_PAIRWISE_EPS = 1.0e-6


@dataclass(frozen=True)
class UtilityListwiseExample:
    event_id: str
    split: str
    claim: str
    candidates: list[dict[str, Any]]
    candidate_scores: list[dict[str, Any]]
    delta_margins: list[float]
    positive_indices: list[int]
    oracle_selected_indices: list[int]
    fingerprint: str
    oracle_example: Stage2OracleExample

    def as_candidate_group(self) -> ListwiseCandidateGroup:
        return ListwiseCandidateGroup(
            claim=self.claim,
            candidates=[dict(item) for item in self.candidates],
            candidate_scores=[dict(item) for item in self.candidate_scores],
        )


def load_utility_listwise_examples(
    vig_cache: str,
    oracle_examples: list[Stage2OracleExample],
    *,
    split: str,
    max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
    sample_limit: int | None = None,
    strict: bool = True,
) -> list[UtilityListwiseExample]:
    rows = read_jsonl(vig_cache)
    return build_utility_listwise_examples(
        rows,
        oracle_examples,
        split=split,
        max_candidates=max_candidates,
        sample_limit=sample_limit,
        strict=strict,
    )


def build_utility_listwise_examples(
    vig_rows: list[dict[str, Any]],
    oracle_examples: list[Stage2OracleExample],
    *,
    split: str,
    max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
    sample_limit: int | None = None,
    strict: bool = True,
) -> list[UtilityListwiseExample]:
    rows_by_event: dict[str, dict[int, dict[str, Any]]] = {}
    for row in vig_rows:
        if int(row.get("step", -1)) != 0:
            continue
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        candidate_idx = _safe_int(row.get("candidate_idx"), -1)
        if candidate_idx < 0 or candidate_idx >= int(max_candidates):
            continue
        rows_by_event.setdefault(event_id, {})[candidate_idx] = dict(row)

    examples: list[UtilityListwiseExample] = []
    for oracle in oracle_examples:
        if sample_limit is not None and len(examples) >= int(sample_limit):
            break
        event_rows = rows_by_event.get(oracle.event_id)
        if not event_rows:
            if strict:
                raise ValueError(f"No step=0 VIG rows for event_id={oracle.event_id}.")
            continue
        n_candidates = min(len(oracle.candidates), int(max_candidates))
        missing = [idx for idx in range(n_candidates) if idx not in event_rows]
        if missing:
            if strict:
                raise ValueError(
                    f"Missing step=0 VIG rows for event_id={oracle.event_id}: {missing}."
                )
            continue
        candidates = [dict(oracle.candidates[idx]) for idx in range(n_candidates)]
        candidate_scores = []
        deltas: list[float] = []
        for idx in range(n_candidates):
            row = event_rows[idx]
            score = dict(oracle.candidate_scores[idx]) if idx < len(oracle.candidate_scores) else {}
            score.setdefault("candidate_idx", idx)
            score.setdefault("hybrid_rank", idx)
            for key in ("hybrid_score", "dense_score", "lexical_score", "bm25_score"):
                if key in row:
                    score[key] = _safe_float(row.get(key), _safe_float(score.get(key), 0.0))
            candidate_scores.append(score)
            deltas.append(_safe_float(row.get("delta_margin"), 0.0))
        examples.append(
            UtilityListwiseExample(
                event_id=oracle.event_id,
                split=str(split),
                claim=oracle.claim,
                candidates=candidates,
                candidate_scores=candidate_scores,
                delta_margins=deltas,
                positive_indices=utility_positive_indices(deltas),
                oracle_selected_indices=[int(idx) for idx in oracle.selected_indices],
                fingerprint=oracle.fingerprint,
                oracle_example=oracle,
            )
        )
    return examples


def utility_positive_indices(
    delta_margins: list[float] | torch.Tensor,
    *,
    positive_best_margin: float = DEFAULT_UTILITY_POSITIVE_BEST_MARGIN,
) -> list[int]:
    values = _float_list(delta_margins)
    if not values:
        return []
    best_delta = max(values)
    cutoff = best_delta - float(positive_best_margin)
    positives = [
        idx
        for idx, delta in enumerate(values)
        if float(delta) > 0.0 or float(delta) >= cutoff
    ]
    best_idx = max(range(len(values)), key=lambda idx: (values[idx], -idx))
    if best_idx not in positives:
        positives.append(best_idx)
    return sorted(set(int(idx) for idx in positives))


def utility_soft_targets(
    delta_margins: list[float] | torch.Tensor,
    *,
    tau: float = DEFAULT_UTILITY_SOFT_TAU,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(delta_margins, torch.Tensor):
        deltas = delta_margins.to(device=device or delta_margins.device, dtype=dtype or delta_margins.dtype)
    else:
        deltas = torch.tensor(delta_margins, dtype=dtype or torch.float32, device=device)
    if deltas.numel() == 0:
        return deltas
    temperature = max(float(tau), 1.0e-6)
    return torch.softmax(deltas / temperature, dim=0)


def utility_listwise_loss(
    score_groups: list[torch.Tensor],
    delta_groups: list[list[float] | torch.Tensor],
    *,
    pairwise_weight: float = 1.0,
    soft_ce_weight: float = 0.2,
    bce_weight: float = 0.2,
    soft_tau: float = DEFAULT_UTILITY_SOFT_TAU,
    positive_best_margin: float = DEFAULT_UTILITY_POSITIVE_BEST_MARGIN,
    pairwise_eps: float = DEFAULT_UTILITY_PAIRWISE_EPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    pairwise_losses: list[torch.Tensor] = []
    soft_ce_losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    metric_scores: list[torch.Tensor] = []
    metric_deltas: list[torch.Tensor] = []

    for scores, raw_deltas in zip(score_groups, delta_groups):
        if scores.numel() == 0:
            continue
        deltas = _delta_tensor(raw_deltas, scores)
        if deltas.numel() != scores.numel():
            raise ValueError(
                f"Score/delta length mismatch: scores={scores.numel()} deltas={deltas.numel()}."
            )
        pairwise_loss = _pairwise_delta_loss(scores, deltas, pairwise_eps=float(pairwise_eps))
        if pairwise_loss is not None:
            pairwise_losses.append(pairwise_loss)
        soft_targets = utility_soft_targets(deltas, tau=float(soft_tau), dtype=scores.dtype, device=scores.device)
        soft_ce_losses.append(-(soft_targets * torch.log_softmax(scores, dim=0)).sum())
        labels = torch.zeros_like(scores)
        positives = utility_positive_indices(deltas, positive_best_margin=float(positive_best_margin))
        if positives:
            labels[torch.tensor(positives, dtype=torch.long, device=scores.device)] = 1.0
        bce_losses.append(nn.functional.binary_cross_entropy_with_logits(scores, labels))
        metric_scores.append(scores.detach())
        metric_deltas.append(deltas.detach())

    if not metric_scores:
        raise ValueError("No valid utility listwise groups for loss.")
    device = metric_scores[0].device
    zero = torch.zeros((), dtype=metric_scores[0].dtype, device=device)
    pairwise_loss = torch.stack(pairwise_losses).mean() if pairwise_losses else zero
    soft_ce_loss = torch.stack(soft_ce_losses).mean() if soft_ce_losses else zero
    bce_loss = torch.stack(bce_losses).mean() if bce_losses else zero
    total = (
        float(pairwise_weight) * pairwise_loss
        + float(soft_ce_weight) * soft_ce_loss
        + float(bce_weight) * bce_loss
    )
    metrics = utility_rank_metrics(metric_scores, metric_deltas, positive_best_margin=positive_best_margin)
    metrics.update(
        {
            "loss": float(total.detach().float().cpu()),
            "pairwise_loss": float(pairwise_loss.detach().float().cpu()),
            "soft_ce_loss": float(soft_ce_loss.detach().float().cpu()),
            "bce_loss": float(bce_loss.detach().float().cpu()),
        }
    )
    return total, metrics


def utility_rank_metrics(
    score_groups: list[torch.Tensor],
    delta_groups: list[list[float] | torch.Tensor],
    *,
    positive_best_margin: float = DEFAULT_UTILITY_POSITIVE_BEST_MARGIN,
    pairwise_eps: float = DEFAULT_UTILITY_PAIRWISE_EPS,
) -> dict[str, float]:
    pair_correct = 0
    pair_total = 0
    top1_match = 0
    positive_hit = 0
    n_groups = 0
    pearsons: list[float] = []
    spearmans: list[float] = []

    for scores, raw_deltas in zip(score_groups, delta_groups):
        if scores.numel() == 0:
            continue
        deltas = _delta_tensor(raw_deltas, scores).detach().float().cpu()
        values = deltas.tolist()
        pred = scores.detach().float().cpu()
        if pred.numel() != deltas.numel():
            continue
        n_groups += 1
        top_pred = int(torch.argmax(pred).item())
        top_delta = max(range(len(values)), key=lambda idx: (values[idx], -idx))
        top1_match += int(top_pred == top_delta)
        positives = set(utility_positive_indices(values, positive_best_margin=float(positive_best_margin)))
        positive_hit += int(top_pred in positives)
        for left in range(len(values)):
            for right in range(len(values)):
                if values[left] > values[right] + float(pairwise_eps):
                    pair_total += 1
                    pair_correct += int(float(pred[left]) > float(pred[right]))
        pearsons.append(_corrcoef(pred.tolist(), values))
        spearmans.append(_corrcoef(_ordinal_ranks(pred.tolist()), _ordinal_ranks(values)))

    return {
        "n_groups": float(n_groups),
        "pairwise_accuracy": float(pair_correct / pair_total) if pair_total else 0.0,
        "n_pairwise_pairs": float(pair_total),
        "top1_delta_match": float(top1_match / max(n_groups, 1)),
        "positive_hit@1": float(positive_hit / max(n_groups, 1)),
        "pearson": _safe_mean(pearsons),
        "spearman": _safe_mean(spearmans),
    }


def _pairwise_delta_loss(
    scores: torch.Tensor,
    deltas: torch.Tensor,
    *,
    pairwise_eps: float,
) -> torch.Tensor | None:
    delta_diff = deltas[:, None] - deltas[None, :]
    mask = delta_diff > float(pairwise_eps)
    if not bool(mask.any()):
        return None
    score_diff = scores[:, None] - scores[None, :]
    return nn.functional.softplus(-score_diff[mask]).mean()


def _delta_tensor(raw_deltas: list[float] | torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if isinstance(raw_deltas, torch.Tensor):
        return raw_deltas.to(device=scores.device, dtype=scores.dtype)
    return torch.tensor(raw_deltas, dtype=scores.dtype, device=scores.device)


def _float_list(values: list[float] | torch.Tensor) -> list[float]:
    if isinstance(values, torch.Tensor):
        return [float(x) for x in values.detach().float().cpu().tolist()]
    return [float(x) for x in values]


def _ordinal_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0 for _ in values]
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _corrcoef(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 0.0
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mean(values: list[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else 0.0


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

