from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


WEIGHT_SCHEMA_VERSION = "mrec_learned_marginal_proxy_weights_v0_2"
REWARD_WEIGHT_SCHEMA_VERSION = "mrec_learned_marginal_reward_weights_v0_2"
SUPPORTED_WEIGHT_SCHEMA_VERSIONS = {WEIGHT_SCHEMA_VERSION, REWARD_WEIGHT_SCHEMA_VERSION}
SUPERVISION_MODE_LEGACY_HYBRID = "legacy_hybrid"
SUPERVISION_MODE_STRUCTURE_ONLY = "structure_only"
SUPPORTED_PROXY_SUPERVISION_MODES = {
    SUPERVISION_MODE_LEGACY_HYBRID,
    SUPERVISION_MODE_STRUCTURE_ONLY,
}
_STRUCTURE_ONLY_CANDIDATE_FIELDS = (
    "candidate_key",
    "candidate_uid",
    "evidence_id",
    "selector_candidate_idx",
    "candidate_idx",
    "text",
    "evidence_text",
    "covered_atom_ids",
    "map_relation",
    "relation",
    "map_directness",
    "directness",
    "map_confidence",
    "evidence_map_quality_score",
    "hybrid_score",
    "baseline_hybrid_score",
    "base_score",
    "mrec_token_cost",
    "token_cost",
    "prompt_token_count",
    "evidence_token_count",
    "duplicate_group",
    "source_group",
    "source_report",
    "report_id",
    "source_id",
)
_STRUCTURE_ONLY_ALIGNMENT_FIELDS = (
    "evidence_id",
    "atom_id",
    "relation",
    "directness",
    "confidence",
)
FEATURE_NAMES = (
    "resolution_delta",
    "entropy_reduction",
    "new_atom_coverage",
    "new_relation_for_atom",
    "stance_tension",
    "corroboration_gain",
    "source_novelty",
    "text_novelty",
    "map_confidence",
    "map_quality",
    "retrieval_score",
    "cost_ratio",
)
POSITIVE_FEATURE_NAMES = tuple(name for name in FEATURE_NAMES if name != "cost_ratio")

_RESOLVING_RELATION_STATES = {"S", "R", "Q"}
_RELATION_GROUPS = {
    "support": "support",
    "supports": "support",
    "supported_by": "support",
    "entails": "support",
    "consistent": "support",
    "refute": "refute",
    "refutes": "refute",
    "contradict": "refute",
    "contradicts": "refute",
    "counter": "refute",
    "conflict": "refute",
    "qualify": "qualify",
    "qualifies": "qualify",
    "qualified": "qualify",
    "condition": "qualify",
    "hedge": "qualify",
    "mixed": "qualify",
    "partially_supports": "qualify",
    "partial": "qualify",
    "insufficient": "insufficient",
    "background": "background",
    "irrelevant": "irrelevant",
}
_RELATION_TO_STATE = {
    "support": "S",
    "refute": "R",
    "qualify": "Q",
}
_DIRECTNESS_FACTOR = {
    "direct": 1.0,
    "full": 1.0,
    "partial": 0.65,
    "medium": 0.4,
    "context": 0.25,
    "none": 0.0,
}


@dataclass(frozen=True)
class LearnedMarginalWeights:
    feature_weights: dict[str, float]
    cost_weight: float
    schema_version: str = WEIGHT_SCHEMA_VERSION
    bias: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LearnedMarginalWeights":
        feature_weights = {
            name: max(0.0, _float_or_default(self.feature_weights.get(name), 0.0))
            for name in POSITIVE_FEATURE_NAMES
        }
        return LearnedMarginalWeights(
            feature_weights=feature_weights,
            cost_weight=max(0.0, _float_or_default(self.cost_weight, 0.0)),
            schema_version=str(self.schema_version or WEIGHT_SCHEMA_VERSION),
            bias=_float_or_default(self.bias, 0.0),
            metadata=dict(self.metadata),
        )

    def to_json_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "schema_version": normalized.schema_version,
            "feature_names": list(FEATURE_NAMES),
            "feature_weights": {
                name: float(normalized.feature_weights.get(name, 0.0))
                for name in POSITIVE_FEATURE_NAMES
            },
            "cost_weight": float(normalized.cost_weight),
            "bias": float(normalized.bias),
            "metadata": dict(normalized.metadata),
        }


def initial_learned_marginal_weights() -> LearnedMarginalWeights:
    return LearnedMarginalWeights(
        feature_weights={
            "resolution_delta": 4.0,
            "entropy_reduction": 3.0,
            "new_atom_coverage": 1.8,
            "new_relation_for_atom": 1.5,
            "stance_tension": 1.4,
            "corroboration_gain": 1.0,
            "source_novelty": 0.8,
            "text_novelty": 0.8,
            "map_confidence": 0.7,
            "map_quality": 0.6,
            "retrieval_score": 0.3,
        },
        cost_weight=0.2,
        metadata={"initialized_from": "hand_seed_v0_2"},
    )


def initial_neutral_learned_marginal_weights() -> LearnedMarginalWeights:
    return LearnedMarginalWeights(
        feature_weights={name: 1.0 for name in POSITIVE_FEATURE_NAMES},
        cost_weight=1.0,
        metadata={"initialized_from": "equal_weight_neutral_v0_1"},
    )


def initial_learned_marginal_reward_weights() -> LearnedMarginalWeights:
    seed = initial_learned_marginal_weights().normalized()
    return LearnedMarginalWeights(
        feature_weights=dict(seed.feature_weights),
        cost_weight=float(seed.cost_weight),
        schema_version=REWARD_WEIGHT_SCHEMA_VERSION,
        bias=0.0,
        metadata={"initialized_from": "proxy_hand_seed_v0_2"},
    )


def load_learned_marginal_weights(path: str | Path | None, *, allow_default: bool = True) -> LearnedMarginalWeights:
    if path is None or not str(path):
        if allow_default:
            return initial_learned_marginal_weights()
        raise ValueError("learned_marginal_proxy requires a weight file")
    return _load_learned_marginal_weights_cached(str(path)).normalized()


@lru_cache(maxsize=16)
def _load_learned_marginal_weights_cached(path: str) -> LearnedMarginalWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version") or "")
    if schema_version not in SUPPORTED_WEIGHT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported learned marginal weight schema: {payload.get('schema_version')!r}")
    feature_weights = payload.get("feature_weights") or {}
    if not isinstance(feature_weights, Mapping):
        raise ValueError("learned marginal weight file has invalid feature_weights")
    return LearnedMarginalWeights(
        feature_weights={name: _float_or_default(feature_weights.get(name), 0.0) for name in POSITIVE_FEATURE_NAMES},
        cost_weight=_float_or_default(payload.get("cost_weight"), 0.0),
        schema_version=schema_version,
        bias=_float_or_default(payload.get("bias"), 0.0),
        metadata=dict(payload.get("metadata") or {}),
    )


def save_learned_marginal_weights(path: str | Path, weights: LearnedMarginalWeights) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(weights.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def learned_marginal_weight_fingerprint(weights_or_path: LearnedMarginalWeights | str | Path | None) -> str:
    if weights_or_path is None or not str(weights_or_path):
        weights = initial_learned_marginal_weights()
        payload = weights.to_json_dict()
    elif isinstance(weights_or_path, LearnedMarginalWeights):
        payload = weights_or_path.to_json_dict()
    else:
        weights = load_learned_marginal_weights(weights_or_path)
        payload = weights.to_json_dict()
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def hard_state_to_soft_state(atom_states: Mapping[str, str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for atom_id, state in atom_states.items():
        normalized = str(state or "U").upper()
        out[str(atom_id)] = {normalized if normalized in {"U", "S", "R", "Q", "C"} else "U": 1.0}
    return out


def update_soft_state_from_relation(
    soft_state: Mapping[str, Mapping[str, float]],
    *,
    atom_id: str,
    relation: str,
) -> dict[str, dict[str, float]]:
    out = {str(key): {str(k): float(v) for k, v in value.items()} for key, value in soft_state.items()}
    relation_state = _state_for_relation(relation)
    if atom_id not in out or relation_state not in _RESOLVING_RELATION_STATES:
        return out
    current = _state_distribution(out.get(atom_id) or {"U": 1.0})
    if current.get("U", 0.0) >= 0.5:
        out[atom_id] = {relation_state: 1.0}
        return out
    dominant = max(current, key=lambda key: current.get(key, 0.0))
    if dominant in {"S", "R"} and relation_state in {"S", "R"} and dominant != relation_state:
        out[atom_id] = {"C": 1.0}
    elif relation_state == "Q" and dominant in {"S", "R", "Q", "C"}:
        out[atom_id] = {"Q": 1.0}
    else:
        out[atom_id] = {dominant: 1.0}
    return out


def extract_marginal_features(
    candidate: Mapping[str, Any],
    *,
    selected_steps: Sequence[Mapping[str, Any]],
    soft_state: Mapping[str, Mapping[str, float]],
    token_budget: int | None,
    pool_max_token_cost: int | None,
    map_ablation_mode: str = "full",
) -> dict[str, float]:
    pairs = _candidate_atom_pairs(candidate, soft_state)
    selected_atom_ids = _selected_atom_ids(selected_steps)
    selected_relations = _selected_relations_by_atom(selected_steps)
    selected_sources = _selected_sources(selected_steps)
    selected_texts = _selected_texts(selected_steps)
    selected_duplicate_groups = _selected_duplicate_groups(selected_steps)
    source_key = _source_key(candidate)
    text_key = _normalize_text(candidate.get("text") or candidate.get("evidence_text") or "")
    duplicate_group = _compact(candidate.get("duplicate_group") or "")

    total_atoms = max(len(soft_state), 1)
    resolution_gains: dict[str, float] = {}
    entropy_gains: dict[str, float] = {}
    new_relation_hits = 0
    tension = 0.0
    corroboration = 0.0
    map_confidences: list[float] = []

    for pair in pairs:
        atom_id = str(pair.get("atom_id") or "")
        relation = _relation_group(pair.get("relation"))
        relation_state = _state_for_relation(relation)
        directness = _directness_factor(pair.get("directness"))
        confidence = _clip01(_float_or_default(pair.get("confidence"), _float_or_default(candidate.get("map_confidence"), 0.0)))
        # Evidence-map ablation: degrade individual map signals so the
        # learned selector is forced to rely on the remaining ones.
        if map_ablation_mode == "no_directness":
            directness = _directness_factor("medium")
        elif map_ablation_mode == "no_confidence":
            confidence = 1.0
        elif map_ablation_mode == "no_relation":
            relation = "background"
            relation_state = _state_for_relation(relation)
        map_confidences.append(confidence)
        if relation_state in _RESOLVING_RELATION_STATES and atom_id in soft_state:
            unresolved_mass = _state_distribution(soft_state.get(atom_id) or {}).get("U", 0.0)
            gain = _clip01(unresolved_mass * directness * max(confidence, 0.5))
            resolution_gains[atom_id] = max(resolution_gains.get(atom_id, 0.0), gain)
            entropy_gains[atom_id] = max(
                entropy_gains.get(atom_id, 0.0),
                _entropy_reduction_for_pair(soft_state.get(atom_id) or {}, relation_state=relation_state, gain=gain),
            )
        if atom_id and relation and relation not in selected_relations.get(atom_id, set()):
            new_relation_hits += 1
        seen_states = {_state_for_relation(rel) for rel in selected_relations.get(atom_id, set())}
        state_dist = _state_distribution(soft_state.get(atom_id) or {})
        dominant_states = {state for state, mass in state_dist.items() if mass >= 0.5}
        if _has_stance_tension(relation_state, seen_states | dominant_states):
            tension = max(tension, directness * max(confidence, 0.5))
        if relation_state in seen_states and relation_state in _RESOLVING_RELATION_STATES:
            corroboration = max(corroboration, directness * max(confidence, 0.5))

    covered_atoms = {str(pair.get("atom_id") or "") for pair in pairs if str(pair.get("atom_id") or "")}
    new_covered_atoms = covered_atoms - selected_atom_ids
    source_novelty = 0.0 if source_key and source_key in selected_sources else 1.0
    text_repeated = bool(text_key and text_key in selected_texts) or bool(duplicate_group and duplicate_group in selected_duplicate_groups)
    text_novelty = 0.0 if text_repeated else 1.0
    if corroboration > 0.0:
        corroboration *= max(source_novelty, text_novelty)

    token_cost = _token_cost(candidate)
    denominator = int(token_budget or 0) if token_budget is not None else int(pool_max_token_cost or 0)
    cost_ratio = _clip01(float(token_cost) / float(max(denominator, 1))) if token_cost > 0 else 0.0
    features = {
        "resolution_delta": _clip01(sum(resolution_gains.values()) / total_atoms),
        "entropy_reduction": _clip01(sum(entropy_gains.values()) / total_atoms),
        "new_atom_coverage": _clip01(len(new_covered_atoms) / total_atoms),
        "new_relation_for_atom": 1.0 if new_relation_hits > 0 else 0.0,
        "stance_tension": _clip01(tension),
        "corroboration_gain": _clip01(corroboration),
        "source_novelty": _clip01(source_novelty),
        "text_novelty": _clip01(text_novelty),
        "map_confidence": _clip01(max(map_confidences) if map_confidences else _float_or_default(candidate.get("map_confidence"), 0.0)),
        "map_quality": _clip01(_float_or_default(candidate.get("evidence_map_quality_score"), 0.0)),
        "retrieval_score": _retrieval_score(candidate),
        "cost_ratio": cost_ratio,
    }
    if map_ablation_mode == "no_map":
        for k in (
            "resolution_delta",
            "entropy_reduction",
            "new_atom_coverage",
            "new_relation_for_atom",
            "stance_tension",
            "corroboration_gain",
            "map_confidence",
            "map_quality",
        ):
            features[k] = 0.0
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def score_marginal_features(features: Mapping[str, Any], weights: LearnedMarginalWeights | Mapping[str, Any]) -> float:
    learned = _coerce_weights(weights)
    score = float(learned.bias)
    for name in POSITIVE_FEATURE_NAMES:
        score += float(learned.feature_weights.get(name, 0.0)) * _clip01(_float_or_default(features.get(name), 0.0))
    score -= float(learned.cost_weight) * _clip01(_float_or_default(features.get("cost_ratio"), 0.0))
    return float(score)


def rank_candidates_by_proxy(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_steps: Sequence[Mapping[str, Any]],
    soft_state: Mapping[str, Mapping[str, float]],
    oracle_ordered_keys: Sequence[str] | None = None,
    token_budget: int | None = None,
    pool_max_token_cost: int | None = None,
    map_ablation_mode: str = "full",
    supervision_mode: str = SUPERVISION_MODE_LEGACY_HYBRID,
) -> list[int]:
    mode = _normalize_proxy_supervision_mode(supervision_mode)
    oracle_rank = (
        {str(key): idx for idx, key in enumerate(oracle_ordered_keys or []) if str(key)}
        if mode == SUPERVISION_MODE_LEGACY_HYBRID
        else {}
    )
    has_oracle_hit = any(_candidate_key(candidate) in oracle_rank for candidate in candidates)
    rows: list[tuple[tuple[Any, ...], int]] = []
    for idx, candidate in enumerate(candidates):
        key = _candidate_key(candidate)
        if has_oracle_hit and key in oracle_rank:
            rows.append(((0, oracle_rank[key], idx), idx))
            continue
        features = extract_marginal_features(
            candidate,
            selected_steps=selected_steps,
            soft_state=soft_state,
            token_budget=token_budget,
            pool_max_token_cost=pool_max_token_cost,
            map_ablation_mode=map_ablation_mode,
        )
        best_directness = _best_directness(candidate, soft_state)
        direct_resolving = 1.0 if features["resolution_delta"] > 0.0 and best_directness >= 1.0 else 0.0
        partial_resolving = 1.0 if features["resolution_delta"] > 0.0 and 0.0 < best_directness < 1.0 else 0.0
        rows.append(
            (
                (
                    1 if has_oracle_hit else 0,
                    -direct_resolving,
                    -partial_resolving,
                    -features["new_atom_coverage"],
                    -features["new_relation_for_atom"],
                    -features["map_quality"],
                    -features["retrieval_score"],
                    idx,
                ),
                idx,
            )
        )
    rows.sort(key=lambda item: item[0])
    return [idx for _, idx in rows]


def build_proxy_pairwise_preferences(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_steps: Sequence[Mapping[str, Any]],
    soft_state: Mapping[str, Mapping[str, float]],
    oracle_ordered_keys: Sequence[str] | None = None,
    token_budget: int | None = None,
    pool_max_token_cost: int | None = None,
    map_ablation_mode: str = "full",
    supervision_mode: str = SUPERVISION_MODE_LEGACY_HYBRID,
) -> list[tuple[int, int]]:
    order = rank_candidates_by_proxy(
        candidates,
        selected_steps=selected_steps,
        soft_state=soft_state,
        oracle_ordered_keys=oracle_ordered_keys,
        token_budget=token_budget,
        pool_max_token_cost=pool_max_token_cost,
        map_ablation_mode=map_ablation_mode,
        supervision_mode=supervision_mode,
    )
    return build_winner_vs_rest_preferences(order)


def build_winner_vs_rest_preferences(order: Sequence[int]) -> list[tuple[int, int]]:
    """Turn any supervision-specific order into the shared pairwise targets."""
    if len(order) < 2:
        return []
    best = int(order[0])
    return [(best, int(other)) for other in order[1:]]


def build_structure_proxy_selected_record(
    candidate: Mapping[str, Any],
    *,
    soft_state: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Expose the exact state-replay record used by the structure-only teacher."""
    record = _proxy_selected_record(candidate, soft_state=soft_state)
    pairs = _candidate_atom_pairs(candidate, soft_state)
    pairs.sort(
        key=lambda pair: (
            -_directness_factor(pair.get("directness")),
            -_clip01(
                _float_or_default(
                    pair.get("confidence"),
                    _float_or_default(candidate.get("map_confidence"), 0.0),
                )
            ),
            str(pair.get("atom_id") or ""),
        )
    )
    if pairs:
        record["directness"] = str(pairs[0].get("directness") or "")
        record["confidence"] = pairs[0].get("confidence")
    return record


def train_learned_marginal_proxy_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 50,
    learning_rate: float = 0.05,
    candidate_top_n: int = 20,
    rollout_steps: int = 5,
    map_ablation_mode: str = "full",
    supervision_mode: str = SUPERVISION_MODE_LEGACY_HYBRID,
) -> tuple[LearnedMarginalWeights, dict[str, Any]]:
    mode = _normalize_proxy_supervision_mode(supervision_mode)
    positive_features, negative_features, supervision_metrics = _collect_proxy_pairwise_features(
        rows,
        candidate_top_n=candidate_top_n,
        rollout_steps=rollout_steps,
        map_ablation_mode=map_ablation_mode,
        supervision_mode=mode,
    )
    initial = (
        initial_neutral_learned_marginal_weights()
        if mode == SUPERVISION_MODE_STRUCTURE_ONLY
        else initial_learned_marginal_weights()
    )
    metadata = {
        "trained_from": "proxy_pairwise",
        "pair_count": int(len(positive_features)),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "map_ablation_mode": str(map_ablation_mode),
    }
    if mode == SUPERVISION_MODE_STRUCTURE_ONLY:
        metadata.update(
            {
                "supervision_mode": mode,
                "supervision_fingerprint": supervision_metrics["supervision_fingerprint"],
                "initialized_from": "equal_weight_neutral_v0_1",
            }
        )
    learned, optimization_metrics = fit_pairwise_marginal_scorer(
        positive_features,
        negative_features,
        initial_weights=initial,
        epochs=epochs,
        learning_rate=learning_rate,
        metadata=metadata,
    )
    metrics = dict(supervision_metrics)
    metrics.update(optimization_metrics)
    return learned, metrics


def evaluate_learned_marginal_proxy_weights(
    rows: Sequence[Mapping[str, Any]],
    weights: LearnedMarginalWeights | Mapping[str, Any],
    *,
    candidate_top_n: int = 20,
    rollout_steps: int = 5,
    map_ablation_mode: str = "full",
    supervision_mode: str = SUPERVISION_MODE_LEGACY_HYBRID,
) -> dict[str, Any]:
    mode = _normalize_proxy_supervision_mode(supervision_mode)
    positive_features, negative_features, metrics = _collect_proxy_pairwise_features(
        rows,
        candidate_top_n=candidate_top_n,
        rollout_steps=rollout_steps,
        map_ablation_mode=map_ablation_mode,
        supervision_mode=mode,
    )
    if mode == SUPERVISION_MODE_LEGACY_HYBRID:
        correct = 0
        scored_rows = 0
        for row in rows:
            candidates = _row_candidates(row, candidate_top_n=candidate_top_n)
            if len(candidates) < 2:
                continue
            oracle_keys = _oracle_ordered_keys(row, candidates)
            if not oracle_keys:
                continue
            soft_state = _initial_soft_state_from_row(row)
            pool_max_token_cost = max([_token_cost(candidate) for candidate in candidates] or [1])
            scores = [
                score_marginal_features(
                    extract_marginal_features(
                        candidate,
                        selected_steps=[],
                        soft_state=soft_state,
                        token_budget=None,
                        pool_max_token_cost=pool_max_token_cost,
                    ),
                    weights,
                )
                for candidate in candidates
            ]
            best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
            correct += int(_candidate_key(candidates[best_idx]) == oracle_keys[0])
            scored_rows += 1
        out = dict(metrics)
        out.update(
            {
                "scored_row_count": int(scored_rows),
                "scored_pair_count": 0,
                "pair_accuracy": float(correct / scored_rows) if scored_rows else 0.0,
                "evaluation_target": "legacy_oracle_first_key_top1",
            }
        )
        return out

    correct = 0
    for positive, negative in zip(positive_features, negative_features):
        positive_score = score_marginal_features(dict(zip(FEATURE_NAMES, positive)), weights)
        negative_score = score_marginal_features(dict(zip(FEATURE_NAMES, negative)), weights)
        correct += int(positive_score > negative_score)
    out = dict(metrics)
    out.update(
        {
            "scored_row_count": int(metrics["eligible_row_count"]),
            "scored_pair_count": int(len(positive_features)),
            "pair_accuracy": float(correct / len(positive_features)) if positive_features else 0.0,
            "evaluation_target": "structure_winner_vs_rest",
        }
    )
    return out


def fit_pairwise_marginal_scorer(
    positive_features: Sequence[Sequence[float]],
    negative_features: Sequence[Sequence[float]],
    *,
    initial_weights: LearnedMarginalWeights,
    epochs: int,
    learning_rate: float,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[LearnedMarginalWeights, dict[str, Any]]:
    """Fit the shared positive-feature/cost pairwise scorer.

    The supervision builder is deliberately outside this function so a future
    utility-only implementation can reuse the exact optimizer and loss.
    """
    import torch
    import torch.nn.functional as F

    if len(positive_features) != len(negative_features):
        raise ValueError("positive and negative pairwise feature counts must match")
    pair_count = len(positive_features)
    if pair_count == 0:
        return initial_weights, {"pair_count": 0, "final_loss": 0.0, "epochs": 0}

    expected_width = len(FEATURE_NAMES)
    if any(len(row) != expected_width for row in positive_features) or any(
        len(row) != expected_width for row in negative_features
    ):
        raise ValueError(f"pairwise feature vectors must have width {expected_width}")

    pos_tensor = torch.tensor(positive_features, dtype=torch.float32)
    neg_tensor = torch.tensor(negative_features, dtype=torch.float32)
    initial = initial_weights.normalized()
    theta = torch.nn.Parameter(torch.tensor([
        _inverse_softplus(float(initial.feature_weights.get(name, 0.0)))
        for name in POSITIVE_FEATURE_NAMES
    ], dtype=torch.float32))
    theta_cost = torch.nn.Parameter(torch.tensor(_inverse_softplus(initial.cost_weight), dtype=torch.float32))
    optimizer = torch.optim.Adam([theta, theta_cost], lr=float(learning_rate))
    final_loss = 0.0
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad()
        positive_weights = F.softplus(theta)
        cost_weight = F.softplus(theta_cost)
        pos_score = (pos_tensor[:, :-1] * positive_weights).sum(dim=1) - pos_tensor[:, -1] * cost_weight
        neg_score = (neg_tensor[:, :-1] * positive_weights).sum(dim=1) - neg_tensor[:, -1] * cost_weight
        loss = F.softplus(-(pos_score - neg_score)).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    learned = LearnedMarginalWeights(
        feature_weights={
            name: float(F.softplus(theta).detach().cpu()[idx].item())
            for idx, name in enumerate(POSITIVE_FEATURE_NAMES)
        },
        cost_weight=float(F.softplus(theta_cost).detach().cpu().item()),
        schema_version=initial.schema_version,
        bias=float(initial.bias),
        metadata=dict(metadata or initial.metadata),
    )
    return learned, {"pair_count": int(pair_count), "final_loss": final_loss, "epochs": int(epochs)}


def _collect_proxy_pairwise_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_top_n: int,
    rollout_steps: int,
    map_ablation_mode: str,
    supervision_mode: str,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
    mode = _normalize_proxy_supervision_mode(supervision_mode)
    positive_features: list[list[float]] = []
    negative_features: list[list[float]] = []
    eligible_row_count = 0
    candidate_count = 0
    preference_step_count = 0
    oracle_preference_step_count = 0
    structure_preference_step_count = 0
    oracle_read_row_count = 0
    oracle_ordered_row_count = 0
    oracle_hit_rows: set[int] = set()
    digest = hashlib.sha1(mode.encode("utf-8"))

    for row_index, row in enumerate(rows):
        candidates = (
            _structure_only_row_candidates(row, candidate_top_n=candidate_top_n)
            if mode == SUPERVISION_MODE_STRUCTURE_ONLY
            else _row_candidates(row, candidate_top_n=candidate_top_n)
        )
        candidate_count += len(candidates)
        if len(candidates) < 2:
            continue
        eligible_row_count += 1
        soft_state = _initial_soft_state_from_row(row)
        selected_steps: list[dict[str, Any]] = []
        remaining = list(range(len(candidates)))
        if mode == SUPERVISION_MODE_LEGACY_HYBRID:
            oracle_keys = _oracle_ordered_keys(row, candidates)
            oracle_read_row_count += 1
            oracle_ordered_row_count += int(bool(oracle_keys))
        else:
            oracle_keys = None
        pool_max_token_cost = max([_token_cost(candidate) for candidate in candidates] or [1])
        for step_index in range(max(1, int(rollout_steps))):
            if len(remaining) < 2:
                break
            remaining_candidates = [candidates[idx] for idx in remaining]
            has_oracle_hit = bool(
                mode == SUPERVISION_MODE_LEGACY_HYBRID
                and oracle_keys
                and any(_candidate_key(candidate) in set(oracle_keys) for candidate in remaining_candidates)
            )
            preferences = build_proxy_pairwise_preferences(
                remaining_candidates,
                selected_steps=selected_steps,
                soft_state=soft_state,
                oracle_ordered_keys=oracle_keys,
                token_budget=None,
                pool_max_token_cost=pool_max_token_cost,
                map_ablation_mode=map_ablation_mode,
                supervision_mode=mode,
            )
            if not preferences:
                break
            preference_step_count += 1
            oracle_preference_step_count += int(has_oracle_hit)
            structure_preference_step_count += int(not has_oracle_hit)
            if has_oracle_hit:
                oracle_hit_rows.add(row_index)
            feature_cache = [
                extract_marginal_features(
                    candidate,
                    selected_steps=selected_steps,
                    soft_state=soft_state,
                    token_budget=None,
                    pool_max_token_cost=pool_max_token_cost,
                    map_ablation_mode=map_ablation_mode,
                )
                for candidate in remaining_candidates
            ]
            winner_local = int(preferences[0][0])
            winner_global = remaining[winner_local]
            for local_pos, local_neg in preferences:
                positive = _feature_vector(feature_cache[local_pos])
                negative = _feature_vector(feature_cache[local_neg])
                positive_features.append(positive)
                negative_features.append(negative)
                digest.update(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "step": step_index,
                            "positive_index": remaining[local_pos],
                            "negative_index": remaining[local_neg],
                            "positive_features": positive,
                            "negative_features": negative,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            selected_record = _proxy_selected_record(candidates[winner_global], soft_state=soft_state)
            selected_steps.append(selected_record)
            if selected_record.get("atom_id"):
                soft_state = update_soft_state_from_relation(
                    soft_state,
                    atom_id=str(selected_record.get("atom_id") or ""),
                    relation=str(selected_record.get("relation") or ""),
                )
            remaining = [idx for idx in remaining if idx != winner_global]

    metrics = {
        "supervision_mode": mode,
        "supervision_fingerprint": digest.hexdigest()[:12],
        "row_count": int(len(rows)),
        "eligible_row_count": int(eligible_row_count),
        "candidate_count": int(candidate_count),
        "preference_step_count": int(preference_step_count),
        "pair_count": int(len(positive_features)),
        "oracle_read_row_count": int(oracle_read_row_count),
        "oracle_ordered_row_count": int(oracle_ordered_row_count),
        "oracle_hit_row_count": int(len(oracle_hit_rows)),
        "oracle_preference_step_count": int(oracle_preference_step_count),
        "structure_preference_step_count": int(structure_preference_step_count),
        "gold_label_read_count": 0,
        "teacher_read_count": 0,
        "utility_read_count": 0,
        "reward_read_count": 0,
    }
    return positive_features, negative_features, metrics


def _normalize_proxy_supervision_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in SUPPORTED_PROXY_SUPERVISION_MODES:
        supported = ", ".join(sorted(SUPPORTED_PROXY_SUPERVISION_MODES))
        raise ValueError(f"unsupported proxy supervision_mode {value!r}; expected one of: {supported}")
    return mode


def train_learned_marginal_reward_weights(
    reward_rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 30,
    learning_rate: float = 0.03,
    pairwise_weight: float = 1.0,
    listwise_weight: float = 0.2,
    huber_weight: float = 0.2,
    prior_weight: float = 0.02,
    soft_tau: float = 0.3,
    pairwise_eps: float = 1.0e-6,
    max_pairs_per_group: int = 64,
    prior_weights: LearnedMarginalWeights | Mapping[str, Any] | None = None,
) -> tuple[LearnedMarginalWeights, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    groups = _reward_training_groups(reward_rows)
    pair_pos: list[list[float]] = []
    pair_neg: list[list[float]] = []
    group_features: list[list[list[float]]] = []
    group_deltas: list[list[float]] = []
    all_deltas: list[float] = []

    for rows in groups.values():
        if len(rows) < 2:
            continue
        features = [_feature_vector(_reward_row_features(row)) for row in rows]
        deltas = [_float_or_default(row.get("delta_margin"), 0.0) for row in rows]
        group_features.append(features)
        group_deltas.append(deltas)
        all_deltas.extend(deltas)

        ordered = sorted(range(len(rows)), key=lambda idx: (deltas[idx], -idx), reverse=True)
        added = 0
        for pos_rank, pos_idx in enumerate(ordered):
            for neg_idx in ordered[pos_rank + 1 :]:
                if deltas[pos_idx] <= deltas[neg_idx] + float(pairwise_eps):
                    continue
                pair_pos.append(features[pos_idx])
                pair_neg.append(features[neg_idx])
                added += 1
                if int(max_pairs_per_group) > 0 and added >= int(max_pairs_per_group):
                    break
            if int(max_pairs_per_group) > 0 and added >= int(max_pairs_per_group):
                break

    if not group_features or not all_deltas:
        weights = initial_learned_marginal_reward_weights()
        return weights, {
            "row_count": int(len(reward_rows)),
            "group_count": 0,
            "pair_count": 0,
            "final_loss": 0.0,
            "epochs": 0,
        }

    delta_scale = _robust_delta_scale(all_deltas)
    initial = _coerce_weights(prior_weights) if prior_weights is not None else initial_learned_marginal_reward_weights()
    theta = torch.nn.Parameter(torch.tensor([
        _inverse_softplus(float(initial.feature_weights.get(name, 0.0)))
        for name in POSITIVE_FEATURE_NAMES
    ], dtype=torch.float32))
    theta_cost = torch.nn.Parameter(torch.tensor(_inverse_softplus(initial.cost_weight), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(float(getattr(initial, "bias", 0.0)), dtype=torch.float32))
    optimizer = torch.optim.Adam([theta, theta_cost, bias], lr=float(learning_rate))

    pos_tensor = torch.tensor(pair_pos, dtype=torch.float32) if pair_pos else None
    neg_tensor = torch.tensor(pair_neg, dtype=torch.float32) if pair_neg else None
    feature_tensors = [torch.tensor(group, dtype=torch.float32) for group in group_features]
    delta_tensors = [torch.tensor(deltas, dtype=torch.float32) for deltas in group_deltas]
    target_tensors = [
        torch.clamp(delta_tensor / float(delta_scale), min=-5.0, max=5.0)
        for delta_tensor in delta_tensors
    ]
    prior = initial.normalized()
    prior_feature_weights = torch.tensor([
        float(prior.feature_weights.get(name, 0.0)) for name in POSITIVE_FEATURE_NAMES
    ], dtype=torch.float32)
    prior_cost_weight = torch.tensor(float(prior.cost_weight), dtype=torch.float32)

    final_loss = 0.0
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad()
        positive_weights = F.softplus(theta)
        cost_weight = F.softplus(theta_cost)
        loss = torch.tensor(0.0, dtype=torch.float32)

        if pos_tensor is not None and neg_tensor is not None and pos_tensor.numel() > 0:
            pos_score = _score_feature_tensor(pos_tensor, positive_weights, cost_weight, bias)
            neg_score = _score_feature_tensor(neg_tensor, positive_weights, cost_weight, bias)
            loss = loss + float(pairwise_weight) * F.softplus(-(pos_score - neg_score)).mean()

        if float(listwise_weight) > 0.0 or float(huber_weight) > 0.0:
            listwise_losses: list[torch.Tensor] = []
            huber_losses: list[torch.Tensor] = []
            for feature_tensor, delta_tensor, target_tensor in zip(feature_tensors, delta_tensors, target_tensors):
                scores = _score_feature_tensor(feature_tensor, positive_weights, cost_weight, bias)
                if float(listwise_weight) > 0.0:
                    targets = torch.softmax(delta_tensor / max(float(soft_tau), 1.0e-6), dim=0)
                    log_probs = torch.log_softmax(scores, dim=0)
                    listwise_losses.append(-(targets * log_probs).sum())
                if float(huber_weight) > 0.0:
                    huber_losses.append(F.smooth_l1_loss(scores, target_tensor))
            if listwise_losses:
                loss = loss + float(listwise_weight) * torch.stack(listwise_losses).mean()
            if huber_losses:
                loss = loss + float(huber_weight) * torch.stack(huber_losses).mean()

        if float(prior_weight) > 0.0:
            prior_loss = F.mse_loss(positive_weights, prior_feature_weights) + F.mse_loss(cost_weight, prior_cost_weight)
            loss = loss + float(prior_weight) * prior_loss

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    learned = LearnedMarginalWeights(
        feature_weights={
            name: float(F.softplus(theta).detach().cpu()[idx].item())
            for idx, name in enumerate(POSITIVE_FEATURE_NAMES)
        },
        cost_weight=float(F.softplus(theta_cost).detach().cpu().item()),
        schema_version=REWARD_WEIGHT_SCHEMA_VERSION,
        bias=float(bias.detach().cpu().item()),
        metadata={
            "trained_from": "verifier_delta_margin_reward",
            "row_count": int(len(reward_rows)),
            "group_count": int(len(group_features)),
            "pair_count": int(len(pair_pos)),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "delta_scale": float(delta_scale),
            "loss_weights": {
                "pairwise": float(pairwise_weight),
                "listwise": float(listwise_weight),
                "huber": float(huber_weight),
                "prior": float(prior_weight),
            },
            "soft_tau": float(soft_tau),
        },
    )
    return learned, {
        "row_count": int(len(reward_rows)),
        "group_count": int(len(group_features)),
        "pair_count": int(len(pair_pos)),
        "final_loss": final_loss,
        "epochs": int(epochs),
        "delta_scale": float(delta_scale),
    }


def evaluate_learned_marginal_reward_weights(
    reward_rows: Sequence[Mapping[str, Any]],
    weights: LearnedMarginalWeights | Mapping[str, Any],
) -> dict[str, Any]:
    groups = _reward_training_groups(reward_rows)
    total_pairs = 0
    correct_pairs = 0
    total_groups = 0
    top1_matches = 0
    scored_rows = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        scored = [
            (
                score_marginal_features(_reward_row_features(row), weights),
                _float_or_default(row.get("delta_margin"), 0.0),
                idx,
            )
            for idx, row in enumerate(rows)
        ]
        scored_rows += len(scored)
        best_score_idx = max(range(len(scored)), key=lambda idx: (scored[idx][0], -idx))
        best_delta_idx = max(range(len(scored)), key=lambda idx: (scored[idx][1], -idx))
        top1_matches += int(best_score_idx == best_delta_idx)
        total_groups += 1
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                delta_i = scored[i][1]
                delta_j = scored[j][1]
                if abs(delta_i - delta_j) <= 1.0e-6:
                    continue
                total_pairs += 1
                score_i = scored[i][0]
                score_j = scored[j][0]
                correct_pairs += int((delta_i > delta_j and score_i > score_j) or (delta_j > delta_i and score_j > score_i))
    return {
        "row_count": int(len(reward_rows)),
        "scored_row_count": int(scored_rows),
        "group_count": int(total_groups),
        "pair_count": int(total_pairs),
        "pair_accuracy": float(correct_pairs / total_pairs) if total_pairs else 0.0,
        "step_top1_match": float(top1_matches / total_groups) if total_groups else 0.0,
    }


def _coerce_weights(weights: LearnedMarginalWeights | Mapping[str, Any]) -> LearnedMarginalWeights:
    if isinstance(weights, LearnedMarginalWeights):
        return weights.normalized()
    feature_weights = weights.get("feature_weights") if isinstance(weights, Mapping) else {}
    return LearnedMarginalWeights(
        feature_weights=dict(feature_weights or {}),
        cost_weight=_float_or_default(weights.get("cost_weight") if isinstance(weights, Mapping) else 0.0, 0.0),
        schema_version=str(weights.get("schema_version") or WEIGHT_SCHEMA_VERSION) if isinstance(weights, Mapping) else WEIGHT_SCHEMA_VERSION,
        bias=_float_or_default(weights.get("bias") if isinstance(weights, Mapping) else 0.0, 0.0),
    ).normalized()


def _score_feature_tensor(features: Any, positive_weights: Any, cost_weight: Any, bias: Any) -> Any:
    return (features[:, :-1] * positive_weights).sum(dim=1) - features[:, -1] * cost_weight + bias


def _reward_training_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        event_id = _compact(row.get("event_id") or "")
        if not event_id:
            continue
        try:
            step = int(row.get("step", 0))
        except (TypeError, ValueError):
            continue
        features = _reward_row_features(row)
        if not features:
            continue
        if row.get("delta_margin") is None:
            continue
        groups.setdefault((event_id, step), []).append(row)
    return groups


def _reward_row_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("mrec_features", "utility_features", "features"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {
        name: row.get(name)
        for name in FEATURE_NAMES
        if row.get(name) is not None
    }


def _robust_delta_scale(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 1.0
    median = ordered[len(ordered) // 2]
    abs_devs = sorted(abs(value - median) for value in ordered)
    mad = abs_devs[len(abs_devs) // 2] if abs_devs else 0.0
    if mad > 1.0e-6:
        return max(1.4826 * mad, 1.0e-3)
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / max(len(ordered), 1)
    return max(math.sqrt(variance), 1.0e-3)


def _feature_vector(features: Mapping[str, Any]) -> list[float]:
    return [_clip01(_float_or_default(features.get(name), 0.0)) for name in POSITIVE_FEATURE_NAMES] + [
        _clip01(_float_or_default(features.get("cost_ratio"), 0.0))
    ]


def _row_candidates(row: Mapping[str, Any], *, candidate_top_n: int) -> list[dict[str, Any]]:
    raw_candidates = row.get("candidate_pool") or row.get("candidates") or row.get("selected_candidates") or []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_candidates):
        if int(candidate_top_n) > 0 and len(out) >= int(candidate_top_n):
            break
        if not isinstance(raw, Mapping):
            continue
        candidate = dict(raw)
        candidate.setdefault("selector_candidate_idx", idx)
        candidate.setdefault("candidate_idx", idx)
        candidate.setdefault("evidence_id", str(candidate.get("candidate_uid") or f"E{idx + 1:02d}"))
        out.append(candidate)
    return out


def _structure_only_row_candidates(row: Mapping[str, Any], *, candidate_top_n: int) -> list[dict[str, Any]]:
    """Project candidates onto deployment-time structural fields only."""
    raw_candidates = row.get("candidate_pool") or row.get("candidates") or row.get("selected_candidates") or []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_candidates):
        if int(candidate_top_n) > 0 and len(out) >= int(candidate_top_n):
            break
        if not isinstance(raw, Mapping):
            continue
        candidate = {
            key: raw.get(key)
            for key in _STRUCTURE_ONLY_CANDIDATE_FIELDS
            if raw.get(key) is not None
        }
        raw_alignments = raw.get("candidate_atom_alignments")
        if isinstance(raw_alignments, (list, tuple)):
            candidate["candidate_atom_alignments"] = [
                {
                    key: alignment.get(key)
                    for key in _STRUCTURE_ONLY_ALIGNMENT_FIELDS
                    if alignment.get(key) is not None
                }
                for alignment in raw_alignments
                if isinstance(alignment, Mapping)
            ]
        candidate.setdefault("selector_candidate_idx", idx)
        candidate.setdefault("candidate_idx", idx)
        candidate.setdefault("evidence_id", str(candidate.get("candidate_uid") or f"E{idx + 1:02d}"))
        out.append(candidate)
    return out


def _initial_soft_state_from_row(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw_atoms = (row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or []
    atom_ids: list[str] = []
    for idx, atom in enumerate(raw_atoms, start=1):
        if not isinstance(atom, Mapping):
            continue
        atom_id = _compact(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        if atom_id:
            atom_ids.append(atom_id)
    if not atom_ids:
        atom_ids.append("A1")
    return {atom_id: {"U": 1.0} for atom_id in atom_ids}


def _oracle_ordered_keys(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [str(key) for key in row.get("oracle_ordered_keys") or [] if str(key)]
    if keys:
        return keys
    out: list[str] = []
    for idx in row.get("oracle_ordered_indices") or []:
        try:
            candidate = candidates[int(idx)]
        except (IndexError, TypeError, ValueError):
            continue
        key = _candidate_key(candidate)
        if key:
            out.append(key)
    return out


def _proxy_selected_record(candidate: Mapping[str, Any], *, soft_state: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    pairs = _candidate_atom_pairs(candidate, soft_state)
    if pairs:
        pairs.sort(
            key=lambda pair: (
                -_directness_factor(pair.get("directness")),
                -_clip01(_float_or_default(pair.get("confidence"), _float_or_default(candidate.get("map_confidence"), 0.0))),
                str(pair.get("atom_id") or ""),
            )
        )
        pair = pairs[0]
        atom_id = str(pair.get("atom_id") or "")
        relation = _relation_group(pair.get("relation"))
    else:
        atom_id = next(iter(soft_state.keys()), "")
        relation = _relation_group(candidate.get("map_relation") or candidate.get("relation") or "")
    return {
        "atom_id": atom_id,
        "relation": relation,
        "evidence_text": str(candidate.get("text") or candidate.get("evidence_text") or ""),
        "duplicate_group": _compact(candidate.get("duplicate_group") or ""),
        "source_group": _source_key(candidate),
    }


def _candidate_atom_pairs(candidate: Mapping[str, Any], soft_state: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    pair_rows = candidate.get("candidate_atom_alignments") or []
    if isinstance(pair_rows, (list, tuple)):
        candidate_eid = _compact(candidate.get("evidence_id") or candidate.get("candidate_uid") or "")
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in pair_rows:
            if not isinstance(raw, Mapping):
                continue
            evidence_id = _compact(raw.get("evidence_id") or "")
            if evidence_id and candidate_eid and evidence_id != candidate_eid:
                continue
            atom_id = _compact(raw.get("atom_id") or "")
            if atom_id not in soft_state:
                continue
            relation = _relation_group(raw.get("relation"))
            directness = _compact(raw.get("directness") or "none").lower()
            key = (atom_id, relation, directness)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "atom_id": atom_id,
                    "relation": relation,
                    "directness": directness,
                    "confidence": raw.get("confidence"),
                }
            )
        if out:
            return out

    covered_atoms = [atom_id for atom_id in _string_list(candidate.get("covered_atom_ids")) if atom_id in soft_state]
    relation = _relation_group(candidate.get("map_relation") or candidate.get("relation") or "")
    directness = _compact(candidate.get("map_directness") or candidate.get("directness") or "none").lower()
    return [
        {
            "atom_id": atom_id,
            "relation": relation,
            "directness": directness,
            "confidence": candidate.get("map_confidence"),
        }
        for atom_id in covered_atoms
    ]


def _selected_atom_ids(selected_steps: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for step in selected_steps:
        atom_id = _compact(step.get("atom_id") or "")
        if atom_id:
            out.add(atom_id)
        out.update(_string_list(step.get("covered_atom_ids")))
    return out


def _selected_relations_by_atom(selected_steps: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for step in selected_steps:
        relation = _relation_group(step.get("relation") or step.get("map_relation") or "")
        atom_ids = _string_list(step.get("covered_atom_ids"))
        atom_id = _compact(step.get("atom_id") or "")
        if atom_id and atom_id not in atom_ids:
            atom_ids.append(atom_id)
        for item in atom_ids:
            if item and relation:
                out.setdefault(item, set()).add(relation)
    return out


def _selected_sources(selected_steps: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        _source_key(step)
        for step in selected_steps
        if _source_key(step)
    }


def _selected_texts(selected_steps: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        _normalize_text(step.get("text") or step.get("evidence_text") or "")
        for step in selected_steps
        if _normalize_text(step.get("text") or step.get("evidence_text") or "")
    }


def _selected_duplicate_groups(selected_steps: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        _compact(step.get("duplicate_group") or "")
        for step in selected_steps
        if _compact(step.get("duplicate_group") or "")
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_key", "candidate_uid", "evidence_id"):
        value = _compact(candidate.get(key) or "")
        if value:
            return value
    return ""


def _source_key(candidate: Mapping[str, Any]) -> str:
    for key in ("source_group", "source_report", "report_id", "source_id"):
        value = _compact(candidate.get(key) or "")
        if value:
            return value
    candidate_key = _compact(candidate.get("candidate_key") or "")
    if ":" in candidate_key:
        return candidate_key.split(":", 1)[0]
    return ""


def _relation_group(value: Any) -> str:
    relation = _compact(value).lower()
    return _RELATION_GROUPS.get(relation, relation)


def _state_for_relation(value: Any) -> str:
    return _RELATION_TO_STATE.get(_relation_group(value), "U")


def _has_stance_tension(relation_state: str, seen_states: set[str]) -> bool:
    if relation_state == "Q" and seen_states & {"S", "R", "Q", "C"}:
        return True
    if relation_state == "S" and seen_states & {"R", "C"}:
        return True
    if relation_state == "R" and seen_states & {"S", "C"}:
        return True
    return False


def _entropy_reduction_for_pair(state_dist: Mapping[str, float], *, relation_state: str, gain: float) -> float:
    before = _state_distribution(state_dist)
    after = dict(before)
    moved = min(after.get("U", 0.0), _clip01(gain))
    after["U"] = max(0.0, after.get("U", 0.0) - moved)
    after[relation_state] = after.get(relation_state, 0.0) + moved
    # In this selector the main uncertainty is unresolved atom mass, not a
    # calibrated stance distribution. Treat moved unresolved mass as the local
    # entropy proxy so a direct resolving pair is learnably preferred.
    return _clip01(before.get("U", 0.0) - after.get("U", 0.0))


def _state_distribution(value: Mapping[str, Any]) -> dict[str, float]:
    dist = {str(key).upper(): max(0.0, _float_or_default(raw, 0.0)) for key, raw in value.items()}
    total = sum(dist.values())
    if total <= 0.0:
        return {"U": 1.0}
    return {key: val / total for key, val in dist.items() if val > 0.0}


def _normalized_entropy(dist: Mapping[str, float]) -> float:
    values = [max(0.0, float(value)) for value in dist.values() if float(value) > 0.0]
    if len(values) <= 1:
        return 0.0 if values and values[0] >= 0.999 else 1.0
    entropy = -sum(value * math.log(value) for value in values)
    return _clip01(entropy / math.log(max(len(values), 2)))


def _best_directness(candidate: Mapping[str, Any], soft_state: Mapping[str, Mapping[str, float]]) -> float:
    pairs = _candidate_atom_pairs(candidate, soft_state)
    if not pairs:
        return _directness_factor(candidate.get("map_directness") or candidate.get("directness") or "")
    return max(_directness_factor(pair.get("directness")) for pair in pairs)


def _directness_factor(value: Any) -> float:
    directness = _compact(value).lower()
    return _DIRECTNESS_FACTOR.get(directness, 0.0)


def _retrieval_score(candidate: Mapping[str, Any]) -> float:
    for key in ("hybrid_score", "baseline_hybrid_score", "base_score"):
        if candidate.get(key) is not None:
            return _clip01(_float_or_default(candidate.get(key), 0.0))
    return 0.0


def _token_cost(candidate: Mapping[str, Any]) -> int:
    for key in ("mrec_token_cost", "token_cost", "prompt_token_count", "evidence_token_count"):
        if candidate.get(key) is not None:
            return max(0, _int_or_default(candidate.get(key), 0))
    text = str(candidate.get("text") or candidate.get("evidence_text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-6)
    if value > 20:
        return value
    return math.log(math.exp(value) - 1.0)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return []


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_text(value: Any) -> str:
    return _compact(value).lower()


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
