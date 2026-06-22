from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


WEIGHT_SCHEMA_VERSION = "mrec_learned_marginal_proxy_weights_v0_2"
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
    "context": 0.25,
    "none": 0.0,
}


@dataclass(frozen=True)
class LearnedMarginalWeights:
    feature_weights: dict[str, float]
    cost_weight: float
    schema_version: str = WEIGHT_SCHEMA_VERSION
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


def load_learned_marginal_weights(path: str | Path | None, *, allow_default: bool = True) -> LearnedMarginalWeights:
    if path is None or not str(path):
        if allow_default:
            return initial_learned_marginal_weights()
        raise ValueError("learned_marginal_proxy requires a weight file")
    return _load_learned_marginal_weights_cached(str(path)).normalized()


@lru_cache(maxsize=16)
def _load_learned_marginal_weights_cached(path: str) -> LearnedMarginalWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != WEIGHT_SCHEMA_VERSION:
        raise ValueError(f"unsupported learned marginal weight schema: {payload.get('schema_version')!r}")
    feature_weights = payload.get("feature_weights") or {}
    if not isinstance(feature_weights, Mapping):
        raise ValueError("learned marginal weight file has invalid feature_weights")
    return LearnedMarginalWeights(
        feature_weights={name: _float_or_default(feature_weights.get(name), 0.0) for name in POSITIVE_FEATURE_NAMES},
        cost_weight=_float_or_default(payload.get("cost_weight"), 0.0),
        schema_version=str(payload.get("schema_version") or WEIGHT_SCHEMA_VERSION),
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
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def score_marginal_features(features: Mapping[str, Any], weights: LearnedMarginalWeights | Mapping[str, Any]) -> float:
    learned = _coerce_weights(weights)
    score = 0.0
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
) -> list[int]:
    oracle_rank = {str(key): idx for idx, key in enumerate(oracle_ordered_keys or []) if str(key)}
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
) -> list[tuple[int, int]]:
    order = rank_candidates_by_proxy(
        candidates,
        selected_steps=selected_steps,
        soft_state=soft_state,
        oracle_ordered_keys=oracle_ordered_keys,
        token_budget=token_budget,
        pool_max_token_cost=pool_max_token_cost,
    )
    if len(order) < 2:
        return []
    best = int(order[0])
    return [(best, int(other)) for other in order[1:]]


def train_learned_marginal_proxy_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 50,
    learning_rate: float = 0.05,
    candidate_top_n: int = 20,
    rollout_steps: int = 5,
) -> tuple[LearnedMarginalWeights, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    positive_features: list[list[float]] = []
    negative_features: list[list[float]] = []
    for row in rows:
        candidates = _row_candidates(row, candidate_top_n=candidate_top_n)
        if len(candidates) < 2:
            continue
        soft_state = _initial_soft_state_from_row(row)
        selected_steps: list[dict[str, Any]] = []
        remaining = list(range(len(candidates)))
        oracle_keys = _oracle_ordered_keys(row, candidates)
        pool_max_token_cost = max([_token_cost(candidate) for candidate in candidates] or [1])
        for _ in range(max(1, int(rollout_steps))):
            if len(remaining) < 2:
                break
            remaining_candidates = [candidates[idx] for idx in remaining]
            preferences = build_proxy_pairwise_preferences(
                remaining_candidates,
                selected_steps=selected_steps,
                soft_state=soft_state,
                oracle_ordered_keys=oracle_keys,
                token_budget=None,
                pool_max_token_cost=pool_max_token_cost,
            )
            if not preferences:
                break
            feature_cache = [
                extract_marginal_features(
                    candidate,
                    selected_steps=selected_steps,
                    soft_state=soft_state,
                    token_budget=None,
                    pool_max_token_cost=pool_max_token_cost,
                )
                for candidate in remaining_candidates
            ]
            for local_pos, local_neg in preferences:
                positive_features.append(_feature_vector(feature_cache[local_pos]))
                negative_features.append(_feature_vector(feature_cache[local_neg]))
            winner_local = preferences[0][0]
            winner_global = remaining[winner_local]
            selected_record = _proxy_selected_record(candidates[winner_global], soft_state=soft_state)
            selected_steps.append(selected_record)
            if selected_record.get("atom_id"):
                soft_state = update_soft_state_from_relation(
                    soft_state,
                    atom_id=str(selected_record.get("atom_id") or ""),
                    relation=str(selected_record.get("relation") or ""),
                )
            remaining = [idx for idx in remaining if idx != winner_global]

    pair_count = len(positive_features)
    if pair_count == 0:
        weights = initial_learned_marginal_weights()
        return weights, {"pair_count": 0, "final_loss": 0.0, "epochs": 0}

    pos_tensor = torch.tensor(positive_features, dtype=torch.float32)
    neg_tensor = torch.tensor(negative_features, dtype=torch.float32)
    initial = initial_learned_marginal_weights()
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
        metadata={
            "trained_from": "proxy_pairwise",
            "pair_count": int(pair_count),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
        },
    )
    return learned, {"pair_count": int(pair_count), "final_loss": final_loss, "epochs": int(epochs)}


def _coerce_weights(weights: LearnedMarginalWeights | Mapping[str, Any]) -> LearnedMarginalWeights:
    if isinstance(weights, LearnedMarginalWeights):
        return weights.normalized()
    feature_weights = weights.get("feature_weights") if isinstance(weights, Mapping) else {}
    return LearnedMarginalWeights(
        feature_weights=dict(feature_weights or {}),
        cost_weight=_float_or_default(weights.get("cost_weight") if isinstance(weights, Mapping) else 0.0, 0.0),
    ).normalized()


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
