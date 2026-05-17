from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from fact_checking.build.candidates import (
    _chunk_mmr_config_fingerprint,
    _load_pickle,
    canonicalize_sentence,
    compute_hybrid_scores,
)


RETAINED_LABELS = ("pants-fire", "false", "barely-true", "half-true")
TRUE_SIDE_LABELS = ("mostly-true", "true")

DEFAULT_FEATURE_NAMES = [
    "dense_score",
    "lexical_score",
    "bm25_score",
    "hybrid_score",
    "rank_by_hybrid",
    "rank_norm",
    "n_candidates",
    "candidate_text_len",
    "candidate_word_count",
    "claim_candidate_dense",
    "mean_sim_to_pool",
    "max_sim_to_pool",
    "same_report_count",
    "source_report_count",
    "oracle_pool_size",
]


@dataclass
class CandidatePool:
    event_id: str
    claim: str
    gold_label: str
    candidates: list[dict[str, Any]]
    features: list[dict[str, float]]
    positive_local_indices: set[int]
    matched_positive_count: int
    oracle_positive_count: int
    source_candidate_count: int
    pool_mode: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_build_config(
    config_path: str,
    *,
    config_overrides: str | None = None,
    model_base_path: str | None = None,
) -> dict[str, Any]:
    """Load the build config the same way oracle search does.

    This intentionally avoids Hydra compose because the plan uses an explicit
    experiment YAML path.
    """
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "configs" / "build" / "default.yaml"
    exp_path = Path(config_path)
    if not exp_path.is_absolute():
        exp_path = project_root / exp_path
    default_cfg = OmegaConf.to_container(OmegaConf.load(default_path), resolve=False)
    exp_cfg = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=False)
    build_default = dict(default_cfg.get("build", {}) or {})
    build_exp = dict(exp_cfg.get("build", {}) or {})
    build_cfg = _deep_merge(build_default, build_exp)

    if config_overrides:
        for override in config_overrides.split(","):
            override = override.strip()
            if not override or "=" not in override:
                continue
            key_path, value = override.split("=", 1)
            target = build_cfg
            keys = key_path.split(".")
            if keys and keys[0] == "build":
                keys = keys[1:]
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = _parse_scalar(value)

    if model_base_path:
        retrieval = build_cfg.get("retrieval", {}) or {}
        for key in ("embedder_model",):
            if retrieval.get(key):
                retrieval[key] = _resolve_model_path(str(retrieval[key]), model_base_path)
        chunking = retrieval.get("chunking", {}) or {}
        if chunking.get("embedder_model"):
            chunking["embedder_model"] = _resolve_model_path(
                str(chunking["embedder_model"]), model_base_path
            )
        prompt = build_cfg.get("prompt", {}) or {}
        if prompt.get("model_name_or_path"):
            prompt["model_name_or_path"] = _resolve_model_path(
                str(prompt["model_name_or_path"]), model_base_path
            )

    return build_cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _resolve_model_path(raw: str, base_path: str) -> str:
    if raw.startswith("/data/models/"):
        return raw.replace("/data/models/", base_path.rstrip("/") + "/", 1)
    return raw


def resolve_chunk_cache_path(
    build_cfg: dict[str, Any],
    *,
    split: str,
    cache_root: str | Path = "outputs/cache/chunk_mmr",
    explicit_path: str | None = None,
    allow_single_fallback: bool = True,
) -> tuple[Path, dict[str, Any]]:
    if explicit_path:
        path = Path(explicit_path)
        return path, {"mode": "explicit", "path": str(path)}

    cache_root = Path(cache_root)
    fp = _chunk_mmr_config_fingerprint(build_cfg)
    resolved = cache_root / fp / f"{split}.pkl"
    if resolved.exists():
        return resolved, {"mode": "fingerprint", "fingerprint": fp, "path": str(resolved)}

    matches = sorted(cache_root.glob(f"*/{split}.pkl"))
    if allow_single_fallback and len(matches) == 1:
        return matches[0], {
            "mode": "single_available_fallback",
            "expected_fingerprint": fp,
            "path": str(matches[0]),
        }
    raise FileNotFoundError(
        f"Chunk-MMR cache not found at {resolved}. "
        f"Found {len(matches)} candidate {split}.pkl files under {cache_root}; "
        "pass --chunk-mmr-cache explicitly."
    )


def oracle_filter_passes(rec: dict[str, Any], preset: str) -> bool:
    if preset == "all":
        return True
    if preset != "v1a":
        raise ValueError(f"Unknown filter preset: {preset}")
    return (
        bool(rec.get("is_correct"))
        and str(rec.get("gold_label", "")).lower() in RETAINED_LABELS
        and float(rec.get("final_logprob", -1e9)) >= -0.5
        and int(rec.get("n_candidates", 0)) > 5
    )


def load_chunk_samples_by_event(cache_path: str | Path) -> dict[str, Any]:
    samples = _load_pickle(Path(cache_path))
    return {str(sample.event_id): sample for sample in samples}


def build_candidate_pool(
    sample: Any,
    oracle_rec: dict[str, Any],
    *,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    pool_mode: str = "oracle_n_top_hybrid_with_positives",
    fallback_pool_size: int = 15,
) -> CandidatePool:
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    if n <= 0:
        return CandidatePool(
            event_id=str(sample.event_id),
            claim=str(sample.claim),
            gold_label=str(sample.label),
            candidates=[],
            features=[],
            positive_local_indices=set(),
            matched_positive_count=0,
            oracle_positive_count=len(oracle_rec.get("selected_texts") or []),
            source_candidate_count=0,
            pool_mode=pool_mode,
        )

    source_indices = _dedup_source_indices(sample, scored["hybrid_scores"], n)
    positive_source_indices = _match_positive_indices(sample, source_indices, oracle_rec)
    oracle_n = int(oracle_rec.get("n_candidates") or fallback_pool_size)
    target_size = max(1, oracle_n if pool_mode.startswith("oracle_n") else fallback_pool_size)
    target_size = max(target_size, len(positive_source_indices))

    ranked_by_hybrid = sorted(
        source_indices,
        key=lambda idx: float(scored["hybrid_scores"][idx]),
        reverse=True,
    )
    selected_source: list[int] = []
    seen: set[int] = set()
    for idx in positive_source_indices:
        if idx not in seen:
            selected_source.append(idx)
            seen.add(idx)
    for idx in ranked_by_hybrid:
        if len(selected_source) >= target_size:
            break
        if idx not in seen:
            selected_source.append(idx)
            seen.add(idx)

    # Keep the model-facing pool ordered by hybrid rank; labels remain attached.
    selected_source.sort(key=lambda idx: float(scored["hybrid_scores"][idx]), reverse=True)
    positive_local = {i for i, idx in enumerate(selected_source) if idx in set(positive_source_indices)}
    features = _pool_features(sample, scored, selected_source, oracle_n=oracle_n)
    candidates = [dict(sample.candidates[idx]) for idx in selected_source]

    return CandidatePool(
        event_id=str(sample.event_id),
        claim=str(sample.claim),
        gold_label=str(sample.label),
        candidates=candidates,
        features=features,
        positive_local_indices=positive_local,
        matched_positive_count=len(positive_source_indices),
        oracle_positive_count=len(oracle_rec.get("selected_texts") or []),
        source_candidate_count=len(source_indices),
        pool_mode=pool_mode,
    )


def _dedup_source_indices(sample: Any, hybrid_scores: np.ndarray, n: int) -> list[int]:
    best_by_text: dict[str, int] = {}
    for idx in range(n):
        text = str(sample.candidates[idx].get("text", ""))
        key = canonicalize_sentence(text)
        if not key:
            continue
        old = best_by_text.get(key)
        if old is None or float(hybrid_scores[idx]) > float(hybrid_scores[old]):
            best_by_text[key] = idx
    return list(best_by_text.values())


def _match_positive_indices(sample: Any, source_indices: list[int], oracle_rec: dict[str, Any]) -> list[int]:
    selected_texts = [str(t) for t in (oracle_rec.get("selected_texts") or [])]
    by_key: dict[str, list[int]] = defaultdict(list)
    source_texts: dict[int, str] = {}
    for idx in source_indices:
        text = str(sample.candidates[idx].get("text", ""))
        source_texts[idx] = text
        by_key[canonicalize_sentence(text)].append(idx)

    matched: list[int] = []
    used: set[int] = set()
    for text in selected_texts:
        key = canonicalize_sentence(text)
        candidates = [idx for idx in by_key.get(key, []) if idx not in used]
        if not candidates:
            candidates = [
                idx for idx, source_text in source_texts.items()
                if idx not in used and _text_matches(text, source_text)
            ]
        if not candidates:
            continue
        idx = candidates[0]
        matched.append(idx)
        used.add(idx)
    return matched


def _text_matches(oracle_text: str, source_text: str) -> bool:
    a = canonicalize_sentence(oracle_text)
    b = canonicalize_sentence(source_text)
    if not a or not b:
        return False
    return a in b or b in a


def _pool_features(
    sample: Any,
    scored: dict[str, Any],
    selected_source: list[int],
    *,
    oracle_n: int,
) -> list[dict[str, float]]:
    if not selected_source:
        return []
    chunk_emb = np.asarray(scored["chunk_emb"], dtype=np.float32)
    emb = chunk_emb[selected_source]
    sim = emb @ emb.T if emb.size else np.zeros((len(selected_source), len(selected_source)), dtype=np.float32)
    report_counts = Counter(
        str(sample.candidates[idx].get("report_id") or _source_report_id(sample.candidates[idx]))
        for idx in selected_source
    )
    hybrid_scores = scored["hybrid_scores"]
    rank_map = {
        idx: rank
        for rank, idx in enumerate(
            sorted(selected_source, key=lambda i: float(hybrid_scores[i]), reverse=True)
        )
    }
    n_pool = len(selected_source)
    rows: list[dict[str, float]] = []
    for local_idx, source_idx in enumerate(selected_source):
        candidate = sample.candidates[source_idx]
        text = str(candidate.get("text", ""))
        if n_pool > 1:
            others = np.delete(sim[local_idx], local_idx)
            mean_sim = float(np.mean(others))
            max_sim = float(np.max(others))
        else:
            mean_sim = 0.0
            max_sim = 0.0
        report_id = str(candidate.get("report_id") or _source_report_id(candidate))
        rank = int(rank_map[source_idx])
        rows.append({
            "dense_score": float(scored["dense_scores"][source_idx]),
            "lexical_score": float(scored["lexical_scores"][source_idx]),
            "bm25_score": float(scored["bm25_scores"][source_idx]),
            "hybrid_score": float(scored["hybrid_scores"][source_idx]),
            "rank_by_hybrid": float(rank),
            "rank_norm": float(rank / max(n_pool - 1, 1)),
            "n_candidates": float(n_pool),
            "candidate_text_len": float(len(text)),
            "candidate_word_count": float(len(text.split())),
            "claim_candidate_dense": float(scored["dense_scores"][source_idx]),
            "mean_sim_to_pool": mean_sim,
            "max_sim_to_pool": max_sim,
            "same_report_count": float(report_counts[report_id]),
            "source_report_count": float(len(report_counts)),
            "oracle_pool_size": float(oracle_n),
        })
    return rows


def _source_report_id(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_report")
    if isinstance(source, dict):
        return str(source.get("report_id", ""))
    return ""


def pool_to_pointwise_rows(pool: CandidatePool, oracle_rec: dict[str, Any], filter_bucket: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (candidate, features) in enumerate(zip(pool.candidates, pool.features)):
        rows.append({
            "event_id": pool.event_id,
            "claim": pool.claim,
            "gold_label": pool.gold_label,
            "candidate_idx": i,
            "source_text": str(candidate.get("text", "")),
            "report_id": str(candidate.get("report_id") or _source_report_id(candidate)),
            "is_oracle_selected": int(i in pool.positive_local_indices),
            "features": {name: float(features.get(name, 0.0)) for name in DEFAULT_FEATURE_NAMES},
            "oracle_final_logprob": float(oracle_rec.get("final_logprob", 0.0)),
            "oracle_correct": bool(oracle_rec.get("is_correct")),
            "oracle_n_candidates": int(oracle_rec.get("n_candidates", 0)),
            "matched_positive_count": int(pool.matched_positive_count),
            "oracle_positive_count": int(pool.oracle_positive_count),
            "filter_bucket": filter_bucket,
            "pool_mode": pool.pool_mode,
        })
    return rows


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        feats = row.get("features", {})
        for j, name in enumerate(feature_names):
            x[i, j] = float(feats.get(name, 0.0))
    return x


def labels_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(row.get("is_oracle_selected", 0)) for row in rows], dtype=np.float32)


def group_rows_by_event(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row["event_id"])].append(i)
    return dict(grouped)


def split_event_ids_by_label(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    labels_by_event: dict[str, str] = {}
    for row in rows:
        labels_by_event.setdefault(str(row["event_id"]), str(row["gold_label"]))
    by_label: dict[str, list[str]] = defaultdict(list)
    for eid, label in labels_by_event.items():
        by_label[label].append(eid)

    rng = np.random.default_rng(seed)
    train: set[str] = set()
    val: set[str] = set()
    for _label, eids in by_label.items():
        shuffled = list(eids)
        rng.shuffle(shuffled)
        n_val = min(max(1, int(round(len(shuffled) * val_fraction))), max(len(shuffled) - 1, 0))
        val.update(shuffled[:n_val])
        train.update(shuffled[n_val:])
    return train, val


def compute_row_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    grouped = group_rows_by_event(rows)
    label_by_event = {eid: str(rows[idxs[0]]["gold_label"]) for eid, idxs in grouped.items()}
    label_counts = Counter(label_by_event.values())
    n_labels = max(len(label_counts), 1)
    label_weights = {
        label: len(grouped) / (n_labels * count)
        for label, count in label_counts.items()
    }
    weights = np.ones(len(rows), dtype=np.float32)
    for eid, idxs in grouped.items():
        ys = [int(rows[i].get("is_oracle_selected", 0)) for i in idxs]
        n_pos = max(sum(ys), 1)
        n_neg = max(len(ys) - sum(ys), 1)
        lw = float(label_weights[label_by_event[eid]])
        for i, y in zip(idxs, ys):
            weights[i] = lw * (0.5 / n_pos if y else 0.5 / n_neg)
    mean = float(weights.mean()) if weights.size else 1.0
    if mean > 0:
        weights /= mean
    return weights


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float32)
    if y.size == 0 or float(y.sum()) <= 0:
        return 0.0
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float32)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / y.sum())


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bce_loss(y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        return float((loss * w).sum() / max(float(w.sum()), 1e-8))
    return float(loss.mean())


def claim_selection_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    top_k: int,
    score_name: str,
) -> dict[str, Any]:
    grouped = group_rows_by_event(rows)
    per_claim: list[dict[str, Any]] = []
    per_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for eid, idxs in grouped.items():
        y = np.array([int(rows[i]["is_oracle_selected"]) for i in idxs], dtype=np.int32)
        if int(y.sum()) == 0:
            continue
        local_scores = np.asarray([scores[i] for i in idxs], dtype=np.float32)
        k = min(top_k, len(idxs))
        pred_local = set(np.argsort(-local_scores)[:k].tolist())
        true_local = {i for i, v in enumerate(y.tolist()) if v == 1}
        inter = len(pred_local & true_local)
        union = len(pred_local | true_local)
        item = {
            "event_id": eid,
            "gold_label": str(rows[idxs[0]]["gold_label"]),
            "recall_at_k": inter / max(len(true_local), 1),
            "precision_at_k": inter / max(k, 1),
            "jaccard_at_k": inter / max(union, 1),
            "n_candidates": len(idxs),
            "n_positive": len(true_local),
        }
        per_claim.append(item)
        per_label[item["gold_label"]].append(item)

    def _mean(items: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(x[key]) for x in items])) if items else 0.0

    label_metrics = {
        label: {
            "n_claims": len(items),
            "recall_at_k": _mean(items, "recall_at_k"),
            "precision_at_k": _mean(items, "precision_at_k"),
            "jaccard_at_k": _mean(items, "jaccard_at_k"),
        }
        for label, items in sorted(per_label.items())
    }
    return {
        "score_name": score_name,
        "n_claims": len(per_claim),
        "recall_at_k": _mean(per_claim, "recall_at_k"),
        "precision_at_k": _mean(per_claim, "precision_at_k"),
        "jaccard_at_k": _mean(per_claim, "jaccard_at_k"),
        "macro_recall_at_k": float(np.mean([m["recall_at_k"] for m in label_metrics.values()])) if label_metrics else 0.0,
        "macro_jaccard_at_k": float(np.mean([m["jaccard_at_k"] for m in label_metrics.values()])) if label_metrics else 0.0,
        "by_label": label_metrics,
    }


def selected_evidence_rows(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    grouped = group_rows_by_event(rows)
    out: list[dict[str, Any]] = []
    for eid, idxs in grouped.items():
        local_scores = np.asarray([scores[i] for i in idxs], dtype=np.float32)
        order = np.argsort(-local_scores)[: min(top_k, len(idxs))]
        selected = []
        for rank, local_idx in enumerate(order.tolist()):
            row = rows[idxs[local_idx]]
            selected.append({
                "rank": rank,
                "candidate_idx": int(row["candidate_idx"]),
                "score": float(local_scores[local_idx]),
                "is_oracle_selected": int(row["is_oracle_selected"]),
                "text": row.get("source_text", ""),
                "report_id": row.get("report_id", ""),
            })
        out.append({
            "event_id": eid,
            "gold_label": rows[idxs[0]]["gold_label"],
            "selected": selected,
        })
    return out


def summarize_filtering(
    oracle_records: list[dict[str, Any]],
    kept_event_ids: set[str],
    *,
    matched_counts: list[tuple[int, int]],
    pool_mode: str,
    cache_path: str,
) -> dict[str, Any]:
    before = Counter(str(r.get("gold_label", "")) for r in oracle_records)
    after = Counter(str(r.get("gold_label", "")) for r in oracle_records if str(r.get("event_id")) in kept_event_ids)
    total_pos = sum(total for _matched, total in matched_counts)
    matched = sum(matched for matched, _total in matched_counts)
    return {
        "pool_mode": pool_mode,
        "cache_path": cache_path,
        "n_oracle_records": len(oracle_records),
        "n_kept_claims": len(kept_event_ids),
        "labels_before": dict(before),
        "labels_after": dict(after),
        "positive_text_match": {
            "matched": int(matched),
            "total": int(total_pos),
            "rate": float(matched / total_pos) if total_pos else 0.0,
        },
    }


def finite_or_zero(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(value)
