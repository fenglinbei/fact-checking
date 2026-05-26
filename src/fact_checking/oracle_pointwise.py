from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from fact_checking.build.cache import chunk_mmr_config_fingerprint, load_pickle
from fact_checking.build.candidates import canonicalize_sentence, compute_hybrid_scores


RETAINED_LABELS = ("pants-fire", "false", "barely-true", "half-true")
TRUE_SIDE_LABELS = ("mostly-true", "true")
ANCHOR_WEIGHTS = {
    "mostly-true": 0.25,
    "true": 0.10,
}

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

PIPELINE_POOL_MODE = "pipeline_hybrid_topk"
LEGACY_POSITIVE_INJECTION_POOL_MODE = "oracle_n_top_hybrid_with_positives"


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
    chunk_mmr_fingerprint: str = ""
    candidate_pool_source: str = ""
    candidate_pool_fingerprint: str = ""
    candidate_pool_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PointwiseSelectorModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_build_config(
    config_path: str,
    *,
    config_overrides: str | None = None,
    model_base_path: str | None = None,
) -> dict[str, Any]:
    """Load an experiment build config, expanding Hydra defaults when possible."""
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "configs" / "build" / "default.yaml"
    exp_path = Path(config_path)
    if not exp_path.is_absolute():
        exp_path = project_root / exp_path
    build_cfg = _load_hydra_build_config(project_root, exp_path)
    if build_cfg is None:
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


def _load_hydra_build_config(project_root: Path, exp_path: Path) -> dict[str, Any] | None:
    experiment_dir = project_root / "configs" / "experiment"
    try:
        rel = exp_path.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        return None
    if len(rel.parts) != 1 or rel.suffix not in {".yaml", ".yml"}:
        return None
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
            cfg = compose(
                config_name="pipeline/default",
                overrides=[f"experiment={rel.stem}"],
            )
        build_cfg = OmegaConf.to_container(cfg.get("build", {}), resolve=False)
        return dict(build_cfg or {})
    except Exception:
        return None


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
    expected_fingerprint: str | None = None,
    allow_single_fallback: bool = False,
    allow_explicit_mismatch: bool = False,
) -> tuple[Path, dict[str, Any]]:
    fp = expected_fingerprint or chunk_mmr_config_fingerprint(build_cfg)
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Explicit Chunk-MMR cache not found: {path}")
        actual_fp = path.parent.name
        if actual_fp != fp and not allow_explicit_mismatch:
            raise ValueError(
                f"Explicit Chunk-MMR cache fingerprint mismatch for split={split}: "
                f"expected {fp}, got {actual_fp} from {path}. "
                "Use a config that resolves to the same cache fingerprint."
            )
        return path, {
            "mode": "explicit",
            "fingerprint": actual_fp,
            "expected_fingerprint": fp,
            "path": str(path),
        }

    cache_root = Path(cache_root)
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
    return supervision_policy_for_record(rec, preset).get("keep", False)


def supervision_policy_for_record(
    rec: dict[str, Any],
    preset: str,
    *,
    fixed_correct_event_ids: set[str] | None = None,
    true_side_anchor_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if preset == "all":
        return {
            "keep": True,
            "bucket": "all",
            "supervision_weight": 1.0,
            "anchor_source": "all",
        }
    if preset not in {"v1a", "v1b"}:
        raise ValueError(f"Unknown filter preset: {preset}")

    retained_keep = (
        bool(rec.get("is_correct"))
        and str(rec.get("gold_label", "")).lower() in RETAINED_LABELS
        and float(rec.get("final_logprob", -1e9)) >= -0.5
        and int(rec.get("n_candidates", 0)) > 5
    )
    if retained_keep:
        return {
            "keep": True,
            "bucket": "retained_oracle_positive",
            "supervision_weight": 1.0,
            "anchor_source": "oracle_correct",
        }

    if preset == "v1a":
        return {
            "keep": False,
            "bucket": "filtered",
            "supervision_weight": 0.0,
            "anchor_source": "",
        }

    label = str(rec.get("gold_label", "")).lower()
    if label not in TRUE_SIDE_LABELS or int(rec.get("n_candidates", 0)) <= 5:
        return {
            "keep": False,
            "bucket": "filtered",
            "supervision_weight": 0.0,
            "anchor_source": "",
        }

    event_id = str(rec.get("event_id", ""))
    fixed_correct = bool(fixed_correct_event_ids and event_id in fixed_correct_event_ids)
    oracle_correct = bool(rec.get("is_correct"))
    if not oracle_correct and not fixed_correct:
        return {
            "keep": False,
            "bucket": "filtered",
            "supervision_weight": 0.0,
            "anchor_source": "",
        }

    weights = dict(ANCHOR_WEIGHTS)
    if true_side_anchor_weights:
        weights.update(true_side_anchor_weights)
    return {
        "keep": True,
        "bucket": f"{label}_anchor",
        "supervision_weight": float(weights.get(label, 0.10)),
        "anchor_source": "oracle_correct" if oracle_correct else "fixed_mmr_correct",
    }


def load_chunk_samples_by_event(cache_path: str | Path) -> dict[str, Any]:
    samples = load_pickle(Path(cache_path))
    return {str(sample.event_id): sample for sample in samples}


def build_candidate_pool(
    sample: Any,
    oracle_rec: dict[str, Any],
    *,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    pool_mode: str = LEGACY_POSITIVE_INJECTION_POOL_MODE,
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
            candidate_pool_source="legacy_reconstructed_with_positive_injection",
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
        candidate_pool_source="legacy_reconstructed_with_positive_injection",
    )


def build_pipeline_style_candidate_pool(
    sample: Any,
    oracle_rec: dict[str, Any],
    *,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    candidate_pool_size: int | None = None,
    fallback_pool_size: int = 15,
    require_oracle_candidate_pool: bool = True,
    expected_chunk_mmr_fingerprint: str | None = None,
) -> CandidatePool:
    """Rebuild the model-facing pool with the production pipeline contract.

    The contract is: deduplicate source chunks, rank by hybrid score, truncate to
    the candidate pool size, then let the pointwise model choose topK. No oracle
    positives are injected before truncation.
    """
    oracle_pool = oracle_rec.get("candidate_pool") or []
    metadata = dict(oracle_rec.get("candidate_pool_metadata") or {})
    oracle_fp = str(metadata.get("chunk_mmr_fingerprint") or "")
    if expected_chunk_mmr_fingerprint:
        if not oracle_fp:
            raise ValueError(
                f"Oracle record {oracle_rec.get('event_id')} has no chunk_mmr_fingerprint "
                "in candidate_pool_metadata."
            )
        if oracle_fp != expected_chunk_mmr_fingerprint:
            raise ValueError(
                f"Oracle record {oracle_rec.get('event_id')} chunk cache fingerprint mismatch: "
                f"expected {expected_chunk_mmr_fingerprint}, got {oracle_fp}."
            )
    if require_oracle_candidate_pool and not oracle_pool:
        raise ValueError(
            f"Oracle record {oracle_rec.get('event_id')} has no saved candidate_pool; "
            "pipeline-style selector data must be built from oracle-search candidate pools."
        )

    effective_pool_size = candidate_pool_size
    if effective_pool_size is None or int(effective_pool_size) <= 0:
        effective_pool_size = len(oracle_pool) if oracle_pool else int(
            oracle_rec.get("n_candidates") or fallback_pool_size
        )
    candidates, features, source_indices, source_count = build_pointwise_inference_pool(
        sample,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        candidate_pool_size=int(effective_pool_size),
    )
    if not candidates:
        return CandidatePool(
            event_id=str(sample.event_id),
            claim=str(sample.claim),
            gold_label=str(sample.label),
            candidates=[],
            features=[],
            positive_local_indices=set(),
            matched_positive_count=0,
            oracle_positive_count=len(oracle_rec.get("selected_indices") or []),
            source_candidate_count=int(source_count),
            pool_mode=PIPELINE_POOL_MODE,
            chunk_mmr_fingerprint=oracle_fp,
            candidate_pool_source="oracle_results_candidate_pool",
            candidate_pool_fingerprint=str(oracle_rec.get("candidate_pool_fingerprint") or ""),
            candidate_pool_metadata=metadata,
        )

    oracle_source_indices: list[int] = []
    if oracle_pool:
        oracle_source_indices = _validate_pipeline_pool_matches_oracle(
            event_id=str(sample.event_id),
            candidates=candidates,
            source_indices=source_indices,
            oracle_pool=oracle_pool,
        )

    selected_indices = _oracle_selected_indices(oracle_rec)
    if oracle_pool and selected_indices:
        max_index = len(oracle_pool) - 1
        bad = [idx for idx in selected_indices if idx < 0 or idx > max_index]
        if bad:
            raise ValueError(
                f"Oracle record {sample.event_id} has selected_indices outside candidate_pool: {bad}"
            )
        if oracle_source_indices:
            source_to_local = {source_idx: local_idx for local_idx, source_idx in enumerate(source_indices)}
            positive_local = {
                source_to_local[oracle_source_indices[idx]]
                for idx in selected_indices
                if idx < len(oracle_source_indices) and oracle_source_indices[idx] in source_to_local
            }
        else:
            positive_local = {idx for idx in selected_indices if 0 <= idx < len(candidates)}
    else:
        positive_local = set()
    if not positive_local and not selected_indices:
        positive_source_indices = _match_positive_indices(sample, source_indices, oracle_rec)
        source_to_local = {source_idx: local_idx for local_idx, source_idx in enumerate(source_indices)}
        positive_local = {
            source_to_local[idx]
            for idx in positive_source_indices
            if idx in source_to_local
        }

    return CandidatePool(
        event_id=str(sample.event_id),
        claim=str(sample.claim),
        gold_label=str(sample.label),
        candidates=candidates,
        features=features,
        positive_local_indices=positive_local,
        matched_positive_count=len(positive_local),
        oracle_positive_count=len(selected_indices or (oracle_rec.get("selected_texts") or [])),
        source_candidate_count=int(source_count),
        pool_mode=PIPELINE_POOL_MODE,
        chunk_mmr_fingerprint=oracle_fp,
        candidate_pool_source="oracle_results_candidate_pool",
        candidate_pool_fingerprint=str(oracle_rec.get("candidate_pool_fingerprint") or ""),
        candidate_pool_metadata=metadata,
    )


def _oracle_selected_indices(oracle_rec: dict[str, Any]) -> list[int]:
    selected: list[int] = []
    for raw in oracle_rec.get("selected_indices") or []:
        try:
            selected.append(int(raw))
        except (TypeError, ValueError):
            continue
    return selected


def _validate_pipeline_pool_matches_oracle(
    *,
    event_id: str,
    candidates: list[dict[str, Any]],
    source_indices: list[int],
    oracle_pool: list[dict[str, Any]],
) -> list[int]:
    if len(candidates) != len(oracle_pool):
        raise ValueError(
            f"Pipeline-style candidate pool size mismatch for {event_id}: "
            f"rebuilt {len(candidates)}, oracle saved {len(oracle_pool)}."
        )

    oracle_source_indices = [item.get("source_index") for item in oracle_pool]
    if all(idx is not None for idx in oracle_source_indices):
        expected = [int(idx) for idx in oracle_source_indices]
        actual = [int(idx) for idx in source_indices]
        if actual != expected:
            if set(actual) != set(expected):
                raise ValueError(
                    f"Pipeline-style candidate pool membership mismatch for {event_id}: "
                    f"rebuilt source_index[:5]={actual[:5]}, oracle source_index[:5]={expected[:5]}."
                )
        return expected

    oracle_uids = [str(item.get("candidate_uid") or "") for item in oracle_pool]
    actual_uids = [str(item.get("candidate_uid") or "") for item in candidates]
    if all(oracle_uids) and actual_uids != oracle_uids:
        raise ValueError(
            f"Pipeline-style candidate pool uid mismatch for {event_id}: "
            f"rebuilt uid[:3]={actual_uids[:3]}, oracle uid[:3]={oracle_uids[:3]}."
        )
    return []


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


def load_pointwise_selector_model(
    path: str | Path,
    *,
    expected_chunk_mmr_fingerprint: str | None = None,
    strict_fingerprint: bool = True,
) -> PointwiseSelectorModel:
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / "model.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Pointwise selector model not found: {model_path}")

    data = np.load(model_path, allow_pickle=True)
    metadata = _load_pointwise_model_metadata(model_path, data)
    model_fp = str(metadata.get("chunk_mmr_fingerprint") or "")
    if expected_chunk_mmr_fingerprint:
        if not model_fp and strict_fingerprint:
            raise ValueError(
                f"Pointwise selector model {model_path} has no chunk_mmr_fingerprint metadata; "
                f"expected {expected_chunk_mmr_fingerprint}."
            )
        if model_fp and model_fp != expected_chunk_mmr_fingerprint and strict_fingerprint:
            raise ValueError(
                f"Pointwise selector model chunk cache fingerprint mismatch: "
                f"expected {expected_chunk_mmr_fingerprint}, got {model_fp} from {model_path}."
            )
    return PointwiseSelectorModel(
        weights=data["weights"].astype(np.float32),
        bias=float(data["bias"][0]),
        feature_mean=data["feature_mean"].astype(np.float32),
        feature_std=data["feature_std"].astype(np.float32),
        feature_names=[str(x) for x in data["feature_names"].tolist()],
        path=str(model_path),
        metadata=metadata,
    )


def _load_pointwise_model_metadata(model_path: Path, data: Any) -> dict[str, Any]:
    if "metadata_json" in getattr(data, "files", []):
        raw = data["metadata_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.tolist()[0])
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
    return {}


def build_pointwise_inference_pool(
    sample: Any,
    *,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    candidate_pool_size: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[int], int]:
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    if n <= 0:
        return [], [], [], 0

    source_indices = _dedup_source_indices(sample, scored["hybrid_scores"], n)
    source_candidate_count = len(source_indices)
    source_indices.sort(key=lambda idx: float(scored["hybrid_scores"][idx]), reverse=True)
    if candidate_pool_size is not None and candidate_pool_size > 0:
        source_indices = source_indices[: int(candidate_pool_size)]

    features = _pool_features(sample, scored, source_indices, oracle_n=len(source_indices))
    candidates: list[dict[str, Any]] = []
    for pool_rank, source_idx in enumerate(source_indices):
        candidate = dict(sample.candidates[source_idx])
        candidate.update({
            "source_index": int(source_idx),
            "candidate_pool_rank": int(pool_rank),
            "dense_score": float(scored["dense_scores"][source_idx]),
            "lexical_score": float(scored["lexical_scores"][source_idx]),
            "bm25_score": float(scored["bm25_scores"][source_idx]),
            "hybrid_score": float(scored["hybrid_scores"][source_idx]),
        })
        candidates.append(candidate)
    return candidates, features, source_indices, source_candidate_count


def score_pointwise_features(
    features: list[dict[str, float]],
    model: PointwiseSelectorModel,
) -> np.ndarray:
    if not features:
        return np.zeros((0,), dtype=np.float32)
    rows = [{"features": item} for item in features]
    x_raw = feature_matrix(rows, model.feature_names)
    x = (x_raw - model.feature_mean) / model.feature_std
    return sigmoid(x @ model.weights + model.bias).astype(np.float32, copy=False)


def resolve_pointwise_candidate_pool_size(
    *,
    top_k: int,
    candidate_pool_size: int | None = None,
    candidate_pool_multiplier: int = 3,
) -> int:
    if candidate_pool_size is not None and int(candidate_pool_size) > 0:
        return max(int(candidate_pool_size), int(top_k))
    return max(int(top_k), int(top_k) * max(int(candidate_pool_multiplier), 1))


def select_candidates_pointwise_oracle(
    sample: Any,
    model: PointwiseSelectorModel,
    *,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    candidate_pool_size: int | None = None,
    candidate_pool_multiplier: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_pool_size = resolve_pointwise_candidate_pool_size(
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        candidate_pool_multiplier=candidate_pool_multiplier,
    )
    candidates, features, _source_indices, source_candidate_count = build_pointwise_inference_pool(
        sample,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        candidate_pool_size=effective_pool_size,
    )
    if not candidates:
        row = {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }
        trace = {
            "event_id": sample.event_id,
            "label": sample.label,
            "top_k": int(top_k),
            "candidate_pool_size": int(effective_pool_size),
            "n_source_candidates": int(source_candidate_count),
            "n_pool_candidates": 0,
            "selected": [],
        }
        return row, trace

    scores = score_pointwise_features(features, model)
    order = np.argsort(-scores)[: min(int(top_k), len(candidates))]
    selected: list[dict[str, Any]] = []
    trace_selected: list[dict[str, Any]] = []
    for rank, pool_idx in enumerate(order.tolist()):
        candidate = dict(candidates[pool_idx])
        pointwise_score = float(scores[pool_idx])
        candidate.update({
            "pointwise_score": pointwise_score,
            "pointwise_rank": int(rank),
            "pointwise_candidate_pool_size": int(len(candidates)),
        })
        selected.append(candidate)
        trace_selected.append({
            "rank": int(rank),
            "candidate_pool_rank": int(candidate.get("candidate_pool_rank", pool_idx)),
            "source_index": int(candidate.get("source_index", -1)),
            "pointwise_score": pointwise_score,
            "hybrid_score": float(candidate.get("hybrid_score", 0.0)),
            "text": str(candidate.get("text", "")),
            "report_id": str(candidate.get("report_id") or _source_report_id(candidate)),
        })

    trace = {
        "event_id": sample.event_id,
        "label": sample.label,
        "top_k": int(top_k),
        "candidate_pool_size": int(effective_pool_size),
        "n_source_candidates": int(source_candidate_count),
        "n_pool_candidates": int(len(candidates)),
        "score_mean": float(scores.mean()) if scores.size else 0.0,
        "score_max": float(scores.max()) if scores.size else 0.0,
        "selected": trace_selected,
        "model_path": model.path,
        "feature_names": model.feature_names,
    }
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": selected,
    }, trace


def _source_report_id(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_report")
    if isinstance(source, dict):
        return str(source.get("report_id", ""))
    return ""


def pool_to_pointwise_rows(
    pool: CandidatePool,
    oracle_rec: dict[str, Any],
    filter_bucket: str,
    *,
    supervision_weight: float = 1.0,
    anchor_source: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (candidate, features) in enumerate(zip(pool.candidates, pool.features)):
        rows.append({
            "event_id": pool.event_id,
            "claim": pool.claim,
            "gold_label": pool.gold_label,
            "candidate_idx": i,
            "candidate_uid": str(candidate.get("candidate_uid") or ""),
            "source_index": int(candidate.get("source_index", i)),
            "candidate_pool_rank": int(candidate.get("candidate_pool_rank", i)),
            "source_text": str(candidate.get("text", "")),
            "report_id": str(candidate.get("report_id") or _source_report_id(candidate)),
            "is_oracle_selected": int(i in pool.positive_local_indices),
            "features": {name: float(features.get(name, 0.0)) for name in DEFAULT_FEATURE_NAMES},
            "oracle_final_logprob": float(oracle_rec.get("final_logprob", 0.0)),
            "oracle_margin": float(oracle_rec.get("margin", 0.0)),
            "oracle_correct": bool(oracle_rec.get("is_correct")),
            "oracle_n_candidates": int(oracle_rec.get("n_candidates", 0)),
            "matched_positive_count": int(pool.matched_positive_count),
            "oracle_positive_count": int(pool.oracle_positive_count),
            "filter_bucket": filter_bucket,
            "supervision_weight": float(supervision_weight),
            "anchor_source": anchor_source,
            "pool_mode": pool.pool_mode,
            "candidate_pool_source": pool.candidate_pool_source,
            "candidate_pool_fingerprint": pool.candidate_pool_fingerprint,
            "chunk_mmr_fingerprint": pool.chunk_mmr_fingerprint,
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
    weight_by_event = {
        eid: float(rows[idxs[0]].get("supervision_weight", 1.0))
        for eid, idxs in grouped.items()
    }
    balanced_events = [
        eid
        for eid, label in label_by_event.items()
        if label in RETAINED_LABELS and weight_by_event.get(eid, 1.0) >= 0.999
    ]
    if balanced_events:
        label_counts = Counter(label_by_event[eid] for eid in balanced_events)
    else:
        label_counts = Counter(label_by_event.values())
    n_labels = max(len(label_counts), 1)
    label_weights = {
        label: len(balanced_events or grouped) / (n_labels * count)
        for label, count in label_counts.items()
    }
    weights = np.ones(len(rows), dtype=np.float32)
    for eid, idxs in grouped.items():
        ys = [int(rows[i].get("is_oracle_selected", 0)) for i in idxs]
        n_pos = max(sum(ys), 1)
        n_neg = max(len(ys) - sum(ys), 1)
        lw = float(label_weights.get(label_by_event[eid], 1.0))
        sw = float(weight_by_event.get(eid, 1.0))
        for i, y in zip(idxs, ys):
            weights[i] = sw * lw * (0.5 / n_pos if y else 0.5 / n_neg)
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
                "candidate_uid": str(row.get("candidate_uid", "")),
                "source_index": int(row.get("source_index", row["candidate_idx"])),
                "candidate_pool_rank": int(row.get("candidate_pool_rank", row["candidate_idx"])),
                "score": float(local_scores[local_idx]),
                "is_oracle_selected": int(row["is_oracle_selected"]),
                "text": row.get("source_text", ""),
                "report_id": row.get("report_id", ""),
            })
        out.append({
            "event_id": eid,
            "gold_label": rows[idxs[0]]["gold_label"],
            "pool_mode": rows[idxs[0]].get("pool_mode", ""),
            "chunk_mmr_fingerprint": rows[idxs[0]].get("chunk_mmr_fingerprint", ""),
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
