from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from fact_checking.build.candidates import canonicalize_sentence
from fact_checking.data.constants import LABEL2ID, LABELS, LABEL_LETTERS, LETTER2LABEL, LETTER_ORDER
from fact_checking.pipeline.artifacts import fingerprint


DEFAULT_DIRECT_VERIFIER_RUN_DIR = (
    "outputs/oracle_direct_verifier/stage2_sentence/train/"
    "b3_oracle_sentence_direct_verifier_1024_20260519-200709"
)
DEFAULT_VERIFIER_CHECKPOINT = "best"
DEFAULT_VERIFIER_BASE_MODEL = "/data/models/Qwen2.5-7B-Instruct"
DEFAULT_LABEL_POLICY = "anchor2_delta"


@dataclass(frozen=True)
class VerifierCheckpointInfo:
    run_dir: str
    checkpoint: str
    checkpoint_dir: str
    adapter_config_path: str
    adapter_model_path: str
    tokenizer_config_path: str
    adapter_sha256: str
    base_model_name_or_path: str
    label_prefix: str
    label_token_ids: dict[str, int]


def require_verifier_checkpoint(
    run_dir: str | Path,
    checkpoint: str = DEFAULT_VERIFIER_CHECKPOINT,
    *,
    label_prefix: str = "Label:",
) -> VerifierCheckpointInfo:
    checkpoint = str(checkpoint)
    if checkpoint == "final":
        raise ValueError(
            "VERIFIER_CHECKPOINT=final is not allowed for verifier-proxy v0. "
            "Use best or checkpoint-600 after syncing the adapter weights."
        )
    run_path = Path(run_dir)
    ckpt_dir = run_path / checkpoint
    required = {
        "adapter_config": ckpt_dir / "adapter_config.json",
        "adapter_model": ckpt_dir / "adapter_model.safetensors",
        "tokenizer_config": ckpt_dir / "tokenizer_config.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Verifier checkpoint is incomplete. Sync the missing file(s) before running "
            "verifier-proxy labeling:\n" + "\n".join(f"- {item}" for item in missing)
        )

    with required["adapter_config"].open(encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)
    base_model = str(adapter_cfg.get("base_model_name_or_path") or DEFAULT_VERIFIER_BASE_MODEL)
    label_token_ids = load_label_token_ids(run_path, label_prefix=label_prefix)
    return VerifierCheckpointInfo(
        run_dir=str(run_path),
        checkpoint=checkpoint,
        checkpoint_dir=str(ckpt_dir),
        adapter_config_path=str(required["adapter_config"]),
        adapter_model_path=str(required["adapter_model"]),
        tokenizer_config_path=str(required["tokenizer_config"]),
        adapter_sha256=sha256_file(required["adapter_model"]),
        base_model_name_or_path=base_model,
        label_prefix=str(label_prefix),
        label_token_ids=label_token_ids,
    )


def load_label_token_ids(run_dir: str | Path, *, label_prefix: str = "Label:") -> dict[str, int]:
    meta_path = Path(run_dir) / "label_token_ce_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing label token metadata: {meta_path}")
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    if str(meta.get("label_prefix") or label_prefix) != str(label_prefix):
        raise ValueError(
            f"label_prefix mismatch: expected {label_prefix!r}, got {meta.get('label_prefix')!r}."
        )
    ids = {str(key): int(value) for key, value in dict(meta.get("label_token_ids") or {}).items()}
    missing = [letter for letter in LETTER_ORDER if letter not in ids]
    if missing:
        raise ValueError(f"label_token_ce_meta.json is missing label token ids for: {missing}")
    return {letter: ids[letter] for letter in LETTER_ORDER}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_fingerprint(payload: dict[str, Any], *, length: int = 16) -> str:
    return fingerprint(payload, length=length, algorithm="sha256")


def verifier_config_fingerprint(info: VerifierCheckpointInfo) -> str:
    return stable_fingerprint(
        {
            "run_dir": info.run_dir,
            "checkpoint": info.checkpoint,
            "adapter_sha256": info.adapter_sha256,
            "base_model_name_or_path": info.base_model_name_or_path,
            "label_prefix": info.label_prefix,
            "label_token_ids": info.label_token_ids,
        }
    )


def prompt_config_fingerprint(prompt_cfg: dict[str, Any]) -> str:
    return stable_fingerprint(dict(prompt_cfg or {}))


def candidate_key(candidate: dict[str, Any]) -> str:
    key = str(candidate.get("canonical_text") or "").strip()
    if key:
        return key
    return canonicalize_sentence(str(candidate.get("text") or ""))


def select_anchor_candidates(candidates: Sequence[dict[str, Any]], *, anchor_k: int = 2) -> list[dict[str, Any]]:
    baseline = [dict(candidate) for candidate in candidates if candidate.get("from_baseline")]
    baseline.sort(key=lambda item: int(item.get("baseline_rank") or 10**9))
    if baseline:
        return baseline[: min(int(anchor_k), len(baseline))]
    ordered = [dict(candidate) for candidate in candidates]
    ordered.sort(key=lambda item: int(item.get("union_pool_rank") or 10**9))
    return ordered[: min(int(anchor_k), len(ordered))]


def dedupe_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate_key(candidate)
        if not key or key in seen:
            continue
        item = dict(candidate)
        item["canonical_text"] = key
        out.append(item)
        seen.add(key)
    return out


def evidence_set_hash(candidate_keys: Sequence[str]) -> str:
    return stable_fingerprint({"candidate_keys": list(candidate_keys)}, length=24)


def score_margin(label_logprobs: dict[str, float], gold_label: str) -> dict[str, Any]:
    gold = str(gold_label).strip().lower()
    if gold not in LABEL2ID:
        raise ValueError(f"Unknown gold_label={gold_label!r}")
    gold_letter = LABEL_LETTERS[gold]
    scores = {letter: float(label_logprobs[letter]) for letter in LETTER_ORDER}
    pred_letter = max(LETTER_ORDER, key=lambda letter: scores[letter])
    wrong_scores = [score for letter, score in scores.items() if letter != gold_letter]
    best_wrong = max(wrong_scores) if wrong_scores else float("-inf")
    gold_score = float(scores[gold_letter])
    return {
        "gold_logprob": gold_score,
        "best_wrong_logprob": float(best_wrong),
        "margin": float(gold_score - best_wrong),
        "pred_label": LETTER2LABEL[pred_letter],
        "is_correct": bool(pred_letter == gold_letter),
    }


def cache_key_for_score(
    *,
    split: str,
    event_id: str,
    evidence_set_hash_value: str,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    label_policy: str,
) -> str:
    return stable_fingerprint(
        {
            "split": str(split),
            "event_id": str(event_id),
            "evidence_set_hash": str(evidence_set_hash_value),
            "verifier_config_fingerprint": str(verifier_fingerprint),
            "prompt_config_fingerprint": str(prompt_fingerprint),
            "label_policy": str(label_policy),
        },
        length=32,
    )


def load_score_cache(path: str | Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    cache: dict[str, dict[str, Any]] = {}
    invalid = 0
    duplicates = 0
    path = Path(path)
    if not path.exists():
        return cache, invalid, duplicates
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            key = str(row.get("cache_key") or "")
            if not key or row.get("status") != "completed":
                invalid += 1
                continue
            if key in cache:
                duplicates += 1
            cache[key] = row
    return cache, invalid, duplicates


def append_score_cache_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_anchor2_delta_rows(
    *,
    split: str,
    union_row: dict[str, Any],
    oracle_row: dict[str, Any],
    score_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    label_policy: str = DEFAULT_LABEL_POLICY,
    positive_close_delta: float = 0.02,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if label_policy != DEFAULT_LABEL_POLICY:
        raise ValueError(f"Unsupported verifier-proxy label_policy={label_policy!r}")
    event_id = str(union_row.get("event_id") or oracle_row.get("event_id") or "")
    claim = str(union_row.get("claim") or oracle_row.get("claim") or "")
    gold_label = str(oracle_row.get("gold_label") or "").strip().lower()
    candidates = dedupe_candidates(list(union_row.get("candidates") or []))
    anchor = dedupe_candidates(select_anchor_candidates(candidates, anchor_k=2))
    anchor_keys = [candidate_key(candidate) for candidate in anchor]
    anchor_score = score_fn(anchor)
    anchor_margin = float(anchor_score["margin"])

    rows: list[dict[str, Any]] = []
    raw_scores = [anchor_score]
    anchor_key_set = set(anchor_keys)
    for candidate_idx, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        if key in anchor_key_set:
            scored = [item for item in anchor if candidate_key(item) != key]
            evidence_policy = "anchor_leave_one_out"
        else:
            scored = dedupe_candidates([*anchor, candidate])
            evidence_policy = "anchor_add_one"
        scored_keys = [candidate_key(item) for item in scored]
        score = score_fn(scored)
        raw_scores.append(score)
        utility = anchor_margin - float(score["margin"]) if key in anchor_key_set else float(score["margin"]) - anchor_margin
        rows.append(
            {
                "split": str(split),
                "event_id": event_id,
                "claim": claim,
                "gold_label": gold_label,
                "candidate_idx": int(candidate_idx),
                "candidate_key": key,
                "candidate_text": str(candidate.get("text") or ""),
                "canonical_text": key,
                "union_source": str(candidate.get("union_source") or ""),
                "from_baseline": bool(candidate.get("from_baseline")),
                "from_qd": bool(candidate.get("from_qd")),
                "baseline_rank": candidate.get("baseline_rank"),
                "qd_pool_rank": candidate.get("qd_pool_rank"),
                "union_pool_rank": candidate.get("union_pool_rank"),
                "baseline_hybrid_score": candidate.get("baseline_hybrid_score"),
                "qd_rrf_score": candidate.get("qd_rrf_score"),
                "qd_question_hit_count": candidate.get("qd_question_hit_count"),
                "qd_max_question_hybrid": candidate.get("qd_max_question_hybrid"),
                "evidence_set_policy": evidence_policy,
                "anchor_candidate_keys": anchor_keys,
                "scored_candidate_keys": scored_keys,
                "evidence_set_hash": evidence_set_hash(scored_keys),
                "label_logprobs": dict(score.get("label_logprobs") or {}),
                "pred_label": score.get("pred_label"),
                "is_correct": bool(score.get("is_correct")),
                "gold_logprob": float(score.get("gold_logprob", 0.0)),
                "best_wrong_logprob": float(score.get("best_wrong_logprob", 0.0)),
                "margin": float(score["margin"]),
                "anchor_margin": anchor_margin,
                "target_utility": float(utility),
                "target_positive": False,
            }
        )
    mark_target_positives(rows, close_delta=float(positive_close_delta))
    return rows, raw_scores


def mark_target_positives(rows: list[dict[str, Any]], *, close_delta: float = 0.02) -> None:
    if not rows:
        return
    best = max(float(row.get("target_utility", 0.0)) for row in rows)
    for row in rows:
        utility = float(row.get("target_utility", 0.0))
        row["target_positive"] = bool(utility > 0.0 or utility >= best - float(close_delta))
    if not any(bool(row.get("target_positive")) for row in rows):
        best_idx = max(range(len(rows)), key=lambda idx: float(rows[idx].get("target_utility", 0.0)))
        rows[best_idx]["target_positive"] = True


def build_grouped_rows(flat_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in flat_rows:
        grouped.setdefault(str(row.get("event_id") or ""), []).append(dict(row))
    out: list[dict[str, Any]] = []
    for event_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("candidate_idx") or 0))
        out.append(
            {
                "event_id": event_id,
                "claim": rows[0].get("claim", "") if rows else "",
                "gold_label": rows[0].get("gold_label", "") if rows else "",
                "anchor_candidate_keys": rows[0].get("anchor_candidate_keys", []) if rows else [],
                "candidates": rows,
            }
        )
    return out


def verifier_proxy_cross_encoder_loss(
    grouped_scores: Sequence[torch.Tensor],
    grouped_utilities: Sequence[Sequence[float]],
    grouped_positive_masks: Sequence[Sequence[bool]],
    *,
    utility_epsilon: float = 1e-4,
    soft_tau: float = 0.3,
    soft_ce_weight: float = 0.2,
    regression_weight: float = 0.2,
    bce_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_losses: list[torch.Tensor] = []
    soft_losses: list[torch.Tensor] = []
    reg_losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    n_pairs = 0
    for scores, utilities_raw, positives_raw in zip(grouped_scores, grouped_utilities, grouped_positive_masks):
        if scores.numel() == 0:
            continue
        utilities = torch.as_tensor(utilities_raw, dtype=scores.dtype, device=scores.device)
        positives = torch.as_tensor(positives_raw, dtype=scores.dtype, device=scores.device)
        n = min(scores.numel(), utilities.numel(), positives.numel())
        scores = scores[:n]
        utilities = utilities[:n]
        positives = positives[:n]
        for i in range(n):
            better = torch.nonzero(utilities[i] > utilities + float(utility_epsilon), as_tuple=False).flatten()
            for j in better.tolist():
                pair_losses.append(nn.functional.softplus(-(scores[i] - scores[int(j)])))
                n_pairs += 1
        target = torch.softmax(utilities / max(float(soft_tau), 1e-6), dim=0)
        soft_losses.append(-(target * torch.log_softmax(scores, dim=0)).sum())
        reg_losses.append(nn.functional.huber_loss(scores, utilities, reduction="mean"))
        bce_losses.append(nn.functional.binary_cross_entropy_with_logits(scores, positives))

    device = grouped_scores[0].device if grouped_scores else torch.device("cpu")
    zero = torch.zeros((), device=device)
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    soft_loss = torch.stack(soft_losses).mean() if soft_losses else zero
    reg_loss = torch.stack(reg_losses).mean() if reg_losses else zero
    bce_loss = torch.stack(bce_losses).mean() if bce_losses else zero
    total = (
        pair_loss
        + float(soft_ce_weight) * soft_loss
        + float(regression_weight) * reg_loss
        + float(bce_weight) * bce_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "soft_ce_loss": float(soft_loss.detach().cpu()),
        "regression_loss": float(reg_loss.detach().cpu()),
        "bce_loss": float(bce_loss.detach().cpu()),
        "n_pairs": float(n_pairs),
    }


def pairwise_utility_accuracy(scores: Sequence[float], utilities: Sequence[float], *, epsilon: float = 1e-4) -> tuple[int, int]:
    correct = 0
    total = 0
    for i, utility_i in enumerate(utilities):
        for j, utility_j in enumerate(utilities):
            if float(utility_i) <= float(utility_j) + float(epsilon):
                continue
            total += 1
            if float(scores[i]) > float(scores[j]):
                correct += 1
    return correct, total


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(xa.std()) < 1e-12 or float(ya.std()) < 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    return pearson_corr(_rankdata(x), _rankdata(y))


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: float(item[1]))
    ranks = [0.0] * len(indexed)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and float(indexed[j][1]) == float(indexed[i][1]):
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
