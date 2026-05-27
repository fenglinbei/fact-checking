from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


PROMPT_VERSION = "stance_bucket_teacher_v0"
SYSTEM_PROMPT = """You are a stance annotation model for fact-checking evidence selection.
Given a political claim and one candidate evidence sentence, estimate how the evidence relates to the claim.
Return JSON only.
Do not use external knowledge. Judge only the relation between the claim and the evidence text.
Return exactly two numeric fields:
1. stance_score: integer or float from 1 to 10, where 1 means the evidence strongly opposes the claim, 5 to 6 means ambiguous/partial/context-dependent, and 10 means the evidence strongly supports the claim.
2. semantic_completeness: integer or float from 0 to 10, where 0 means fragmentary or unusable as evidence, and 10 means a complete, self-contained sentence.
Do not include the gold veracity label or any hidden oracle information in the answer."""

USER_PROMPT_TEMPLATE = """Annotate the claim-evidence stance score and semantic completeness.

claim:
{claim}

candidate_evidence:
{evidence_text}

Return only this JSON object:
{{
  "stance_score": <number from 1 to 10>,
  "semantic_completeness": <number from 0 to 10>
}}"""


class TeacherAnnotationError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherAnnotation:
    stance_score: float
    semantic_completeness: float
    stance_score_clamped: bool = False
    semantic_completeness_clamped: bool = False


def format_user_prompt(*, claim: str, evidence_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(claim=str(claim or "").strip(), evidence_text=str(evidence_text or "").strip())


def annotation_key(
    *,
    event_id: str,
    candidate_uid: str,
    prompt_version: str = PROMPT_VERSION,
    model: str,
) -> str:
    payload = f"{event_id}|{candidate_uid}|{prompt_version}|{model}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def bucket_names(n_stance_buckets: int) -> list[str]:
    n = int(n_stance_buckets)
    if n == 3:
        return ["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"]
    if n == 5:
        return [
            "strong_oppose_claim_bucket",
            "weak_oppose_claim_bucket",
            "ambiguous_claim_bucket",
            "weak_support_claim_bucket",
            "strong_support_claim_bucket",
        ]
    if n == 7:
        return [
            "strong_oppose_claim_bucket",
            "moderate_oppose_claim_bucket",
            "weak_oppose_claim_bucket",
            "ambiguous_claim_bucket",
            "weak_support_claim_bucket",
            "moderate_support_claim_bucket",
            "strong_support_claim_bucket",
        ]
    if n < 2:
        raise ValueError("n_stance_buckets must be at least 2.")
    return [f"stance_bucket_{idx:02d}" for idx in range(n)]


def bucket_centers(n_stance_buckets: int) -> list[float]:
    n = int(n_stance_buckets)
    if n < 2:
        raise ValueError("n_stance_buckets must be at least 2.")
    return [1.0 + 9.0 * float(idx) / float(n - 1) for idx in range(n)]


def stance_score_to_probs(
    stance_score: float,
    *,
    n_stance_buckets: int,
    tau: float = 2.0,
) -> dict[str, float]:
    score = _clamp(float(stance_score), 1.0, 10.0)
    centers = bucket_centers(n_stance_buckets)
    logits = [-((score - center) ** 2) / max(float(tau), 1e-6) for center in centers]
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    denom = sum(exp_values) or 1.0
    return {
        name: float(value / denom)
        for name, value in zip(bucket_names(n_stance_buckets), exp_values)
    }


def derive_stance_fields(
    *,
    stance_score: float,
    n_stance_buckets: int,
    tau: float = 2.0,
) -> dict[str, Any]:
    probs = stance_score_to_probs(stance_score, n_stance_buckets=n_stance_buckets, tau=tau)
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    max_prob = ordered[0][1] if ordered else 0.0
    second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
    entropy = normalized_entropy(list(probs.values()))
    expected_score = sum(center * probs[name] for name, center in zip(bucket_names(n_stance_buckets), bucket_centers(n_stance_buckets)))
    return {
        "n_stance_buckets": int(n_stance_buckets),
        "teacher_stance_probs": probs,
        "stance_bucket_derived": ordered[0][0] if ordered else "",
        "stance_strength": float(max_prob - second_prob),
        "stance_entropy": float(entropy),
        "stance_expected_score": float(expected_score),
    }


def normalized_entropy(probs: Sequence[float]) -> float:
    values = [max(float(prob), 0.0) for prob in probs]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    normed = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in normed if value > 0.0)
    return float(entropy / math.log(len(normed)))


def parse_teacher_content(raw_text: str) -> TeacherAnnotation:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = _strip_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TeacherAnnotationError(f"Invalid JSON: {exc}") from exc
    return validate_teacher_payload(payload)


def validate_teacher_payload(payload: dict[str, Any]) -> TeacherAnnotation:
    if not isinstance(payload, dict):
        raise TeacherAnnotationError("Teacher payload must be a JSON object.")
    allowed = {"stance_score", "semantic_completeness"}
    extras = set(payload) - allowed
    missing = allowed - set(payload)
    if extras or missing:
        raise TeacherAnnotationError(f"Teacher payload keys mismatch; missing={sorted(missing)} extras={sorted(extras)}")
    stance_raw = _finite_float(payload.get("stance_score"), "stance_score")
    completeness_raw = _finite_float(payload.get("semantic_completeness"), "semantic_completeness")
    stance = _clamp(stance_raw, 1.0, 10.0)
    completeness = _clamp(completeness_raw, 0.0, 10.0)
    return TeacherAnnotation(
        stance_score=float(stance),
        semantic_completeness=float(completeness),
        stance_score_clamped=abs(stance - stance_raw) > 1e-9,
        semantic_completeness_clamped=abs(completeness - completeness_raw) > 1e-9,
    )


def teacher_annotation_payload(annotation: TeacherAnnotation) -> dict[str, Any]:
    return {
        "stance_score": float(annotation.stance_score),
        "semantic_completeness": float(annotation.semantic_completeness),
        "stance_score_clamped": bool(annotation.stance_score_clamped),
        "semantic_completeness_clamped": bool(annotation.semantic_completeness_clamped),
    }


def _finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TeacherAnnotationError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise TeacherAnnotationError(f"{field_name} must be finite.")
    return float(parsed)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
