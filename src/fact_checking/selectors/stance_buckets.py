from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


PROMPT_VERSION = "stance_bucket_teacher_v0"
PROMPT_VERSION_V02 = "stance_bucket_teacher_v02"
SUPPORTED_PROMPT_VERSIONS = {PROMPT_VERSION, PROMPT_VERSION_V02}
EVIDENCE_ROLES = {
    "direct_support_claim",
    "direct_refute_claim",
    "partial_support",
    "partial_refute",
    "contextual_background",
    "not_relevant",
}
ROLE_EVIDENCE_SCORES = {
    "direct_support_claim": 1.0,
    "direct_refute_claim": 1.0,
    "partial_support": 0.75,
    "partial_refute": 0.75,
    "contextual_background": 0.35,
    "not_relevant": 0.0,
}
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

SYSTEM_PROMPT_V02 = """You are an evidence-quality annotation model for fact-checking evidence selection.
Given a political claim and one candidate evidence sentence, judge only the relation between the claim and the evidence text.
Return JSON only. Do not use external knowledge, the gold veracity label, or any hidden oracle information.
Use the fields to distinguish direct claim-specific evidence from merely stance-related background.
Return exactly these fields:
1. stance_score: number from 1 to 10, where 1 strongly refutes the claim, 5 to 6 is ambiguous/partial/context-dependent, and 10 strongly supports the claim.
2. semantic_completeness: number from 0 to 10, where 0 is fragmentary/unusable and 10 is complete/self-contained.
3. claim_specificity: number from 0 to 10, where 10 is tightly about the exact entities, quantities, time, relation, and action in the claim.
4. direct_evidence: number from 0 to 10, where 10 directly helps verify or falsify the claim without needing distant context.
5. background_only: number from 0 to 10, where 10 is only general context/background, even if topical.
6. key_fact_overlap: number from 0 to 10, where 10 covers the claim's key factual atoms.
7. evidence_role: one of direct_support_claim, direct_refute_claim, partial_support, partial_refute, contextual_background, not_relevant."""

USER_PROMPT_TEMPLATE_V02 = """Annotate the claim-evidence relation and evidence quality.

claim:
{claim}

candidate_evidence:
{evidence_text}

Return only this JSON object:
{{
  "stance_score": <number from 1 to 10>,
  "semantic_completeness": <number from 0 to 10>,
  "claim_specificity": <number from 0 to 10>,
  "direct_evidence": <number from 0 to 10>,
  "background_only": <number from 0 to 10>,
  "key_fact_overlap": <number from 0 to 10>,
  "evidence_role": "<direct_support_claim|direct_refute_claim|partial_support|partial_refute|contextual_background|not_relevant>"
}}"""


class TeacherAnnotationError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherAnnotation:
    stance_score: float
    semantic_completeness: float
    claim_specificity: float | None = None
    direct_evidence: float | None = None
    background_only: float | None = None
    key_fact_overlap: float | None = None
    evidence_role: str | None = None
    stance_score_clamped: bool = False
    semantic_completeness_clamped: bool = False
    claim_specificity_clamped: bool = False
    direct_evidence_clamped: bool = False
    background_only_clamped: bool = False
    key_fact_overlap_clamped: bool = False


def system_prompt_for_version(prompt_version: str = PROMPT_VERSION) -> str:
    version = normalize_prompt_version(prompt_version)
    if version == PROMPT_VERSION_V02:
        return SYSTEM_PROMPT_V02
    return SYSTEM_PROMPT


def format_user_prompt(*, claim: str, evidence_text: str, prompt_version: str = PROMPT_VERSION) -> str:
    version = normalize_prompt_version(prompt_version)
    template = USER_PROMPT_TEMPLATE_V02 if version == PROMPT_VERSION_V02 else USER_PROMPT_TEMPLATE
    return template.format(claim=str(claim or "").strip(), evidence_text=str(evidence_text or "").strip())


def normalize_prompt_version(prompt_version: str | None) -> str:
    version = str(prompt_version or PROMPT_VERSION)
    if version not in SUPPORTED_PROMPT_VERSIONS:
        raise TeacherAnnotationError(f"Unsupported prompt_version: {version}")
    return version


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


def parse_teacher_content(raw_text: str, *, prompt_version: str = PROMPT_VERSION) -> TeacherAnnotation:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = _strip_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TeacherAnnotationError(f"Invalid JSON: {exc}") from exc
    return validate_teacher_payload(payload, prompt_version=prompt_version)


def validate_teacher_payload(payload: dict[str, Any], *, prompt_version: str = PROMPT_VERSION) -> TeacherAnnotation:
    if not isinstance(payload, dict):
        raise TeacherAnnotationError("Teacher payload must be a JSON object.")
    version = normalize_prompt_version(prompt_version)
    allowed = {"stance_score", "semantic_completeness"}
    if version == PROMPT_VERSION_V02:
        allowed = {
            "stance_score",
            "semantic_completeness",
            "claim_specificity",
            "direct_evidence",
            "background_only",
            "key_fact_overlap",
            "evidence_role",
        }
    extras = set(payload) - allowed
    missing = allowed - set(payload)
    if extras or missing:
        raise TeacherAnnotationError(f"Teacher payload keys mismatch; missing={sorted(missing)} extras={sorted(extras)}")
    stance_raw = _finite_float(payload.get("stance_score"), "stance_score")
    completeness_raw = _finite_float(payload.get("semantic_completeness"), "semantic_completeness")
    stance = _clamp(stance_raw, 1.0, 10.0)
    completeness = _clamp(completeness_raw, 0.0, 10.0)
    extra: dict[str, Any] = {}
    if version == PROMPT_VERSION_V02:
        specificity_raw = _finite_float(payload.get("claim_specificity"), "claim_specificity")
        direct_raw = _finite_float(payload.get("direct_evidence"), "direct_evidence")
        background_raw = _finite_float(payload.get("background_only"), "background_only")
        overlap_raw = _finite_float(payload.get("key_fact_overlap"), "key_fact_overlap")
        specificity = _clamp(specificity_raw, 0.0, 10.0)
        direct = _clamp(direct_raw, 0.0, 10.0)
        background = _clamp(background_raw, 0.0, 10.0)
        overlap = _clamp(overlap_raw, 0.0, 10.0)
        role = normalize_evidence_role(payload.get("evidence_role"))
        extra = {
            "claim_specificity": float(specificity),
            "direct_evidence": float(direct),
            "background_only": float(background),
            "key_fact_overlap": float(overlap),
            "evidence_role": role,
            "claim_specificity_clamped": abs(specificity - specificity_raw) > 1e-9,
            "direct_evidence_clamped": abs(direct - direct_raw) > 1e-9,
            "background_only_clamped": abs(background - background_raw) > 1e-9,
            "key_fact_overlap_clamped": abs(overlap - overlap_raw) > 1e-9,
        }
    return TeacherAnnotation(
        stance_score=float(stance),
        semantic_completeness=float(completeness),
        stance_score_clamped=abs(stance - stance_raw) > 1e-9,
        semantic_completeness_clamped=abs(completeness - completeness_raw) > 1e-9,
        **extra,
    )


def teacher_annotation_payload(annotation: TeacherAnnotation) -> dict[str, Any]:
    payload = {
        "stance_score": float(annotation.stance_score),
        "semantic_completeness": float(annotation.semantic_completeness),
        "stance_score_clamped": bool(annotation.stance_score_clamped),
        "semantic_completeness_clamped": bool(annotation.semantic_completeness_clamped),
    }
    if annotation.evidence_role is not None:
        payload.update(
            {
                "claim_specificity": float(annotation.claim_specificity or 0.0),
                "direct_evidence": float(annotation.direct_evidence or 0.0),
                "background_only": float(annotation.background_only or 0.0),
                "key_fact_overlap": float(annotation.key_fact_overlap or 0.0),
                "evidence_role": normalize_evidence_role(annotation.evidence_role),
                "role_evidence_score": role_evidence_score(annotation.evidence_role),
                "claim_specificity_clamped": bool(annotation.claim_specificity_clamped),
                "direct_evidence_clamped": bool(annotation.direct_evidence_clamped),
                "background_only_clamped": bool(annotation.background_only_clamped),
                "key_fact_overlap_clamped": bool(annotation.key_fact_overlap_clamped),
            }
        )
    return payload


def normalize_evidence_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    role = role.replace("-", "_").replace(" ", "_")
    if role not in EVIDENCE_ROLES:
        raise TeacherAnnotationError(f"Invalid evidence_role: {value}")
    return role


def role_evidence_score(value: Any) -> float:
    try:
        role = normalize_evidence_role(value)
    except TeacherAnnotationError:
        return 0.0
    return float(ROLE_EVIDENCE_SCORES.get(role, 0.0))


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
