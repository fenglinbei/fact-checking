from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


ASPECT_EXTRACTION_VERSION = "rule_aspect_v1"
LLM_DECOMP_PLUS_VERSION = "llm_decomp_plus_v1"
FULL_CLAIM_ASPECT_TYPE = "full_claim"

_MIN_DEFAULT_TOKENS = 4
_MAX_DEFAULT_TOKENS = 24
_MIN_DEFAULT_RETRIEVABILITY = 2

_BAD_ENTITY_SPANS = {
    "Says",
    "When",
    "On",
    "The",
    "Every",
    "Two",
    "Their",
    "Bible",
    "Gov",
}
_CONTEXT_LEADING_CUES = {"instead", "otherwise", "where", "because", "but", "then"}
_ACTION_WORDS = {
    "abandoning",
    "advocated",
    "allowed",
    "chartered",
    "cost",
    "die",
    "ending",
    "lowered",
    "prohibited",
    "providing",
    "remove",
    "rent",
    "scuttling",
    "set",
    "spent",
    "started",
    "stop",
    "sued",
    "take",
    "use",
    "working",
}
_POLICY_TOPIC_WORDS = {
    "bible",
    "congress",
    "department",
    "health",
    "income",
    "insurance",
    "jobs",
    "justice",
    "kuran",
    "medicaid",
    "reform",
    "school",
    "security",
    "social",
    "tax",
    "treatment",
    "voters",
}

_QUANTITY_RE = re.compile(
    r"(?i)(?:"
    r"\$\s?\d[\d,]*(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s?(?:percent|%|million|billion|people|jobs|days|years|dollars)?"
    r"|\b(?:half|two thirds|three quarters|first 100 days|70s|08 race)\b"
    r")"
)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'.-]+(?:\s+(?:of|the|and)?\s*[A-Z][A-Za-z0-9'.-]+)*)\b"
)
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|,\s+(?:but|and|where|because|otherwise|instead|then)\b|\b(?:but instead|otherwise)\b)\s*",
    re.IGNORECASE,
)
_CUE_PATTERNS = {
    "negation": re.compile(r"\b(?:not|no|never|without|did not|would not|cannot|can't)\b", re.IGNORECASE),
    "comparison": re.compile(
        r"\b(?:less|more|nearly|average|bottom|similar|than|half|thirds|quarters|first)\b",
        re.IGNORECASE,
    ),
    "causal_condition": re.compile(
        r"\b(?:because|cause|harkens back|where|if|otherwise|but|instead)\b",
        re.IGNORECASE,
    ),
    "policy_action": re.compile(
        r"\b(?:set limits|providing|advocated|abandoning|ending|lowered|spent|chartered|"
        r"remove|take them back|prohibited|allowed to die|sued|rent)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class ClaimAspect:
    aspect_id: str
    type: str
    text: str
    raw_span: str
    source: str = "rule"
    added_context: list[str] = field(default_factory=list)
    is_atomic: bool = True
    is_decontextualized: bool = True
    retrievability_score: int = 0
    quality: str = "diagnostic"
    drop_reason: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimAspectBundle:
    event_id: str
    claim: str
    extraction_version: str
    full_claim_anchor: ClaimAspect | None
    aspects: list[ClaimAspect]
    dropped_aspects: list[ClaimAspect]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "claim": self.claim,
            "extraction_version": self.extraction_version,
            "full_claim_anchor": self.full_claim_anchor.to_dict() if self.full_claim_anchor else None,
            "aspects": [aspect.to_dict() for aspect in self.aspects],
            "dropped_aspects": [aspect.to_dict() for aspect in self.dropped_aspects],
        }


@dataclass(frozen=True)
class _RawAspect:
    type: str
    text: str
    raw_span: str
    added_context: list[str] = field(default_factory=list)
    priority: int = 100


def extract_claim_aspects(
    claim: str,
    *,
    event_id: str = "",
    max_local_aspects: int = 8,
    include_full_claim: bool = True,
    min_retrievability_score: int = _MIN_DEFAULT_RETRIEVABILITY,
    min_tokens: int = _MIN_DEFAULT_TOKENS,
    max_tokens: int = _MAX_DEFAULT_TOKENS,
) -> ClaimAspectBundle:
    cleaned = clean_text(claim)
    base_id = str(event_id or _stable_id(cleaned))
    full_anchor = None
    if include_full_claim:
        full_features = aspect_feature_flags(cleaned)
        full_anchor = ClaimAspect(
            aspect_id=f"{base_id}:full",
            type=FULL_CLAIM_ASPECT_TYPE,
            text=cleaned,
            raw_span=cleaned,
            is_atomic=False,
            is_decontextualized=True,
            retrievability_score=retrievability_score(cleaned),
            quality="full_claim_anchor",
            features=full_features,
        )

    raw_aspects = _raw_aspect_candidates(cleaned)
    kept: list[ClaimAspect] = []
    dropped: list[ClaimAspect] = []
    seen_signatures: list[str] = []
    for raw in sorted(raw_aspects, key=lambda item: (item.priority, len(item.text), item.text.lower())):
        text = clean_text(raw.text)
        if not text:
            continue
        sig = _signature(text)
        if _is_duplicate_signature(sig, seen_signatures):
            dropped.append(
                _materialize_aspect(
                    raw,
                    aspect_id=f"{base_id}:d{len(dropped):02d}",
                    quality="debug_only",
                    drop_reason="duplicate_local_aspect",
                )
            )
            continue
        seen_signatures.append(sig)
        aspect = _materialize_aspect(raw, aspect_id=f"{base_id}:a{len(kept):02d}")
        drop_reason = _quality_drop_reason(
            aspect,
            min_retrievability_score=int(min_retrievability_score),
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
        )
        if drop_reason:
            dropped.append(
                _materialize_aspect(
                    raw,
                    aspect_id=f"{base_id}:d{len(dropped):02d}",
                    quality="debug_only",
                    drop_reason=drop_reason,
                )
            )
            continue
        kept.append(aspect)
        if len(kept) >= int(max_local_aspects):
            break

    return ClaimAspectBundle(
        event_id=str(event_id or ""),
        claim=cleaned,
        extraction_version=ASPECT_EXTRACTION_VERSION,
        full_claim_anchor=full_anchor,
        aspects=kept,
        dropped_aspects=dropped,
    )


def build_claim_aspect_bundle_from_texts(
    claim: str,
    aspect_texts: list[str],
    *,
    event_id: str = "",
    extraction_version: str = LLM_DECOMP_PLUS_VERSION,
    source: str = "llm_decomp_plus",
    aspect_type: str = "llm_subclaim",
    max_local_aspects: int = 5,
    include_full_claim: bool = True,
    min_tokens: int = 3,
    max_tokens: int = 60,
) -> ClaimAspectBundle:
    """Convert externally generated sub-claims into the local aspect schema."""
    cleaned_claim = clean_text(claim)
    base_id = str(event_id or _stable_id(cleaned_claim))
    full_anchor = None
    if include_full_claim:
        full_anchor = ClaimAspect(
            aspect_id=f"{base_id}:full",
            type=FULL_CLAIM_ASPECT_TYPE,
            text=cleaned_claim,
            raw_span=cleaned_claim,
            source=source,
            is_atomic=False,
            is_decontextualized=True,
            retrievability_score=retrievability_score(cleaned_claim),
            quality="full_claim_anchor",
            features=aspect_feature_flags(cleaned_claim),
        )

    kept: list[ClaimAspect] = []
    dropped: list[ClaimAspect] = []
    seen_signatures: list[str] = []
    for raw_idx, raw_text in enumerate(aspect_texts):
        text = clean_text(raw_text)
        if not text:
            continue
        sig = _signature(text)
        features = aspect_feature_flags(text)
        token_count = int(features.get("token_count", 0))
        drop_reason = None
        if _is_duplicate_signature(sig, seen_signatures):
            drop_reason = "duplicate_local_aspect"
        elif token_count < int(min_tokens):
            drop_reason = "too_short"
        elif token_count > int(max_tokens):
            drop_reason = "too_long"
        elif features.get("starts_with_context_cue"):
            drop_reason = "not_decontextualized"

        aspect = ClaimAspect(
            aspect_id=f"{base_id}:a{len(kept):02d}" if drop_reason is None else f"{base_id}:d{len(dropped):02d}",
            type=aspect_type,
            text=text,
            raw_span=text,
            source=source,
            is_atomic=bool(token_count <= int(max_tokens) and int(features.get("connector_count", 0)) <= 3),
            is_decontextualized=not bool(features.get("starts_with_context_cue")),
            retrievability_score=retrievability_score(text),
            quality="diagnostic" if drop_reason is None else "debug_only",
            drop_reason=drop_reason,
            features={**features, "llm_order": int(raw_idx)},
        )
        if drop_reason is not None:
            dropped.append(aspect)
            continue
        seen_signatures.append(sig)
        kept.append(aspect)
        if len(kept) >= int(max_local_aspects):
            break

    return ClaimAspectBundle(
        event_id=str(event_id or ""),
        claim=cleaned_claim,
        extraction_version=extraction_version,
        full_claim_anchor=full_anchor,
        aspects=kept,
        dropped_aspects=dropped,
    )


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:\"")


def aspect_feature_flags(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    tokens = content_tokens(cleaned)
    return {
        "token_count": len(tokens),
        "has_entity": bool(_valid_entities(cleaned)),
        "has_number_or_time": bool(_QUANTITY_RE.search(cleaned)),
        "has_action_verb": _has_action_word(cleaned),
        "has_policy_or_topic_noun": any(token.lower() in _POLICY_TOPIC_WORDS for token in tokens),
        "has_unresolved_pronoun": _has_unresolved_pronoun(cleaned),
        "starts_with_context_cue": _starts_with_context_cue(cleaned),
        "connector_count": _connector_count(cleaned),
        "quantity_count": len(_QUANTITY_RE.findall(cleaned)),
    }


def retrievability_score(text: str) -> int:
    features = aspect_feature_flags(text)
    score = 0
    score += 1 if features["has_entity"] else 0
    score += 1 if features["has_number_or_time"] else 0
    score += 1 if features["has_action_verb"] else 0
    score += 1 if features["has_policy_or_topic_noun"] else 0
    score += 1 if 5 <= int(features["token_count"]) <= 18 else 0
    score -= 2 if features["has_unresolved_pronoun"] and not features["has_entity"] else 0
    score -= 2 if features["starts_with_context_cue"] else 0
    score -= 1 if int(features["token_count"]) < 4 else 0
    return int(score)


def content_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%][A-Za-z0-9$%'.-]*", clean_text(text))


def _raw_aspect_candidates(claim: str) -> list[_RawAspect]:
    body = _strip_says_prefix(claim)
    main_subject = _main_subject(body)
    raw: list[_RawAspect] = []

    raw.extend(_contrast_aspects(body, main_subject))
    raw.extend(_otherwise_aspects(body))
    raw.extend(_parallel_action_aspects(body, main_subject))

    clauses = _split_clauses(body)
    previous_clause = ""
    for clause in clauses:
        if len(content_tokens(clause)) >= 4 and _signature(clause) != _signature(claim):
            text, added_context = _decontextualize_clause(
                clause,
                previous_clause=previous_clause,
                main_subject=main_subject,
            )
            raw.append(
                _RawAspect(
                    type="subclaim_clause",
                    text=text,
                    raw_span=clause,
                    added_context=added_context,
                    priority=10,
                )
            )
        previous_clause = clause

    for match in _QUANTITY_RE.finditer(claim):
        raw.append(
            _RawAspect(
                type="quantity_time",
                text=_window(claim, match.start(), match.end(), width=6),
                raw_span=match.group(),
                priority=20,
            )
        )

    for entity in _valid_entities(claim):
        start = claim.find(entity)
        if start >= 0:
            raw.append(
                _RawAspect(
                    type="entity_context",
                    text=_window(claim, start, start + len(entity), width=5),
                    raw_span=entity,
                    priority=30,
                )
            )

    for aspect_type, pattern in _CUE_PATTERNS.items():
        for match in pattern.finditer(claim):
            raw.append(
                _RawAspect(
                    type=aspect_type,
                    text=_window(claim, match.start(), match.end(), width=6),
                    raw_span=match.group(),
                    priority=_cue_priority(aspect_type),
                )
            )
    return raw


def _parallel_action_aspects(body: str, main_subject: str) -> list[_RawAspect]:
    match = re.match(
        r"^(?P<subject>.+?)\s+(?:has|have|had)?\s*"
        r"(?P<verb>advocated|promised|proposed|supported|opposed|called for|voted for|voted against)\s+"
        r"(?P<items>.+)$",
        body,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    subject = clean_text(match.group("subject")) or main_subject
    verb = clean_text(match.group("verb")).lower()
    items = _split_parallel_items(match.group("items"))
    if len(items) <= 1:
        return []
    output: list[_RawAspect] = []
    for item in items:
        item = clean_text(item)
        if len(content_tokens(item)) < 2:
            continue
        output.append(
            _RawAspect(
                type="entity_action",
                text=f"{subject} {verb} {item}",
                raw_span=item,
                added_context=[subject, verb],
                priority=5,
            )
        )
    return output


def _contrast_aspects(body: str, main_subject: str) -> list[_RawAspect]:
    match = re.search(r"(?P<previous>.+?)\bbut\s+instead\s+(?P<current>[^.;]+)", body, flags=re.IGNORECASE)
    if not match or not main_subject:
        return []
    previous = clean_text(match.group("previous"))
    current = clean_text(match.group("current"))
    candidate = _decontextualize_instead(f"instead {current}", previous_clause=previous, main_subject=main_subject)
    if not candidate:
        return []
    return [
        _RawAspect(
            type="contrast_alternative",
            text=candidate,
            raw_span=clean_text(match.group(0)),
            added_context=[main_subject],
            priority=4,
        )
    ]


def _otherwise_aspects(body: str) -> list[_RawAspect]:
    match = re.search(r"(?P<previous>.+?);\s*otherwise,?\s*(?P<current>[^.;]+)", body, flags=re.IGNORECASE)
    if not match:
        return []
    previous_context = _short_condition(match.group("previous"))
    current = clean_text(match.group("current"))
    if not previous_context or not current:
        return []
    return [
        _RawAspect(
            type="condition_contrast",
            text=f"{previous_context}; otherwise, {current}",
            raw_span=clean_text(match.group(0)),
            added_context=[previous_context],
            priority=4,
        )
    ]


def _split_parallel_items(text: str) -> list[str]:
    normalized = re.sub(r"\s+and\s+", ", ", clean_text(text), flags=re.IGNORECASE)
    return [clean_text(item) for item in normalized.split(",") if clean_text(item)]


def _split_clauses(body: str) -> list[str]:
    return [clean_text(part) for part in _CLAUSE_SPLIT_RE.split(body) if clean_text(part)]


def _decontextualize_clause(
    clause: str,
    *,
    previous_clause: str,
    main_subject: str,
) -> tuple[str, list[str]]:
    cleaned = clean_text(clause)
    added_context: list[str] = []
    lowered = cleaned.lower()
    if lowered.startswith("instead") and main_subject:
        candidate = _decontextualize_instead(cleaned, previous_clause=previous_clause, main_subject=main_subject)
        if candidate:
            return candidate, [main_subject]
    if lowered.startswith("otherwise") and previous_clause:
        previous_context = _short_condition(previous_clause)
        remainder = clean_text(re.sub(r"^otherwise,?\s*", "", cleaned, flags=re.IGNORECASE))
        if previous_context and remainder:
            return f"{previous_context}; otherwise, {remainder}", [previous_context]
    cleaned = _replace_person_pronouns(cleaned, main_subject)
    if main_subject and not _has_subject_anchor(cleaned):
        return f"{main_subject} {cleaned}", [main_subject]
    return cleaned, added_context


def _decontextualize_instead(clause: str, *, previous_clause: str, main_subject: str) -> str:
    previous = clean_text(previous_clause)
    current = clean_text(re.sub(r"^instead\s+", "", clause, flags=re.IGNORECASE))
    previous_match = re.search(r"(?:did\s+not|would\s+not|not)\s+use\s+(?P<object>.+)$", previous, flags=re.IGNORECASE)
    if not previous_match or not current:
        return ""
    old_object = clean_text(previous_match.group("object"))
    new_object = re.sub(r"\s*\(.+\)$", "", current).strip()
    return clean_text(f"{main_subject} used {new_object} instead of {old_object}")


def _short_condition(text: str) -> str:
    cleaned = clean_text(text)
    if "," in cleaned:
        cleaned = clean_text(cleaned.split(",", 1)[0])
    if len(content_tokens(cleaned)) > 14:
        cleaned = " ".join(content_tokens(cleaned)[:14])
    return cleaned


def _materialize_aspect(
    raw: _RawAspect,
    *,
    aspect_id: str,
    quality: str = "diagnostic",
    drop_reason: str | None = None,
) -> ClaimAspect:
    text = clean_text(raw.text)
    features = aspect_feature_flags(text)
    return ClaimAspect(
        aspect_id=aspect_id,
        type=raw.type,
        text=text,
        raw_span=clean_text(raw.raw_span),
        source="rule",
        added_context=[clean_text(item) for item in raw.added_context if clean_text(item)],
        is_atomic=_is_atomic(text, features),
        is_decontextualized=_is_decontextualized(text, features),
        retrievability_score=retrievability_score(text),
        quality=quality,
        drop_reason=drop_reason,
        features=features,
    )


def _quality_drop_reason(
    aspect: ClaimAspect,
    *,
    min_retrievability_score: int,
    min_tokens: int,
    max_tokens: int,
) -> str | None:
    token_count = int(aspect.features.get("token_count", 0))
    if token_count < int(min_tokens):
        return "too_short"
    if token_count > int(max_tokens):
        return "too_long"
    if not aspect.is_atomic:
        return "not_atomic"
    if not aspect.is_decontextualized:
        return "not_decontextualized"
    if int(aspect.retrievability_score) < int(min_retrievability_score):
        return "low_retrievability"
    return None


def _valid_entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in _ENTITY_RE.finditer(text):
        entity = clean_text(match.group())
        entity = re.sub(r"^(?:Says|When|On)\s+", "", entity, flags=re.IGNORECASE)
        if not entity or entity in _BAD_ENTITY_SPANS or len(entity) <= 2:
            continue
        if entity.lower() in {"says", "when"}:
            continue
        entities.append(entity)
    return entities


def _main_subject(text: str) -> str:
    entities = _valid_entities(text)
    return entities[0] if entities else ""


def _has_subject_anchor(text: str) -> bool:
    cleaned = clean_text(text)
    if _valid_entities(cleaned):
        return True
    return bool(re.match(r"(?i)^(?:people|armed civilians|congress|we|the|a|an)\b", cleaned))


def _replace_person_pronouns(text: str, subject: str) -> str:
    if not subject:
        return clean_text(text)
    output = re.sub(r"(?i)\b(he|him|she)\b", subject, text)
    output = re.sub(r"(?i)\b(his|her)\b", f"{subject}'s", output)
    return clean_text(output)


def _is_decontextualized(text: str, features: dict[str, Any]) -> bool:
    if features.get("starts_with_context_cue"):
        return False
    if features.get("has_unresolved_pronoun"):
        return False
    return True


def _is_atomic(text: str, features: dict[str, Any]) -> bool:
    token_count = int(features.get("token_count", 0))
    connector_count = int(features.get("connector_count", 0))
    quantity_count = int(features.get("quantity_count", 0))
    if token_count > 28:
        return False
    if connector_count > 2:
        return False
    if quantity_count > 2:
        return False
    return True


def _has_action_word(text: str) -> bool:
    tokens = {token.lower() for token in content_tokens(text)}
    if tokens & _ACTION_WORDS:
        return True
    return any(token.endswith("ed") or token.endswith("ing") for token in tokens)


def _has_unresolved_pronoun(text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:he|him|his|she|her|they|their|them|it|its)\b", text))


def _starts_with_context_cue(text: str) -> bool:
    tokens = content_tokens(text)
    return bool(tokens and tokens[0].lower() in _CONTEXT_LEADING_CUES)


def _connector_count(text: str) -> int:
    return len(re.findall(r"(?i)\b(?:and|but|because|where|otherwise|instead|then)\b", text))


def _window(text: str, start: int, end: int, *, width: int) -> str:
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return clean_text(text)
    center = 0
    for idx, (tok_start, tok_end, _) in enumerate(tokens):
        if tok_start <= start < tok_end or tok_start >= start:
            center = idx
            break
    lo = max(0, center - int(width))
    hi = min(len(tokens), center + int(width) + 1)
    return clean_text(" ".join(token for _, _, token in tokens[lo:hi]))


def _cue_priority(aspect_type: str) -> int:
    return {
        "negation": 40,
        "comparison": 50,
        "policy_action": 60,
        "causal_condition": 70,
    }.get(aspect_type, 90)


def _strip_says_prefix(text: str) -> str:
    return clean_text(re.sub(r"^Says\s+", "", clean_text(text), flags=re.IGNORECASE))


def _signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def _is_duplicate_signature(signature: str, seen_signatures: list[str]) -> bool:
    if not signature:
        return True
    for seen in seen_signatures:
        if signature == seen:
            return True
        if len(signature) > 26 and (signature in seen or seen in signature):
            return True
    return False


def _stable_id(text: str) -> str:
    return hashlib.sha1(clean_text(text).encode("utf-8")).hexdigest()[:12]
