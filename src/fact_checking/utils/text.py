from __future__ import annotations

import math
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$]+")
_PLACEHOLDER_DOT = "<DOT>"
_COMMON_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "i.e.", "e.g.", "u.s.", "u.k.", "no.", "fig.",
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Z0-9])")
_WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

def robust_sentence_split(text: str) -> list[str]:
    """Split report content into sentences with simple abbreviation protection."""
    text = clean_text(text)
    if not text:
        return []

    protected = text
    for abbr in _COMMON_ABBREVIATIONS:
        escaped = re.escape(abbr)
        protected = re.sub(
            escaped,
            lambda m: m.group(0).replace(".", _PLACEHOLDER_DOT),
            protected,
            flags=re.IGNORECASE,
        )

    parts = [p.strip() for p in _SENT_SPLIT_RE.split(protected) if p.strip()]
    sentences = [p.replace(_PLACEHOLDER_DOT, ".").strip() for p in parts]
    return [s for s in sentences if s]

def word_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(clean_text(text))]


def jaccard(a: str, b: str) -> float:
    sa = set(word_tokens(a))
    sb = set(word_tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def cosine_from_numpy(a, b) -> float:
    import numpy as np
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denom)


def safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


CLAUSE_CONNECTORS = {
    "and", "or", "but", "while", "although", "because", "if", "when", "since",
    "after", "before", "whereas", "unless", "despite", "however", "than"
}
TIME_HINTS = {
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "today", "yesterday", "tomorrow", "before", "after", "during"
}
COMPARISON_HINTS = {
    "more", "less", "fewer", "higher", "lower", "greater", "smaller",
    "increase", "decrease", "double", "half", "compared", "than"
}


def count_complexity_signals(text: str) -> dict[str, int]:
    tokens = word_tokens(text)
    token_set = set(tokens)
    connector_count = sum(1 for t in tokens if t in CLAUSE_CONNECTORS)
    time_count = sum(1 for t in tokens if t in TIME_HINTS)
    comp_count = sum(1 for t in tokens if t in COMPARISON_HINTS)
    digit_count = sum(ch.isdigit() for ch in text)
    punctuation_count = sum(ch in ",;:()" for ch in text)
    return {
        "num_tokens": len(tokens),
        "connector_count": connector_count,
        "time_count": time_count,
        "comparison_count": comp_count,
        "digit_count": digit_count,
        "punctuation_count": punctuation_count,
        "has_percent": 1 if "%" in text else 0,
        "has_money": 1 if "$" in text else 0,
    }
