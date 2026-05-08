from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "on", "in", "for", "by", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "as",
    "at", "from", "will", "would", "could", "should", "can", "may", "might", "do",
    "does", "did", "have", "has", "had", "not", "but", "if", "than", "then", "into",
    "their", "there", "about", "literally",
}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in _STOPWORDS]


def content_tokens_counter(text: str) -> tuple[Counter, int]:
    """Return (Counter of content tokens, total token count)."""
    toks = content_tokens(text)
    return Counter(toks), len(toks)


def lexical_overlap_f1(query: str, sentence: str) -> float:
    q_ctr, q_len = content_tokens_counter(query)
    s_ctr, s_len = content_tokens_counter(sentence)
    return lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)


def lexical_overlap_f1_from_counters(
    q_ctr: Counter, s_ctr: Counter, q_len: int, s_len: int
) -> float:
    if q_len == 0 or s_len == 0:
        return 0.0
    overlap = sum(min(q_ctr[k], s_ctr[k]) for k in q_ctr.keys() & s_ctr.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / s_len
    recall = overlap / q_len
    return 2.0 * precision * recall / max(1e-8, precision + recall)


def bm25_like_score(query: str, sentence: str) -> float:
    q_ctr, _ = content_tokens_counter(query)
    s_ctr, s_len = content_tokens_counter(sentence)
    return bm25_like_score_from_counters(q_ctr, s_ctr, s_len)


def bm25_like_score_from_counters(
    q_ctr: Counter, s_ctr: Counter, s_len: int
) -> float:
    if s_len == 0:
        return 0.0
    score = 0.0
    k1 = 1.2
    b = 0.75
    avgdl = 18.0
    dl = max(1, s_len)
    for term in q_ctr:
        tf = s_ctr.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1.0 + (1.0 / (1.0 + tf))) + 0.5
        denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
        score += idf * (tf * (k1 + 1.0) / denom)
    return score
