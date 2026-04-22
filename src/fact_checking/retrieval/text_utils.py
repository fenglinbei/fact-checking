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



def lexical_overlap_f1(query: str, sentence: str) -> float:
    q = content_tokens(query)
    s = content_tokens(sentence)
    if not q or not s:
        return 0.0
    q_count = Counter(q)
    s_count = Counter(s)
    overlap = sum(min(q_count[k], s_count[k]) for k in q_count.keys() & s_count.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(s))
    recall = overlap / max(1, len(q))
    return 2.0 * precision * recall / max(1e-8, precision + recall)



def bm25_like_score(query: str, sentence: str) -> float:
    q = content_tokens(query)
    s = content_tokens(sentence)
    if not q or not s:
        return 0.0
    s_count = Counter(s)
    score = 0.0
    k1 = 1.2
    b = 0.75
    avgdl = 18.0
    dl = max(1, len(s))
    for term in set(q):
        tf = s_count.get(term, 0)
        if tf == 0:
            continue
        # Local-only approximation: use a bounded pseudo-IDF.
        idf = math.log(1.0 + (1.0 / (1.0 + tf))) + 0.5
        denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
        score += idf * (tf * (k1 + 1.0) / denom)
    return score
