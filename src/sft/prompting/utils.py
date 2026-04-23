import re
from typing import Any
from fact_checking.utils.text import robust_sentence_split

_WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

def _context_window(content: str, sent_idx: int, k: int) -> str:
    sents = robust_sentence_split(content)
    if not sents:
        return ""
    idx = min(max(int(sent_idx), 0), len(sents) - 1)
    left = max(0, idx - k)
    right = min(len(sents), idx + k + 1)
    snippet = [sents[pos] for pos in range(left, right)]
    return " ".join(snippet)


def build_evidence_block(row: dict[str, Any], top_k: int, use_context: bool, context_k: int) -> str:
    candidates = sorted(
        row.get("candidates", []),
        key=lambda x: float(x.get("hybrid_score", 0.0)),
        reverse=True,
    )[:top_k]
    lines: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        sent = str(cand.get("text", "")).strip()
        if use_context:
            report = cand.get("source_report", {}) if isinstance(cand.get("source_report", {}), dict) else {}
            content = str(report.get("content", ""))
            ctx = _context_window(content=content, sent_idx=int(cand.get("sent_idx", 0)), k=context_k)
            evidence_text = ctx or sent
        else:
            evidence_text = sent
        lines.append(f"[{i}] {evidence_text}")
    return "\n".join(lines)