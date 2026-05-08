import re
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_evidence_items(row: dict[str, Any], top_k: int) -> list[str]:
    candidates = sorted(
        row.get("candidates", []),
        key=lambda x: float(x.get("hybrid_score", 0.0)),
        reverse=True,
    )[:top_k]
    return [str(cand.get("text", "")).strip() for cand in candidates]


def build_evidence_block(row: dict[str, Any], top_k: int) -> str:
    lines: list[str] = []
    for i, evidence_text in enumerate(
        build_evidence_items(row, top_k=top_k),
        start=1,
    ):
        lines.append(f"[{i}] {evidence_text}")
    return "\n".join(lines)
