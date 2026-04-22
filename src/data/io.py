from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from src import LABEL2ID
from data.types import SampleRecord, SentenceRecord

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


_PLACEHOLDER_DOT = "<DOT>"
_COMMON_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "i.e.", "e.g.", "u.s.", "u.k.", "no.", "fig.",
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Z0-9])")


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



def load_split(path: str | Path) -> list[SampleRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = clean_text(str(item["label"])).lower()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown label: {label!r} in {path}")
        records.append(
            SampleRecord(
                event_id=str(item["event_id"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explain", ""))),
                reports=item.get("reports", []),
            )
        )
    return records



def iter_sentences(sample: SampleRecord, min_char_len: int = 10) -> Iterable[SentenceRecord]:
    for report in sample.reports:
        report_id = report.get("report_id", "unknown")
        link = report.get("link")
        domain = report.get("domain")
        content = clean_text(str(report.get("content", "")))
        for sent_idx, sent in enumerate(robust_sentence_split(content)):
            if len(sent) < min_char_len:
                continue
            yield SentenceRecord(
                event_id=sample.event_id,
                report_id=report_id,
                sent_idx=sent_idx,
                text=sent,
                link=link,
                domain=domain,
                raw=report,
            )
