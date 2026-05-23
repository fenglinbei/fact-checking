from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fact_checking.data.constants import LABEL2ID
from fact_checking.data.types import SampleRecord, SentenceRecord
from fact_checking.utils.text import clean_text, robust_sentence_split


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

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _coerce_evidence_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _iter_content_sentences(sample: SampleRecord, min_char_len: int) -> Iterable[SentenceRecord]:
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


def _iter_tokenized_sentences(sample: SampleRecord, min_char_len: int) -> Iterable[SentenceRecord]:
    for report_order, report in enumerate(sample.reports):
        report_id = report.get("report_id", "unknown")
        link = report.get("link")
        domain = report.get("domain")
        tokenized = report.get("tokenized")
        if not isinstance(tokenized, list):
            continue
        for sent_idx, item in enumerate(tokenized):
            if not isinstance(item, dict):
                continue
            text = clean_text(str(item.get("sent") or item.get("sentence") or item.get("text") or ""))
            if len(text) < min_char_len:
                continue
            yield SentenceRecord(
                event_id=sample.event_id,
                report_id=report_id,
                sent_idx=sent_idx,
                text=text,
                link=link,
                domain=domain,
                raw={
                    "raw_sentence_source": "tokenized",
                    "raw_is_evidence": _coerce_evidence_label(item.get("is_evidence", 0)),
                    "raw_evidence_label": item.get("is_evidence", 0),
                    "raw_report_order": report_order,
                    "raw_sent_order": sent_idx,
                },
            )


def iter_sentences(
    sample: SampleRecord,
    min_char_len: int = 10,
    source: str = "content",
) -> Iterable[SentenceRecord]:
    source = str(source or "content").strip().lower()
    if source in {"tokenized", "raw_tokenized", "raw"}:
        yield from _iter_tokenized_sentences(sample, min_char_len)
        return
    if source != "content":
        raise ValueError(f"Unsupported sentence source: {source!r}. Use content or tokenized.")
    yield from _iter_content_sentences(sample, min_char_len)
