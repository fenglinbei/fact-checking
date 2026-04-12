from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from liar_raw import LABEL2ID
from liar_raw.data.types import SampleRecord, SentenceRecord

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def naive_sentence_split(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sentences



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
        tokenized = report.get("tokenized")
        if tokenized:
            for sent_idx, token in enumerate(tokenized):
                sent = clean_text(str(token.get("sent", "")))
                if len(sent) < min_char_len:
                    continue
                yield SentenceRecord(
                    event_id=sample.event_id,
                    report_id=report_id,
                    sent_idx=sent_idx,
                    text=sent,
                    link=link,
                    domain=domain,
                    raw=token,
                )
        else:
            content = clean_text(str(report.get("content", "")))
            for sent_idx, sent in enumerate(naive_sentence_split(content)):
                if len(sent) < min_char_len:
                    continue
                yield SentenceRecord(
                    event_id=sample.event_id,
                    report_id=report_id,
                    sent_idx=sent_idx,
                    text=sent,
                    link=link,
                    domain=domain,
                    raw={},
                )
