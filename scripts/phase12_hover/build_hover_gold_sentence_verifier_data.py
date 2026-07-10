#!/usr/bin/env python3
"""Build HoVer S2 gold-evidence verifier data.

This is the HoVer S2 diagnostic path: it uses official HoVer
``supporting_facts`` to fetch sentence-level evidence from the HotpotQA
processed Wikipedia corpus, then writes prebuilt ``build_<split>.jsonl`` files
and a ``train.resolved.yaml`` consumable by ``sft.label_token_trainer``.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.prompts import build_training_row, load_prompt_tokenizer
from fact_checking.config import save_yaml
from fact_checking.data.io import load_split
from fact_checking.utils.io import save_json, write_jsonl
from fact_checking.utils.text import clean_text


DEFAULT_MODEL = "/data/models/Ministral-3-8B-Instruct-2512"
DEFAULT_OUTPUT_DIR = "outputs/sentence_trace_method/hover__ministral3_8b__gold_sentences_minmax9_9"
_PLACEHOLDER_DOT = "<DOT>"
_COMMON_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "i.e.", "e.g.", "u.s.", "u.k.", "no.", "fig.",
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Z0-9])")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HoVer S2 gold sentence verifier data.")
    p.add_argument("--train-raw", default="data/raw/HoVer/train.json")
    p.add_argument("--val-raw", default="data/raw/HoVer/val.json")
    p.add_argument("--wiki-root", default="data/raw/HoVer/wiki")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    p.add_argument("--deepspeed-config", default="configs/deepspeed/deepspeed_zero2_bsz1_ga4.json")
    p.add_argument("--evidence-mode", default="gold_sentences", choices=["gold_sentences", "gold_docs"])
    p.add_argument("--sentence-window", type=int, default=0)
    p.add_argument("--max-doc-sentences", type=int, default=20)
    p.add_argument("--missing-policy", default="error", choices=["error", "skip"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    build_dir = output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    required_titles = collect_required_titles([Path(args.train_raw), Path(args.val_raw)])
    wiki_pages = load_required_wiki_pages(Path(args.wiki_root), required_titles)
    tokenizer = load_prompt_tokenizer(str(args.model_name_or_path))
    prompt_cfg = build_prompt_config(model_name_or_path=str(args.model_name_or_path))

    split_paths: dict[str, str] = {}
    split_reports: dict[str, Any] = {}
    for split, raw_path in (("train", Path(args.train_raw)), ("val", Path(args.val_raw))):
        retrieval_rows, report = build_split_retrieval_rows(
            split=split,
            raw_path=raw_path,
            wiki_pages=wiki_pages,
            evidence_mode=str(args.evidence_mode),
            sentence_window=int(args.sentence_window),
            max_doc_sentences=int(args.max_doc_sentences),
            missing_policy=str(args.missing_policy),
            sample_limit=args.sample_limit,
            show_progress=not args.no_progress,
        )
        training_rows = [
            _build_training_row_with_hover_metadata(row, tokenizer=tokenizer, prompt_cfg=prompt_cfg)
            for row in retrieval_rows
        ]
        out_path = build_dir / f"build_{split}.jsonl"
        write_jsonl(training_rows, out_path)
        split_paths[split] = str(out_path)
        split_reports[split] = {
            **report,
            "build_path": str(out_path),
            "prompt_truncation_rate": _rate(row.get("was_truncated") for row in training_rows),
            "prompt_token_count": _summary([int(row.get("prompt_token_count", 0)) for row in training_rows]),
            "evidence_count": _summary([int(row.get("evidence_count", 0)) for row in training_rows]),
        }

    train_config = build_train_config(
        output_dir=output_dir,
        split_paths=split_paths,
        model_name_or_path=str(args.model_name_or_path),
        deepspeed_config=str(args.deepspeed_config),
    )
    train_config_path = output_dir / "train.resolved.yaml"
    save_yaml(train_config, train_config_path)

    report = {
        "status": "completed",
        "dataset": "hover",
        "label_schema": "hover2",
        "evidence_mode": str(args.evidence_mode),
        "sentence_window": int(args.sentence_window),
        "max_doc_sentences": int(args.max_doc_sentences),
        "wiki_root": str(args.wiki_root),
        "required_title_count": len(required_titles),
        "loaded_title_count": len(wiki_pages),
        "split_paths": split_paths,
        "train_config": str(train_config_path),
        "splits": split_reports,
        "notes": [
            "S2 uses official HoVer supporting_facts as gold evidence.",
            "Official HoVer test is claim-only and is intentionally not used.",
            "This diagnostic excludes open-domain retrieval and MREC selector errors.",
        ],
    }
    save_json(report, output_dir / "build_report.json")

    print(f"Wrote HoVer S2 gold verifier data to {output_dir}")
    for split, split_report in split_reports.items():
        print(
            "{split}: rows={rows} skipped={skipped} missing_sf={missing} trunc={trunc:.4f}".format(
                split=split,
                rows=split_report["n_rows"],
                skipped=split_report["skipped_total"],
                missing=split_report["missing_supporting_facts"],
                trunc=split_report["prompt_truncation_rate"],
            )
        )
    print(f"Train config: {train_config_path}")


def build_prompt_config(*, model_name_or_path: str) -> dict[str, Any]:
    return {
        "model_name_or_path": model_name_or_path,
        "max_length": 1024,
        "output_mode": "label_only",
        "label_format": "letter",
        "label_schema": "hover2",
        "system_prompt": None,
        "chat_template": {
            "mode": "tokenizer_default",
            "add_generation_prompt": True,
            "template_kwargs": {},
            "migration_note": "hover_s2_gold_sentence_ministral3",
        },
    }


def collect_required_titles(raw_paths: Iterable[Path]) -> set[str]:
    titles: set[str] = set()
    for raw_path in raw_paths:
        for sample in load_split(raw_path, dataset="hover", label_schema="hover2"):
            for title, _sent_idx in _supporting_fact_pairs(sample.metadata.get("supporting_facts")):
                titles.add(title)
    return titles


def load_required_wiki_pages(wiki_root: Path, required_titles: set[str]) -> dict[str, list[str]]:
    if not wiki_root.exists():
        raise FileNotFoundError(f"HoVer wiki corpus root does not exist: {wiki_root}")
    required_by_key = {_title_key(title): title for title in required_titles}
    pages: dict[str, list[str]] = {}
    for db_path in _sqlite_wiki_paths(wiki_root):
        missing_titles = set(required_titles) - set(pages)
        if not missing_titles:
            return pages
        pages.update(_load_required_sqlite_pages(db_path, missing_titles))

    required_by_key = {
        _title_key(title): title for title in required_titles if title not in pages
    }
    for item in _iter_wiki_records(wiki_root):
        title = _record_title(item)
        if not title:
            continue
        required_title = required_by_key.get(_title_key(title))
        if required_title is None or required_title in pages:
            continue
        sentences = _record_sentences(item)
        if sentences:
            pages[required_title] = sentences
        if len(pages) >= len(required_titles):
            break
    return pages


def _sqlite_wiki_paths(wiki_root: Path) -> list[Path]:
    if wiki_root.is_file():
        return [wiki_root] if wiki_root.suffix.lower() == ".db" else []
    preferred = wiki_root / "wiki_wo_links.db"
    paths: list[Path] = []
    if preferred.is_file():
        paths.append(preferred)
    paths.extend(path for path in sorted(wiki_root.glob("*.db")) if path != preferred)
    return paths


def _load_required_sqlite_pages(db_path: Path, required_titles: set[str]) -> dict[str, list[str]]:
    pages: dict[str, list[str]] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        for title in sorted(required_titles):
            row = _fetch_wiki_document(cursor, title)
            if row is None:
                continue
            sentences = _coerce_sentences(row[1])
            if sentences:
                pages[title] = sentences
    finally:
        conn.close()
    return pages


def _fetch_wiki_document(cursor: sqlite3.Cursor, title: str) -> tuple[Any, ...] | None:
    variants = [
        unicodedata.normalize("NFD", title),
        title,
        unicodedata.normalize("NFD", title.replace(" ", "_")),
        title.replace(" ", "_"),
    ]
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        row = cursor.execute("SELECT * FROM documents WHERE id=(?)", (variant,)).fetchone()
        if row is not None:
            return row
    return None


def build_split_retrieval_rows(
    *,
    split: str,
    raw_path: Path,
    wiki_pages: Mapping[str, list[str]],
    evidence_mode: str,
    sentence_window: int,
    max_doc_sentences: int,
    missing_policy: str,
    sample_limit: int | None,
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_split(raw_path, dataset="hover", label_schema="hover2")
    if sample_limit is not None:
        samples = samples[: int(sample_limit)]

    out_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    hop_counts: Counter[str] = Counter()
    missing_supporting_facts = 0
    recovered_supporting_facts = 0

    iterator = tqdm(
        samples,
        desc=f"hover-s2 [{split}]",
        unit="claim",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for sample in iterator:
        facts = _supporting_fact_pairs(sample.metadata.get("supporting_facts"))
        if not facts:
            skipped["no_supporting_facts"] += 1
            continue
        candidates, missing, recovered = _gold_candidates_for_sample(
            facts=facts,
            wiki_pages=wiki_pages,
            evidence_mode=evidence_mode,
            sentence_window=sentence_window,
            max_doc_sentences=max_doc_sentences,
        )
        missing_supporting_facts += missing
        recovered_supporting_facts += recovered
        if missing and missing_policy == "error":
            raise ValueError(
                f"{split}:{sample.event_id} missing supporting fact text for "
                f"{missing}/{len(facts)} supporting facts"
            )
        if not candidates:
            skipped["missing_supporting_fact"] += 1
            continue

        row = {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "label_schema": "hover2",
            "explain": sample.explain,
            "candidates": candidates,
            "hover_gold_evidence": {
                "mode": evidence_mode,
                "split": split,
                "supporting_fact_count": len(facts),
                "missing_supporting_fact_count": int(missing),
                "recovered_supporting_fact_count": int(recovered),
                "num_hops": sample.metadata.get("num_hops"),
                "hpqa_id": sample.metadata.get("hpqa_id"),
            },
        }
        out_rows.append(row)
        label_counts[str(sample.label)] += 1
        hop_counts[str(sample.metadata.get("num_hops", ""))] += 1

    report = {
        "split": split,
        "raw_path": str(raw_path),
        "n_raw_rows": len(samples),
        "n_rows": len(out_rows),
        "skipped": dict(skipped),
        "skipped_total": int(sum(skipped.values())),
        "labels": dict(label_counts),
        "num_hops": dict(hop_counts),
        "missing_supporting_facts": int(missing_supporting_facts),
        "recovered_supporting_facts": int(recovered_supporting_facts),
        "evidence_mode": evidence_mode,
        "sentence_window": int(sentence_window),
        "max_doc_sentences": int(max_doc_sentences),
    }
    return out_rows, report


def build_train_config(
    *,
    output_dir: Path,
    split_paths: Mapping[str, str],
    model_name_or_path: str,
    deepspeed_config: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    return {
        "label_schema": "hover2",
        "output_dir": str(output_dir / "train"),
        "eval_output_dir": str(output_dir / "eval"),
        "prompt_stats_output_dir": str(output_dir / "prompt_stats"),
        "data": {
            "train_candidates": str(split_paths["train"]),
            "val_candidates": str(split_paths["val"]),
            "test_candidates": str(split_paths.get("test") or split_paths["val"]),
        },
        "model_name_or_path": str(model_name_or_path),
        "baseline": {
            "variant": "hover_gold_sentences_minmax9_9",
            "chunking_strategy": "hover_gold_sentence",
            "label_schema": "hover2",
        },
        "sft_train": {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-5,
            "num_train_epochs": 12,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "bf16": True,
            "max_length": 1024,
            "logging_steps": 2,
            "save_steps": 100,
            "eval_steps": 100,
            "dataloader_num_workers": 4,
            "gradient_checkpointing": True,
            "use_flash_attention_2": True,
            "lr_scheduler_type": "cosine_with_restarts",
            "lr_scheduler_kwargs": {"num_cycles": 2},
            "max_grad_norm": 1.0,
            "padding": "longest",
            "use_length_bucket": True,
            "empty_cache_steps": 0,
            "empty_cache_on_eval": True,
            "empty_cache_on_save": True,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "early_stopping_patience": 8,
            "eval_log_predictions": 0,
            "label_schema": "hover2",
            "logit_adjust": {
                "enabled": False,
                "tau": 1.0,
            },
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "modules_to_save": None,
            },
            "label_token_ce": {
                "label_prefix": "Label:",
                "early_stopping_metric": "macro_f1",
                "class_weights": {
                    "supported": 1.0,
                    "not_supported": 1.0,
                },
                "ordinal_loss": {
                    "enabled": False,
                    "alpha": 0.0,
                    "normalize_distance": True,
                    "alpha_warmup_ratio": 0.0,
                },
            },
            "resolved_output_dir": True,
            "save_latest_state": True,
            "resume_latest_state": True,
        },
        "tracking": {
            "enabled": True,
            "backend": "swanlab",
        },
        "swanlab": {
            "project": "fact-checking-sentence-trace-method-hover",
            "experiment_name": "hover__ministral3_8b__gold_sentences_minmax9_9",
        },
        "train": {
            "deepspeed_config": str(deepspeed_config),
        },
    }


def _build_training_row_with_hover_metadata(
    retrieval_row: dict[str, Any],
    *,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    row = build_training_row(retrieval_row, tokenizer, prompt_cfg)
    if "hover_gold_evidence" in retrieval_row:
        row["hover_gold_evidence"] = retrieval_row["hover_gold_evidence"]
    return row


def _gold_candidates_for_sample(
    *,
    facts: list[tuple[str, int]],
    wiki_pages: Mapping[str, list[str]],
    evidence_mode: str,
    sentence_window: int,
    max_doc_sentences: int,
) -> tuple[list[dict[str, Any]], int, int]:
    seen_facts: set[tuple[str, int]] = set()
    candidates: list[dict[str, Any]] = []
    missing = 0
    recovered = 0
    if evidence_mode == "gold_docs":
        seen_titles: set[str] = set()
        for title, _sent_idx in facts:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sentences = wiki_pages.get(title)
            if not sentences:
                missing += 1
                continue
            kept = [sent for sent in sentences[: max(1, int(max_doc_sentences))] if sent.strip()]
            if not kept:
                missing += 1
                continue
            candidates.append(
                {
                    "report_id": title,
                    "sent_idx": 0,
                    "chunk_sent_indices": list(range(len(kept))),
                    "text": f"{title}: " + " ".join(kept),
                    "source_report": {"report_id": title, "domain": "wikipedia", "link": None},
                    "hover_page_title": title,
                    "hover_evidence_mode": "gold_docs",
                    "hover_gold_doc": True,
                    "hybrid_score": 1.0,
                }
            )
        return candidates, missing, recovered

    if evidence_mode != "gold_sentences":
        raise ValueError(f"Unsupported evidence_mode={evidence_mode!r}")

    window = max(0, int(sentence_window))
    for title, sent_idx in facts:
        key = (title, int(sent_idx))
        if key in seen_facts:
            continue
        seen_facts.add(key)
        sentences = wiki_pages.get(title)
        if not sentences or sent_idx < 0:
            missing += 1
            continue
        resolved_sent_idx = int(sent_idx)
        index_recovered = False
        if resolved_sent_idx >= len(sentences):
            resolved_sent_idx = len(sentences) - 1
            index_recovered = True
            recovered += 1
        start = max(0, resolved_sent_idx - window)
        end = min(len(sentences), resolved_sent_idx + window + 1)
        selected = [sent for sent in sentences[start:end] if sent.strip()]
        if not selected:
            missing += 1
            continue
        candidates.append(
            {
                "report_id": title,
                "sent_idx": resolved_sent_idx,
                "chunk_sent_indices": list(range(start, end)),
                "text": f"{title}: " + " ".join(selected),
                "source_report": {"report_id": title, "domain": "wikipedia", "link": None},
                "hover_page_title": title,
                "hover_requested_sent_idx": int(sent_idx),
                "hover_sent_idx": resolved_sent_idx,
                "hover_sentence_index_recovered": index_recovered,
                "hover_sentence_window": window,
                "hover_evidence_mode": "gold_sentences",
                "hover_gold_supporting_fact": True,
                "hybrid_score": 1.0,
            }
        )
    return candidates, missing, recovered


def _supporting_fact_pairs(value: Any) -> list[tuple[str, int]]:
    facts: list[tuple[str, int]] = []
    if not isinstance(value, list):
        return facts
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = clean_text(str(item[0])).replace("_", " ").strip()
        try:
            sent_idx = int(item[1])
        except (TypeError, ValueError):
            continue
        if title:
            facts.append((title, sent_idx))
    return facts


def _iter_wiki_records(root: Path) -> Iterator[dict[str, Any]]:
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.name.startswith(".") or not _is_supported_wiki_file(path):
            continue
        yield from _iter_json_objects(path)


def _is_supported_wiki_file(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return any(suffix in {".json", ".jsonl", ".gz", ".bz2"} for suffix in suffixes)


def _iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed_json = suffixes[-1:] in ([".bz2"], [".gz"]) and ".json" in suffixes and ".jsonl" not in suffixes
    if ".jsonl" in suffixes or (suffixes[-1:] in ([".bz2"], [".gz"]) and not compressed_json):
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return

    with _open_text(path) as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[":
            payload = json.load(f)
            for item in _records_from_payload(payload):
                yield item
            return
        if first == "{":
            text = f.read()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            yield item
                return
            for item in _records_from_payload(payload):
                yield item
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _records_from_payload(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        if _record_title(payload):
            yield payload
            return
        for title, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("title", title)
                yield item
            elif isinstance(value, list):
                yield {"title": title, "text": value}
            elif isinstance(value, str):
                yield {"title": title, "text": value}


def _record_title(item: Mapping[str, Any]) -> str:
    for key in ("title", "page_title", "doc_title"):
        value = str(item.get(key) or "").replace("_", " ").strip()
        if value:
            return clean_text(value)
    return ""


def _record_sentences(item: Mapping[str, Any]) -> list[str]:
    for key in ("sentences", "text", "document", "paragraphs"):
        if key not in item:
            continue
        sentences = _coerce_sentences(item.get(key))
        if sentences:
            return sentences
    return []


def _coerce_sentences(value: Any) -> list[str]:
    if isinstance(value, str):
        return [sent for sent in _split_hover_wiki_sentences(value) if sent.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = clean_text(item)
                if text:
                    out.append(text)
            elif isinstance(item, list):
                out.extend(_coerce_sentences(item))
            elif isinstance(item, dict):
                for key in ("sent", "sentence", "text"):
                    if key in item:
                        out.extend(_coerce_sentences(item[key]))
                        break
        return out
    return []


def _split_hover_wiki_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    protected = text
    for abbr in _COMMON_ABBREVIATIONS:
        escaped = re.escape(abbr)
        protected = re.sub(
            rf"(?<![A-Za-z0-9]){escaped}",
            lambda m: m.group(0).replace(".", _PLACEHOLDER_DOT),
            protected,
            flags=re.IGNORECASE,
        )

    parts = [part.strip() for part in _SENT_SPLIT_RE.split(protected) if part.strip()]
    return [part.replace(_PLACEHOLDER_DOT, ".").strip() for part in parts if part.strip()]


def _title_key(title: str) -> str:
    return clean_text(str(title)).replace("_", " ").casefold()


def _rate(values: Iterable[Any]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(1 for item in items if bool(item)) / len(items))


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


if __name__ == "__main__":
    main()
