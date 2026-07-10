#!/usr/bin/env python3
"""Build HoVer S3 open-domain BM25 page + sentence-MMR verifier data."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.prompts import build_training_row, load_prompt_tokenizer
from fact_checking.config import save_yaml
from fact_checking.data.io import load_split
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)
from fact_checking.utils.io import save_json, write_jsonl
from fact_checking.utils.text import clean_text
from scripts.phase12_hover.build_hover_gold_sentence_verifier_data import (
    DEFAULT_MODEL,
    _coerce_sentences,
    _fetch_wiki_document,
    _sqlite_wiki_paths,
    _summary,
    _supporting_fact_pairs,
    build_prompt_config,
    build_train_config,
)


DEFAULT_OUTPUT_DIR = "outputs/sentence_trace_method/hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9"
DEFAULT_INDEX_DB = "outputs/cache/hover/wiki_index/wiki_fts.db"
_WORKER_INDEX_CONN: sqlite3.Connection | None = None
_WORKER_WIKI_CONN: sqlite3.Connection | None = None
_WORKER_PAGE_CACHE: OrderedDict[str, list[str]] | None = None
_WORKER_PAGE_CACHE_SIZE = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HoVer S3 BM25 page retrieval baseline.")
    p.add_argument("--train-raw", default="data/raw/HoVer/train.json")
    p.add_argument("--val-raw", default="data/raw/HoVer/val.json")
    p.add_argument("--wiki-root", default="data/raw/HoVer/wiki")
    p.add_argument("--index-db", default=DEFAULT_INDEX_DB)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    p.add_argument("--deepspeed-config", default="configs/deepspeed/deepspeed_zero2_bsz1_ga4.json")
    p.add_argument("--page-top-k", type=int, default=100)
    p.add_argument("--sentence-pool-k", type=int, default=128)
    p.add_argument("--top-k", type=int, default=9)
    p.add_argument("--mmr-lambda", type=float, default=0.70)
    p.add_argument("--splits", default="train,val", help="Comma-separated subset of train,val.")
    p.add_argument("--retrieval-stage", choices=("sentences", "pages"), default="sentences")
    p.add_argument("--page-query-mode", choices=("text", "title"), default="text")
    p.add_argument("--page-cache-size", type=int, default=50000)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--force-index", action="store_true")
    p.add_argument("--index-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wiki_db = _resolve_wiki_db(Path(args.wiki_root))
    index_db = Path(args.index_db)
    output_dir = Path(args.output_dir)
    build_dir = output_dir / "build"
    retrieval_dir = output_dir / "retrieval"
    build_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    index_report = build_fts_index(
        wiki_db=wiki_db,
        index_db=index_db,
        force=bool(args.force_index),
        limit=args.index_limit,
        show_progress=not args.no_progress,
    )

    split_sources = _parse_requested_splits(
        args.splits,
        {
            "train": Path(args.train_raw),
            "val": Path(args.val_raw),
        },
    )
    tokenizer = None
    prompt_cfg: dict[str, Any] | None = None
    if args.retrieval_stage == "sentences":
        tokenizer = load_prompt_tokenizer(str(args.model_name_or_path))
        prompt_cfg = build_prompt_config(model_name_or_path=str(args.model_name_or_path))

    split_paths: dict[str, str] = {}
    retrieval_paths: dict[str, str] = {}
    split_reports: dict[str, Any] = {}

    for split, raw_path in split_sources.items():
        if args.retrieval_stage == "pages":
            retrieval_rows, report = build_split_page_retrieval_rows(
                split=split,
                raw_path=raw_path,
                index_db=index_db,
                page_top_k=int(args.page_top_k),
                page_query_mode=str(args.page_query_mode),
                sample_limit=args.sample_limit,
                show_progress=not args.no_progress,
                num_workers=int(args.num_workers),
            )
            retrieval_path = retrieval_dir / f"page_retrieval_{split}.jsonl"
            write_jsonl(retrieval_rows, retrieval_path)
            retrieval_paths[split] = str(retrieval_path)
            split_reports[split] = {
                **report,
                "retrieval_path": str(retrieval_path),
            }
            continue

        retrieval_rows, report = build_split_open_retrieval_rows(
            split=split,
            raw_path=raw_path,
            wiki_db=wiki_db,
            index_db=index_db,
            page_top_k=int(args.page_top_k),
            sentence_pool_k=int(args.sentence_pool_k),
            top_k=int(args.top_k),
            mmr_lambda=float(args.mmr_lambda),
            sample_limit=args.sample_limit,
            show_progress=not args.no_progress,
            page_cache_size=int(args.page_cache_size),
            page_query_mode=str(args.page_query_mode),
            num_workers=int(args.num_workers),
        )
        retrieval_path = retrieval_dir / f"retrieval_{split}.jsonl"
        write_jsonl(retrieval_rows, retrieval_path)
        training_rows = [
            _build_training_row_with_s3_metadata(row, tokenizer=tokenizer, prompt_cfg=prompt_cfg or {})
            for row in retrieval_rows
        ]
        build_path = build_dir / f"build_{split}.jsonl"
        write_jsonl(training_rows, build_path)
        retrieval_paths[split] = str(retrieval_path)
        split_paths[split] = str(build_path)
        split_reports[split] = {
            **report,
            "retrieval_path": str(retrieval_path),
            "build_path": str(build_path),
            "prompt_truncation_rate": _rate(row.get("was_truncated") for row in training_rows),
            "prompt_token_count": _summary([int(row.get("prompt_token_count", 0)) for row in training_rows]),
            "evidence_count": _summary([int(row.get("evidence_count", 0)) for row in training_rows]),
        }

    train_config_path: Path | None = None
    if args.retrieval_stage == "sentences" and {"train", "val"} <= set(split_paths):
        train_config = build_train_config(
            output_dir=output_dir,
            split_paths=split_paths,
            model_name_or_path=str(args.model_name_or_path),
            deepspeed_config=str(args.deepspeed_config),
        )
        train_config["baseline"]["variant"] = "hover_bm25_page_mmr_sentence_minmax9_9"
        train_config["baseline"]["chunking_strategy"] = "hover_bm25_page_sentence"
        train_config["swanlab"]["experiment_name"] = "hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9"
        train_config_path = output_dir / "train.resolved.yaml"
        save_yaml(train_config, train_config_path)

    report = {
        "status": "completed",
        "dataset": "hover",
        "label_schema": "hover2",
        "retrieval_mode": "bm25_page_mmr_sentence"
        if args.retrieval_stage == "sentences"
        else "bm25_page_only",
        "retrieval_stage": str(args.retrieval_stage),
        "wiki_db": str(wiki_db),
        "index_db": str(index_db),
        "index": index_report,
        "requested_splits": list(split_sources),
        "page_top_k": int(args.page_top_k),
        "page_query_mode": str(args.page_query_mode),
        "sentence_pool_k": int(args.sentence_pool_k),
        "top_k": int(args.top_k),
        "mmr_lambda": float(args.mmr_lambda),
        "page_cache_size": int(args.page_cache_size),
        "num_workers": int(args.num_workers),
        "retrieval_paths": retrieval_paths,
        "split_paths": split_paths,
        "train_config": str(train_config_path) if train_config_path is not None else None,
        "splits": split_reports,
        "notes": [
            "S3 uses claim-only BM25/FTS page retrieval over the HoVer Wikipedia corpus.",
            "Sentence evidence is selected from retrieved pages using lexical/BM25 hybrid MMR.",
            "Official HoVer test is claim-only and is intentionally not used.",
        ],
    }
    save_json(report, output_dir / "build_report.json")
    print(f"Wrote HoVer S3 retrieval baseline data to {output_dir}")
    for split, split_report in split_reports.items():
        passage = split_report["passage"]
        sentence = split_report["sentence"]
        print(
            f"{split}: rows={split_report['n_rows']} "
            f"passage_all_recall@{args.page_top_k}={passage['all_recall_at_k']:.4f} "
            f"sentence_selected_recall={sentence['selected_recall']:.4f}"
        )
    if train_config_path is not None:
        print(f"Train config: {train_config_path}")


def build_fts_index(
    *,
    wiki_db: Path,
    index_db: Path,
    force: bool = False,
    limit: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    if index_db.exists() and not force:
        count = _index_count(index_db)
        if count > 0:
            title_report = _ensure_title_fts_index(
                wiki_db=wiki_db,
                index_db=index_db,
                limit=limit,
                show_progress=show_progress,
            )
            return {
                "status": "reused",
                "indexed_documents": count,
                "path": str(index_db),
                "title_index": title_report,
            }
    index_db.parent.mkdir(parents=True, exist_ok=True)
    if index_db.exists():
        index_db.unlink()

    src = sqlite3.connect(str(wiki_db))
    dst = sqlite3.connect(str(index_db))
    try:
        dst.execute("CREATE VIRTUAL TABLE wiki_fts USING fts5(title UNINDEXED, text)")
        dst.execute("CREATE VIRTUAL TABLE wiki_title_fts USING fts5(title)")
        cursor = src.execute("SELECT id, text FROM documents")
        n = 0
        iterator = cursor if limit is None else _limited(cursor, int(limit))
        for title, text in tqdm(iterator, desc="hover-s3 index", unit="page", disable=not show_progress):
            clean_title = clean_text(str(title)).replace("_", " ")
            dst.execute("INSERT INTO wiki_fts(title, text) VALUES (?, ?)", (str(title), str(text or "")))
            dst.execute("INSERT INTO wiki_title_fts(title) VALUES (?)", (clean_title,))
            n += 1
            if n % 5000 == 0:
                dst.commit()
        dst.commit()
    finally:
        src.close()
        dst.close()
    return {
        "status": "built",
        "indexed_documents": int(n),
        "path": str(index_db),
        "title_index": {"status": "built", "indexed_titles": int(n)},
    }


def query_pages(
    index_db: Path,
    claim: str,
    top_k: int,
    page_query_mode: str = "text",
) -> list[dict[str, Any]]:
    conn = _open_readonly_connection(index_db)
    try:
        return query_pages_from_conn(conn, claim, top_k, page_query_mode=page_query_mode)
    finally:
        conn.close()


def query_pages_from_conn(
    conn: sqlite3.Connection,
    claim: str,
    top_k: int,
    page_query_mode: str = "text",
) -> list[dict[str, Any]]:
    if page_query_mode == "title":
        return _query_title_pages_from_conn(conn, claim=claim, top_k=top_k)
    if page_query_mode != "text":
        raise ValueError(f"Unsupported page_query_mode={page_query_mode!r}")
    query = _fts_query(claim)
    if not query:
        return []
    rows = conn.execute(
        """
        SELECT title, bm25(wiki_fts) AS score
        FROM wiki_fts
        WHERE wiki_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, int(top_k)),
    ).fetchall()
    return [
        {"title": clean_text(str(title)).replace("_", " "), "rank": idx + 1, "bm25_score": float(score)}
        for idx, (title, score) in enumerate(rows)
    ]


def _query_title_pages_from_conn(conn: sqlite3.Connection, *, claim: str, top_k: int) -> list[dict[str, Any]]:
    query = _fts_query(claim, min_token_len=3, max_tokens=24)
    if not query:
        return []
    rows = conn.execute(
        """
        SELECT title, bm25(wiki_title_fts) AS score
        FROM wiki_title_fts
        WHERE wiki_title_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, int(top_k)),
    ).fetchall()
    return [
        {"title": clean_text(str(title)).replace("_", " "), "rank": idx + 1, "bm25_score": float(score)}
        for idx, (title, score) in enumerate(rows)
    ]


def build_split_page_retrieval_rows(
    *,
    split: str,
    raw_path: Path,
    index_db: Path,
    page_top_k: int,
    sample_limit: int | None,
    page_query_mode: str = "text",
    show_progress: bool = False,
    num_workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_split(raw_path, dataset="hover", label_schema="hover2")
    if sample_limit is not None:
        samples = samples[: int(sample_limit)]
    payloads = [_sample_payload(sample) for sample in samples]

    rows: list[dict[str, Any]] = []
    metrics = _MetricAccumulator()
    label_counts: Counter[str] = Counter()
    hop_counts: Counter[str] = Counter()

    if int(num_workers) > 1 and len(payloads) > 1:
        tasks = (
            (payload, split, int(page_top_k), str(page_query_mode))
            for payload in payloads
        )
        with ProcessPoolExecutor(
            max_workers=int(num_workers),
            initializer=_init_page_worker,
            initargs=(str(index_db),),
        ) as executor:
            results = executor.map(_process_page_payload_worker, tasks, chunksize=_chunksize(len(payloads), int(num_workers)))
            iterator = tqdm(
                results,
                total=len(payloads),
                desc=f"hover-s3 pages [{split}]",
                unit="claim",
                disable=not show_progress,
            )
            for result in iterator:
                _record_retrieval_result(result, rows=rows, metrics=metrics, label_counts=label_counts, hop_counts=hop_counts)
    else:
        index_conn = _open_readonly_connection(index_db)
        try:
            iterator = tqdm(payloads, desc=f"hover-s3 pages [{split}]", unit="claim", disable=not show_progress)
            for payload in iterator:
                result = _page_result_for_payload(
                    payload=payload,
                    split=split,
                    index_conn=index_conn,
                    page_top_k=int(page_top_k),
                    page_query_mode=str(page_query_mode),
                )
                _record_retrieval_result(result, rows=rows, metrics=metrics, label_counts=label_counts, hop_counts=hop_counts)
        finally:
            index_conn.close()

    report = {
        "split": split,
        "raw_path": str(raw_path),
        "n_rows": len(rows),
        "labels": dict(label_counts),
        "num_hops": dict(hop_counts),
        "page_query_mode": str(page_query_mode),
        **metrics.report(),
    }
    return rows, report


def build_split_open_retrieval_rows(
    *,
    split: str,
    raw_path: Path,
    wiki_db: Path,
    index_db: Path,
    page_top_k: int,
    sentence_pool_k: int,
    top_k: int,
    mmr_lambda: float,
    sample_limit: int | None,
    show_progress: bool = False,
    page_cache_size: int = 0,
    page_query_mode: str = "text",
    num_workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_split(raw_path, dataset="hover", label_schema="hover2")
    if sample_limit is not None:
        samples = samples[: int(sample_limit)]
    payloads = [_sample_payload(sample) for sample in samples]

    rows: list[dict[str, Any]] = []
    metrics = _MetricAccumulator()
    label_counts: Counter[str] = Counter()
    hop_counts: Counter[str] = Counter()

    if int(num_workers) > 1 and len(payloads) > 1:
        tasks = (
            (
                payload,
                split,
                int(page_top_k),
                int(sentence_pool_k),
                int(top_k),
                float(mmr_lambda),
                str(page_query_mode),
            )
            for payload in payloads
        )
        with ProcessPoolExecutor(
            max_workers=int(num_workers),
            initializer=_init_sentence_worker,
            initargs=(str(index_db), str(wiki_db), int(page_cache_size)),
        ) as executor:
            results = executor.map(
                _process_sentence_payload_worker,
                tasks,
                chunksize=_chunksize(len(payloads), int(num_workers)),
            )
            iterator = tqdm(
                results,
                total=len(payloads),
                desc=f"hover-s3 [{split}]",
                unit="claim",
                disable=not show_progress,
            )
            for result in iterator:
                _record_retrieval_result(result, rows=rows, metrics=metrics, label_counts=label_counts, hop_counts=hop_counts)
    else:
        conn = _open_readonly_connection(wiki_db)
        index_conn = _open_readonly_connection(index_db)
        page_cache: OrderedDict[str, list[str]] | None = OrderedDict() if page_cache_size > 0 else None
        try:
            iterator = tqdm(payloads, desc=f"hover-s3 [{split}]", unit="claim", disable=not show_progress)
            for payload in iterator:
                result = _sentence_result_for_payload(
                    payload=payload,
                    split=split,
                    wiki_conn=conn,
                    index_conn=index_conn,
                    page_top_k=int(page_top_k),
                    sentence_pool_k=int(sentence_pool_k),
                    top_k=int(top_k),
                    mmr_lambda=float(mmr_lambda),
                    page_query_mode=str(page_query_mode),
                    page_cache=page_cache,
                    page_cache_size=int(page_cache_size),
                )
                _record_retrieval_result(result, rows=rows, metrics=metrics, label_counts=label_counts, hop_counts=hop_counts)
        finally:
            conn.close()
            index_conn.close()

    report = {
        "split": split,
        "raw_path": str(raw_path),
        "n_rows": len(rows),
        "labels": dict(label_counts),
        "num_hops": dict(hop_counts),
        "page_query_mode": str(page_query_mode),
        **metrics.report(),
    }
    return rows, report


def _sample_payload(sample: Any) -> dict[str, Any]:
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "metadata": dict(sample.metadata or {}),
    }


def _open_readonly_connection(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    for pragma in (
        "PRAGMA query_only=ON",
        "PRAGMA cache_size=-200000",
        "PRAGMA mmap_size=1073741824",
    ):
        try:
            conn.execute(pragma)
        except sqlite3.Error:
            pass
    return conn


def _init_page_worker(index_db: str) -> None:
    global _WORKER_INDEX_CONN
    _WORKER_INDEX_CONN = _open_readonly_connection(index_db)


def _init_sentence_worker(index_db: str, wiki_db: str, page_cache_size: int) -> None:
    global _WORKER_INDEX_CONN, _WORKER_WIKI_CONN, _WORKER_PAGE_CACHE, _WORKER_PAGE_CACHE_SIZE
    _WORKER_INDEX_CONN = _open_readonly_connection(index_db)
    _WORKER_WIKI_CONN = _open_readonly_connection(wiki_db)
    _WORKER_PAGE_CACHE_SIZE = int(page_cache_size)
    _WORKER_PAGE_CACHE = OrderedDict() if int(page_cache_size) > 0 else None


def _process_page_payload_worker(args: tuple[dict[str, Any], str, int, str]) -> dict[str, Any]:
    payload, split, page_top_k, page_query_mode = args
    if _WORKER_INDEX_CONN is None:
        raise RuntimeError("page worker was not initialized")
    return _page_result_for_payload(
        payload=payload,
        split=split,
        index_conn=_WORKER_INDEX_CONN,
        page_top_k=page_top_k,
        page_query_mode=page_query_mode,
    )


def _process_sentence_payload_worker(
    args: tuple[dict[str, Any], str, int, int, int, float, str]
) -> dict[str, Any]:
    payload, split, page_top_k, sentence_pool_k, top_k, mmr_lambda, page_query_mode = args
    if _WORKER_INDEX_CONN is None or _WORKER_WIKI_CONN is None:
        raise RuntimeError("sentence worker was not initialized")
    return _sentence_result_for_payload(
        payload=payload,
        split=split,
        wiki_conn=_WORKER_WIKI_CONN,
        index_conn=_WORKER_INDEX_CONN,
        page_top_k=page_top_k,
        sentence_pool_k=sentence_pool_k,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        page_query_mode=page_query_mode,
        page_cache=_WORKER_PAGE_CACHE,
        page_cache_size=_WORKER_PAGE_CACHE_SIZE,
    )


def _page_result_for_payload(
    *,
    payload: Mapping[str, Any],
    split: str,
    index_conn: sqlite3.Connection,
    page_top_k: int,
    page_query_mode: str,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    gold_pairs = _supporting_fact_pairs(metadata.get("supporting_facts"))
    page_hits = query_pages_from_conn(
        index_conn,
        str(payload.get("claim") or ""),
        top_k=page_top_k,
        page_query_mode=page_query_mode,
    )
    row = {
        "event_id": payload.get("event_id"),
        "claim": payload.get("claim"),
        "label": payload.get("label"),
        "label_schema": "hover2",
        "explain": payload.get("explain"),
        "candidates": [],
        "hover_s3_retrieval": {
            "mode": "bm25_page_only",
            "split": split,
            "page_query_mode": str(page_query_mode),
            "retrieved_page_count": len(page_hits),
            "page_top_k": int(page_top_k),
            "supporting_facts": [[title, idx] for title, idx in gold_pairs],
            "num_hops": metadata.get("num_hops"),
            "hpqa_id": metadata.get("hpqa_id"),
        },
        "retrieved_pages": page_hits,
    }
    return _retrieval_result(
        row=row,
        page_hits=page_hits,
        selected_candidates=[],
        gold_pairs=gold_pairs,
        label=str(payload.get("label")),
        hop=str(metadata.get("num_hops", "")),
    )


def _sentence_result_for_payload(
    *,
    payload: Mapping[str, Any],
    split: str,
    wiki_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    page_top_k: int,
    sentence_pool_k: int,
    top_k: int,
    mmr_lambda: float,
    page_query_mode: str,
    page_cache: OrderedDict[str, list[str]] | None,
    page_cache_size: int,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    claim = str(payload.get("claim") or "")
    gold_pairs = _supporting_fact_pairs(metadata.get("supporting_facts"))
    page_hits = query_pages_from_conn(
        index_conn,
        claim,
        top_k=page_top_k,
        page_query_mode=page_query_mode,
    )
    pages = _load_hit_pages(
        wiki_conn,
        page_hits,
        page_cache=page_cache,
        page_cache_size=page_cache_size,
    )
    sentence_pool = _candidate_sentences_for_pages(
        claim=claim,
        pages=pages,
        sentence_pool_k=sentence_pool_k,
    )
    candidates = _select_sentence_mmr(sentence_pool, top_k=top_k, mmr_lambda=mmr_lambda)
    row = {
        "event_id": payload.get("event_id"),
        "claim": payload.get("claim"),
        "label": payload.get("label"),
        "label_schema": "hover2",
        "explain": payload.get("explain"),
        "candidates": candidates,
        "hover_s3_retrieval": {
            "mode": "bm25_page_mmr_sentence",
            "split": split,
            "page_query_mode": str(page_query_mode),
            "retrieved_page_count": len(page_hits),
            "page_top_k": int(page_top_k),
            "sentence_pool_k": int(sentence_pool_k),
            "top_k": int(top_k),
            "mmr_lambda": float(mmr_lambda),
            "supporting_facts": [[title, idx] for title, idx in gold_pairs],
            "num_hops": metadata.get("num_hops"),
            "hpqa_id": metadata.get("hpqa_id"),
        },
        "retrieved_pages": page_hits,
    }
    return _retrieval_result(
        row=row,
        page_hits=page_hits,
        selected_candidates=candidates,
        gold_pairs=gold_pairs,
        label=str(payload.get("label")),
        hop=str(metadata.get("num_hops", "")),
    )


def _retrieval_result(
    *,
    row: dict[str, Any],
    page_hits: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    gold_pairs: list[tuple[str, int]],
    label: str,
    hop: str,
) -> dict[str, Any]:
    return {
        "row": row,
        "retrieved_titles": [str(hit["title"]) for hit in page_hits],
        "selected_candidates": selected_candidates,
        "gold_titles": [title for title, _idx in gold_pairs],
        "gold_sentence_keys": [[title, int(idx)] for title, idx in gold_pairs],
        "label": label,
        "hop": hop,
    }


def _record_retrieval_result(
    result: Mapping[str, Any],
    *,
    rows: list[dict[str, Any]],
    metrics: "_MetricAccumulator",
    label_counts: Counter[str],
    hop_counts: Counter[str],
) -> None:
    row = dict(result["row"])
    rows.append(row)
    gold_titles = {str(title) for title in result.get("gold_titles") or []}
    gold_sentence_keys = {
        (str(item[0]), int(item[1]))
        for item in result.get("gold_sentence_keys") or []
        if isinstance(item, (list, tuple)) and len(item) >= 2
    }
    metrics.add(
        retrieved_titles=[str(title) for title in result.get("retrieved_titles") or []],
        selected_candidates=list(result.get("selected_candidates") or []),
        gold_titles=gold_titles,
        gold_sentence_keys=gold_sentence_keys,
        hop=str(result.get("hop") or ""),
    )
    label_counts[str(result.get("label") or "")] += 1
    hop_counts[str(result.get("hop") or "")] += 1


def _chunksize(n_items: int, n_workers: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(16, math.ceil(n_items / max(1, n_workers * 16))))


def _parse_requested_splits(splits: str, split_paths: Mapping[str, Path]) -> dict[str, Path]:
    requested = [part.strip() for part in str(splits).split(",") if part.strip()]
    if not requested:
        raise ValueError("--splits must include at least one split")
    unknown = [split for split in requested if split not in split_paths]
    if unknown:
        raise ValueError(f"Unsupported --splits values: {unknown}; expected one or more of {sorted(split_paths)}")
    return {split: split_paths[split] for split in split_paths if split in requested}


def _resolve_wiki_db(wiki_root: Path) -> Path:
    paths = _sqlite_wiki_paths(wiki_root)
    if not paths:
        raise FileNotFoundError(f"Could not find wiki_wo_links.db under {wiki_root}")
    return paths[0]


def _limited(iterator: Iterable[Any], limit: int) -> Iterator[Any]:
    for idx, item in enumerate(iterator):
        if idx >= limit:
            break
        yield item


def _index_count(index_db: Path) -> int:
    conn = sqlite3.connect(str(index_db))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM wiki_fts").fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _ensure_title_fts_index(
    *,
    wiki_db: Path,
    index_db: Path,
    limit: int | None,
    show_progress: bool,
) -> dict[str, Any]:
    existing = _title_index_count(index_db)
    expected = int(limit) if limit is not None else None
    if existing > 0 and (expected is None or existing >= expected):
        return {"status": "reused", "indexed_titles": int(existing)}

    src = sqlite3.connect(str(wiki_db))
    dst = sqlite3.connect(str(index_db))
    try:
        dst.execute("DROP TABLE IF EXISTS wiki_title_fts")
        dst.execute("CREATE VIRTUAL TABLE wiki_title_fts USING fts5(title)")
        cursor = src.execute("SELECT id FROM documents")
        iterator = cursor if limit is None else _limited(cursor, int(limit))
        n = 0
        for (title,) in tqdm(iterator, desc="hover-s3 title index", unit="title", disable=not show_progress):
            clean_title = clean_text(str(title)).replace("_", " ")
            dst.execute("INSERT INTO wiki_title_fts(title) VALUES (?)", (clean_title,))
            n += 1
            if n % 10000 == 0:
                dst.commit()
        dst.commit()
    finally:
        src.close()
        dst.close()
    return {"status": "built", "indexed_titles": int(n)}


def _title_index_count(index_db: Path) -> int:
    conn = sqlite3.connect(str(index_db))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM wiki_title_fts").fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _fts_query(text: str, *, min_token_len: int = 2, max_tokens: int = 32) -> str:
    toks = []
    seen: set[str] = set()
    for tok in content_tokens(text):
        tok = "".join(ch for ch in tok if ch.isalnum() or ch == "_")
        if len(tok) < int(min_token_len) or tok in seen:
            continue
        seen.add(tok)
        toks.append(tok)
    return " OR ".join(toks[: int(max_tokens)])


def _load_hit_pages(
    conn: sqlite3.Connection,
    page_hits: list[dict[str, Any]],
    *,
    page_cache: OrderedDict[str, list[str]] | None = None,
    page_cache_size: int = 0,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    cursor = conn.cursor()
    for hit in page_hits:
        hit_title = str(hit["title"])
        cache_key = clean_text(hit_title).replace("_", " ")
        if page_cache is not None and cache_key in page_cache:
            sentences = page_cache.pop(cache_key)
            page_cache[cache_key] = sentences
            title = cache_key
        else:
            row = _fetch_wiki_document(cursor, hit_title)
            if row is None:
                sentences = []
                title = cache_key
            else:
                title = clean_text(str(row[0])).replace("_", " ")
                sentences = _coerce_sentences(row[1])
            if page_cache is not None:
                page_cache[cache_key] = sentences
                while len(page_cache) > int(page_cache_size):
                    page_cache.popitem(last=False)
        if not sentences:
            continue
        pages.append(
            {
                "title": title,
                "rank": int(hit["rank"]),
                "page_bm25_score": float(hit["bm25_score"]),
                "sentences": sentences,
            }
        )
    return pages


def _candidate_sentences_for_pages(
    *,
    claim: str,
    pages: list[dict[str, Any]],
    sentence_pool_k: int,
) -> list[dict[str, Any]]:
    q_ctr, q_len = content_tokens_counter(claim)
    candidates: list[dict[str, Any]] = []
    for page in pages:
        for sent_idx, sentence in enumerate(page["sentences"]):
            text = clean_text(str(sentence))
            if not text:
                continue
            s_ctr, s_len = content_tokens_counter(text)
            lexical = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
            bm25 = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)
            page_rank_score = 1.0 / max(1, int(page["rank"]))
            hybrid = 0.60 * _squash(bm25) + 0.30 * lexical + 0.10 * page_rank_score
            candidates.append(
                {
                    "report_id": page["title"],
                    "sent_idx": sent_idx,
                    "chunk_sent_indices": [sent_idx],
                    "text": f"{page['title']}: {text}",
                    "source_report": {"report_id": page["title"], "domain": "wikipedia", "link": None},
                    "hover_page_title": page["title"],
                    "hover_sent_idx": sent_idx,
                    "hover_evidence_mode": "bm25_page_mmr_sentence",
                    "hover_page_rank": int(page["rank"]),
                    "dense_score": 0.0,
                    "lexical_score": float(lexical),
                    "bm25_score": float(bm25),
                    "hybrid_score": float(hybrid),
                    "_tokens": set(content_tokens(text)),
                }
            )
    candidates.sort(key=lambda row: float(row["hybrid_score"]), reverse=True)
    return candidates[: max(1, int(sentence_pool_k))]


def _select_sentence_mmr(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    mmr_lambda: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < int(top_k):
        best_idx = 0
        best_score = -math.inf
        for idx, cand in enumerate(remaining):
            redundancy = 0.0
            for chosen in selected:
                redundancy = max(redundancy, _token_jaccard(cand.get("_tokens"), chosen.get("_tokens")))
            score = float(mmr_lambda) * float(cand["hybrid_score"]) - (1.0 - float(mmr_lambda)) * redundancy
            if score > best_score:
                best_idx = idx
                best_score = score
        item = dict(remaining.pop(best_idx))
        item.pop("_tokens", None)
        item["mmr_score"] = float(best_score)
        item["mmr_rank"] = len(selected) + 1
        selected.append(item)
    selected.sort(key=lambda row: int(row["mmr_rank"]))
    return selected


def _token_jaccard(a: Any, b: Any) -> float:
    sa = set(a or [])
    sb = set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _squash(value: float) -> float:
    return float(value / (1.0 + abs(value)))


def _build_training_row_with_s3_metadata(
    retrieval_row: dict[str, Any],
    *,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    row = build_training_row(retrieval_row, tokenizer, prompt_cfg)
    for key in ("hover_s3_retrieval", "retrieved_pages"):
        if key in retrieval_row:
            row[key] = retrieval_row[key]
    return row


class _MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.passage_all_recall = 0.0
        self.passage_f1 = 0.0
        self.sentence_recall = 0.0
        self.by_hop: dict[str, "_MetricAccumulator"] = {}

    def add(
        self,
        *,
        retrieved_titles: list[str],
        selected_candidates: list[dict[str, Any]],
        gold_titles: set[str],
        gold_sentence_keys: set[tuple[str, int]],
        hop: str,
    ) -> None:
        retrieved_set = set(retrieved_titles)
        selected_keys = {
            (str(c.get("hover_page_title") or c.get("report_id")), int(c.get("hover_sent_idx", -1)))
            for c in selected_candidates
        }
        self.n += 1
        self.passage_all_recall += 1.0 if gold_titles and gold_titles <= retrieved_set else 0.0
        self.passage_f1 += _set_f1(gold_titles, retrieved_set)
        self.sentence_recall += _set_recall(gold_sentence_keys, selected_keys)
        if hop:
            if hop not in self.by_hop:
                self.by_hop[hop] = _MetricAccumulator()
            self.by_hop[hop].add(
                retrieved_titles=retrieved_titles,
                selected_candidates=selected_candidates,
                gold_titles=gold_titles,
                gold_sentence_keys=gold_sentence_keys,
                hop="",
            )

    def report(self) -> dict[str, Any]:
        n = max(1, self.n)
        return {
            "passage": {
                "all_recall_at_k": self.passage_all_recall / n,
                "set_f1": self.passage_f1 / n,
            },
            "sentence": {
                "selected_recall": self.sentence_recall / n,
            },
            "by_hop": {hop: acc.report() for hop, acc in sorted(self.by_hop.items())},
        }


def _set_recall(gold: set[Any], pred: set[Any]) -> float:
    if not gold:
        return 0.0
    return len(gold & pred) / len(gold)


def _set_f1(gold: set[Any], pred: set[Any]) -> float:
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / max(1e-8, precision + recall)


def _rate(values: Iterable[Any]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(1 for item in items if bool(item)) / len(items))


if __name__ == "__main__":
    main()
