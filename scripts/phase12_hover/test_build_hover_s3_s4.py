from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

from scripts.phase12_hover.build_hover_s3_retrieval_baseline import (
    build_fts_index,
    build_split_open_retrieval_rows,
    build_split_page_retrieval_rows,
    query_pages,
)
from scripts.phase12_hover.prepare_hover_s4_mrec_sources import (
    build_s4_sources_for_split,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_wiki_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT)")
        conn.executemany(
            "INSERT INTO documents VALUES (?, ?)",
            [
                ("Paris", "Paris is a city in France. The River starts in Paris."),
                ("Alice", "Alice was born in Paris. Alice later moved away."),
                ("Unused", "A page about a mountain and a lake."),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_s3_retrieves_gold_pages_and_builds_open_retrieval_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Alice and Paris explain where the river starts.",
                "label": "SUPPORTED",
                "num_hops": 2,
                "supporting_facts": [["Alice", 0], ["Paris", 1]],
            }
        ],
    )
    wiki_db = tmp_path / "wiki" / "wiki_wo_links.db"
    index_db = tmp_path / "wiki_index" / "wiki_fts.db"
    _make_wiki_db(wiki_db)
    report = build_fts_index(wiki_db=wiki_db, index_db=index_db, force=True)

    assert report["indexed_documents"] == 3
    page_hits = query_pages(index_db, "Alice river Paris", top_k=2)
    assert {hit["title"] for hit in page_hits} == {"Alice", "Paris"}
    title_hits = query_pages(index_db, "Alice river Paris", top_k=2, page_query_mode="title")
    assert {hit["title"] for hit in title_hits} == {"Alice", "Paris"}

    rows, split_report = build_split_open_retrieval_rows(
        split="train",
        raw_path=raw_path,
        wiki_db=wiki_db,
        index_db=index_db,
        page_top_k=2,
        sentence_pool_k=8,
        top_k=3,
        mmr_lambda=0.7,
        sample_limit=None,
    )

    assert split_report["n_rows"] == 1
    assert split_report["passage"]["all_recall_at_k"] == 1.0
    assert split_report["sentence"]["selected_recall"] == 1.0
    assert rows[0]["event_id"] == "hover-1"
    assert rows[0]["hover_s3_retrieval"]["retrieved_page_count"] == 2
    assert {"Alice", "Paris"} <= {c["hover_page_title"] for c in rows[0]["candidates"]}
    assert all("hybrid_score" in c for c in rows[0]["candidates"])


def test_s3_page_retrieval_gate_skips_sentence_loading(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_val.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Alice and Paris explain where the river starts.",
                "label": "SUPPORTED",
                "num_hops": 2,
                "supporting_facts": [["Alice", 0], ["Paris", 1]],
            }
        ],
    )
    wiki_db = tmp_path / "wiki" / "wiki_wo_links.db"
    index_db = tmp_path / "wiki_index" / "wiki_fts.db"
    _make_wiki_db(wiki_db)
    build_fts_index(wiki_db=wiki_db, index_db=index_db, force=True)

    rows, split_report = build_split_page_retrieval_rows(
        split="val",
        raw_path=raw_path,
        index_db=index_db,
        page_top_k=2,
        page_query_mode="title",
        sample_limit=None,
        num_workers=2,
    )

    assert split_report["n_rows"] == 1
    assert split_report["passage"]["all_recall_at_k"] == 1.0
    assert split_report["sentence"]["selected_recall"] == 0.0
    assert rows[0]["event_id"] == "hover-1"
    assert rows[0]["hover_s3_retrieval"]["mode"] == "bm25_page_only"
    assert rows[0]["candidates"] == []
    assert {hit["title"] for hit in rows[0]["retrieved_pages"]} == {"Alice", "Paris"}


def test_s4_sources_mark_gold_evidence_and_proxy_pairs(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_train.json"
    retrieval_path = tmp_path / "retrieval_train.jsonl"
    out_dir = tmp_path / "s4_sources"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Alice was born in Paris.",
                "label": "SUPPORTED",
                "supporting_facts": [["Alice", 0]],
            }
        ],
    )
    _write_jsonl(
        retrieval_path,
        [
            {
                "event_id": "hover-1",
                "claim": "Alice was born in Paris.",
                "label": "supported",
                "candidates": [
                    {
                        "text": "Alice: Alice was born in Paris.",
                        "hover_page_title": "Alice",
                        "hover_sent_idx": 0,
                        "hybrid_score": 1.0,
                    },
                    {
                        "text": "Unused: A page about a lake.",
                        "hover_page_title": "Unused",
                        "hover_sent_idx": 0,
                        "hybrid_score": 0.1,
                    },
                ],
            }
        ],
    )

    report = build_s4_sources_for_split(
        split="train",
        raw_path=raw_path,
        retrieval_path=retrieval_path,
        output_dir=out_dir,
        sample_limit=None,
    )

    assert report["n_rows"] == 1
    assert report["gold_sentence_candidate_count"] == 1
    evidence_rows = [
        json.loads(line)
        for line in (out_dir / "04_evidence_map" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert evidence_rows[0]["candidate_labels"][0]["is_gold_sentence"] is True
    assert evidence_rows[0]["candidate_labels"][1]["is_gold_title"] is False
    proxy_rows = [
        json.loads(line)
        for line in (out_dir / "05_mrec_v0_2_learned_marginal_proxy_fullpool" / "train_proxy_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert proxy_rows == [
        {
            "event_id": "hover-1",
            "winner_candidate_index": 0,
            "loser_candidate_index": 1,
            "reason": "gold_sentence_over_non_gold",
        }
    ]


def test_s3_and_s4_wrappers_dry_run() -> None:
    root = Path(__file__).resolve().parents[2]
    for script in (
        "scripts/phase12_hover/run_hover_s3_bm25_page_mmr_sentence_minmax9_9.sh",
        "scripts/phase12_hover/run_hover_s4_prepare_mrec_sources.sh",
    ):
        result = subprocess.run(
            ["bash", script],
            cwd=root,
            env={**os.environ, "DRY_RUN": "true"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "phase12_hover" in result.stdout


def test_s3_wrapper_dry_run_passes_fast_gate_options() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", "scripts/phase12_hover/run_hover_s3_bm25_page_mmr_sentence_minmax9_9.sh"],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "true",
            "SPLITS": "val",
            "RETRIEVAL_STAGE": "pages",
            "PAGE_QUERY_MODE": "title",
            "NUM_WORKERS": "4",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "--splits val" in result.stdout
    assert "--retrieval-stage pages" in result.stdout
    assert "--page-query-mode title" in result.stdout
    assert "--num-workers 4" in result.stdout
