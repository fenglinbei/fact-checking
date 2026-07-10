from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import unicodedata
from pathlib import Path

import pytest
import yaml

from scripts.phase12_hover.build_hover_gold_sentence_verifier_data import (
    build_split_retrieval_rows,
    build_train_config,
    load_required_wiki_pages,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_split_retrieval_rows_uses_hover_supporting_fact_sentences(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Alice was born in the city where the river starts.",
                "label": "SUPPORTED",
                "num_hops": 2,
                "hpqa_id": "hpqa-1",
                "supporting_facts": [["Alice", 1], ["River", 0], ["Alice", 1]],
            }
        ],
    )
    wiki_pages = {
        "Alice": ["Alice is a scientist.", "Alice was born in Paris.", "Alice moved later."],
        "River": ["The River starts in Paris.", "It ends elsewhere."],
    }

    rows, report = build_split_retrieval_rows(
        split="train",
        raw_path=raw_path,
        wiki_pages=wiki_pages,
        evidence_mode="gold_sentences",
        sentence_window=0,
        max_doc_sentences=20,
        missing_policy="error",
        sample_limit=None,
    )

    assert report["n_rows"] == 1
    assert report["missing_supporting_facts"] == 0
    assert rows[0]["event_id"] == "hover-1"
    assert rows[0]["label_schema"] == "hover2"
    assert rows[0]["label"] == "supported"
    assert [c["text"] for c in rows[0]["candidates"]] == [
        "Alice: Alice was born in Paris.",
        "River: The River starts in Paris.",
    ]
    assert rows[0]["candidates"][0]["hover_page_title"] == "Alice"
    assert rows[0]["candidates"][0]["hover_sent_idx"] == 1
    assert rows[0]["hover_gold_evidence"]["num_hops"] == 2


def test_load_required_wiki_pages_reads_only_requested_titles_from_jsonl(tmp_path: Path) -> None:
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "part.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"title": "Alice", "text": ["A0.", "A1."]}),
                json.dumps({"title": "Unused", "text": ["U0."]}),
                json.dumps({"title": "River", "sentences": ["R0.", "R1."]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pages = load_required_wiki_pages(wiki_root, {"Alice", "River"})

    assert pages == {"Alice": ["A0.", "A1."], "River": ["R0.", "R1."]}


def test_load_required_wiki_pages_reads_official_sqlite_documents_table(tmp_path: Path) -> None:
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    db_path = wiki_root / "wiki_wo_links.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT)")
        conn.execute(
            "INSERT INTO documents VALUES (?, ?)",
            (
                unicodedata.normalize("NFD", "Café"),
                "Café is a city. It has a river.",
            ),
        )
        conn.execute("INSERT INTO documents VALUES (?, ?)", ("Unused", "Unused text."))
        conn.commit()
    finally:
        conn.close()

    pages = load_required_wiki_pages(wiki_root, {"Café", "Missing"})

    assert pages == {"Café": ["Café is a city.", "It has a river."]}


def test_load_required_wiki_pages_does_not_treat_scientist_suffix_as_st_abbreviation(
    tmp_path: Path,
) -> None:
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    db_path = wiki_root / "wiki_wo_links.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT)")
        conn.execute(
            "INSERT INTO documents VALUES (?, ?)",
            (
                "Scientist",
                "Ada was a scientist. He cited her work. The final sentence remains.",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    pages = load_required_wiki_pages(wiki_root, {"Scientist"})

    assert pages["Scientist"] == [
        "Ada was a scientist.",
        "He cited her work.",
        "The final sentence remains.",
    ]


def test_build_train_config_is_hover2_val_only_eval(tmp_path: Path) -> None:
    split_paths = {
        "train": str(tmp_path / "build_train.jsonl"),
        "val": str(tmp_path / "build_val.jsonl"),
    }
    cfg = build_train_config(
        output_dir=tmp_path,
        split_paths=split_paths,
        model_name_or_path="/data/models/Ministral-3-8B-Instruct-2512",
        deepspeed_config="configs/deepspeed/deepspeed_zero2_bsz1_ga4.json",
    )

    assert cfg["label_schema"] == "hover2"
    assert cfg["data"]["train_candidates"] == split_paths["train"]
    assert cfg["data"]["val_candidates"] == split_paths["val"]
    assert cfg["data"]["test_candidates"] == split_paths["val"]
    assert cfg["sft_train"]["label_schema"] == "hover2"
    assert cfg["sft_train"]["label_token_ce"]["class_weights"] == {
        "supported": 1.0,
        "not_supported": 1.0,
    }
    assert cfg["sft_train"]["label_token_ce"]["ordinal_loss"]["enabled"] is False
    assert yaml.safe_load(yaml.safe_dump(cfg))["baseline"]["variant"] == "hover_gold_sentences_minmax9_9"


def test_missing_supporting_fact_can_skip_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Claim",
                "label": "SUPPORTED",
                "supporting_facts": [["Missing", 0]],
            }
        ],
    )

    rows, report = build_split_retrieval_rows(
        split="train",
        raw_path=raw_path,
        wiki_pages={},
        evidence_mode="gold_sentences",
        sentence_window=0,
        max_doc_sentences=20,
        missing_policy="skip",
        sample_limit=None,
    )

    assert rows == []
    assert report["n_rows"] == 0
    assert report["skipped"]["missing_supporting_fact"] == 1


def test_missing_supporting_fact_errors_by_default(tmp_path: Path) -> None:
    raw_path = tmp_path / "hover_train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Claim",
                "label": "SUPPORTED",
                "supporting_facts": [["Missing", 0]],
            }
        ],
    )

    with pytest.raises(ValueError, match="missing supporting fact"):
        build_split_retrieval_rows(
            split="train",
            raw_path=raw_path,
            wiki_pages={},
            evidence_mode="gold_sentences",
            sentence_window=0,
            max_doc_sentences=20,
            missing_policy="error",
            sample_limit=None,
        )


def test_out_of_range_supporting_fact_index_uses_last_available_sentence(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "hover_train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "hover-1",
                "claim": "Claim",
                "label": "SUPPORTED",
                "supporting_facts": [["Page", 2]],
            }
        ],
    )

    rows, report = build_split_retrieval_rows(
        split="train",
        raw_path=raw_path,
        wiki_pages={"Page": ["First sentence.", "Last sentence."]},
        evidence_mode="gold_sentences",
        sentence_window=0,
        max_doc_sentences=20,
        missing_policy="error",
        sample_limit=None,
    )

    assert report["missing_supporting_facts"] == 0
    assert report["recovered_supporting_facts"] == 1
    assert rows[0]["candidates"][0]["text"] == "Page: Last sentence."
    assert rows[0]["candidates"][0]["hover_requested_sent_idx"] == 2
    assert rows[0]["candidates"][0]["hover_sent_idx"] == 1
    assert rows[0]["candidates"][0]["hover_sentence_index_recovered"] is True
    assert rows[0]["hover_gold_evidence"]["recovered_supporting_fact_count"] == 1


def test_hover_s2_wrapper_dry_run_reaches_training_launch() -> None:
    script = Path("scripts/phase12_hover/run_hover_s2_gold_sentences_minmax9_9_lora.sh")
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "MODE": "full",
        "PYTHON_BIN": "/opt/hover-python",
        "ACCELERATE_BIN": "/opt/accelerate",
        "OUTPUT_DIR": "outputs/sentence_trace_method/hover_test_s2",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "build_hover_gold_sentence_verifier_data.py" in result.stdout
    assert "--evidence-mode gold_sentences" in result.stdout
    assert "outputs/sentence_trace_method/hover_test_s2/train.resolved.yaml" in result.stdout
    assert "-m sft.label_token_trainer" in result.stdout
