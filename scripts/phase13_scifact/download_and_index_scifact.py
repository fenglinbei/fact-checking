#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
REQUIRED_FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official SciFact data and build a local SQLite corpus index.")
    parser.add_argument("--data-root", default="data/raw/SciFact")
    parser.add_argument("--processed-root", default="data/processed/SciFact")
    parser.add_argument("--output-manifest", default="outputs/selectors/scifact_atom_anchor/00_data/download_manifest.json")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    data_root = Path(args.data_root)
    processed_root = Path(args.processed_root)
    manifest_path = Path(args.output_manifest)
    data_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    if args.force_download or not _has_required_files(data_root):
        _download_and_extract(str(args.url), data_root=data_root)
    _require_files(data_root)

    sqlite_path = processed_root / "scifact_corpus.sqlite"
    if args.force_index or not sqlite_path.exists():
        corpus_stats = _build_sqlite_index(data_root / "corpus.jsonl", sqlite_path)
    else:
        corpus_stats = _sqlite_stats(sqlite_path)

    split_stats = {path.name: _count_jsonl(path) for path in sorted(data_root.glob("claims_*.jsonl"))}
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "url": str(args.url),
        "data_root": str(data_root),
        "processed_root": str(processed_root),
        "sqlite_path": str(sqlite_path),
        "required_files": {name: str(data_root / name) for name in REQUIRED_FILES},
        "corpus_stats": corpus_stats,
        "split_stats": split_stats,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote SciFact manifest: {manifest_path}")
    print(f"Wrote SciFact SQLite index: {sqlite_path}")
    return 0


def _has_required_files(data_root: Path) -> bool:
    return all((data_root / name).is_file() and (data_root / name).stat().st_size > 0 for name in REQUIRED_FILES)


def _require_files(data_root: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SciFact files under {data_root}: {missing}")


def _download_and_extract(url: str, *, data_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="scifact_download_") as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / "data.tar.gz"
        print(f"Downloading SciFact data: {url}")
        urllib.request.urlretrieve(url, archive)
        extract_root = tmp_dir / "extract"
        extract_root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, extract_root)
        source_dir = extract_root / "data"
        if not source_dir.exists():
            source_dir = extract_root
        for name in REQUIRED_FILES:
            src = source_dir / name
            if not src.exists():
                found = list(extract_root.glob(f"**/{name}"))
                if not found:
                    raise FileNotFoundError(f"Downloaded tarball did not contain {name}")
                src = found[0]
            shutil.copy2(src, data_root / name)


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    root = path.resolve()
    for member in tar.getmembers():
        target = (path / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Unsafe path in tarball: {member.name}")
    tar.extractall(path)


def _build_sqlite_index(corpus_path: Path, sqlite_path: Path) -> dict[str, Any]:
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE abstracts (
              doc_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              abstract_json TEXT NOT NULL,
              structured INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE sentences (
              doc_id TEXT NOT NULL,
              title TEXT NOT NULL,
              sent_idx INTEGER NOT NULL,
              sentence TEXT NOT NULL,
              structured INTEGER NOT NULL,
              PRIMARY KEY (doc_id, sent_idx)
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE sentence_fts USING fts5(
              doc_id UNINDEXED,
              title,
              sentence
            )
            """
        )

        n_docs = 0
        n_sentences = 0
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = str(row["doc_id"])
                title = str(row.get("title") or "")
                abstract = [str(sent).strip() for sent in row.get("abstract") or [] if str(sent).strip()]
                structured = 1 if bool(row.get("structured")) else 0
                conn.execute(
                    "INSERT INTO abstracts(doc_id, title, abstract_json, structured) VALUES (?, ?, ?, ?)",
                    (doc_id, title, json.dumps(abstract, ensure_ascii=False), structured),
                )
                for sent_idx, sentence in enumerate(abstract):
                    conn.execute(
                        "INSERT INTO sentences(doc_id, title, sent_idx, sentence, structured) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, title, int(sent_idx), sentence, structured),
                    )
                    conn.execute(
                        "INSERT INTO sentence_fts(doc_id, title, sentence) VALUES (?, ?, ?)",
                        (doc_id, title, sentence),
                    )
                    n_sentences += 1
                n_docs += 1
        conn.commit()
        return {"documents": n_docs, "sentences": n_sentences}
    finally:
        conn.close()


def _sqlite_stats(sqlite_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(sqlite_path)
    try:
        docs = conn.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
        sentences = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        return {"documents": int(docs), "sentences": int(sentences), "reused": True}
    finally:
        conn.close()


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
