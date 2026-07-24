#!/usr/bin/env python3
"""Prepare blinded bilingual Label Studio tasks for Exp1 adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_DIR = ROOT / "results" / "exp1_reliability_pre_adjudication"
DEFAULT_BLIND_QUEUE = SOURCE_DIR / "adjudication_tasks_blind.jsonl"
DEFAULT_SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
DEFAULT_METRICS = SOURCE_DIR / "metrics.json"
DEFAULT_UNIVERSE = ROOT / "data" / "exp1_tasks_flat_zh.jsonl"
DEFAULT_ATOM_CONFIG = ROOT / "config" / "exp1_adjudication_atom.xml"
DEFAULT_COMPLETENESS_CONFIG = ROOT / "config" / "exp1_adjudication_completeness.xml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "exp1_adjudication_v1"

FORBIDDEN_KEYS = {"annotator_a", "annotator_b", "disagreements"}
FIELD_DISPLAY = {
    "faithfulness": "忠实性 Faithfulness",
    "atomicity": "原子性 Atomicity",
    "completeness_missed": "完整性 Completeness",
}
QUESTION_SPEC = {
    "faithfulness": {
        "field": "faithfulness",
        "title": "忠实性 / Faithfulness",
        "question_zh": "当前 atom 的全部语义是否都能由 claim 支持，且未引入 claim 中不存在的信息？",
        "question_en": "Is the full meaning of the current atom supported by the claim, without adding information absent from the claim?",
        "criteria": "yes=忠实改写或拆分；no=存在任何新增、扭曲或无法由 claim 支持的内容。",
    },
    "atomicity": {
        "field": "atomicity",
        "title": "原子性 / Atomicity",
        "question_zh": "当前 atom 是否已经是最小可独立核验命题，不能再拆成两个或更多可分别核验的事实断言？",
        "question_en": "Is the current atom a minimal independently verifiable proposition that cannot be split into two or more separately verifiable factual assertions?",
        "criteria": "yes=最小可核验粒度；no=仍黏合了两个或更多可独立核验断言。",
    },
}
QUESTION_ORDER = ("faithfulness", "atomicity")
TRANSLATION_FALLBACK = "（暂无中文翻译，请以英文为准。）"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def assert_no_forbidden_keys(value: Any, context: str) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"Blinding violation in {context}: {sorted(leaked)}")
        for key, child in value.items():
            assert_no_forbidden_keys(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_forbidden_keys(child, f"{context}[{index}]")


def validate_source_manifest(manifest_path: Path, blind_queue_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("Source reliability manifest is not complete")
    artifact = manifest.get("artifacts", {}).get(blind_queue_path.name)
    if not artifact:
        raise ValueError(f"Source manifest does not list {blind_queue_path.name}")
    actual = sha256_file(blind_queue_path)
    if actual != artifact.get("sha256"):
        raise ValueError(
            f"Blind queue hash mismatch: expected={artifact.get('sha256')}, actual={actual}"
        )
    return manifest


def load_universe(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    universe: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = (str(row.get("dataset", "")), str(row.get("event_id", "")), str(row.get("atom_id", "")))
        if any(not value or value == "-" for value in key):
            raise ValueError(f"Invalid universe key: {key}")
        if key in universe:
            raise ValueError(f"Duplicate universe key: {key}")
        required = (
            "claim",
            "proposition",
            "type",
            "all_atoms_text",
        )
        missing = [field for field in required if not isinstance(row.get(field), str) or not row[field].strip()]
        if missing:
            raise ValueError(f"Universe row {key} has invalid fields: {missing}")
        for field in ("claim_zh", "proposition_zh", "all_atoms_text_zh"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                row[field] = TRANSLATION_FALLBACK
        universe[key] = row
    if len(universe) != 257:
        raise ValueError(f"Expected 257 universe atoms, found {len(universe)}")
    return universe


def atom_sort_key(atom_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"A(\d+)", atom_id)
    return (int(match.group(1)), atom_id) if match else (10**9, atom_id)


def prepare_tasks(
    blind_rows: list[dict[str, Any]],
    universe: dict[tuple[str, str, str], dict[str, Any]],
    protocol_version: str,
    analysis_input_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(blind_rows) != 47:
        raise ValueError(f"Expected 47 blind tasks, found {len(blind_rows)}")
    assert_no_forbidden_keys(blind_rows, "blind_rows")
    ids = [row.get("adjudication_id") for row in blind_rows]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Blind adjudication IDs must be non-empty and unique")

    by_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, row in universe.items():
        by_claim[(key[0], key[1])].append(row)

    atom_tasks: list[dict[str, Any]] = []
    completeness_tasks: list[dict[str, Any]] = []
    for row in blind_rows:
        unit = row.get("unit")
        dataset = str(row.get("dataset", ""))
        event_id = str(row.get("event_id", ""))
        fields = row.get("fields_to_adjudicate")
        if not isinstance(fields, list) or not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Invalid fields_to_adjudicate for {row.get('adjudication_id')}: {fields}")
        unknown = sorted(set(fields) - set(FIELD_DISPLAY))
        if unknown:
            raise ValueError(f"Unknown adjudication fields: {unknown}")

        common = {
            "adjudication_id": row["adjudication_id"],
            "adjudication_protocol_version": protocol_version,
            "source_analysis_input_sha256": analysis_input_sha256,
            "unit": unit,
            "dataset": dataset,
            "event_id": event_id,
            "fields_to_adjudicate": fields,
            "fields_to_adjudicate_text": " + ".join(FIELD_DISPLAY[field] for field in fields),
        }
        if unit == "atom":
            if not set(fields).issubset({"faithfulness", "atomicity"}):
                raise ValueError(f"Atom task has non-atom field: {row}")
            atom_id = str(row.get("atom_id", ""))
            source = universe.get((dataset, event_id, atom_id))
            if source is None:
                raise ValueError(f"Atom task missing from universe: {(dataset, event_id, atom_id)}")
            if row.get("claim") != source["claim"] or row.get("proposition") != source["proposition"]:
                raise ValueError(f"Blind/universe text mismatch: {(dataset, event_id, atom_id)}")
            atom_tasks.append(
                {
                    **common,
                    "atom_id": atom_id,
                    "atom_type": source["type"],
                    "claim": source["claim"],
                    "claim_zh": source["claim_zh"],
                    "all_atoms_text": source["all_atoms_text"],
                    "all_atoms_text_zh": source["all_atoms_text_zh"],
                    "proposition": source["proposition"],
                    "proposition_zh": source["proposition_zh"],
                    "questions": [QUESTION_SPEC[field] for field in QUESTION_ORDER if field in fields],
                }
            )
        elif unit == "claim":
            if fields != ["completeness_missed"]:
                raise ValueError(
                    f"Claim task must adjudicate completeness only: {row.get('adjudication_id')} {fields}"
                )
            atoms = sorted(by_claim.get((dataset, event_id), []), key=lambda item: atom_sort_key(item["atom_id"]))
            if not atoms:
                raise ValueError(f"Claim task missing from universe: {(dataset, event_id)}")
            if row.get("claim") != atoms[0]["claim"]:
                raise ValueError(f"Claim text mismatch: {(dataset, event_id)}")
            expected_atoms = [(atom["atom_id"], atom["proposition"]) for atom in atoms]
            blind_atoms = [(atom.get("atom_id"), atom.get("proposition")) for atom in row.get("atoms", [])]
            if blind_atoms != expected_atoms:
                raise ValueError(f"Claim atom panorama mismatch: {(dataset, event_id)}")
            for field in ("claim", "claim_zh", "all_atoms_text", "all_atoms_text_zh"):
                if len({atom[field] for atom in atoms}) != 1:
                    raise ValueError(f"Claim-level field differs across atoms: {(dataset, event_id)} {field}")
            completeness_tasks.append(
                {
                    **common,
                    "claim": atoms[0]["claim"],
                    "claim_zh": atoms[0]["claim_zh"],
                    "all_atoms_text": atoms[0]["all_atoms_text"],
                    "all_atoms_text_zh": atoms[0]["all_atoms_text_zh"],
                    "atom_count": len(atoms),
                }
            )
        else:
            raise ValueError(f"Unknown adjudication unit: {unit!r}")

    atom_tasks.sort(key=lambda row: (row["dataset"], row["event_id"], atom_sort_key(row["atom_id"])))
    completeness_tasks.sort(key=lambda row: (row["dataset"], row["event_id"]))
    if len(atom_tasks) != 37 or len(completeness_tasks) != 10:
        raise ValueError(
            f"Unexpected task counts: atom={len(atom_tasks)}, completeness={len(completeness_tasks)}"
        )
    expected_combinations = Counter({("atomicity",): 26, ("faithfulness",): 8, ("atomicity", "faithfulness"): 3})
    actual_combinations = Counter(tuple(row["fields_to_adjudicate"]) for row in atom_tasks)
    if actual_combinations != expected_combinations:
        raise ValueError(f"Unexpected atom field combinations: {actual_combinations}")
    assert_no_forbidden_keys(atom_tasks, "atom_tasks")
    assert_no_forbidden_keys(completeness_tasks, "completeness_tasks")
    return atom_tasks, completeness_tasks


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def prepare(
    blind_queue: Path,
    source_manifest_path: Path,
    metrics_path: Path,
    universe_path: Path,
    atom_config: Path,
    completeness_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = validate_source_manifest(source_manifest_path, blind_queue)
    metrics = load_json(metrics_path)
    if metrics.get("analysis_status") != "pre_adjudication":
        raise ValueError("Expected pre-adjudication source metrics")
    protocol_version = metrics["methodology"]["gold_resolution_protocol_version"]
    analysis_input_sha256 = metrics["snapshot"]["analysis_input_sha256"]
    if source_manifest.get("analysis_input_sha256") != analysis_input_sha256:
        raise ValueError("Source manifest/metrics analysis hash mismatch")
    universe = load_universe(universe_path)
    atom_tasks, completeness_tasks = prepare_tasks(
        load_jsonl(blind_queue), universe, protocol_version, analysis_input_sha256
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atom_path = output_dir / "atom_tasks.jsonl"
    completeness_path = output_dir / "completeness_tasks.jsonl"
    atomic_write(atom_path, jsonl_text(atom_tasks))
    atomic_write(completeness_path, jsonl_text(completeness_tasks))
    artifacts = {
        atom_path.name: {"sha256": sha256_file(atom_path), "rows": len(atom_tasks)},
        completeness_path.name: {"sha256": sha256_file(completeness_path), "rows": len(completeness_tasks)},
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adjudication_protocol_version": protocol_version,
        "source_analysis_input_sha256": analysis_input_sha256,
        "source_reliability_generation_id": source_manifest["generation_id"],
        "source_blind_queue_sha256": sha256_file(blind_queue),
        "source_universe_sha256": sha256_file(universe_path),
        "config_sha256": {
            "atom": sha256_file(atom_config),
            "completeness": sha256_file(completeness_config),
        },
        "counts": {
            "atom": len(atom_tasks),
            "completeness": len(completeness_tasks),
            "total": len(atom_tasks) + len(completeness_tasks),
        },
        "artifacts": artifacts,
        "blinding": {
            "forbidden_keys_checked_recursively": sorted(FORBIDDEN_KEYS),
            "contains_rater_labels": False,
        },
    }
    atomic_write(
        output_dir / "task_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-queue", type=Path, default=DEFAULT_BLIND_QUEUE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--atom-config", type=Path, default=DEFAULT_ATOM_CONFIG)
    parser.add_argument("--completeness-config", type=Path, default=DEFAULT_COMPLETENESS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(
        args.blind_queue,
        args.source_manifest,
        args.metrics,
        args.universe,
        args.atom_config,
        args.completeness_config,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
