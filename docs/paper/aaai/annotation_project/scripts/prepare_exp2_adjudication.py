#!/usr/bin/env python3
"""Prepare the blinded 125-pair Exp2 exact-gold adjudication task set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_DIR = ROOT / "results" / "exp2_reliability_pre_adjudication"
DEFAULT_BLIND_QUEUE = SOURCE_DIR / "exact_gold_adjudication_tasks_blind.jsonl"
DEFAULT_SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
DEFAULT_METRICS = SOURCE_DIR / "metrics.json"
DEFAULT_CONFIG = ROOT / "config" / "exp2_adjudication_exact_gold.xml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "exp2_adjudication_v1"
PROTOCOL_VERSION = "exp2-exact-gold-resolution-v1-20260719"

ALLOWED_FIELDS = {"gold_relation", "gold_directness"}
FORBIDDEN_KEYS = {
    "annotator_a",
    "annotator_b",
    "llm_relation",
    "llm_directness",
    "llm_confidence",
    "llm_evidence_role",
}
REQUIRED_CORE_TEXT_FIELDS = (
    "dataset",
    "event_id",
    "atom_id",
    "evidence_id",
    "claim",
    "atom_proposition",
    "evidence_text",
)
TRANSLATION_FIELDS = (
    "claim_zh",
    "atom_proposition_zh",
    "evidence_text_zh",
)
TRANSLATION_FALLBACK = "（暂无中文翻译，请以英文为准。）"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(canonical_json(row) + "\n" for row in rows),
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object: {path}")
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


def assert_blinded(value: Any, context: str) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"Blinding violation in {context}: {sorted(leaked)}")
        for key, child in value.items():
            if key.startswith("llm_"):
                raise ValueError(f"Blinding violation in {context}: {key}")
            assert_blinded(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_blinded(child, f"{context}[{index}]")


def validate_sources(
    blind_queue_path: Path,
    source_manifest_path: Path,
    metrics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = load_json(source_manifest_path)
    artifact = source_manifest.get("artifacts", {}).get(blind_queue_path.name)
    if artifact is None:
        raise ValueError("Source manifest does not list exact-gold blind queue")
    if artifact.get("sha256") != sha256_file(blind_queue_path):
        raise ValueError("Source blind queue hash mismatch")
    if artifact.get("line_count") != 125:
        raise ValueError(f"Unexpected source blind queue count: {artifact}")
    metrics = load_json(metrics_path)
    expected = metrics.get("adjudication", {}).get("exact_gold_recommended", {})
    expected_values = {
        "unique_pair_count": 125,
        "relation_field_count": 121,
        "directness_field_count": 86,
        "field_decision_count": 207,
    }
    for key, value in expected_values.items():
        if expected.get(key) != value:
            raise ValueError(f"Unexpected exact-gold metric {key}: {expected.get(key)}")
    return source_manifest, metrics


def normalize_task(row: dict[str, Any]) -> dict[str, Any]:
    assert_blinded(row, "source task")
    adjudication_id = row.get("adjudication_task_id")
    if not isinstance(adjudication_id, str) or not adjudication_id.startswith("exp2-"):
        raise ValueError(f"Invalid adjudication task id: {adjudication_id}")
    missing_text = [
        field
        for field in REQUIRED_CORE_TEXT_FIELDS
        if not isinstance(row.get(field), str) or not row[field].strip()
    ]
    if missing_text:
        raise ValueError(f"Task {adjudication_id} has invalid text fields: {missing_text}")
    fields = row.get("fields_to_adjudicate")
    if (
        not isinstance(fields, list)
        or not fields
        or len(fields) != len(set(fields))
        or not set(fields).issubset(ALLOWED_FIELDS)
    ):
        raise ValueError(f"Task {adjudication_id} has invalid fields: {fields}")
    relation_needed = "gold_relation" in fields
    directness_needed = "gold_directness" in fields
    field_labels = {
        "gold_relation": "Relation（关系类型）",
        "gold_directness": "Directness（直接程度）",
    }
    task = {
        "adjudication_id": adjudication_id,
        "adjudication_protocol_version": PROTOCOL_VERSION,
        **{field: row[field] for field in REQUIRED_CORE_TEXT_FIELDS},
        **{
            field: (
                row.get(field)
                if isinstance(row.get(field), str) and row[field].strip()
                else TRANSLATION_FALLBACK
            )
            for field in TRANSLATION_FIELDS
        },
        "fields_to_adjudicate": fields,
        "fields_to_adjudicate_text": " + ".join(field_labels[field] for field in fields),
        "relation_questions": (
            [
                {
                    "field": "gold_relation",
                    "title": "一、Relation（关系类型）",
                    "question": "只看本页 evidence：它对当前 atom 的语义关系是什么？",
                }
            ]
            if relation_needed
            else []
        ),
        "directness_questions": (
            [
                {
                    "field": "gold_directness",
                    "title": "二、Directness（直接程度）",
                    "question": "只看本页 evidence：它涉及当前 atom 真值的直接程度是什么？",
                }
            ]
            if directness_needed
            else []
        ),
        "unit": "evidence_atom_pair",
    }
    assert_blinded(task, f"prepared task {adjudication_id}")
    return task


def prepare(
    blind_queue_path: Path,
    source_manifest_path: Path,
    metrics_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest, metrics = validate_sources(
        blind_queue_path, source_manifest_path, metrics_path
    )
    source_rows = load_jsonl(blind_queue_path)
    if len(source_rows) != 125:
        raise ValueError(f"Expected 125 source tasks, found {len(source_rows)}")
    tasks = [normalize_task(row) for row in source_rows]
    empty_translation_count = sum(
        not isinstance(row.get(field), str) or not row[field].strip()
        for row in source_rows
        for field in TRANSLATION_FIELDS
    )
    inherited_control_character_count = sum(
        any(0x80 <= ord(character) <= 0x9F for character in row[field])
        for row in source_rows
        for field in REQUIRED_CORE_TEXT_FIELDS
    )
    ids = [task["adjudication_id"] for task in tasks]
    if len(set(ids)) != 125:
        raise ValueError("Adjudication task IDs must be unique")

    patterns = Counter(tuple(task["fields_to_adjudicate"]) for task in tasks)
    expected_patterns = Counter(
        {
            ("gold_relation",): 39,
            ("gold_directness",): 4,
            ("gold_relation", "gold_directness"): 82,
        }
    )
    if patterns != expected_patterns:
        raise ValueError(f"Unexpected task field patterns: {patterns}")
    relation_count = sum("gold_relation" in task["fields_to_adjudicate"] for task in tasks)
    directness_count = sum(
        "gold_directness" in task["fields_to_adjudicate"] for task in tasks
    )
    if (relation_count, directness_count) != (121, 86):
        raise ValueError(
            f"Unexpected decision counts: relation={relation_count}, directness={directness_count}"
        )

    config_content = config_path.read_text(encoding="utf-8")
    required_config_tokens = (
        "$relation_questions",
        "$directness_questions",
        "relation_decision_{{idx}}",
        "directness_decision_{{idx}}",
        "review_complete",
    )
    missing_tokens = [token for token in required_config_tokens if token not in config_content]
    if missing_tokens:
        raise ValueError(f"Adjudication config is missing controls: {missing_tokens}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "tasks.jsonl"
    write_jsonl(task_path, tasks)
    task_hash = sha256_file(task_path)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adjudication_protocol_version": PROTOCOL_VERSION,
        "counts": {
            "relation_only_pairs": patterns[("gold_relation",)],
            "directness_only_pairs": patterns[("gold_directness",)],
            "both_fields_pairs": patterns[("gold_relation", "gold_directness")],
            "total_pairs": len(tasks),
            "relation_decisions": relation_count,
            "directness_decisions": directness_count,
            "total_field_decisions": relation_count + directness_count,
        },
        "blinding": {
            "contains_rater_labels": False,
            "contains_llm_labels": False,
            "forbidden_keys_checked_recursively": sorted(FORBIDDEN_KEYS),
        },
        "source_data_notes": {
            "empty_translations_replaced_with_fallback": empty_translation_count,
            "inherited_c1_control_character_rows": inherited_control_character_count,
            "blocking_issue_count": 0,
        },
        "source": {
            "analysis_generated_at_utc": source_manifest.get("generated_at_utc"),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_blind_queue_sha256": sha256_file(blind_queue_path),
            "source_metrics_sha256": sha256_file(metrics_path),
            "authoritative_tasks_sha256": metrics["authoritative_inputs"]["tasks_sha256"],
        },
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "tasks.jsonl": {
                "rows": len(tasks),
                "sha256": task_hash,
            }
        },
    }
    write_json(output_dir / "task_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-queue", type=Path, default=DEFAULT_BLIND_QUEUE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(
        args.blind_queue,
        args.source_manifest,
        args.metrics,
        args.config,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
