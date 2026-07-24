#!/usr/bin/env python3
"""Audit Exp1 adjudication, resolve exact gold labels, and publish final metrics.

The two formal annotation projects remain the source of pre-adjudication IAA.
For each primary field, exact A/B consensus is retained and every disagreement
or claim-level internal conflict is resolved from the blinded third-rater
projects.  Dynamic ``decision_i`` controls are mapped through
``task.data.questions[i].field``; ``fields_to_adjudicate`` order is never used
as a result-control order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from analyze_exp1_reliability import (
    ProjectSpec,
    analysis_input_sha256,
    load_authoritative_universe,
    load_draft_issues,
    load_project_annotations,
    validate_distinct_project_pair,
    validate_project_universe,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRE_DIR = ROOT / "results" / "exp1_reliability_pre_adjudication"
PREPARED_DIR = ROOT / "results" / "exp1_adjudication_v1"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "exp1_reliability_final"
DEFAULT_DB = ROOT / "label_studio_data" / "label_studio.sqlite3"
DEFAULT_UNIVERSE = ROOT / "data" / "exp1_tasks_flat_zh.jsonl"
WRITING_ANCHOR = ROOT.parent / "writing_outline_v0.4.2_structure_only.md"
ATOM_CONFIG = ROOT / "config" / "exp1_adjudication_atom.xml"
COMPLETENESS_CONFIG = ROOT / "config" / "exp1_adjudication_completeness.xml"

EXPECTED_SOURCE_PROJECTS = (
    ProjectSpec(14, "Yulin", "1849812973@qq.com"),
    ProjectSpec(15, "Zhiqiang", "3180643570@qq.com"),
)
EXPECTED_ADJUDICATOR_EMAIL = "1349410043@qq.com"
EXPECTED_ADJUDICATOR_USER_ID = 3
EXPECTED_PROJECT_MEMBERS = {1, 3}
EXPECTED_PROTOCOL = "exp1-exact-gold-resolution-v1-20260717"
EXPECTED_PROJECTS = {
    20: {
        "title": "[ZIJIE ONLY] Exp1-Atom-Adjudication-v1",
        "unit": "atom",
        "prepared": PREPARED_DIR / "atom_tasks.jsonl",
        "config": ATOM_CONFIG,
        "task_count": 37,
    },
    21: {
        "title": "[ZIJIE ONLY] Exp1-Completeness-Adjudication-v1",
        "unit": "claim",
        "prepared": PREPARED_DIR / "completeness_tasks.jsonl",
        "config": COMPLETENESS_CONFIG,
        "task_count": 10,
    },
}
ATOM_FIELDS = ("faithfulness", "atomicity")
COMPLETENESS_VALUES = ("0", "1", "2", "3+")
FORBIDDEN_BLIND_KEYS = {"annotator_a", "annotator_b", "disagreements"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def artifact_entry(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "sha256": sha256_bytes(content),
        "bytes": len(content),
        "line_count": content.count(b"\n"),
    }


def assert_no_forbidden_keys(value: Any, context: str) -> None:
    if isinstance(value, dict):
        leaked = sorted(FORBIDDEN_BLIND_KEYS.intersection(value))
        if leaked:
            raise ValueError(f"Blinding violation in {context}: {leaked}")
        for key, child in value.items():
            assert_no_forbidden_keys(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_forbidden_keys(child, f"{context}[{index}]")


def validate_manifest_artifact(manifest: dict[str, Any], path: Path) -> None:
    entry = manifest.get("artifacts", {}).get(path.name)
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest does not list {path.name}")
    actual = sha256_file(path)
    if actual != entry.get("sha256"):
        raise ValueError(
            f"Artifact hash mismatch for {path.name}: expected={entry.get('sha256')}, actual={actual}"
        )
    expected_rows = entry.get("rows")
    if expected_rows is not None and len(load_jsonl(path)) != expected_rows:
        raise ValueError(f"Artifact row-count mismatch for {path.name}")


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[int, list[dict[str, Any]]]]:
    source_manifest = load_json(PRE_DIR / "manifest.json")
    prepared_manifest = load_json(PREPARED_DIR / "task_manifest.json")
    if source_manifest.get("complete") is not True or prepared_manifest.get("complete") is not True:
        raise ValueError("Source or prepared manifest is not complete")
    if source_manifest.get("gold_resolution_protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected source gold-resolution protocol")
    if prepared_manifest.get("adjudication_protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected prepared adjudication protocol")
    if prepared_manifest.get("source_analysis_input_sha256") != source_manifest.get(
        "analysis_input_sha256"
    ):
        raise ValueError("Source/prepared analysis hashes differ")

    source_paths = (
        PRE_DIR / "metrics.json",
        PRE_DIR / "atom_annotations_a.jsonl",
        PRE_DIR / "atom_annotations_b.jsonl",
        PRE_DIR / "claim_annotations.jsonl",
        PRE_DIR / "adjudication_tasks_blind.jsonl",
    )
    for path in source_paths:
        validate_manifest_artifact(source_manifest, path)
    for project_id, spec in EXPECTED_PROJECTS.items():
        path = spec["prepared"]
        validate_manifest_artifact(prepared_manifest, path)
        expected_config_hash = prepared_manifest.get("config_sha256", {}).get(
            "atom" if project_id == 20 else "completeness"
        )
        if sha256_file(spec["config"]) != expected_config_hash:
            raise ValueError(f"Config hash mismatch for project {project_id}")

    if sha256_file(PRE_DIR / "adjudication_tasks_blind.jsonl") != prepared_manifest.get(
        "source_blind_queue_sha256"
    ):
        raise ValueError("Prepared manifest does not point to the frozen blind queue")
    if sha256_file(DEFAULT_UNIVERSE) != prepared_manifest.get("source_universe_sha256"):
        raise ValueError("Prepared manifest universe hash mismatch")

    prepared_by_project = {
        project_id: load_jsonl(spec["prepared"])
        for project_id, spec in EXPECTED_PROJECTS.items()
    }
    all_ids = [row["adjudication_id"] for rows in prepared_by_project.values() for row in rows]
    if len(all_ids) != 47 or len(set(all_ids)) != 47:
        raise ValueError("Prepared adjudication IDs must contain 47 unique values")
    assert_no_forbidden_keys(prepared_by_project, "prepared_tasks")

    blind_rows = load_jsonl(PRE_DIR / "adjudication_tasks_blind.jsonl")
    blind_identity = {
        row["adjudication_id"]: (
            row["unit"],
            row["dataset"],
            row["event_id"],
            row.get("atom_id"),
            tuple(row["fields_to_adjudicate"]),
        )
        for row in blind_rows
    }
    prepared_identity = {
        row["adjudication_id"]: (
            row["unit"],
            row["dataset"],
            row["event_id"],
            row.get("atom_id"),
            tuple(row["fields_to_adjudicate"]),
        )
        for rows in prepared_by_project.values()
        for row in rows
    }
    if blind_identity != prepared_identity:
        raise ValueError("Prepared task identities differ from the frozen blind queue")
    return source_manifest, prepared_manifest, prepared_by_project


def _choice(entry: dict[str, Any], field: str, allowed: Sequence[str]) -> str:
    choices = (entry.get("value") or {}).get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0] not in allowed:
        raise ValueError(f"Invalid choice for {field}: {choices}")
    return choices[0]


def parse_adjudication_result(data: dict[str, Any], raw_result: Any) -> tuple[dict[str, str], list[str]]:
    result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    if not isinstance(result, list):
        raise ValueError("Adjudication result must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in result:
        if not isinstance(entry, dict) or not isinstance(entry.get("from_name"), str):
            raise ValueError("Malformed adjudication result entry")
        name = entry["from_name"]
        if name in by_name:
            raise ValueError(f"Duplicate adjudication control: {name}")
        by_name[name] = entry

    review = by_name.pop("review_complete", None)
    if review is None or _choice(review, "review_complete", ("confirmed",)) != "confirmed":
        raise ValueError("Missing confirmed review_complete")
    notes: list[str] = []
    note_entry = by_name.pop("notes", None)
    if note_entry is not None:
        raw_notes = (note_entry.get("value") or {}).get("text", [])
        if isinstance(raw_notes, str):
            raw_notes = [raw_notes]
        if not isinstance(raw_notes, list) or not all(isinstance(item, str) for item in raw_notes):
            raise ValueError("Invalid adjudication notes")
        notes = [item for item in raw_notes if item]

    decisions: dict[str, str] = {}
    if data.get("unit") == "atom":
        questions = data.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Atom adjudication task has no questions")
        question_fields = []
        for index, question in enumerate(questions):
            field = question.get("field") if isinstance(question, dict) else None
            if field not in ATOM_FIELDS or field in question_fields:
                raise ValueError(f"Invalid atom question field: {field}")
            question_fields.append(field)
            name = f"decision_{index}"
            entry = by_name.pop(name, None)
            if entry is None:
                raise ValueError(f"Missing dynamic control {name}")
            decisions[field] = _choice(entry, field, ("yes", "no"))
        if set(question_fields) != set(data.get("fields_to_adjudicate", [])):
            raise ValueError("Dynamic questions do not match fields_to_adjudicate")
    elif data.get("unit") == "claim":
        if data.get("fields_to_adjudicate") != ["completeness_missed"]:
            raise ValueError("Claim adjudication must resolve completeness only")
        entry = by_name.pop("completeness_missed", None)
        if entry is None:
            raise ValueError("Missing completeness_missed control")
        decisions["completeness_missed"] = _choice(
            entry, "completeness_missed", COMPLETENESS_VALUES
        )
    else:
        raise ValueError(f"Unknown adjudication unit: {data.get('unit')}")
    if by_name:
        raise ValueError(f"Unexpected adjudication controls: {sorted(by_name)}")
    return decisions, notes


def audit_project(
    connection: sqlite3.Connection,
    project_id: int,
    prepared_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = EXPECTED_PROJECTS[project_id]
    project = connection.execute(
        """
        SELECT id, title, label_config, maximum_annotations, deleted_at, result_count
        FROM project WHERE id = ?
        """,
        (project_id,),
    ).fetchone()
    if project is None or project["deleted_at"] is not None:
        raise ValueError(f"Missing or archived adjudication project {project_id}")
    if project["title"] != spec["title"] or project["maximum_annotations"] != 1:
        raise ValueError(f"Project {project_id} metadata differs from launch contract")
    if project["label_config"] != spec["config"].read_text(encoding="utf-8"):
        raise ValueError(f"Project {project_id} label config differs from frozen XML")
    members = {
        row[0]
        for row in connection.execute(
            "SELECT user_id FROM projects_projectmember WHERE project_id = ?", (project_id,)
        ).fetchall()
    }
    if members != EXPECTED_PROJECT_MEMBERS:
        raise ValueError(f"Unexpected project {project_id} members: {sorted(members)}")

    expected_by_id = {row["adjudication_id"]: row for row in prepared_rows}
    task_rows = connection.execute(
        """
        SELECT id, inner_id, data, is_labeled, total_annotations, cancelled_annotations,
               total_predictions, comment_count
        FROM task WHERE project_id = ? ORDER BY inner_id, id
        """,
        (project_id,),
    ).fetchall()
    if len(task_rows) != spec["task_count"]:
        raise ValueError(f"Project {project_id} task count is {len(task_rows)}")
    task_ids = [row["id"] for row in task_rows]
    placeholders = ",".join("?" for _ in task_ids)
    completion_rows = connection.execute(
        f"""
        SELECT tc.*, u.email AS completed_by_email
        FROM task_completion AS tc
        LEFT JOIN htx_user AS u ON u.id = tc.completed_by_id
        WHERE tc.task_id IN ({placeholders})
        ORDER BY tc.task_id, tc.id
        """,
        tuple(task_ids),
    ).fetchall()
    drafts = connection.execute(
        f"SELECT COUNT(*) FROM tasks_annotationdraft WHERE task_id IN ({placeholders})",
        tuple(task_ids),
    ).fetchone()[0]
    if drafts:
        raise ValueError(f"Project {project_id} still has {drafts} drafts")
    by_task: dict[int, list[sqlite3.Row]] = defaultdict(list)
    cancelled = 0
    for row in completion_rows:
        if row["was_cancelled"]:
            cancelled += 1
        else:
            by_task[row["task_id"]].append(row)
    if cancelled:
        raise ValueError(f"Project {project_id} has cancelled completions")

    exported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for task_row in task_rows:
        data = json.loads(task_row["data"])
        adjudication_id = data.get("adjudication_id")
        if adjudication_id not in expected_by_id or adjudication_id in seen_ids:
            raise ValueError(f"Unexpected/duplicate adjudication ID: {adjudication_id}")
        seen_ids.add(adjudication_id)
        if data != expected_by_id[adjudication_id]:
            raise ValueError(f"Live task data differs from prepared queue: {adjudication_id}")
        if (
            task_row["is_labeled"] != 1
            or task_row["total_annotations"] != 1
            or task_row["cancelled_annotations"] != 0
        ):
            raise ValueError(f"Task state is incomplete for {adjudication_id}")
        active = by_task.get(task_row["id"], [])
        if len(active) != 1:
            raise ValueError(f"Expected one active completion for {adjudication_id}, found {len(active)}")
        completion = active[0]
        if (
            completion["project_id"] != project_id
            or completion["completed_by_id"] != EXPECTED_ADJUDICATOR_USER_ID
            or completion["completed_by_email"] != EXPECTED_ADJUDICATOR_EMAIL
        ):
            raise ValueError(f"Unexpected adjudicator for {adjudication_id}")
        decisions, notes = parse_adjudication_result(data, completion["result"])
        exported.append(
            {
                "adjudication_id": adjudication_id,
                "project_id": project_id,
                "task_id": task_row["id"],
                "task_inner_id": task_row["inner_id"],
                "annotation_id": completion["id"],
                "completed_by_user_id": completion["completed_by_id"],
                "completed_by_email": completion["completed_by_email"],
                "created_at_utc": completion["created_at"],
                "updated_at_utc": completion["updated_at"],
                "lead_time_seconds": completion["lead_time"],
                "unit": data["unit"],
                "dataset": data["dataset"],
                "event_id": data["event_id"],
                "atom_id": data.get("atom_id"),
                "fields_to_adjudicate": data["fields_to_adjudicate"],
                "decisions": decisions,
                "review_complete": "confirmed",
                "notes": notes,
            }
        )
    if seen_ids != set(expected_by_id):
        raise ValueError(f"Project {project_id} is missing prepared IDs")
    last_updated = max(row["updated_at_utc"] for row in exported)
    lead_times = sorted(float(row["lead_time_seconds"] or 0.0) for row in exported)
    summary = {
        "project_id": project_id,
        "title": project["title"],
        "expected_tasks": spec["task_count"],
        "completed_tasks": len(exported),
        "active_annotations": len(exported),
        "cancelled_annotations": cancelled,
        "drafts": drafts,
        "confirmed_reviews": sum(row["review_complete"] == "confirmed" for row in exported),
        "notes_count": sum(bool(row["notes"]) for row in exported),
        "completed_by_user_id": EXPECTED_ADJUDICATOR_USER_ID,
        "completed_by_email": EXPECTED_ADJUDICATOR_EMAIL,
        "project_members": sorted(members),
        "last_annotation_updated_at_utc": last_updated,
        "lead_time_seconds_median": lead_times[len(lead_times) // 2],
        "lead_time_seconds_max": max(lead_times),
        "project_result_count_observed": project["result_count"],
        "project_result_count_note": (
            "aggregate field was not refreshed; task/completion rows are authoritative"
            if project["result_count"] != len(exported)
            else None
        ),
    }
    return exported, summary


def audit_prior_exposure(
    connection: sqlite3.Connection,
    adjudications: Sequence[dict[str, Any]],
    pilot_project_id: int = 18,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT t.data, tc.id AS annotation_id, tc.result
        FROM task AS t
        JOIN task_completion AS tc ON tc.task_id = t.id
        WHERE t.project_id = ? AND tc.was_cancelled = 0 AND tc.completed_by_id = ?
        ORDER BY t.inner_id, tc.id
        """,
        (pilot_project_id, EXPECTED_ADJUDICATOR_USER_ID),
    ).fetchall()
    pilot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        data = json.loads(row["data"])
        result = json.loads(row["result"])
        labels: dict[str, Any] = {}
        for entry in result:
            name = entry.get("from_name")
            choices = (entry.get("value") or {}).get("choices", [])
            if isinstance(choices, list) and len(choices) == 1:
                labels[name] = choices[0]
        pilot[(data.get("dataset"), data.get("event_id"))].append(
            {
                "atom_id": data.get("atom_id"),
                "annotation_id": row["annotation_id"],
                "labels": labels,
            }
        )
    overlaps: list[dict[str, Any]] = []
    for adjudication in adjudications:
        claim_key = (adjudication["dataset"], adjudication["event_id"])
        candidates = pilot.get(claim_key, [])
        if adjudication["unit"] == "atom":
            candidates = [row for row in candidates if row["atom_id"] == adjudication["atom_id"]]
        if not candidates:
            continue
        relevant_fields = adjudication["fields_to_adjudicate"]
        overlaps.append(
            {
                "adjudication_id": adjudication["adjudication_id"],
                "unit": adjudication["unit"],
                "dataset": adjudication["dataset"],
                "event_id": adjudication["event_id"],
                "atom_id": adjudication.get("atom_id"),
                "fields_to_adjudicate": relevant_fields,
                "pilot_project_id": pilot_project_id,
                "prior_rows": [
                    {
                        "atom_id": row["atom_id"],
                        "annotation_id": row["annotation_id"],
                        "relevant_labels": {
                            field: row["labels"].get(field)
                            for field in relevant_fields
                            if field in row["labels"]
                        },
                    }
                    for row in candidates
                ],
            }
        )
    return overlaps


def load_frozen_annotations() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    a_rows = load_jsonl(PRE_DIR / "atom_annotations_a.jsonl")
    b_rows = load_jsonl(PRE_DIR / "atom_annotations_b.jsonl")
    claim_rows = load_jsonl(PRE_DIR / "claim_annotations.jsonl")
    if len(a_rows) != 257 or len(b_rows) != 257 or len(claim_rows) != 200:
        raise ValueError("Unexpected frozen annotation counts")
    return a_rows, b_rows, claim_rows


def _unique_by(rows: Sequence[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any], name: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def resolve_gold(
    a_rows: Sequence[dict[str, Any]],
    b_rows: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
    adjudications: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    atom_key = lambda row: (row["dataset"], row["event_id"], row["atom_id"])
    claim_key = lambda row: (row["dataset"], row["event_id"])
    a_by_key = _unique_by(a_rows, atom_key, "annotator A atom")
    b_by_key = _unique_by(b_rows, atom_key, "annotator B atom")
    claims_by_key = _unique_by(claim_rows, claim_key, "claim")
    if set(a_by_key) != set(b_by_key):
        raise ValueError("Frozen A/B atom universes differ")

    atom_adj: dict[tuple[str, str, str], dict[str, Any]] = {}
    claim_adj: dict[tuple[str, str], dict[str, Any]] = {}
    all_decisions: set[tuple[str, str]] = set()
    for row in adjudications:
        target = atom_adj if row["unit"] == "atom" else claim_adj
        key = atom_key(row) if row["unit"] == "atom" else claim_key(row)
        if key in target:
            raise ValueError(f"Duplicate adjudication semantic key: {key}")
        target[key] = row
        for field in row["decisions"]:
            decision_key = (row["adjudication_id"], field)
            if decision_key in all_decisions:
                raise ValueError(f"Duplicate adjudication field decision: {decision_key}")
            all_decisions.add(decision_key)

    consumed: set[tuple[str, str]] = set()
    gold_atoms: list[dict[str, Any]] = []
    for key in sorted(a_by_key):
        a = a_by_key[key]
        b = b_by_key[key]
        for text_field in ("claim", "proposition", "atom_type"):
            if a[text_field] != b[text_field]:
                raise ValueError(f"Frozen A/B text mismatch for {key}: {text_field}")
        gold: dict[str, str] = {}
        resolution: dict[str, dict[str, Any]] = {}
        for field in ATOM_FIELDS:
            value_a = a["labels"][field]
            value_b = b["labels"][field]
            if value_a == value_b:
                value = value_a
                resolution[field] = {
                    "method": "annotator_consensus",
                    "annotator_a": value_a,
                    "annotator_b": value_b,
                    "adjudication_id": None,
                }
            else:
                adjudication = atom_adj.get(key)
                if adjudication is None or field not in adjudication["decisions"]:
                    raise ValueError(f"Missing atom adjudication for {key} {field}")
                value = adjudication["decisions"][field]
                consumed.add((adjudication["adjudication_id"], field))
                resolution[field] = {
                    "method": "blind_third_rater",
                    "annotator_a": value_a,
                    "annotator_b": value_b,
                    "adjudication_id": adjudication["adjudication_id"],
                    "adjudicator": value,
                }
            gold[field] = value
        gold_atoms.append(
            {
                "dataset": key[0],
                "event_id": key[1],
                "atom_id": key[2],
                "claim": a["claim"],
                "proposition": a["proposition"],
                "atom_type": a["atom_type"],
                "gold": gold,
                "resolution": resolution,
            }
        )

    atoms_by_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gold_atoms:
        atoms_by_claim[(row["dataset"], row["event_id"])].append(row)
    gold_claims: list[dict[str, Any]] = []
    for key in sorted(claims_by_key):
        source = claims_by_key[key]
        value_a = source["a"]["completeness_missed"]
        value_b = source["b"]["completeness_missed"]
        if value_a is not None and value_b is not None and value_a == value_b:
            value = value_a
            resolution = {
                "method": "annotator_consensus",
                "annotator_a": value_a,
                "annotator_b": value_b,
                "adjudication_id": None,
            }
        else:
            adjudication = claim_adj.get(key)
            if adjudication is None or "completeness_missed" not in adjudication["decisions"]:
                raise ValueError(f"Missing completeness adjudication for {key}")
            value = adjudication["decisions"]["completeness_missed"]
            consumed.add((adjudication["adjudication_id"], "completeness_missed"))
            resolution = {
                "method": "blind_third_rater",
                "annotator_a": value_a,
                "annotator_b": value_b,
                "annotator_a_by_atom": source["a"]["completeness_missed_by_atom"],
                "annotator_b_by_atom": source["b"]["completeness_missed_by_atom"],
                "adjudication_id": adjudication["adjudication_id"],
                "adjudicator": value,
            }
        claim_atoms = atoms_by_claim.get(key, [])
        if len(claim_atoms) != source["atom_count"]:
            raise ValueError(f"Gold claim atom count mismatch for {key}")
        failure_dimensions: list[str] = []
        if value != "0":
            failure_dimensions.append("completeness")
        if any(atom["gold"]["faithfulness"] != "yes" for atom in claim_atoms):
            failure_dimensions.append("faithfulness")
        if any(atom["gold"]["atomicity"] != "yes" for atom in claim_atoms):
            failure_dimensions.append("atomicity")
        gold_claims.append(
            {
                "dataset": key[0],
                "event_id": key[1],
                "claim": source["claim"],
                "atom_count": source["atom_count"],
                "atom_ids": source["atom_ids"],
                "gold": {
                    "completeness_missed": value,
                    "complete_coverage": value == "0",
                    "strict_all_criteria_pass": not failure_dimensions,
                },
                "resolution": {"completeness_missed": resolution},
                "failure_dimensions": failure_dimensions,
                "auxiliary_claim_complexity": {
                    "annotator_a": source["a"]["claim_complexity"],
                    "annotator_b": source["b"]["claim_complexity"],
                    "gold": None,
                    "note": "not adjudicated and not a primary LLM-quality dimension",
                },
            }
        )
    if consumed != all_decisions:
        missing = sorted(all_decisions - consumed)
        unexpected = sorted(consumed - all_decisions)
        raise ValueError(f"Unconsumed/unexpected adjudication decisions: missing={missing}, extra={unexpected}")
    merge_audit = {
        "expected_field_decisions": len(all_decisions),
        "consumed_field_decisions": len(consumed),
        "atom_field_decisions": sum(len(row["decisions"]) for row in adjudications if row["unit"] == "atom"),
        "claim_field_decisions": sum(len(row["decisions"]) for row in adjudications if row["unit"] == "claim"),
        "consensus_atom_field_labels": sum(
            detail["method"] == "annotator_consensus"
            for row in gold_atoms
            for detail in row["resolution"].values()
        ),
        "adjudicated_atom_field_labels": sum(
            detail["method"] == "blind_third_rater"
            for row in gold_atoms
            for detail in row["resolution"].values()
        ),
        "consensus_claim_labels": sum(
            row["resolution"]["completeness_missed"]["method"] == "annotator_consensus"
            for row in gold_claims
        ),
        "adjudicated_claim_labels": sum(
            row["resolution"]["completeness_missed"]["method"] == "blind_third_rater"
            for row in gold_claims
        ),
    }
    return gold_atoms, gold_claims, merge_audit


def percentile(values: Sequence[float], probability: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        raise ValueError("Cannot calculate percentile of empty data")
    position = (len(clean) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def stratified_claim_cluster_ci(
    rows: Sequence[dict[str, Any]],
    positive_fn: Callable[[dict[str, Any]], bool],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    by_dataset_claim: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_dataset_claim[row["dataset"]][(row["dataset"], row["event_id"])].append(row)
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(reps):
        sampled: list[dict[str, Any]] = []
        for claims in by_dataset_claim.values():
            keys = list(claims)
            for _draw in range(len(keys)):
                sampled.extend(claims[rng.choice(keys)])
        rates.append(sum(positive_fn(row) for row in sampled) / len(sampled))
    return {
        "method": "95% percentile; dataset-stratified claim-cluster bootstrap",
        "replicates": reps,
        "seed": seed,
        "low": percentile(rates, 0.025),
        "high": percentile(rates, 0.975),
    }


def gold_rate_summary(
    rows: Sequence[dict[str, Any]],
    positive_fn: Callable[[dict[str, Any]], bool],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    positive_count = sum(positive_fn(row) for row in rows)
    result = {
        "n": len(rows),
        "pass_count": positive_count,
        "fail_count": len(rows) - positive_count,
        "pass_rate": positive_count / len(rows),
        "error_rate": (len(rows) - positive_count) / len(rows),
        "claim_cluster_bootstrap_ci95": stratified_claim_cluster_ci(
            rows, positive_fn, reps, seed
        ),
        "by_dataset": {},
    }
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        count = sum(positive_fn(row) for row in subset)
        result["by_dataset"][dataset] = {
            "n": len(subset),
            "pass_count": count,
            "fail_count": len(subset) - count,
            "pass_rate": count / len(subset),
            "error_rate": (len(subset) - count) / len(subset),
        }
    return result


def agreement_summary(source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n",
        "exact_agreement_count",
        "exact_agreement",
        "cohen_kappa",
        "gwet_ac1",
        "annotator_a_positive_rate",
        "annotator_b_positive_rate",
        "minority_label",
        "minority_agreement",
    )
    return {key: source.get(key) for key in keys}


def build_metrics(
    source_manifest: dict[str, Any],
    prepared_manifest: dict[str, Any],
    pre_metrics: dict[str, Any],
    project_summaries: Sequence[dict[str, Any]],
    adjudications: Sequence[dict[str, Any]],
    prior_exposure: Sequence[dict[str, Any]],
    gold_atoms: Sequence[dict[str, Any]],
    gold_claims: Sequence[dict[str, Any]],
    merge_audit: dict[str, Any],
    db_path: Path,
    current_source_hash: str,
    sqlite_quick_check: str,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    faith = gold_rate_summary(
        gold_atoms, lambda row: row["gold"]["faithfulness"] == "yes", bootstrap_reps, seed + 1
    )
    atomic = gold_rate_summary(
        gold_atoms, lambda row: row["gold"]["atomicity"] == "yes", bootstrap_reps, seed + 2
    )
    complete = gold_rate_summary(
        gold_claims, lambda row: row["gold"]["complete_coverage"], bootstrap_reps, seed + 3
    )
    strict = gold_rate_summary(
        gold_claims,
        lambda row: row["gold"]["strict_all_criteria_pass"],
        bootstrap_reps,
        seed + 4,
    )
    completeness_distribution = Counter(
        row["gold"]["completeness_missed"] for row in gold_claims
    )
    field_decisions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in adjudications:
        for field, value in row["decisions"].items():
            field_decisions[field][value] += 1
    failed_claim_dimensions = Counter(
        dimension for row in gold_claims for dimension in row["failure_dimensions"]
    )
    pre_complete = pre_metrics["claim_level"]["complete_coverage"]
    return {
        "schema_version": 1,
        "analysis_status": "final_adjudicated_gold",
        "snapshot": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_path": str(db_path.resolve()),
            "sqlite_quick_check": sqlite_quick_check,
            "writing_anchor_path": str(WRITING_ANCHOR.resolve()),
            "writing_anchor_sha256": sha256_file(WRITING_ANCHOR),
            "source_analysis_input_sha256": source_manifest["analysis_input_sha256"],
            "current_formal_source_analysis_input_sha256": current_source_hash,
            "current_formal_source_matches_frozen": current_source_hash
            == source_manifest["analysis_input_sha256"],
            "source_reliability_generation_id": prepared_manifest["source_reliability_generation_id"],
            "adjudication_protocol_version": EXPECTED_PROTOCOL,
            "prepared_atom_tasks_sha256": prepared_manifest["artifacts"]["atom_tasks.jsonl"][
                "sha256"
            ],
            "prepared_completeness_tasks_sha256": prepared_manifest["artifacts"][
                "completeness_tasks.jsonl"
            ]["sha256"],
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_seed": seed,
        },
        "study_design": {
            "claims": 200,
            "atoms": 257,
            "claims_by_dataset": {"liar_raw": 100, "rawfc": 100},
            "sampling": "70% random and 30% difficulty-prioritized within each dataset validation set",
            "scope": "claim atomization only",
            "human_label_use": "post-hoc reliability audit only; not used in pipeline training, generation, selection, or checkpoint choice",
            "gold_rule": "A/B exact consensus; otherwise blinded third-rater adjudication",
            "claim_complexity": "auxiliary only; not adjudicated and not included in gold quality rates",
        },
        "adjudication_audit": {
            "complete": True,
            "projects": list(project_summaries),
            "tasks_expected": 47,
            "tasks_completed": len(adjudications),
            "field_decisions_expected": 50,
            "field_decisions_completed": sum(len(row["decisions"]) for row in adjudications),
            "field_decision_distribution": {
                field: dict(counts) for field, counts in sorted(field_decisions.items())
            },
            "merge": merge_audit,
            "anomalies": [],
            "non_semantic_observations": [
                "project.result_count remained 0 although task/completion rows are complete",
                "one atom task has a 76919.796-second cross-day page-open lead time",
            ],
            "prior_pilot_exposure": {
                "count": len(prior_exposure),
                "records": list(prior_exposure),
                "interpretation": "A/B labels remained hidden, but the third rater had previously seen these semantic units in pilot project 18",
            },
        },
        "pre_adjudication_agreement": {
            "note": "IAA is computed from the two original annotators only; the adjudicator is not mixed into agreement coefficients",
            "faithfulness": agreement_summary(pre_metrics["atom_level"]["faithfulness"]),
            "atomicity": agreement_summary(pre_metrics["atom_level"]["atomicity"]),
            "complete_coverage": agreement_summary(pre_complete),
            "complete_coverage_excluded_internal_conflict_claims": pre_complete[
                "excluded_internal_conflict_claims"
            ],
        },
        "final_gold": {
            "faithfulness": faith,
            "atomicity": atomic,
            "complete_coverage": complete,
            "completeness_missed_distribution": {
                value: completeness_distribution[value] for value in COMPLETENESS_VALUES
            },
            "strict_all_criteria_pass": {
                **strict,
                "definition": "completeness_missed=0 and every atom has faithfulness=yes and atomicity=yes",
                "failed_claims_by_dimension_nonexclusive": dict(failed_claim_dimensions),
            },
        },
        "interpretation_boundary": {
            "supported": "descriptive reliability and adjudicated quality rates for LLM claim atomization on the designed validation sample",
            "not_supported": [
                "Evidence Map relation/directness/confidence reliability",
                "a causal effect of claim decomposition on downstream fact-checking performance",
                "a population estimate under the original candidate distribution",
            ],
        },
    }


def _rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _gold_ci(result: dict[str, Any]) -> str:
    ci = result["claim_cluster_bootstrap_ci95"]
    return f"{_rate(ci['low'])}--{_rate(ci['high'])}"


def render_report(metrics: dict[str, Any]) -> str:
    gold = metrics["final_gold"]
    pre = metrics["pre_adjudication_agreement"]
    audit = metrics["adjudication_audit"]
    distributions = audit["field_decision_distribution"]
    total_cancelled = sum(project["cancelled_annotations"] for project in audit["projects"])
    total_drafts = sum(project["drafts"] for project in audit["projects"])
    total_notes = sum(project["notes_count"] for project in audit["projects"])
    prior_units = ", ".join(
        f"`{row['dataset']}/{row['event_id']}"
        + (f"/{row['atom_id']}`" if row.get("atom_id") else "`")
        + " "
        + "/".join(row["fields_to_adjudicate"])
        for row in audit["prior_pilot_exposure"]["records"]
    )
    lines = [
        "# Exp1 Claim Atomization 人工可靠性分析（仲裁完成）",
        "",
        f"- 生成时间：`{metrics['snapshot']['generated_at_utc']}`",
        "- 写作锚点：`writing_outline_v0.4.2_structure_only.md`。",
        "- 样本：LIAR-RAW / RAWFC validation 各 100 claims，共 200 claims、257 atoms。",
        "- Gold 规则：A/B 完全一致则保留共识；否则由第三位标注者在看不到 A/B 标签时独立仲裁。",
        "- 完成状态：47/47 仲裁任务、50/50 字段决策均通过结构审计并成功消费。",
        "- 结论边界：Exp1 只审计 claim atomization，不验证 Evidence Map 或 downstream causality。",
        "",
        "## Final gold 质量率",
        "",
        "| 维度 | 单位/N | Gold 通过 | Gold 错误 | 95% claim-cluster CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, unit, result in (
        ("忠实性", "atom", gold["faithfulness"]),
        ("原子性", "atom", gold["atomicity"]),
        ("完整覆盖（missed=0）", "claim", gold["complete_coverage"]),
        ("严格三维全通过（次要）", "claim", gold["strict_all_criteria_pass"]),
    ):
        lines.append(
            f"| {name} | {unit}/{result['n']} | {result['pass_count']}/{result['n']} ({_rate(result['pass_rate'])}) "
            f"| {result['fail_count']}/{result['n']} ({_rate(result['error_rate'])}) | {_gold_ci(result)} |"
        )
    lines.extend(
        [
            "",
            "Strict pass 定义为该 claim 的 `completeness_missed=0`，且其所有 atoms 均同时满足 faithfulness=yes 和 atomicity=yes。它是对 atom 数敏感的逻辑合取，只作次要诊断。",
            "",
            "## 双标 IAA 与最终 gold 的分工",
            "",
            "| 维度 | IAA N | Pre-adj Exact | Cohen κ | Gwet AC1 | Final gold pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, pre_key, gold_key in (
        ("忠实性", "faithfulness", "faithfulness"),
        ("原子性", "atomicity", "atomicity"),
        ("完整覆盖", "complete_coverage", "complete_coverage"),
    ):
        p = pre[pre_key]
        g = gold[gold_key]
        lines.append(
            f"| {name} | {p['n']} | {_rate(p['exact_agreement'])} | {_number(p['cohen_kappa'])} "
            f"| {_number(p['gwet_ac1'])} | {_rate(g['pass_rate'])} |"
        )
    lines.extend(
        [
            "",
            "完整覆盖的 pre-adjudication IAA 分母为 199，因为一个 claim 存在标注者内部重复字段冲突；该 claim 已进入第三人队列，因此 final gold 分母恢复为 200。IAA 始终只由原两位标注者计算，第三人结果仅用于形成唯一 gold。多数通过类明显偏斜，故 Exact、κ 与 AC1 需并列解释。",
            "",
            "## 数据集分层",
            "",
            "| 数据集 | Faithfulness | Atomicity | Complete coverage | Strict pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("liar_raw", "rawfc"):
        lines.append(
            f"| {dataset} | {_rate(gold['faithfulness']['by_dataset'][dataset]['pass_rate'])} "
            f"| {_rate(gold['atomicity']['by_dataset'][dataset]['pass_rate'])} "
            f"| {_rate(gold['complete_coverage']['by_dataset'][dataset]['pass_rate'])} "
            f"| {_rate(gold['strict_all_criteria_pass']['by_dataset'][dataset]['pass_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 仲裁质量控制",
            "",
            f"- Atom 项目：{audit['projects'][0]['completed_tasks']}/{audit['projects'][0]['expected_tasks']}；claim completeness 项目：{audit['projects'][1]['completed_tasks']}/{audit['projects'][1]['expected_tasks']}。",
            f"- 所有任务恰好一条 active annotation，均由指定仲裁者提交，且全部 `review_complete=confirmed`；{total_cancelled} cancelled、{total_drafts} drafts、{total_notes} notes。",
            f"- Atom 决策：atomicity {sum(distributions['atomicity'].values())}（yes={distributions['atomicity'].get('yes', 0)}/no={distributions['atomicity'].get('no', 0)}），faithfulness {sum(distributions['faithfulness'].values())}（yes={distributions['faithfulness'].get('yes', 0)}/no={distributions['faithfulness'].get('no', 0)}）；completeness {sum(distributions['completeness_missed'].values())}（0={distributions['completeness_missed'].get('0', 0)}/1={distributions['completeness_missed'].get('1', 0)}）。",
            "- Live task data、prepared queues、blind queue、XML config 和 formal A/B source snapshot 的哈希/语义指纹均一致。",
            "- 三个双字段任务严格通过 `questions[i].field` 映射 `decision_i`；未使用顺序不同的 `fields_to_adjudicate` 解释结果。",
            "- `project.result_count=0` 是未刷新的聚合字段；逐任务与 completion 表完整。一个跨日挂页产生 76,919.796 秒 lead-time 离群，不影响标签结构。",
            "",
            "## 协议披露",
            "",
            f"第三人始终看不到 A/B 原始标签，但 pilot project 18 与本次 resolution queue 有 {audit['prior_pilot_exposure']['count']} 个语义任务重合（{prior_units}）。因此仲裁对 A/B 标签是盲化的，但这些重合项并非对任务内容的首次接触；该事实保留在 metrics 与最终 gold 产物中，正文不据此作更强的独立性主张。",
            "",
            "## 解释",
            "",
            f"最终 human gold 显示，faithfulness、atomicity 与 complete coverage 的通过率分别为 {_rate(gold['faithfulness']['pass_rate'])}、{_rate(gold['atomicity']['pass_rate'])} 和 {_rate(gold['complete_coverage']['pass_rate'])}，说明当前 LLM atomization 在该审计样本上总体高度符合人工质量判断，可作为后续结构构建的可靠上游输入。Atomicity 错误率为 {_rate(gold['atomicity']['error_rate'])}，仍是主要残余风险。Pre-adjudication IAA 则表明人工判断过程总体稳定；它与 final gold 的 artifact-quality 结论分属不同证据层。该结论不能外推到 Evidence Map 的 relation/directness/confidence，也不能证明 claim decomposition 因果性地改善 downstream F1。",
            "",
            "## 生成文件",
            "",
            "- `metrics.json`：仲裁审计、pre-adj IAA、final gold 指标及 bootstrap 区间。",
            "- `adjudication_annotations.jsonl`：按稳定 ID 导出的第三人结果。",
            "- `gold_atom_annotations.jsonl`：257 个 atom 的唯一 gold 与逐字段 resolution provenance。",
            "- `gold_claim_annotations.jsonl`：200 个 claim 的 completeness gold 与严格诊断。",
            "- `paper_insert_v0.4.2.md`：Exp1 正文、Exp2 占位和 Limitations 替换稿。",
            "- `manifest.json`：最终文件哈希；最后发布。",
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_insert(metrics: dict[str, Any]) -> str:
    gold = metrics["final_gold"]
    pre = metrics["pre_adjudication_agreement"]
    faith = gold["faithfulness"]
    atomic = gold["atomicity"]
    complete = gold["complete_coverage"]
    lines = [
        "## Claim Atomization Reliability Study (Exp1)",
        "",
        "为审计 claim atomization 这一上游输入，我们从 LIAR-RAW 与 RAWFC validation data 各抽取 100 条 claims（70% 随机、30% 困难优先），得到 257 个 atoms。两位标注者独立评估 faithfulness、atomicity 与 completeness；所有主维度 exact mismatches 以及一个 claim-level 内部冲突均由第三位标注者在看不到 A/B 标签时独立仲裁。",
        "",
        "| Dimension | Unit / Gold N | Final gold pass | Pre-adj. Exact | Cohen's $\\kappa$ | Gwet AC1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, unit, gold_key, pre_key in (
        ("Faithfulness", "atom", "faithfulness", "faithfulness"),
        ("Atomicity", "atom", "atomicity", "atomicity"),
        ("Complete coverage", "claim", "complete_coverage", "complete_coverage"),
    ):
        g = gold[gold_key]
        p = pre[pre_key]
        lines.append(
            f"| {name} | {unit} / {g['n']} | {_rate(g['pass_rate'])} | {_rate(p['exact_agreement'])} "
            f"| {_number(p['cohen_kappa'])} | {_number(p['gwet_ac1'])} |"
        )
    lines.extend(
        [
            "",
            f"**结果表述。** 表X表明，自动 claim atomization 在本次 LIAR-RAW/RAWFC 审计样本上高度符合独立人工质量判断。最终 human gold 中，{_rate(faith['pass_rate'])} 的 atoms 被判定为 faithful，{_rate(atomic['pass_rate'])} 满足单一、可独立核验的 atomicity 要求，且 {_rate(complete['pass_rate'])} 的 claims 获得 complete atom coverage；对应的 dataset-stratified claim-cluster bootstrap 95% 区间分别为 {_gold_ci(faith)}、{_gold_ci(atomic)} 和 {_gold_ci(complete)}。即使要求同一 claim 的所有 atoms 同时通过 faithfulness 与 atomicity，并完整覆盖原 claim，仍有 {gold['strict_all_criteria_pass']['pass_count']}/{gold['strict_all_criteria_pass']['n']}（{_rate(gold['strict_all_criteria_pass']['pass_rate'])}）达到 strict pass。因此，Exp1 不仅记录了人工审计过程，也支持在本研究覆盖的数据与抽样范围内，将当前 atomization 视为质量较高且可用于后续 retrieval 与 Evidence Map 构建的可靠上游结构输入。残余错误主要集中在 atomicity（{_rate(atomic['error_rate'])}），说明复合命题的拆分粒度仍是后续质量控制最需要关注的环节。",
            "",
            f"**证据解释。** 上述质量结论来自 final human gold，而人工审计本身的稳定性由 pre-adjudication IAA 单独衡量。两位标注者在三个维度上的 exact agreement 为 88.72%--95.72%，Gwet AC1 为 0.866--0.955，说明独立人工判断的总体模式具有较高可重复性；同时，较低的 Cohen's $\\kappa$ 与少数失败类 agreement 表明错误边界仍弱于多数通过类，不能只凭 AC1 消解这一不确定性。换言之，final gold 支持“模型生成的 atoms 基本符合人工质量判断”，IAA 支持“这一人工判断过程总体稳定”，二者不能混为同一个指标。Complete coverage 的 pre-adjudication IAA 基于 {pre['complete_coverage']['n']} 条 internally consistent claims，冲突项经仲裁后 final gold 分母恢复为 {complete['n']}。该结论不进一步证明 Evidence Map 标注可靠、atomization 带来 downstream performance 提升，或 audit trace 构成模型预测的忠实解释。",
            "",
            "## Evidence Map Annotation Reliability Study (Exp2; Placeholder)",
            "",
            "**占位说明。** Exp2 将独立审计 Evidence Map 中 candidate--atom pair 的 `relation`、`directness` 与 `confidence` 标注。当前版本只冻结结果位置与报告口径；在正式双标、分歧仲裁和 artifact audit 完成前不填入数值，也不据此扩展本文的结果或贡献表述。",
            "",
            "| Field | Human reliability | Planned LLM comparison | Diagnostic |",
            "|---|---|---|---|",
            "| Relation | Cohen's $\\kappa$ | Overall and per-relation accuracy | Confusion matrix |",
            "| Directness | Spearman $\\rho$ | Ordinal agreement (TBD) | Ordinal error analysis |",
            "| Confidence | TBD after target definition | TBD after target definition | Calibration / ECE target TBD |",
            "",
            "表注：Exp2 衡量结构标注本身的可靠性；RQ4 的 component ablation 衡量 map signals 对下游结果的敏感性，二者不是同一个问题。`gold_confidence` 记录的是人工标注者自信度，填入结果前需另行冻结其与 LLM confidence 的比较及校准目标，不能预先把它等同于事实 gold。",
            "",
            "### Limitations replacement (v0.4.2 paragraphs 1--2)",
            "",
            f"首先，claim decomposition 并不总能稳定提高事实核查表现，错误拆分、遗漏限定条件或过度细分仍可能向 retrieval 与 Evidence Map 传播 \\citep{{Hu2025DecompositionDilemmas}}。为审计并量化这一风险，我们在 200 条 claims、257 个 atoms 上完成两位标注者的独立双标与第三人盲化仲裁。最终 gold 的 faithfulness、atomicity 与 complete coverage 通过率分别为 {_rate(faith['pass_rate'])}、{_rate(atomic['pass_rate'])} 和 {_rate(complete['pass_rate'])}。这些结果支持本研究审计样本内的 atomization artifact 高度符合人工质量判断，并将 atomicity 定位为主要残余风险。由于样本来自两个 validation set 的等量抽样且过采样困难样本，该结论不能外推为 claim decomposition 普遍可靠，也不能证明它会因果性地改善 downstream verification。",
            "",
            "第二，Evidence Map 仍依赖 LLM API，且 Exp1 只覆盖 claim atomization，不能外推为对 relation、directness 或 confidence 标注的验证。Exp2 的独立双标、仲裁与校准分析仍待完成；冻结缓存、prompt/schema hash、调用日期与调用元数据提高了可复现性和 artifact-level 可审计性，但不能将这些结构标注等同于人工 gold supervision。",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    db_path: Path,
    output_dir: Path,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    source_manifest, prepared_manifest, prepared_by_project = validate_inputs()
    pre_metrics = load_json(PRE_DIR / "metrics.json")
    if pre_metrics.get("analysis_status") != "pre_adjudication":
        raise ValueError("Expected pre-adjudication source metrics")
    if pre_metrics["snapshot"]["bootstrap_replicates"] != bootstrap_reps:
        raise ValueError("Bootstrap replicate count must match the frozen source analysis")
    if pre_metrics["snapshot"]["bootstrap_seed"] != seed:
        raise ValueError("Bootstrap seed must match the frozen source analysis")

    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick_check}")
        universe = load_authoritative_universe(DEFAULT_UNIVERSE)
        source_a = load_project_annotations(connection, EXPECTED_SOURCE_PROJECTS[0])
        source_b = load_project_annotations(connection, EXPECTED_SOURCE_PROJECTS[1])
        validate_distinct_project_pair(source_a, source_b)
        validate_project_universe(source_a, universe)
        validate_project_universe(source_b, universe)
        draft_issues = load_draft_issues(
            connection,
            (EXPECTED_SOURCE_PROJECTS[0].project_id, EXPECTED_SOURCE_PROJECTS[1].project_id),
        )
        current_source_hash = analysis_input_sha256(
            (source_a, source_b), draft_issues, universe, bootstrap_reps, seed
        )
        if current_source_hash != source_manifest["analysis_input_sha256"]:
            raise ValueError(
                "Formal A/B source labels drifted after adjudication launch: "
                f"frozen={source_manifest['analysis_input_sha256']} current={current_source_hash}"
            )

        adjudications: list[dict[str, Any]] = []
        project_summaries: list[dict[str, Any]] = []
        for project_id in (20, 21):
            rows, summary = audit_project(connection, project_id, prepared_by_project[project_id])
            adjudications.extend(rows)
            project_summaries.append(summary)
        prior_exposure = audit_prior_exposure(connection, adjudications)
    finally:
        connection.close()

    if len(adjudications) != 47:
        raise ValueError(f"Expected 47 adjudication tasks, found {len(adjudications)}")
    if sum(len(row["decisions"]) for row in adjudications) != 50:
        raise ValueError("Expected 50 adjudication field decisions")
    a_rows, b_rows, claim_rows = load_frozen_annotations()
    gold_atoms, gold_claims, merge_audit = resolve_gold(
        a_rows, b_rows, claim_rows, adjudications
    )
    metrics = build_metrics(
        source_manifest,
        prepared_manifest,
        pre_metrics,
        project_summaries,
        adjudications,
        prior_exposure,
        gold_atoms,
        gold_claims,
        merge_audit,
        db_path,
        current_source_hash,
        quick_check,
        bootstrap_reps,
        seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "metrics.json",
        "adjudication_annotations.jsonl",
        "gold_atom_annotations.jsonl",
        "gold_claim_annotations.jsonl",
        "report.md",
        "paper_insert_v0.4.2.md",
    )
    manifest_path = output_dir / "manifest.json"
    manifest_base = {
        "schema_version": 1,
        "analysis_status": "final_adjudicated_gold",
        "source_analysis_input_sha256": source_manifest["analysis_input_sha256"],
        "adjudication_protocol_version": EXPECTED_PROTOCOL,
        "writing_anchor_sha256": sha256_file(WRITING_ANCHOR),
        "finalization_script_sha256": sha256_file(Path(__file__)),
        "expected_artifacts": list(artifact_names),
    }
    write_json(manifest_path, {**manifest_base, "complete": False, "artifacts": {}})
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "adjudication_annotations.jsonl",
        sorted(adjudications, key=lambda row: (row["project_id"], row["task_inner_id"])),
    )
    write_jsonl(output_dir / "gold_atom_annotations.jsonl", gold_atoms)
    write_jsonl(output_dir / "gold_claim_annotations.jsonl", gold_claims)
    atomic_write(output_dir / "report.md", render_report(metrics))
    atomic_write(output_dir / "paper_insert_v0.4.2.md", render_paper_insert(metrics))
    artifacts = {name: artifact_entry(output_dir / name) for name in artifact_names}
    generation_id = sha256_bytes(
        canonical_json(
            {
                "source": source_manifest["analysis_input_sha256"],
                "adjudications": artifacts["adjudication_annotations.jsonl"]["sha256"],
                "script": manifest_base["finalization_script_sha256"],
            }
        ).encode("utf-8")
    )[:16]
    write_json(
        manifest_path,
        {
            **manifest_base,
            "generation_id": generation_id,
            "complete": True,
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = finalize(args.db, args.output_dir, args.bootstrap_reps, args.seed)
    gold = metrics["final_gold"]
    print(f"Wrote final Exp1 reliability analysis to: {args.output_dir.resolve()}")
    print(
        f"faithfulness={gold['faithfulness']['pass_count']}/{gold['faithfulness']['n']} "
        f"atomicity={gold['atomicity']['pass_count']}/{gold['atomicity']['n']} "
        f"complete={gold['complete_coverage']['pass_count']}/{gold['complete_coverage']['n']} "
        f"strict={gold['strict_all_criteria_pass']['pass_count']}/{gold['strict_all_criteria_pass']['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
