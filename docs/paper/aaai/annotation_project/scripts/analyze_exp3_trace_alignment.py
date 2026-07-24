#!/usr/bin/env python3
"""Analyze the blinded EviTrace preference and transition audits.

The analyzer is deliberately read-only with respect to Label Studio.  It binds
the live submissions to the frozen exported tasks and private keys, collapses
only redundant label-identical completions, and keeps both annotators' answers
as separate claim-clustered observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DB = PROJECT_ROOT / "label_studio_data" / "label_studio.sqlite3"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "results" / "exp3_trace_alignment_v1"
DEFAULT_TASK_MANIFEST = DEFAULT_PREPARED_DIR / "task_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp3_trace_alignment_analysis_v1"

PREFERENCE_CHOICES = (
    "strongly_prefer_a",
    "prefer_a",
    "tie",
    "prefer_b",
    "strongly_prefer_b",
)
PREFERENCE_SIDE_SCORE = {
    "strongly_prefer_a": 2,
    "prefer_a": 1,
    "tie": 0,
    "prefer_b": -1,
    "strongly_prefer_b": -2,
}
DATA_ISSUES = (
    "translation",
    "missing_or_malformed_evidence",
    "duplicate_evidence",
    "source_or_format",
    "other",
)
TRANSITION_VALIDITY = ("invalid", "partially_valid", "valid")
MARGINAL_CONTRIBUTION = ("none", "limited", "clear")
OPERATIONS = ("OPEN", "CONTRAST", "BRIDGE", "CORROBORATE", "FALLBACK")
CHANGE_OPERATIONS = frozenset({"OPEN", "CONTRAST"})
SELF_TRANSITION_OPERATIONS = frozenset(
    {"BRIDGE", "CORROBORATE", "FALLBACK"}
)
EXPECTED_FORMAL_COUNTS = {"main": 120, "order_only": 80, "transition": 100}
PREFERENCE_PUBLIC_FIELDS = {
    "blind_task_id",
    "claim_en",
    "claim_zh",
    "sequence_a_html",
    "sequence_b_html",
}
TRANSITION_PUBLIC_FIELDS = {
    "blind_task_id",
    "claim_en",
    "claim_zh",
    "focal_atom_en",
    "focal_atom_zh",
    "state_legend_html",
    "prior_evidence_html",
    "current_evidence_html",
    "proposed_transition",
}


def _json_load(raw: Any, context: str) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {context}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        row = _json_load(raw_line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        json.dumps(
            _finite_json(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(_finite_json(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for row in rows
    ]
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))


def _finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _public_task(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data")
    if isinstance(data, dict) and set(row).issubset({"data", "meta"}):
        return data
    return row


def _unique_rows_by_blind_id(
    rows: Sequence[dict[str, Any]], context: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        blind_id = row.get("blind_task_id")
        if not isinstance(blind_id, str) or not blind_id.strip():
            raise ValueError(f"{context} contains an invalid blind_task_id")
        if blind_id in result:
            raise ValueError(f"Duplicate blind_task_id in {context}: {blind_id}")
        result[blind_id] = row
    return result


def _artifact_records(node: Any, key_path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Flatten several manifest layouts without weakening hash verification."""

    records: list[dict[str, Any]] = []
    if isinstance(node, dict):
        sha = node.get("sha256") or node.get("commitment_sha256")
        raw_path = node.get("path") or node.get("relative_path")
        if isinstance(sha, str):
            records.append(
                {
                    "keys": key_path,
                    "path": raw_path,
                    "sha256": sha,
                    "rows": node.get("rows") or node.get("row_count"),
                }
            )
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                records.extend(_artifact_records(value, (*key_path, str(key))))
            elif isinstance(value, str) and (
                key.endswith("_sha256") or key.endswith("_commitment")
            ):
                records.append(
                    {
                        "keys": (*key_path, str(key)),
                        "path": None,
                        "sha256": value,
                        "rows": None,
                    }
                )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            records.extend(_artifact_records(value, (*key_path, str(index))))
    return records


def _artifact_hash(
    manifest: dict[str, Any], path: Path, logical_names: Sequence[str]
) -> str | None:
    basename = path.name.lower()
    candidates: list[tuple[int, str]] = []
    for record in _artifact_records(manifest):
        keys = "/".join(record["keys"]).lower()
        raw_path = record.get("path")
        record_basename = Path(raw_path).name.lower() if isinstance(raw_path, str) else ""
        score = 0
        if record_basename == basename:
            score += 100
        for name in logical_names:
            normalized = name.lower()
            if normalized in keys or normalized in record_basename:
                score += 10
        if score:
            candidates.append((score, record["sha256"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    hashes = {value for score, value in candidates if score == best_score}
    if len(hashes) != 1:
        raise ValueError(f"Ambiguous manifest commitment for {path.name}")
    return next(iter(hashes))


def _resolve_bundle_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    explicit: Path | None,
    default_relative: str,
    logical_names: Sequence[str],
) -> tuple[Path, str]:
    path = explicit
    if path is None:
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, dict):
            for logical_name in logical_names:
                record = artifacts.get(logical_name)
                if isinstance(record, dict) and isinstance(record.get("path"), str):
                    candidate = Path(record["path"])
                    path = (
                        candidate
                        if candidate.is_absolute()
                        else manifest_path.parent / candidate
                    )
                    break
    if path is None:
        exact_basename_matches = []
        expected_basename = Path(default_relative).name.lower()
        for record in _artifact_records(manifest):
            raw_path = record.get("path")
            if (
                isinstance(raw_path, str)
                and Path(raw_path).name.lower() == expected_basename
            ):
                exact_basename_matches.append(Path(raw_path))
        if len(exact_basename_matches) > 1:
            raise ValueError(
                f"Ambiguous exact artifact path for {default_relative}"
            )
        if exact_basename_matches:
            candidate = exact_basename_matches[0]
            path = (
                candidate
                if candidate.is_absolute()
                else manifest_path.parent / candidate
            )
    if path is None:
        path = manifest_path.parent / default_relative
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing frozen bundle artifact: {path}")
    expected_sha = _artifact_hash(manifest, path, logical_names)
    if expected_sha is None:
        raise ValueError(f"Task manifest has no SHA-256 commitment for {path.name}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha}, got {actual_sha}"
        )
    return path, actual_sha


def load_frozen_bundle(
    task_manifest_path: Path,
    *,
    preference_tasks_path: Path | None = None,
    transition_tasks_path: Path | None = None,
    preference_key_path: Path | None = None,
    transition_key_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _json_load(
        task_manifest_path.read_text(encoding="utf-8"), str(task_manifest_path)
    )
    if not isinstance(manifest, dict):
        raise ValueError("task_manifest.json must contain an object")
    if manifest.get("complete") is not True:
        raise ValueError("Frozen task manifest must have complete=true")
    resolved: dict[str, tuple[Path, str]] = {}
    specifications = (
        (
            "preference_tasks",
            preference_tasks_path,
            "preference_tasks.jsonl",
            ("preference_tasks", "formal_preference"),
        ),
        (
            "transition_tasks",
            transition_tasks_path,
            "transition_tasks.jsonl",
            ("transition_tasks", "formal_transition"),
        ),
        (
            "preference_key",
            preference_key_path,
            "private/blinding_key.jsonl",
            ("blinding_key", "preference_key", "private_preference"),
        ),
        (
            "transition_key",
            transition_key_path,
            "private/transition_key.jsonl",
            ("transition_key", "private_transition"),
        ),
    )
    for logical, explicit, relative, names in specifications:
        resolved[logical] = _resolve_bundle_path(
            task_manifest_path, manifest, explicit, relative, names
        )
    commitments = manifest.get("private_key_commitments")
    if not isinstance(commitments, dict):
        raise ValueError("Task manifest is missing private_key_commitments")
    expected_commitments = {
        "blinding_key_sha256": resolved["preference_key"][1],
        "transition_key_sha256": resolved["transition_key"][1],
    }
    for name, actual in expected_commitments.items():
        if commitments.get(name) != actual:
            raise ValueError(f"Private key commitment differs: {name}")

    preference_public = [
        _public_task(row) for row in load_jsonl(resolved["preference_tasks"][0])
    ]
    transition_public = [
        _public_task(row) for row in load_jsonl(resolved["transition_tasks"][0])
    ]
    preference_key_all = load_jsonl(resolved["preference_key"][0])
    transition_key_all = load_jsonl(resolved["transition_key"][0])
    preference_key = [
        row for row in preference_key_all if row.get("phase", "formal") == "formal"
    ]
    transition_key = [
        row for row in transition_key_all if row.get("phase", "formal") == "formal"
    ]

    public_preference_by_id = _unique_rows_by_blind_id(
        preference_public, "preference_tasks.jsonl"
    )
    public_transition_by_id = _unique_rows_by_blind_id(
        transition_public, "transition_tasks.jsonl"
    )
    preference_key_by_id = _unique_rows_by_blind_id(
        preference_key, "formal preference key"
    )
    transition_key_by_id = _unique_rows_by_blind_id(
        transition_key, "formal transition key"
    )
    for blind_id, row in public_preference_by_id.items():
        if set(row) != PREFERENCE_PUBLIC_FIELDS:
            raise ValueError(
                f"Preference public field whitelist differs for {blind_id}: "
                f"{sorted(set(row) - PREFERENCE_PUBLIC_FIELDS)}"
            )
    for blind_id, row in public_transition_by_id.items():
        if set(row) != TRANSITION_PUBLIC_FIELDS:
            raise ValueError(
                f"Transition public field whitelist differs for {blind_id}: "
                f"{sorted(set(row) - TRANSITION_PUBLIC_FIELDS)}"
            )
    if set(public_preference_by_id) != set(preference_key_by_id):
        raise ValueError("Formal preference public tasks and private key differ")
    if set(public_transition_by_id) != set(transition_key_by_id):
        raise ValueError("Formal transition public tasks and private key differ")
    overlap = set(public_preference_by_id) & set(public_transition_by_id)
    if overlap:
        raise ValueError(f"Blind IDs overlap across task types: {sorted(overlap)[:5]}")

    for blind_id, row in preference_key_by_id.items():
        if row.get("task_type", "preference") != "preference":
            raise ValueError(f"Preference key has wrong task_type: {blind_id}")
        if row.get("comparison_type") not in {"main", "order_only"}:
            raise ValueError(f"Invalid preference comparison_type: {blind_id}")
        _validate_preference_key_row(row)
        _validate_public_fingerprint(row, public_preference_by_id[blind_id])
    for blind_id, row in transition_key_by_id.items():
        if row.get("task_type", "transition") != "transition":
            raise ValueError(f"Transition key has wrong task_type: {blind_id}")
        if row.get("comparison_type", "transition") != "transition":
            raise ValueError(f"Invalid transition comparison_type: {blind_id}")
        _validate_transition_key_row(row)
        _validate_public_fingerprint(row, public_transition_by_id[blind_id])

    main_events = {
        str(row["event_id"])
        for row in preference_key
        if row["comparison_type"] == "main"
    }
    order_events = {
        str(row["event_id"])
        for row in preference_key
        if row["comparison_type"] == "order_only"
    }
    transition_events = {str(row["event_id"]) for row in transition_key}
    if main_events & order_events or (main_events | order_events) & transition_events:
        raise ValueError("Formal main, order-only, and transition claims are not disjoint")
    if len(transition_events) != len(transition_key):
        raise ValueError("Transition audit contains more than one step for a claim")

    artifact_hash = manifest.get("artifact_sha256")
    if not isinstance(artifact_hash, dict) or not artifact_hash:
        raise ValueError("Manifest artifact_sha256 must be a non-empty mapping")
    for row in [*preference_key, *transition_key]:
        split = str(row.get("split", "test"))
        prefix = f"{split}_"
        split_hash = {
            name[len(prefix) :]: value
            for name, value in artifact_hash.items()
            if str(name).startswith(prefix)
        }
        expected = split_hash or artifact_hash
        if row.get("artifact_sha256") != expected:
            raise ValueError(
                f"Artifact SHA mapping differs in key row {row['blind_task_id']}"
            )

    return {
        "manifest": manifest,
        "manifest_path": str(task_manifest_path.resolve()),
        "manifest_sha256": sha256_file(task_manifest_path),
        "paths": {key: str(value[0]) for key, value in resolved.items()},
        "sha256": {key: value[1] for key, value in resolved.items()},
        "preference_public": public_preference_by_id,
        "transition_public": public_transition_by_id,
        "preference_key": preference_key_by_id,
        "transition_key": transition_key_by_id,
    }


def _validate_preference_key_row(row: dict[str, Any]) -> None:
    mapping = row.get("method_to_side")
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing method_to_side for {row.get('blind_task_id')}")
    evi_side = str(mapping.get("evitrace", "")).upper()
    control_side = str(mapping.get("control", "")).upper()
    if {evi_side, control_side} != {"A", "B"}:
        raise ValueError(f"Invalid method_to_side for {row.get('blind_task_id')}")
    for field in ("event_id", "stratum"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"Missing {field} in preference key")
    for field in ("evi_token_count", "control_token_count"):
        value = row.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid {field} in preference key")
    k_visible = row.get("k_visible")
    if not isinstance(k_visible, int) or k_visible <= 0:
        raise ValueError("Invalid k_visible in preference key")
    evi_uids = row.get("evi_candidate_uids")
    control_uids = row.get("control_candidate_uids")
    if not isinstance(evi_uids, list) or not isinstance(control_uids, list):
        raise ValueError("Preference candidate UID fields must be lists")
    if len(evi_uids) != k_visible or len(control_uids) != k_visible:
        raise ValueError(
            f"Matched-count candidate lists differ from k_visible for "
            f"{row.get('blind_task_id')}"
        )
    if (
        len(set(map(str, evi_uids))) != len(evi_uids)
        or len(set(map(str, control_uids))) != len(control_uids)
    ):
        raise ValueError("Preference candidate UID lists contain duplicates")
    stated_difference = row.get("token_count_difference_evi_minus_control")
    actual_difference = row["evi_token_count"] - row["control_token_count"]
    if stated_difference is not None and stated_difference != actual_difference:
        raise ValueError(
            f"Token difference is inconsistent for {row.get('blind_task_id')}"
        )
    if row.get("comparison_type") == "order_only":
        if row.get("same_evidence_set") is not True:
            raise ValueError("Order-only task is not marked as the same evidence set")
        if row.get("evi_token_count") != row.get("control_token_count"):
            raise ValueError("Order-only task does not have identical token counts")
        if set(map(str, evi_uids)) != set(map(str, control_uids)):
            raise ValueError("Order-only candidate UID sets differ")
        if list(map(str, evi_uids)) == list(map(str, control_uids)):
            raise ValueError("Order-only task has an identical ordering")


def _validate_public_fingerprint(
    key_row: dict[str, Any], public_row: dict[str, Any]
) -> None:
    expected = key_row.get("public_task_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            f"Private key lacks public task fingerprint for "
            f"{key_row.get('blind_task_id')}"
        )
    actual = hashlib.sha256(
        json.dumps(
            public_row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError(
            f"Public task fingerprint differs for {key_row.get('blind_task_id')}"
        )


def _validate_transition_key_row(row: dict[str, Any]) -> None:
    if row.get("operation") not in OPERATIONS:
        raise ValueError(f"Invalid transition operation: {row.get('operation')}")
    before = row.get("state_before")
    after = row.get("state_after")
    kind = row.get("transition_kind")
    if kind not in {"change", "self"}:
        raise ValueError(f"Invalid transition_kind: {kind}")
    if (before == after) != (kind == "self"):
        raise ValueError(
            f"Transition kind disagrees with before/after state: {row.get('blind_task_id')}"
        )
    expected_kind = (
        "change" if row["operation"] in CHANGE_OPERATIONS else "self"
    )
    if kind != expected_kind:
        raise ValueError(
            f"Operation {row['operation']} requires a {expected_kind} transition"
        )
    for field in ("event_id", "atom_id", "candidate_uid"):
        if row.get(field) is None or str(row[field]) == "":
            raise ValueError(f"Missing {field} in transition key")


def parse_preference_result(raw: Any) -> dict[str, Any]:
    payload = _json_load(raw, "preference annotation.result")
    if not isinstance(payload, list):
        raise ValueError("Preference annotation.result must be a list")
    preference: str | None = None
    issues: list[str] = []
    notes: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Preference result entries must be objects")
        name = item.get("from_name")
        value = item.get("value") or {}
        if name == "overall_preference":
            if preference is not None:
                raise ValueError("Duplicate overall_preference control")
            choices = value.get("choices", [])
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("overall_preference must contain exactly one choice")
            preference = choices[0]
        elif name == "data_issue":
            choices = value.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("data_issue choices must be a list")
            issues.extend(choices)
        elif name == "notes":
            texts = value.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, list) or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("notes text must be a string list")
            notes.extend(text for text in texts if text.strip())
    if preference not in PREFERENCE_CHOICES:
        if preference is None:
            raise ValueError("Missing overall_preference submission")
        raise ValueError(f"Invalid overall_preference: {preference}")
    if len(issues) != len(set(issues)):
        raise ValueError("Duplicate data_issue choice")
    invalid_issues = sorted(set(issues) - set(DATA_ISSUES))
    if invalid_issues:
        raise ValueError(f"Invalid data_issue choices: {invalid_issues}")
    return {
        "overall_preference": preference,
        "data_issue": sorted(issues),
        "notes": notes,
    }


def parse_transition_result(raw: Any) -> dict[str, Any]:
    payload = _json_load(raw, "transition annotation.result")
    if not isinstance(payload, list):
        raise ValueError("Transition annotation.result must be a list")
    parsed: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Transition result entries must be objects")
        name = item.get("from_name")
        if name not in {"transition_validity", "marginal_contribution"}:
            continue
        if name in parsed:
            raise ValueError(f"Duplicate {name} control")
        choices = (item.get("value") or {}).get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"{name} must contain exactly one choice")
        parsed[name] = choices[0]
    missing = sorted(
        {"transition_validity", "marginal_contribution"} - set(parsed)
    )
    if missing:
        raise ValueError(f"Missing transition submissions: {missing}")
    if parsed["transition_validity"] not in TRANSITION_VALIDITY:
        raise ValueError(
            f"Invalid transition_validity: {parsed['transition_validity']}"
        )
    if parsed["marginal_contribution"] not in MARGINAL_CONTRIBUTION:
        raise ValueError(
            f"Invalid marginal_contribution: {parsed['marginal_contribution']}"
        )
    return parsed


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def load_project_annotations(
    connection: sqlite3.Connection,
    project_id: int,
    public_tasks: dict[str, dict[str, Any]],
    task_type: str,
) -> dict[str, Any]:
    project = connection.execute(
        "SELECT id, title, maximum_annotations, deleted_at FROM project WHERE id=?",
        (project_id,),
    ).fetchone()
    if project is None or project["deleted_at"] is not None:
        raise ValueError(f"Project {project_id} is missing or archived")
    if project["maximum_annotations"] != 1:
        raise ValueError(f"Project {project_id} must have maximum_annotations=1")

    task_rows = connection.execute(
        "SELECT id, inner_id, data FROM task WHERE project_id=? ORDER BY inner_id, id",
        (project_id,),
    ).fetchall()
    if len(task_rows) != len(public_tasks):
        raise ValueError(
            f"Project {project_id} has {len(task_rows)} tasks, expected {len(public_tasks)}"
        )
    tasks_by_id: dict[int, tuple[str, dict[str, Any], sqlite3.Row]] = {}
    seen_blind: set[str] = set()
    for row in task_rows:
        data = _json_load(row["data"], f"task {row['id']}.data")
        if not isinstance(data, dict):
            raise ValueError(f"Task {row['id']} data is not an object")
        blind_id = data.get("blind_task_id")
        if blind_id not in public_tasks:
            raise ValueError(f"Project {project_id} contains unknown blind task {blind_id}")
        if blind_id in seen_blind:
            raise ValueError(
                f"Project {project_id} repeats blind task {blind_id}"
            )
        if data != public_tasks[blind_id]:
            raise ValueError(
                f"Project {project_id} task data differs from frozen task {blind_id}"
            )
        seen_blind.add(blind_id)
        tasks_by_id[row["id"]] = (blind_id, data, row)
    if seen_blind != set(public_tasks):
        raise ValueError(f"Project {project_id} task universe is incomplete")

    user_join = ""
    user_columns = (
        "NULL AS annotator_email, NULL AS annotator_name"
    )
    if _table_exists(connection, "htx_user"):
        user_join = "LEFT JOIN htx_user AS u ON u.id = a.completed_by_id"
        user_columns = (
            "u.email AS annotator_email, "
            "TRIM(COALESCE(u.first_name, '') || ' ' || COALESCE(u.last_name, '')) "
            "AS annotator_name"
        )
    completion_rows = connection.execute(
        f"""
        SELECT
            a.id AS annotation_id,
            a.task_id,
            a.result,
            a.created_at,
            a.updated_at,
            a.completed_by_id,
            a.was_cancelled,
            {user_columns}
        FROM task_completion AS a
        {user_join}
        WHERE a.project_id=?
        ORDER BY a.task_id, a.updated_at, a.id
        """,
        (project_id,),
    ).fetchall()
    active_by_task: dict[int, list[sqlite3.Row]] = defaultdict(list)
    cancelled_count = 0
    for row in completion_rows:
        if row["task_id"] not in tasks_by_id:
            raise ValueError(
                f"Project {project_id} completion points to a foreign task"
            )
        if row["was_cancelled"]:
            cancelled_count += 1
        else:
            active_by_task[row["task_id"]].append(row)

    missing = [
        tasks_by_id[task_id][0]
        for task_id in tasks_by_id
        if not active_by_task.get(task_id)
    ]
    if missing:
        raise ValueError(
            f"Project {project_id} has missing submissions: {sorted(missing)[:10]}"
        )

    parser = (
        parse_preference_result
        if task_type == "preference"
        else parse_transition_result
    )
    records: dict[str, dict[str, Any]] = {}
    duplicate_issues: list[dict[str, Any]] = []
    for task_id, rows in active_by_task.items():
        parsed = [parser(row["result"]) for row in rows]
        fingerprints = {
            json.dumps(value, ensure_ascii=False, sort_keys=True) for value in parsed
        }
        blind_id, _data, task_row = tasks_by_id[task_id]
        if len(fingerprints) != 1:
            raise ValueError(
                f"Conflicting active completions for project {project_id}, "
                f"blind task {blind_id}"
            )
        chosen = rows[-1]
        records[blind_id] = {
            "blind_task_id": blind_id,
            "project_id": project_id,
            "project_title": project["title"],
            "task_id": task_id,
            "inner_id": task_row["inner_id"],
            "annotation_id": chosen["annotation_id"],
            "created_at": chosen["created_at"],
            "updated_at": chosen["updated_at"],
            "completed_by_id": chosen["completed_by_id"],
            "annotator_email": chosen["annotator_email"],
            "annotator_name": chosen["annotator_name"],
            **parsed[-1],
        }
        if len(rows) > 1:
            duplicate_issues.append(
                {
                    "issue_type": "redundant_identical_active_completion",
                    "task_type": task_type,
                    "project_id": project_id,
                    "blind_task_id": blind_id,
                    "annotation_ids": [row["annotation_id"] for row in rows],
                    "collapsed_to_annotation_id": chosen["annotation_id"],
                }
            )
    annotators = {
        (record["completed_by_id"], record["annotator_email"])
        for record in records.values()
    }
    if len(annotators) != 1:
        raise ValueError(
            f"Project {project_id} does not contain exactly one annotator: {annotators}"
        )
    return {
        "project_id": project_id,
        "title": project["title"],
        "maximum_annotations": project["maximum_annotations"],
        "task_type": task_type,
        "task_count": len(task_rows),
        "active_completion_count": sum(len(rows) for rows in active_by_task.values()),
        "semantic_annotation_count": len(records),
        "cancelled_completion_count": cancelled_count,
        "duplicate_completion_count": len(duplicate_issues),
        "annotator": {
            "completed_by_id": next(iter(annotators))[0],
            "email": next(iter(annotators))[1],
        },
        "records": records,
        "duplicate_issues": duplicate_issues,
    }


def validate_project_pair(
    project_a: dict[str, Any], project_b: dict[str, Any]
) -> None:
    if project_a["project_id"] == project_b["project_id"]:
        raise ValueError("Double annotation requires two distinct projects")
    if set(project_a["records"]) != set(project_b["records"]):
        raise ValueError("Double-annotation project task universes differ")
    annotator_a = project_a["annotator"]["completed_by_id"]
    annotator_b = project_b["annotator"]["completed_by_id"]
    if annotator_a is not None and annotator_a == annotator_b:
        raise ValueError("Double annotation requires two distinct annotators")


def unblind_preference_rows(
    projects: Sequence[dict[str, Any]],
    private_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotator_index, project in enumerate(projects):
        if set(project["records"]) != set(private_key):
            raise ValueError(
                f"Preference project {project['project_id']} differs from private key"
            )
        for blind_id in sorted(private_key):
            key = private_key[blind_id]
            annotation = project["records"][blind_id]
            evi_side = str(key["method_to_side"]["evitrace"]).upper()
            side_score = PREFERENCE_SIDE_SCORE[annotation["overall_preference"]]
            score = side_score if evi_side == "A" else -side_score
            outcome = "tie"
            if score > 0:
                outcome = "evitrace"
            elif score < 0:
                outcome = "control"
            token_difference = (
                key["evi_token_count"] - key["control_token_count"]
            )
            rows.append(
                {
                    "task_type": "preference",
                    "blind_task_id": blind_id,
                    "comparison_type": key["comparison_type"],
                    "event_id": str(key["event_id"]),
                    "stratum": key["stratum"],
                    "gold_label": key.get("gold_label"),
                    "complexity": key.get("complexity"),
                    "annotator_index": annotator_index,
                    "project_id": project["project_id"],
                    "annotator_email": annotation["annotator_email"],
                    "completed_by_id": annotation["completed_by_id"],
                    "annotation_id": annotation["annotation_id"],
                    "annotation_created_at": annotation["created_at"],
                    "annotation_updated_at": annotation["updated_at"],
                    "overall_preference": annotation["overall_preference"],
                    "data_issue": annotation["data_issue"],
                    "notes": annotation["notes"],
                    "evitrace_side": evi_side,
                    "control_side": str(
                        key["method_to_side"]["control"]
                    ).upper(),
                    "evitrace_score": score,
                    "collapsed_outcome": outcome,
                    "k_visible": key.get("k_visible"),
                    "evi_candidate_uids": key.get("evi_candidate_uids"),
                    "control_candidate_uids": key.get("control_candidate_uids"),
                    "evi_token_count": key["evi_token_count"],
                    "control_token_count": key["control_token_count"],
                    "token_count_difference_evi_minus_control": token_difference,
                    "same_evidence_set": key.get("same_evidence_set"),
                    "artifact_sha256": key.get("artifact_sha256"),
                }
            )
    return rows


def unblind_transition_rows(
    projects: Sequence[dict[str, Any]],
    private_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotator_index, project in enumerate(projects):
        if set(project["records"]) != set(private_key):
            raise ValueError(
                f"Transition project {project['project_id']} differs from private key"
            )
        for blind_id in sorted(private_key):
            key = private_key[blind_id]
            annotation = project["records"][blind_id]
            rows.append(
                {
                    "task_type": "transition",
                    "blind_task_id": blind_id,
                    "comparison_type": "transition",
                    "event_id": str(key["event_id"]),
                    "gold_label": key.get("gold_label"),
                    "complexity": key.get("complexity"),
                    "annotator_index": annotator_index,
                    "project_id": project["project_id"],
                    "annotator_email": annotation["annotator_email"],
                    "completed_by_id": annotation["completed_by_id"],
                    "annotation_id": annotation["annotation_id"],
                    "annotation_created_at": annotation["created_at"],
                    "annotation_updated_at": annotation["updated_at"],
                    "transition_validity": annotation["transition_validity"],
                    "marginal_contribution": annotation[
                        "marginal_contribution"
                    ],
                    "operation": key["operation"],
                    "atom_id": key["atom_id"],
                    "state_before": key["state_before"],
                    "state_after": key["state_after"],
                    "transition_kind": key["transition_kind"],
                    "step": key.get("step"),
                    "candidate_uid": key["candidate_uid"],
                    "artifact_sha256": key.get("artifact_sha256"),
                }
            )
    return rows


def cohen_kappa(
    a_values: Sequence[Any],
    b_values: Sequence[Any],
    categories: Sequence[Any],
) -> float | None:
    if len(a_values) != len(b_values) or not a_values:
        return None
    n = len(a_values)
    observed = sum(a == b for a, b in zip(a_values, b_values)) / n
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    expected = sum(
        (a_counts[category] / n) * (b_counts[category] / n)
        for category in categories
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def linear_weighted_kappa(
    a_values: Sequence[Any],
    b_values: Sequence[Any],
    categories: Sequence[Any],
) -> float | None:
    if len(a_values) != len(b_values) or not a_values:
        return None
    n = len(a_values)
    positions = {value: index for index, value in enumerate(categories)}
    scale = max(len(categories) - 1, 1)

    def weight(left: Any, right: Any) -> float:
        return 1.0 - abs(positions[left] - positions[right]) / scale

    observed = sum(weight(a, b) for a, b in zip(a_values, b_values)) / n
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    expected = sum(
        weight(a, b) * (a_counts[a] / n) * (b_counts[b] / n)
        for a in categories
        for b in categories
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def agreement_summary(
    a_values: Sequence[Any],
    b_values: Sequence[Any],
    categories: Sequence[Any],
    *,
    ordinal: bool,
) -> dict[str, Any]:
    if len(a_values) != len(b_values):
        raise ValueError("Agreement arrays differ in length")
    exact_count = sum(a == b for a, b in zip(a_values, b_values))
    result = {
        "n": len(a_values),
        "exact_agreement_count": exact_count,
        "exact_agreement": exact_count / len(a_values) if a_values else None,
        "cohen_kappa": cohen_kappa(a_values, b_values, categories),
        "annotator_a_distribution": dict(Counter(map(str, a_values))),
        "annotator_b_distribution": dict(Counter(map(str, b_values))),
    }
    if ordinal:
        result["linear_weighted_cohen_kappa"] = linear_weighted_kappa(
            a_values, b_values, categories
        )
    return result


def preference_agreement(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["blind_task_id"]].append(row)
    a_values: list[int] = []
    b_values: list[int] = []
    for blind_id in sorted(by_task):
        pair = sorted(by_task[blind_id], key=lambda row: row["annotator_index"])
        if len(pair) != 2 or [row["annotator_index"] for row in pair] != [0, 1]:
            raise ValueError(f"Preference task is not double annotated: {blind_id}")
        a_values.append(pair[0]["evitrace_score"])
        b_values.append(pair[1]["evitrace_score"])
    collapsed_a = [1 if value > 0 else -1 if value < 0 else 0 for value in a_values]
    collapsed_b = [1 if value > 0 else -1 if value < 0 else 0 for value in b_values]
    return {
        "five_level": agreement_summary(
            a_values, b_values, (-2, -1, 0, 1, 2), ordinal=True
        ),
        "collapsed_evitrace_tie_control": agreement_summary(
            collapsed_a, collapsed_b, (-1, 0, 1), ordinal=False
        ),
    }


def _basic_preference_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    outcomes = Counter(row["collapsed_outcome"] for row in rows)
    n = len(rows)
    non_tie = outcomes["evitrace"] + outcomes["control"]
    scores = [row["evitrace_score"] for row in rows]
    return {
        "claim_count": len({row["event_id"] for row in rows}),
        "annotation_count": n,
        "evitrace_win_count": outcomes["evitrace"],
        "control_win_count": outcomes["control"],
        "tie_count": outcomes["tie"],
        "evitrace_win_rate": outcomes["evitrace"] / n if n else None,
        "control_win_rate": outcomes["control"] / n if n else None,
        "tie_rate": outcomes["tie"] / n if n else None,
        "non_tie_count": non_tie,
        "conditional_evitrace_win_rate": (
            outcomes["evitrace"] / non_tie if non_tie else None
        ),
        "mean_evitrace_score": sum(scores) / n if n else None,
        "score_distribution": {
            str(score): sum(value == score for value in scores)
            for score in (-2, -1, 0, 1, 2)
        },
    }


def _sampling_block(manifest: dict[str, Any], comparison_type: str) -> Any:
    sampling = manifest.get("sampling", {})
    if not isinstance(sampling, dict):
        return None
    if isinstance(sampling.get("formal"), dict):
        sampling = sampling["formal"]
    aliases = {
        "main": ("main", "preference_main"),
        "order_only": ("order_only", "order", "preference_order"),
        "transition": ("transition",),
    }[comparison_type]
    for alias in aliases:
        if alias in sampling:
            return sampling[alias]
    return None


def design_weights(
    manifest: dict[str, Any],
    comparison_type: str,
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], str]:
    observed = sorted({str(row["stratum"]) for row in rows})
    block = _sampling_block(manifest, comparison_type)
    strata: Any = block.get("strata") if isinstance(block, dict) else None
    if strata is None and comparison_type == "order_only" and isinstance(block, dict):
        allocations = block.get("allocations")
        if isinstance(allocations, list):
            strata = [
                {
                    **entry,
                    "stratum": f"{entry.get('complexity')}|{entry.get('label')}",
                }
                for entry in allocations
                if isinstance(entry, dict)
            ]
    entries: list[dict[str, Any]] = []
    if isinstance(strata, list):
        entries = [entry for entry in strata if isinstance(entry, dict)]
    elif isinstance(strata, dict):
        for name, value in strata.items():
            if isinstance(value, dict):
                entries.append({"stratum": name, **value})
    weights: dict[str, float] = {}
    pool_sizes: dict[str, float] = {}
    for entry in entries:
        name = entry.get("stratum") or entry.get("name")
        if name is None:
            continue
        if isinstance(entry.get("design_weight"), (int, float)):
            weights[str(name)] = float(entry["design_weight"])
        if isinstance(entry.get("pool_size"), (int, float)):
            pool_sizes[str(name)] = float(entry["pool_size"])
    source = "manifest_design_weight"
    if not weights and pool_sizes:
        total = sum(pool_sizes.values())
        if total > 0:
            weights = {name: value / total for name, value in pool_sizes.items()}
            source = "manifest_pool_size"
    if set(weights) != set(observed) or any(value < 0 for value in weights.values()):
        counts = Counter(str(row["stratum"]) for row in rows)
        total = sum(counts.values())
        weights = {name: counts[name] / total for name in observed} if total else {}
        source = "sample_proportion_fallback"
    total_weight = sum(weights.values())
    if total_weight:
        weights = {name: value / total_weight for name, value in weights.items()}
    return weights, source


def _weighted_preference_summary(
    rows: Sequence[dict[str, Any]], weights: dict[str, float]
) -> dict[str, Any]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["stratum"])].append(row)
    summaries = {
        name: _basic_preference_summary(stratum_rows)
        for name, stratum_rows in by_stratum.items()
    }

    def weighted(field: str) -> float | None:
        values = [
            (weights[name], summary[field])
            for name, summary in summaries.items()
            if name in weights and summary[field] is not None
        ]
        denominator = sum(weight for weight, _value in values)
        return (
            sum(weight * value for weight, value in values) / denominator
            if denominator
            else None
        )

    win = weighted("evitrace_win_rate")
    control = weighted("control_win_rate")
    tie = weighted("tie_rate")
    non_tie = (win or 0.0) + (control or 0.0)
    annotation_count = len(rows)
    return {
        "evitrace_win_rate": win,
        "control_win_rate": control,
        "tie_rate": tie,
        "standardized_counts_at_observed_annotation_n": {
            "annotation_n": annotation_count,
            "evitrace_win": (
                win * annotation_count if win is not None else None
            ),
            "control_win": (
                control * annotation_count if control is not None else None
            ),
            "tie": tie * annotation_count if tie is not None else None,
        },
        "conditional_evitrace_win_rate": win / non_tie if non_tie else None,
        "mean_evitrace_score": weighted("mean_evitrace_score"),
    }


def percentile(values: Sequence[float], probability: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def stratified_claim_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    stratum_field: str,
    metric_fn: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    metric_names: Sequence[str],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if reps <= 0 or not rows:
        return {}
    clusters: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    claim_strata: dict[str, str] = {}
    for row in rows:
        claim = str(row["event_id"])
        stratum = str(row[stratum_field])
        previous = claim_strata.setdefault(claim, stratum)
        if previous != stratum:
            raise ValueError(f"Claim {claim} spans multiple bootstrap strata")
        clusters[stratum][claim].append(row)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _replicate in range(reps):
        sampled: list[dict[str, Any]] = []
        for stratum in sorted(clusters):
            claim_rows = clusters[stratum]
            claim_ids = sorted(claim_rows)
            for _draw in range(len(claim_ids)):
                sampled.extend(claim_rows[rng.choice(claim_ids)])
        result = metric_fn(sampled)
        for name in metric_names:
            value = result.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                samples[name].append(float(value))
    return {
        name: {
            "low": percentile(values, 0.025),
            "high": percentile(values, 0.975),
            "valid_replicates": len(values),
        }
        for name, values in samples.items()
    }


def preference_summary(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    comparison_type: str,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    weights, weight_source = design_weights(manifest, comparison_type, rows)
    raw = _basic_preference_summary(rows)
    weighted = _weighted_preference_summary(rows, weights)
    strata = {
        name: _basic_preference_summary(
            [row for row in rows if str(row["stratum"]) == name]
        )
        for name in sorted(weights)
    }

    def metric(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _weighted_preference_summary(sample, weights)

    ci = stratified_claim_bootstrap(
        rows,
        stratum_field="stratum",
        metric_fn=metric,
        metric_names=(
            "evitrace_win_rate",
            "control_win_rate",
            "tie_rate",
            "conditional_evitrace_win_rate",
            "mean_evitrace_score",
        ),
        reps=bootstrap_reps,
        seed=seed,
    )
    return {
        "raw": raw,
        "design_weighted": weighted,
        "design_weights": weights,
        "design_weight_source": weight_source,
        "strata": strata,
        "stratified_claim_cluster_bootstrap_ci95": ci,
        "agreement": preference_agreement(rows),
        "data_issue_annotation_count": sum(bool(row["data_issue"]) for row in rows),
        "data_issue_distribution": dict(
            Counter(issue for row in rows for issue in row["data_issue"])
        ),
        "data_issue_exclusion_applied": False,
    }


def claim_label_swap_randomization(
    rows: Sequence[dict[str, Any]],
    weights: dict[str, float],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    observed = _weighted_preference_summary(rows, weights)["mean_evitrace_score"]
    if observed is None:
        return {}
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_claim[str(row["event_id"])].append(row)
    claim_ids = sorted(by_claim)

    def statistic(signs: Sequence[int]) -> float:
        swapped: list[dict[str, Any]] = []
        for claim, sign in zip(claim_ids, signs):
            for original in by_claim[claim]:
                score = original["evitrace_score"] * sign
                swapped.append(
                    {
                        **original,
                        "evitrace_score": score,
                        "collapsed_outcome": (
                            "evitrace"
                            if score > 0
                            else "control"
                            if score < 0
                            else "tie"
                        ),
                    }
                )
        value = _weighted_preference_summary(swapped, weights)[
            "mean_evitrace_score"
        ]
        assert value is not None
        return value

    if len(claim_ids) <= 20 and 2 ** len(claim_ids) <= max(reps, 1):
        total = 2 ** len(claim_ids)
        extreme = 0
        for mask in range(total):
            signs = [
                -1 if mask & (1 << index) else 1
                for index in range(len(claim_ids))
            ]
            extreme += abs(statistic(signs)) >= abs(observed) - 1e-12
        p_value = extreme / total
        actual_reps = total
        mode = "exact"
    else:
        rng = random.Random(seed)
        extreme = 0
        for _replicate in range(reps):
            signs = [rng.choice((-1, 1)) for _claim in claim_ids]
            extreme += abs(statistic(signs)) >= abs(observed) - 1e-12
        p_value = (extreme + 1) / (reps + 1) if reps else None
        actual_reps = reps
        mode = "monte_carlo"
    return {
        "unit": "claim",
        "statistic": "design_weighted_mean_evitrace_score",
        "observed": observed,
        "alternative": "two-sided",
        "p_value": p_value,
        "mode": mode,
        "replicates": actual_reps,
        "seed": seed,
    }


def _matrix_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be non-empty and square")
    augmented = [
        [float(value) for value in row]
        + [1.0 if row_index == column else 0.0 for column in range(n)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-11:
            raise ValueError("Singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[n:] for row in augmented]


def _matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("Invalid matrix multiplication dimensions")
    return [
        [
            sum(left_value * right_value for left_value, right_value in zip(row, column))
            for column in zip(*right)
        ]
        for row in left
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def logistic_token_sensitivity(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    clean = [row for row in rows if row["evitrace_score"] != 0]
    if not clean:
        return {
            "status": "not_estimable_no_non_ties",
            "n": 0,
            "ties_excluded": len(rows),
        }
    y = [1.0 if row["evitrace_score"] > 0 else 0.0 for row in clean]
    if len(set(y)) < 2:
        return {
            "status": "not_reported_complete_separation",
            "reason": "The non-tie outcome has only one observed class.",
            "n": len(clean),
            "ties_excluded": len(rows) - len(clean),
        }
    annotators = sorted({str(row["annotator_index"]) for row in clean})
    reference = annotators[0]
    dummy_annotators = annotators[1:]
    names = [
        "intercept",
        "token_difference_evi_minus_control_per_64",
        *[f"annotator_{annotator}_fixed_effect" for annotator in dummy_annotators],
    ]
    x = [
        [
            1.0,
            row["token_count_difference_evi_minus_control"] / 64.0,
            *[
                1.0 if str(row["annotator_index"]) == annotator else 0.0
                for annotator in dummy_annotators
            ],
        ]
        for row in clean
    ]
    p = len(names)
    beta = [0.0] * p
    converged = False
    inverse_hessian: list[list[float]] | None = None
    probabilities: list[float] = []
    try:
        for _iteration in range(100):
            probabilities = [
                _sigmoid(sum(coefficient * value for coefficient, value in zip(beta, row)))
                for row in x
            ]
            hessian = [[0.0] * p for _ in range(p)]
            score = [0.0] * p
            for row, outcome, probability in zip(x, y, probabilities):
                weight = max(probability * (1.0 - probability), 1e-12)
                for left in range(p):
                    score[left] += row[left] * (outcome - probability)
                    for right in range(p):
                        hessian[left][right] += (
                            row[left] * weight * row[right]
                        )
            inverse_hessian = _matrix_inverse(hessian)
            delta = [
                sum(inverse_hessian[row][column] * score[column] for column in range(p))
                for row in range(p)
            ]
            beta = [value + change for value, change in zip(beta, delta)]
            if max(abs(change) for change in delta) < 1e-8:
                converged = True
                break
            if max(abs(value) for value in beta) > 30:
                break
    except ValueError:
        return {
            "status": "not_estimable_rank_deficient",
            "reason": "The requested token-difference plus annotator-fixed-effect model is singular.",
            "n": len(clean),
            "ties_excluded": len(rows) - len(clean),
        }
    probabilities = [
        _sigmoid(sum(coefficient * value for coefficient, value in zip(beta, row)))
        for row in x
    ]
    if (
        not converged
        or max(abs(value) for value in beta) > 25
        or all(
            probability < 1e-8 if outcome == 0 else probability > 1 - 1e-8
            for probability, outcome in zip(probabilities, y)
        )
    ):
        return {
            "status": "not_reported_complete_separation",
            "reason": "Unpenalized maximum likelihood diverged under complete or quasi-complete separation.",
            "n": len(clean),
            "ties_excluded": len(rows) - len(clean),
        }
    assert inverse_hessian is not None

    cluster_scores: dict[str, list[float]] = defaultdict(lambda: [0.0] * p)
    for row, design, outcome, probability in zip(clean, x, y, probabilities):
        cluster = str(row["event_id"])
        for index in range(p):
            cluster_scores[cluster][index] += design[index] * (outcome - probability)
    meat = [[0.0] * p for _ in range(p)]
    for score in cluster_scores.values():
        for left in range(p):
            for right in range(p):
                meat[left][right] += score[left] * score[right]
    covariance = _matmul(_matmul(inverse_hessian, meat), inverse_hessian)
    cluster_count = len(cluster_scores)
    n = len(clean)
    if cluster_count > 1 and n > p:
        correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - p))
        covariance = [
            [value * correction for value in row] for row in covariance
        ]
    standard_errors = [
        math.sqrt(max(covariance[index][index], 0.0)) for index in range(p)
    ]
    coefficients: dict[str, dict[str, Any]] = {}
    for name, estimate, standard_error in zip(names, beta, standard_errors):
        low = estimate - 1.96 * standard_error
        high = estimate + 1.96 * standard_error
        coefficients[name] = {
            "log_odds": estimate,
            "cluster_robust_se": standard_error,
            "odds_ratio": math.exp(min(estimate, 700.0)),
            "odds_ratio_ci95": [
                math.exp(max(low, -700.0)),
                math.exp(min(high, 700.0)),
            ],
        }
    return {
        "status": "estimated",
        "outcome": "evitrace_win_among_non_ties",
        "token_scale": "64 tokenizer tokens",
        "annotator_reference": reference,
        "n": n,
        "claim_cluster_count": cluster_count,
        "ties_excluded": len(rows) - len(clean),
        "coefficients": coefficients,
    }


def token_robustness_summary(
    main_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    differences_by_claim: dict[str, int] = {}
    for row in main_rows:
        claim = str(row["event_id"])
        difference = row["token_count_difference_evi_minus_control"]
        previous = differences_by_claim.setdefault(claim, difference)
        if previous != difference:
            raise ValueError(f"Token difference varies within claim {claim}")
    subset = [
        row
        for row in main_rows
        if abs(row["token_count_difference_evi_minus_control"]) <= 64
    ]
    differences = list(differences_by_claim.values())
    return {
        "full_sample": {
            "claim_count": len(differences),
            "mean_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "mean_absolute_difference": (
                sum(abs(value) for value in differences) / len(differences)
                if differences
                else None
            ),
            "minimum": min(differences) if differences else None,
            "maximum": max(differences) if differences else None,
            "differences": dict(Counter(map(str, differences))),
        },
        "pre_registered_absolute_difference_le_64": preference_summary(
            subset,
            manifest,
            "main",
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        ),
        "secondary_logistic_sensitivity": logistic_token_sensitivity(main_rows),
    }


def _paired_values(
    rows: Sequence[dict[str, Any]], field: str
) -> tuple[list[Any], list[Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["blind_task_id"]].append(row)
    a_values: list[Any] = []
    b_values: list[Any] = []
    for blind_id in sorted(by_task):
        pair = sorted(by_task[blind_id], key=lambda row: row["annotator_index"])
        if len(pair) != 2 or [row["annotator_index"] for row in pair] != [0, 1]:
            raise ValueError(f"Transition task is not double annotated: {blind_id}")
        a_values.append(pair[0][field])
        b_values.append(pair[1][field])
    return a_values, b_values


def _basic_transition_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    validity = Counter(row["transition_validity"] for row in rows)
    contribution = Counter(row["marginal_contribution"] for row in rows)
    n = len(rows)
    return {
        "claim_count": len({row["event_id"] for row in rows}),
        "annotation_count": n,
        "validity_distribution": {
            value: validity[value] for value in TRANSITION_VALIDITY
        },
        "valid_rate": validity["valid"] / n if n else None,
        "valid_or_partial_rate": (
            (validity["valid"] + validity["partially_valid"]) / n if n else None
        ),
        "marginal_contribution_distribution": {
            value: contribution[value] for value in MARGINAL_CONTRIBUTION
        },
        "clear_contribution_rate": contribution["clear"] / n if n else None,
        "clear_or_limited_contribution_rate": (
            (contribution["clear"] + contribution["limited"]) / n if n else None
        ),
    }


def transition_summary(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    def metric(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _basic_transition_summary(sample)

    overall = _basic_transition_summary(rows)
    overall_ci = stratified_claim_bootstrap(
        rows,
        stratum_field="operation",
        metric_fn=metric,
        metric_names=(
            "valid_rate",
            "valid_or_partial_rate",
            "clear_contribution_rate",
            "clear_or_limited_contribution_rate",
        ),
        reps=bootstrap_reps,
        seed=seed,
    )
    by_kind: dict[str, Any] = {}
    for offset, kind in enumerate(("change", "self")):
        subset = [row for row in rows if row["transition_kind"] == kind]
        summary = _basic_transition_summary(subset)
        summary["operation_stratified_claim_cluster_bootstrap_ci95"] = (
            stratified_claim_bootstrap(
                subset,
                stratum_field="operation",
                metric_fn=metric,
                metric_names=(
                    "valid_rate",
                    "valid_or_partial_rate",
                    "clear_contribution_rate",
                ),
                reps=bootstrap_reps,
                seed=seed + 1 + offset,
            )
        )
        by_kind[kind] = summary
    by_operation = {
        operation: _basic_transition_summary(
            [row for row in rows if row["operation"] == operation]
        )
        for operation in OPERATIONS
    }
    validity_a, validity_b = _paired_values(rows, "transition_validity")
    contribution_a, contribution_b = _paired_values(
        rows, "marginal_contribution"
    )
    lower = (
        overall_ci.get("valid_rate", {}).get("low")
        if overall_ci
        else None
    )
    if lower is None:
        gate = "uncertain"
    elif lower > 0.5:
        gate = "human_aligned"
    else:
        gate = "mixed_or_uncertain"
    return {
        "overall_balanced_sample": overall,
        "operation_stratified_claim_cluster_bootstrap_ci95": overall_ci,
        "change_step_validity": by_kind["change"],
        "self_transition_appropriateness": by_kind["self"],
        "marginal_contribution": {
            "overall": {
                key: value
                for key, value in overall.items()
                if "contribution" in key
            },
            "by_operation": {
                operation: {
                    key: value
                    for key, value in summary.items()
                    if "contribution" in key
                }
                for operation, summary in by_operation.items()
            },
        },
        "by_operation": by_operation,
        "operation_sample_is_deliberately_balanced": True,
        "natural_operation_distribution_claimed": False,
        "agreement": {
            "transition_validity": agreement_summary(
                validity_a,
                validity_b,
                TRANSITION_VALIDITY,
                ordinal=True,
            ),
            "marginal_contribution": agreement_summary(
                contribution_a,
                contribution_b,
                MARGINAL_CONTRIBUTION,
                ordinal=True,
            ),
        },
        "human_aligned_wording_gate": {
            "criterion": "strict valid-rate claim-cluster bootstrap 95% CI lower bound > 0.5",
            "valid_rate_ci95_lower": lower,
            "result": gate,
        },
    }


def _format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def _format_number(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _ci_text(interval: dict[str, Any] | None) -> str:
    if not interval or interval.get("low") is None or interval.get("high") is None:
        return "NA"
    return f"[{interval['low']:.1%}, {interval['high']:.1%}]"


def locked_interpretation(
    main: dict[str, Any],
    order: dict[str, Any],
    transition: dict[str, Any],
) -> dict[str, Any]:
    main_p = main["claim_level_label_swap_randomization"]["p_value"]
    order_p = order["claim_level_label_swap_randomization"]["p_value"]
    main_weighted = main["design_weighted"]
    order_weighted = order["design_weighted"]
    main_positive = (
        main_p is not None
        and main_p < 0.05
        and main_weighted["evitrace_win_rate"]
        > main_weighted["control_win_rate"]
    )
    order_positive = (
        order_p is not None
        and order_p < 0.05
        and order_weighted["evitrace_win_rate"]
        > order_weighted["control_win_rate"]
    )
    if main_positive and order_positive:
        preference_wording = "improves decision-oriented ordering"
    elif main_positive:
        preference_wording = "improves evidence selection and overall organization"
    else:
        preference_wording = "exploratory preference evidence is mixed or uncertain"
    return {
        "main_positive_at_two_sided_0_05": main_positive,
        "order_only_positive_at_two_sided_0_05": order_positive,
        "allowed_preference_wording": preference_wording,
        "transition_wording": transition["human_aligned_wording_gate"]["result"],
        "prohibited_claims": [
            "improved human fact-checking accuracy",
            "causal explanation of verifier behavior",
            "latent chain-of-thought alignment",
        ],
        "reasoning_trace_scope": (
            "observable evidence selection and atom-state transitions, "
            "not latent chain-of-thought"
        ),
    }


def report_text(metrics: dict[str, Any]) -> str:
    main = metrics["preference"]["main"]
    order = metrics["preference"]["order_only"]
    transition = metrics["transition"]
    main_weighted = main["design_weighted"]
    order_weighted = order["design_weighted"]
    main_ci = main["stratified_claim_cluster_bootstrap_ci95"]
    order_ci = order["stratified_claim_cluster_bootstrap_ci95"]
    valid = transition["overall_balanced_sample"]
    valid_ci = transition[
        "operation_stratified_claim_cluster_bootstrap_ci95"
    ].get("valid_rate")
    completion = metrics["completion_contract"]
    return "\n".join(
        [
            "# EviTrace small-scale double-annotation analysis",
            "",
            "## Snapshot and completion",
            "",
            f"- SQLite quick_check: `{metrics['snapshot']['sqlite_quick_check']}`.",
            f"- Formal semantic tasks: {completion['formal_claim_counts']['main']} main, "
            f"{completion['formal_claim_counts']['order_only']} order-only, "
            f"{completion['formal_claim_counts']['transition']} transition.",
            f"- Raw retained annotations: {completion['raw_annotation_count']}; "
            f"redundant active completions: {completion['redundant_completion_count']}.",
            f"- Completion contract satisfied: `{completion['satisfied']}`. "
            "A complete manifest additionally requires exact task counts, two distinct "
            "system-blind annotators, frozen task/key hashes, and no redundant completion.",
            "",
            "## Main matched-count preference",
            "",
            f"- Design-weighted EviTrace/control/tie rates: "
            f"{_format_rate(main_weighted['evitrace_win_rate'])} / "
            f"{_format_rate(main_weighted['control_win_rate'])} / "
            f"{_format_rate(main_weighted['tie_rate'])}.",
            f"- Conditional EviTrace win rate: "
            f"{_format_rate(main_weighted['conditional_evitrace_win_rate'])} "
            f"(claim-clustered 95% CI "
            f"{_ci_text(main_ci.get('conditional_evitrace_win_rate'))}).",
            f"- Claim-level label-swap randomization: "
            f"p={_format_number(main['claim_level_label_swap_randomization']['p_value'], 4)}.",
            f"- Five-level exact agreement: "
            f"{_format_rate(main['agreement']['five_level']['exact_agreement'])}; "
            f"linear-weighted Cohen kappa="
            f"{_format_number(main['agreement']['five_level']['linear_weighted_cohen_kappa'])}.",
            "",
            "The main comparison matches visible evidence count, not tokenizer length. "
            "The complete token difference distribution, the pre-registered "
            "|T_Evi-T_S4|<=64 subset, and the secondary clustered logistic sensitivity "
            "are retained in metrics.json.",
            "",
            "## Same-set order-only preference",
            "",
            f"- EviTrace-order/control-order/tie rates: "
            f"{_format_rate(order_weighted['evitrace_win_rate'])} / "
            f"{_format_rate(order_weighted['control_win_rate'])} / "
            f"{_format_rate(order_weighted['tie_rate'])}.",
            f"- Conditional EviTrace-order win rate: "
            f"{_format_rate(order_weighted['conditional_evitrace_win_rate'])} "
            f"(95% CI {_ci_text(order_ci.get('conditional_evitrace_win_rate'))}); "
            f"label-swap p="
            f"{_format_number(order['claim_level_label_swap_randomization']['p_value'], 4)}.",
            "- This is the only comparison in which text and tokenizer length are "
            "identical, so it is the only direct ordering contrast.",
            "",
            "## Transition audit",
            "",
            f"- Strict valid rate in the deliberately balanced sample: "
            f"{_format_rate(valid['valid_rate'])} (operation-stratified "
            f"claim-clustered 95% CI {_ci_text(valid_ci)}).",
            f"- Human-aligned wording gate: "
            f"`{transition['human_aligned_wording_gate']['result']}`.",
            "- Change-step validity, self-transition appropriateness, marginal "
            "contribution, and operation-specific results are reported separately. "
            "Operation-balanced frequencies are not estimates of the natural trace mix.",
            "",
            "## Locked interpretation",
            "",
            f"- Allowed preference wording: "
            f"“{metrics['locked_interpretation']['allowed_preference_wording']}”.",
            f"- Transition wording: "
            f"`{metrics['locked_interpretation']['transition_wording']}`.",
            "- These judgments concern observable evidence organization and "
            "atom-state transitions. They do not establish human fact-checking "
            "accuracy gains, verifier causality, or latent chain-of-thought alignment.",
            "",
        ]
    )


def paper_insert_text(metrics: dict[str, Any]) -> str:
    main = metrics["preference"]["main"]
    order = metrics["preference"]["order_only"]
    transition = metrics["transition"]
    rows = []
    for label, result in (("Main", main), ("Order-only", order)):
        weighted = result["design_weighted"]
        ci = result["stratified_claim_cluster_bootstrap_ci95"].get(
            "conditional_evitrace_win_rate"
        )
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(result["raw"]["claim_count"]),
                    _format_rate(weighted["evitrace_win_rate"]),
                    _format_rate(weighted["control_win_rate"]),
                    _format_rate(weighted["tie_rate"]),
                    _format_rate(weighted["conditional_evitrace_win_rate"]),
                    _ci_text(ci),
                    _format_number(
                        result["claim_level_label_swap_randomization"]["p_value"],
                        4,
                    ),
                )
            )
            + " |"
        )
    valid_ci = transition[
        "operation_stratified_claim_cluster_bootstrap_ci95"
    ].get("valid_rate")
    return "\n".join(
        [
            "### Trace-quality human evaluation (paper insert)",
            "",
            "| Comparison | Claims | Evi win | Control win | Tie | "
            "Conditional Evi win | 95% CI | Swap p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Across the deliberately operation-balanced transition audit, the strict "
            f"`valid` rate was "
            f"{_format_rate(transition['overall_balanced_sample']['valid_rate'])} "
            f"(operation-stratified claim-clustered 95% CI {_ci_text(valid_ci)}). "
            f"The pre-specified wording gate is "
            f"`{transition['human_aligned_wording_gate']['result']}`.",
            "",
            f"Locked interpretation: "
            f"{metrics['locked_interpretation']['allowed_preference_wording']}. "
            "The experiment evaluates observable evidence selection, ordering, and "
            "atom-state transitions; it does not measure human fact-checking accuracy "
            "or reveal latent chain-of-thought.",
            "",
        ]
    )


def paper_table_tex(metrics: dict[str, Any]) -> str:
    main = metrics["preference"]["main"]
    order = metrics["preference"]["order_only"]
    transition = metrics["transition"]

    def percent(value: float | None) -> str:
        return "--" if value is None else f"{100.0 * value:.1f}\\%"

    def interval(result: dict[str, Any], field: str) -> str:
        ci = result.get(field, {})
        if ci.get("low") is None or ci.get("high") is None:
            return "--"
        return (
            f"[{100.0 * ci['low']:.1f}, "
            f"{100.0 * ci['high']:.1f}]"
        )

    table_rows: list[str] = []
    for name, result in (("Main", main), ("Order-only", order)):
        weighted = result["design_weighted"]
        ci = result["stratified_claim_cluster_bootstrap_ci95"]
        p_value = result["claim_level_label_swap_randomization"]["p_value"]
        table_rows.append(
            f"{name} & {result['raw']['claim_count']} & "
            f"{percent(weighted['evitrace_win_rate'])} / "
            f"{percent(weighted['control_win_rate'])} / "
            f"{percent(weighted['tie_rate'])} & "
            f"{percent(weighted['conditional_evitrace_win_rate'])} & "
            f"{interval(ci, 'conditional_evitrace_win_rate')} & "
            f"{'--' if p_value is None else f'{p_value:.4f}'} \\\\"
        )
    transition_ci = transition[
        "operation_stratified_claim_cluster_bootstrap_ci95"
    ]
    transition_pooled = transition["overall_balanced_sample"]
    table_rows.append(
        f"Transition valid & {transition_pooled['claim_count']} & -- & "
        f"{percent(transition_pooled['valid_rate'])} & "
        f"{interval(transition_ci, 'valid_rate')} & -- \\\\"
    )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Comparison & $n$ & Evi / Ctrl / Tie & Cond./Valid & 95\% CI & $p$ \\",
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Exploratory, randomized, system-blind, double-annotated "
            r"EviTrace evaluation. Preference rates are design-weighted; confidence "
            r"intervals use stratified claim-clustered bootstrap. The order-only row "
            r"holds evidence text and length fixed. Transition operations were "
            r"deliberately balanced and are not a natural-frequency estimate.}",
            r"\label{tab:evitrace_human_alignment}",
            r"\end{table}",
            "",
        ]
    )


def _project_public_summary(project: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in project.items()
        if key not in {"records", "duplicate_issues"}
    }


def _side_balance(
    preference_key: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for comparison_type in ("main", "order_only"):
        result[comparison_type] = dict(
            Counter(
                str(row["method_to_side"]["evitrace"]).upper()
                for row in preference_key.values()
                if row["comparison_type"] == comparison_type
            )
        )
    return result


def analyze(
    db_path: Path,
    task_manifest_path: Path,
    preference_project_ids: Sequence[int],
    transition_project_ids: Sequence[int],
    output_dir: Path,
    *,
    preference_tasks_path: Path | None = None,
    transition_tasks_path: Path | None = None,
    preference_key_path: Path | None = None,
    transition_key_path: Path | None = None,
    bootstrap_reps: int = 10_000,
    randomization_reps: int = 10_000,
    seed: int = 20_260_724,
) -> dict[str, Any]:
    if len(preference_project_ids) != 2 or len(transition_project_ids) != 2:
        raise ValueError("Exactly two preference and two transition project IDs are required")
    if bootstrap_reps < 0 or randomization_reps < 0:
        raise ValueError("Resampling replicate counts cannot be negative")
    bundle = load_frozen_bundle(
        task_manifest_path,
        preference_tasks_path=preference_tasks_path,
        transition_tasks_path=transition_tasks_path,
        preference_key_path=preference_key_path,
        transition_key_path=transition_key_path,
    )

    uri = f"file:{db_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise ValueError(f"SQLite quick_check failed: {quick_check}")
        preference_projects = [
            load_project_annotations(
                connection,
                project_id,
                bundle["preference_public"],
                "preference",
            )
            for project_id in preference_project_ids
        ]
        transition_projects = [
            load_project_annotations(
                connection,
                project_id,
                bundle["transition_public"],
                "transition",
            )
            for project_id in transition_project_ids
        ]
    finally:
        connection.close()
    validate_project_pair(*preference_projects)
    validate_project_pair(*transition_projects)
    for index in range(2):
        preference_annotator = preference_projects[index]["annotator"][
            "completed_by_id"
        ]
        transition_annotator = transition_projects[index]["annotator"][
            "completed_by_id"
        ]
        if (
            preference_annotator is not None
            and transition_annotator is not None
            and preference_annotator != transition_annotator
        ):
            raise ValueError(
                "Preference and transition project ordering does not preserve annotators"
            )

    preference_rows = unblind_preference_rows(
        preference_projects, bundle["preference_key"]
    )
    transition_rows = unblind_transition_rows(
        transition_projects, bundle["transition_key"]
    )
    main_rows = [
        row for row in preference_rows if row["comparison_type"] == "main"
    ]
    order_rows = [
        row for row in preference_rows if row["comparison_type"] == "order_only"
    ]
    main = preference_summary(
        main_rows,
        bundle["manifest"],
        "main",
        bootstrap_reps=bootstrap_reps,
        seed=seed + 11,
    )
    order = preference_summary(
        order_rows,
        bundle["manifest"],
        "order_only",
        bootstrap_reps=bootstrap_reps,
        seed=seed + 12,
    )
    main["claim_level_label_swap_randomization"] = (
        claim_label_swap_randomization(
            main_rows,
            main["design_weights"],
            reps=randomization_reps,
            seed=seed + 21,
        )
    )
    order["claim_level_label_swap_randomization"] = (
        claim_label_swap_randomization(
            order_rows,
            order["design_weights"],
            reps=randomization_reps,
            seed=seed + 22,
        )
    )
    token_robustness = token_robustness_summary(
        main_rows,
        bundle["manifest"],
        bootstrap_reps=bootstrap_reps,
        seed=seed + 31,
    )
    transition = transition_summary(
        transition_rows, bootstrap_reps=bootstrap_reps, seed=seed + 41
    )

    formal_counts = {
        "main": len({row["blind_task_id"] for row in main_rows}),
        "order_only": len({row["blind_task_id"] for row in order_rows}),
        "transition": len({row["blind_task_id"] for row in transition_rows}),
    }
    side_balance = _side_balance(bundle["preference_key"])
    expected_side_balance = {
        "main": {"A": 60, "B": 60},
        "order_only": {"A": 40, "B": 40},
    }
    main_stratum_claim_counts = dict(
        Counter(
            row["stratum"]
            for row in bundle["preference_key"].values()
            if row["comparison_type"] == "main"
        )
    )
    order_complexity_claim_counts = dict(
        Counter(
            row.get("complexity")
            for row in bundle["preference_key"].values()
            if row["comparison_type"] == "order_only"
        )
    )
    transition_operation_claim_counts = dict(
        Counter(row["operation"] for row in bundle["transition_key"].values())
    )
    expected_transition_operations = {
        "OPEN": 40,
        "CONTRAST": 20,
        "BRIDGE": 20,
        "CORROBORATE": 10,
        "FALLBACK": 10,
    }
    exact_sampling_quotas = (
        len(main_stratum_claim_counts) == 12
        and set(main_stratum_claim_counts.values()) == {10}
        and order_complexity_claim_counts == {"single": 40, "multi": 40}
        and transition_operation_claim_counts == expected_transition_operations
    )
    duplicate_issues = [
        issue
        for project in [*preference_projects, *transition_projects]
        for issue in project["duplicate_issues"]
    ]
    raw_annotation_count = len(preference_rows) + len(transition_rows)
    exact_counts = formal_counts == EXPECTED_FORMAL_COUNTS
    exact_annotations = raw_annotation_count == 600
    exact_balance = side_balance == expected_side_balance
    no_redundant = not duplicate_issues
    validated_invariants = {
        "frozen_bundle_hashes_verified": True,
        "project_task_fingerprints_verified": True,
        "distinct_annotators_verified": True,
        "artifact_source_hash_mapping_verified": True,
        "main_design_weights_from_manifest": (
            main["design_weight_source"]
            in {"manifest_design_weight", "manifest_pool_size"}
            and len(main["design_weights"]) == 12
        ),
    }
    completion_satisfied = (
        exact_counts
        and exact_annotations
        and exact_balance
        and exact_sampling_quotas
        and no_redundant
        and all(validated_invariants.values())
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    metrics: dict[str, Any] = {
        "analysis_type": "evitrace_small_scale_human_alignment",
        "analysis_scope": "exploratory_small_scale_human_evaluation",
        "generated_at_utc": generated_at,
        "resampling": {
            "bootstrap_replicates": bootstrap_reps,
            "randomization_replicates_requested": randomization_reps,
            "seed": seed,
        },
        "snapshot": {
            "sqlite_path": str(db_path.resolve()),
            "sqlite_sha256": sha256_file(db_path),
            "sqlite_quick_check": quick_check,
            "task_manifest_path": bundle["manifest_path"],
            "task_manifest_sha256": bundle["manifest_sha256"],
            "frozen_artifact_paths": bundle["paths"],
            "frozen_artifact_sha256": bundle["sha256"],
        },
        "projects": [
            _project_public_summary(project)
            for project in [*preference_projects, *transition_projects]
        ],
        "preference": {
            "main": main,
            "order_only": order,
            "token_robustness": token_robustness,
        },
        "transition": transition,
        "data_integrity": {
            "public_task_fingerprints_match": True,
            "private_key_commitments_match": True,
            "artifact_sha_mapping_consistent": True,
            "formal_groups_claim_disjoint": True,
            "transition_at_most_one_step_per_claim": True,
            "preference_evitrace_side_balance": side_balance,
            "expected_preference_evitrace_side_balance": expected_side_balance,
            "main_stratum_claim_counts": main_stratum_claim_counts,
            "order_complexity_claim_counts": order_complexity_claim_counts,
            "transition_operation_claim_counts": transition_operation_claim_counts,
            "expected_transition_operation_claim_counts": expected_transition_operations,
            "redundant_completion_issues": duplicate_issues,
            "fatal_issue_count": 0,
        },
        "completion_contract": {
            "expected_formal_claim_counts": EXPECTED_FORMAL_COUNTS,
            "formal_claim_counts": formal_counts,
            "expected_raw_annotation_count": 600,
            "raw_annotation_count": raw_annotation_count,
            "redundant_completion_count": len(duplicate_issues),
            "exact_claim_counts": exact_counts,
            "exact_raw_annotation_count": exact_annotations,
            "exact_side_balance": exact_balance,
            "exact_sampling_quotas": exact_sampling_quotas,
            "no_redundant_active_completions": no_redundant,
            **validated_invariants,
            "satisfied": completion_satisfied,
        },
    }
    metrics["locked_interpretation"] = locked_interpretation(
        main, order, transition
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = sorted(
        [*preference_rows, *transition_rows],
        key=lambda row: (
            row["task_type"],
            row["comparison_type"],
            row["blind_task_id"],
            row["annotator_index"],
        ),
    )
    raw_path = output_dir / "raw_double_annotations.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    paper_path = output_dir / "paper_insert.md"
    paper_table_path = output_dir / "paper_table.tex"
    write_jsonl(raw_path, raw_rows)
    write_json(metrics_path, metrics)
    _atomic_write(report_path, report_text(metrics))
    _atomic_write(paper_path, paper_insert_text(metrics))
    _atomic_write(paper_table_path, paper_table_tex(metrics))

    artifact_paths = {
        path.name: path
        for path in (
            raw_path,
            metrics_path,
            report_path,
            paper_path,
            paper_table_path,
        )
    }
    manifest = {
        "schema_version": "exp3-trace-alignment-analysis-v1",
        "generated_at_utc": generated_at,
        "complete": completion_satisfied,
        "completion_contract": metrics["completion_contract"],
        "inputs": {
            "sqlite_path": str(db_path.resolve()),
            "sqlite_sha256": metrics["snapshot"]["sqlite_sha256"],
            "task_manifest_path": bundle["manifest_path"],
            "task_manifest_sha256": bundle["manifest_sha256"],
            "frozen_artifact_sha256": bundle["sha256"],
            "preference_project_ids": list(preference_project_ids),
            "transition_project_ids": list(transition_project_ids),
        },
        "expected_artifacts": sorted(artifact_paths),
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                **(
                    {"rows": len(raw_rows)}
                    if name.endswith(".jsonl")
                    else {}
                ),
            }
            for name, path in artifact_paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frozen EviTrace preference and transition projects."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST
    )
    parser.add_argument(
        "--preference-project-ids",
        type=int,
        nargs=2,
        required=True,
        metavar=("ANNOTATOR_A", "ANNOTATOR_B"),
    )
    parser.add_argument(
        "--transition-project-ids",
        type=int,
        nargs=2,
        required=True,
        metavar=("ANNOTATOR_A", "ANNOTATOR_B"),
    )
    parser.add_argument("--preference-tasks", type=Path)
    parser.add_argument("--transition-tasks", type=Path)
    parser.add_argument("--preference-key", type=Path)
    parser.add_argument("--transition-key", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--randomization-reps", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=20_260_724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze(
        args.db,
        args.task_manifest,
        args.preference_project_ids,
        args.transition_project_ids,
        args.output_dir,
        preference_tasks_path=args.preference_tasks,
        transition_tasks_path=args.transition_tasks,
        preference_key_path=args.preference_key,
        transition_key_path=args.transition_key,
        bootstrap_reps=args.bootstrap_reps,
        randomization_reps=args.randomization_reps,
        seed=args.analysis_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "complete": metrics["completion_contract"]["satisfied"],
                "formal_claim_counts": metrics["completion_contract"][
                    "formal_claim_counts"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
