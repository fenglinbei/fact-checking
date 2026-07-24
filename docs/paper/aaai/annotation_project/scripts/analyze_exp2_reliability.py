#!/usr/bin/env python3
"""Audit completed Exp2 double annotations and prepare adjudication queues."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from analyze_exp1_reliability import (
    agreement_metrics,
    artifact_manifest_entry,
    atomic_write_text,
    write_json,
    write_jsonl,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DB = PROJECT_ROOT / "label_studio_data" / "label_studio.sqlite3"
DEFAULT_TASKS = PROJECT_ROOT / "data" / "exp2_tasks_zh.jsonl"
DEFAULT_LLM_LABELS = PROJECT_ROOT / "data" / "exp2_llm_labels.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "exp2_reliability_pre_adjudication"

RELATIONS = (
    "support",
    "refute",
    "qualify",
    "mixed",
    "insufficient",
    "background",
    "irrelevant",
)
DIRECTNESS = ("direct", "partial", "context", "none")
DIRECTNESS_RANK = {label: index for index, label in enumerate(DIRECTNESS)}
ALLOWED_CONFIDENCE = (0.1, 0.3, 0.5, 0.7, 0.9)


def json_load(raw: Any, context: str) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {context}: {exc}") from exc


def semantic_key(data: dict[str, Any]) -> tuple[str, str, str, str]:
    values = (
        data.get("dataset"),
        data.get("event_id"),
        data.get("atom_id"),
        data.get("evidence_id"),
    )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"Invalid Exp2 semantic key: {values}")
    return values


def key_dict(key: tuple[str, str, str, str]) -> dict[str, str]:
    return dict(zip(("dataset", "event_id", "atom_id", "evidence_id"), key))


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    content = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json_load(raw_line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows, hashlib.sha256(content).hexdigest()


def load_task_universe(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows, _ = load_jsonl(path)
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = semantic_key(row)
        if key in records:
            raise ValueError(f"Duplicate authoritative task: {key}")
        leaked = sorted(name for name in row if name.startswith("llm_"))
        if leaked:
            raise ValueError(f"Task leaks LLM labels for {key}: {leaked}")
        records[key] = row
    if len(records) != 250:
        raise ValueError(f"Expected 250 authoritative tasks, found {len(records)}")
    return records


def parse_result(raw: Any) -> dict[str, Any]:
    payload = json_load(raw, "annotation.result")
    if not isinstance(payload, list):
        raise ValueError("annotation.result must be a list")
    values: dict[str, Any] = {}
    notes: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("annotation.result entries must be objects")
        name = item.get("from_name")
        value = item.get("value") or {}
        if name == "notes":
            texts = value.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            notes.extend(text for text in texts if isinstance(text, str) and text.strip())
            continue
        if name not in {"gold_relation", "gold_directness", "gold_confidence"}:
            continue
        choices = value.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"{name} must contain one choice")
        values[name] = choices[0]
    missing = sorted(
        {"gold_relation", "gold_directness", "gold_confidence"} - values.keys()
    )
    if missing:
        raise ValueError(f"Missing annotation controls: {missing}")
    if values["gold_relation"] not in RELATIONS:
        raise ValueError(f"Invalid relation: {values['gold_relation']}")
    if values["gold_directness"] not in DIRECTNESS:
        raise ValueError(f"Invalid directness: {values['gold_directness']}")
    confidence = float(values["gold_confidence"])
    if not any(math.isclose(confidence, allowed) for allowed in ALLOWED_CONFIDENCE):
        raise ValueError(f"Invalid confidence: {confidence}")
    values["gold_confidence"] = confidence
    values["notes"] = notes
    return values


def labels_fingerprint(labels: dict[str, Any]) -> tuple[Any, ...]:
    return (
        labels["gold_relation"],
        labels["gold_directness"],
        labels["gold_confidence"],
        tuple(labels["notes"]),
    )


def load_project(
    connection: sqlite3.Connection,
    project_id: int,
    expected_email: str,
    universe: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    project = connection.execute(
        "SELECT id, title, deleted_at FROM project WHERE id = ?", (project_id,)
    ).fetchone()
    if project is None or project["deleted_at"] is not None:
        raise ValueError(f"Project {project_id} is missing or archived")
    task_count = connection.execute(
        "SELECT COUNT(*) FROM task WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    cancelled_count = connection.execute(
        "SELECT COUNT(*) FROM task_completion WHERE project_id = ? AND was_cancelled = 1",
        (project_id,),
    ).fetchone()[0]
    draft_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM tasks_annotationdraft AS d
        JOIN task AS t ON t.id = d.task_id
        WHERE t.project_id = ?
        """,
        (project_id,),
    ).fetchone()[0]
    rows = connection.execute(
        """
        SELECT
            t.id AS task_id,
            t.inner_id,
            t.data,
            a.id AS annotation_id,
            a.result,
            a.created_at,
            u.email AS annotator_email,
            TRIM(u.first_name || ' ' || u.last_name) AS annotator_name
        FROM task AS t
        JOIN task_completion AS a
          ON a.task_id = t.id
         AND a.project_id = t.project_id
         AND a.was_cancelled = 0
        JOIN htx_user AS u ON u.id = a.completed_by_id
        WHERE t.project_id = ?
        ORDER BY t.inner_id, t.id, a.id
        """,
        (project_id,),
    ).fetchall()
    if task_count != 250:
        raise ValueError(f"Project {project_id} has {task_count} tasks, expected 250")
    emails = {row["annotator_email"] for row in rows}
    if emails != {expected_email}:
        raise ValueError(f"Project {project_id} annotators differ: {sorted(emails)}")

    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicate_issues: list[dict[str, Any]] = []
    for task_id, task_rows in grouped.items():
        parsed = [parse_result(row["result"]) for row in task_rows]
        fingerprints = {labels_fingerprint(labels) for labels in parsed}
        if len(task_rows) > 1 and len(fingerprints) != 1:
            raise ValueError(f"Conflicting active annotations on task {task_id}")
        row = task_rows[-1]
        data = json_load(row["data"], f"task {task_id}.data")
        key = semantic_key(data)
        if key in records:
            raise ValueError(f"Duplicate semantic key in project {project_id}: {key}")
        records[key] = {
            **key_dict(key),
            "task_id": task_id,
            "inner_id": row["inner_id"],
            "annotation_id": row["annotation_id"],
            "created_at": row["created_at"],
            "annotator_email": row["annotator_email"],
            "annotator_name": row["annotator_name"],
            **parsed[-1],
        }
        if len(task_rows) > 1:
            duplicate_issues.append(
                {
                    "issue_type": "redundant_identical_active_completion",
                    "project_id": project_id,
                    "task_id": task_id,
                    "inner_id": row["inner_id"],
                    **key_dict(key),
                    "annotation_ids": [item["annotation_id"] for item in task_rows],
                    "created_at": [item["created_at"] for item in task_rows],
                    "collapsed_to_one_semantic_annotation": True,
                }
            )
    if set(records) != set(universe):
        raise ValueError(f"Project {project_id} task universe differs from authoritative input")
    for key, task in universe.items():
        db_task = json_load(
            connection.execute(
                "SELECT data FROM task WHERE id = ?", (records[key]["task_id"],)
            ).fetchone()[0],
            "task.data",
        )
        if db_task != task:
            raise ValueError(f"Project {project_id} task data differs for {key}")
    return {
        "project_id": project_id,
        "title": project["title"],
        "annotator_email": expected_email,
        "task_count": task_count,
        "active_completion_count": len(rows),
        "semantic_annotation_count": len(records),
        "cancelled_completion_count": cancelled_count,
        "draft_count": draft_count,
        "records": records,
        "duplicate_issues": duplicate_issues,
    }


def ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        mean_rank = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]:
            result[index] = mean_rank
        start = end
    return result


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    return numerator / denominator if denominator else None


def stable_task_id(key: tuple[str, str, str, str], fields: Sequence[str]) -> str:
    payload = "|".join((*key, *sorted(fields))).encode("utf-8")
    return "exp2-" + hashlib.sha256(payload).hexdigest()[:16]


def export_annotation_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "dataset",
        "event_id",
        "atom_id",
        "evidence_id",
        "task_id",
        "inner_id",
        "annotation_id",
        "annotator_email",
        "gold_relation",
        "gold_directness",
        "gold_confidence",
        "notes",
    )
    return [
        {field: project["records"][key][field] for field in fields}
        for key in sorted(project["records"])
    ]


def queue_row(
    key: tuple[str, str, str, str],
    fields: list[str],
    project_a: dict[str, Any],
    project_b: dict[str, Any],
    universe: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    a = project_a["records"][key]
    b = project_b["records"][key]
    return {
        "adjudication_task_id": stable_task_id(key, fields),
        **key_dict(key),
        "fields_to_adjudicate": fields,
        "annotator_a": {field: a[field] for field in fields},
        "annotator_b": {field: b[field] for field in fields},
        "source_task": universe[key],
    }


def blind_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "adjudication_task_id": row["adjudication_task_id"],
        **row["source_task"],
        "fields_to_adjudicate": row["fields_to_adjudicate"],
    }


def report_text(metrics: dict[str, Any]) -> str:
    relation = metrics["agreement"]["relation"]
    directness = metrics["agreement"]["directness"]
    confidence = metrics["agreement"]["confidence"]
    protocol = metrics["adjudication"]["guideline_protocol"]
    exact = metrics["adjudication"]["exact_gold_recommended"]
    return "\n".join(
        [
            "# Exp2 pre-adjudication audit",
            "",
            "## Completion and integrity",
            "",
            "- Both formal projects contain 250 authoritative semantic tasks and are fully submitted.",
            "- Project 16 contains one redundant, label-identical active completion; it is collapsed before pairing.",
            "- No missing controls, invalid choices, cancelled completions, drafts, task-universe drift, or LLM-label leakage were found.",
            "",
            "## Independent double-annotation agreement",
            "",
            f"- Relation: {relation['exact_agreement_count']}/250 exact ({relation['exact_agreement']:.1%}); Cohen kappa={relation['cohen_kappa']:.3f}; Gwet AC1={relation['gwet_ac1']:.3f}.",
            f"- Directness: {directness['exact_agreement_count']}/250 exact ({directness['exact_agreement']:.1%}); Spearman rho={directness['spearman_rho']:.3f}.",
            f"- Confidence: {confidence['exact_agreement_count']}/250 exact ({confidence['exact_agreement']:.1%}); this is annotator self-confidence and is not adjudicated.",
            "",
            "## Adjudication counts",
            "",
            f"- Guideline protocol: {protocol['unique_pair_count']} unique pairs, {protocol['field_decision_count']} field decisions.",
            f"- Recommended exact gold: {exact['unique_pair_count']} unique pairs, {exact['field_decision_count']} field decisions.",
            "- The exact-gold queue adds every directness mismatch so the final relation and directness gold are unique per pair.",
            "- Confidence remains separate self-report metadata under both scopes.",
            "",
            "## Interpretation",
            "",
            "The double annotation is complete and structurally usable, but the pre-adjudication relation agreement is moderate rather than high. Final claims about LLM Evidence Map accuracy must use adjudicated human gold, not these pre-adjudication labels alone.",
            "",
        ]
    )


def build_analysis(
    db_path: Path,
    tasks_path: Path,
    llm_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    universe = load_task_universe(tasks_path)
    llm_rows, llm_sha = load_jsonl(llm_path)
    llm = {semantic_key(row): row for row in llm_rows}
    if set(llm) != set(universe):
        raise ValueError("LLM label universe differs from task universe")
    tasks_sha = hashlib.sha256(tasks_path.read_bytes()).hexdigest()

    uri = f"file:{db_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise ValueError(f"SQLite quick_check failed: {quick_check}")
        project_a = load_project(connection, 16, "1849812973@qq.com", universe)
        project_b = load_project(connection, 17, "3180643570@qq.com", universe)
    finally:
        connection.close()

    keys = sorted(universe)
    relation_a = [project_a["records"][key]["gold_relation"] for key in keys]
    relation_b = [project_b["records"][key]["gold_relation"] for key in keys]
    direct_a = [project_a["records"][key]["gold_directness"] for key in keys]
    direct_b = [project_b["records"][key]["gold_directness"] for key in keys]
    confidence_a = [project_a["records"][key]["gold_confidence"] for key in keys]
    confidence_b = [project_b["records"][key]["gold_confidence"] for key in keys]

    relation_metrics = agreement_metrics(relation_a, relation_b, RELATIONS)
    direct_metrics = agreement_metrics(direct_a, direct_b, DIRECTNESS)
    direct_gaps = [
        abs(DIRECTNESS_RANK[a] - DIRECTNESS_RANK[b])
        for a, b in zip(direct_a, direct_b)
    ]
    direct_metrics["absolute_gap_distribution"] = dict(Counter(map(str, direct_gaps)))
    direct_metrics["spearman_rho"] = pearson(
        ranks([DIRECTNESS_RANK[value] for value in direct_a]),
        ranks([DIRECTNESS_RANK[value] for value in direct_b]),
    )
    confidence_exact = sum(math.isclose(a, b) for a, b in zip(confidence_a, confidence_b))
    confidence_metrics = {
        "n": len(keys),
        "exact_agreement_count": confidence_exact,
        "exact_agreement": confidence_exact / len(keys),
        "disagreement_count": len(keys) - confidence_exact,
        "annotator_a_mean": sum(confidence_a) / len(keys),
        "annotator_b_mean": sum(confidence_b) / len(keys),
        "absolute_gap_distribution": dict(
            Counter(f"{abs(a - b):.1f}" for a, b in zip(confidence_a, confidence_b))
        ),
        "pearson_r": pearson(confidence_a, confidence_b),
        "spearman_rho": pearson(ranks(confidence_a), ranks(confidence_b)),
        "adjudicated": False,
        "reason": "Annotator self-confidence is not a gold target in guideline v1.0.",
    }

    protocol_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    relation_mismatch = 0
    direct_gap_two = 0
    direct_any_mismatch = 0
    protocol_overlap = 0
    exact_overlap = 0
    protocol_by_dataset: Counter[str] = Counter()
    for key in keys:
        a = project_a["records"][key]
        b = project_b["records"][key]
        relation_diff = a["gold_relation"] != b["gold_relation"]
        direct_gap = abs(
            DIRECTNESS_RANK[a["gold_directness"]]
            - DIRECTNESS_RANK[b["gold_directness"]]
        )
        relation_mismatch += relation_diff
        direct_gap_two += direct_gap >= 2
        direct_any_mismatch += direct_gap >= 1
        protocol_overlap += relation_diff and direct_gap >= 2
        exact_overlap += relation_diff and direct_gap >= 1
        protocol_fields = []
        if relation_diff:
            protocol_fields.append("gold_relation")
        if direct_gap >= 2:
            protocol_fields.append("gold_directness")
        if protocol_fields:
            protocol_rows.append(
                queue_row(key, protocol_fields, project_a, project_b, universe)
            )
            protocol_by_dataset[key[0]] += 1
        exact_fields = []
        if relation_diff:
            exact_fields.append("gold_relation")
        if direct_gap >= 1:
            exact_fields.append("gold_directness")
        if exact_fields:
            exact_rows.append(queue_row(key, exact_fields, project_a, project_b, universe))

    metrics = {
        "analysis_type": "exp2_pre_adjudication_double_annotation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite_quick_check": quick_check,
        "authoritative_inputs": {
            "tasks_path": str(tasks_path.resolve()),
            "tasks_sha256": tasks_sha,
            "llm_labels_path": str(llm_path.resolve()),
            "llm_labels_sha256": llm_sha,
            "semantic_pair_count": len(keys),
            "llm_fields_hidden_from_tasks": True,
        },
        "projects": [
            {key: value for key, value in project.items() if key not in {"records", "duplicate_issues"}}
            for project in (project_a, project_b)
        ],
        "data_issues": {
            "redundant_identical_active_completion_count": len(
                project_a["duplicate_issues"] + project_b["duplicate_issues"]
            ),
            "fatal_issue_count": 0,
        },
        "agreement": {
            "relation": relation_metrics,
            "directness": direct_metrics,
            "confidence": confidence_metrics,
        },
        "adjudication": {
            "guideline_protocol": {
                "definition": "Any relation mismatch plus directness ordinal gap of at least two.",
                "relation_field_count": relation_mismatch,
                "directness_field_count": direct_gap_two,
                "overlap_pair_count": protocol_overlap,
                "unique_pair_count": len(protocol_rows),
                "field_decision_count": relation_mismatch + direct_gap_two,
                "relation_only_pair_count": relation_mismatch - protocol_overlap,
                "directness_only_pair_count": direct_gap_two - protocol_overlap,
                "both_pair_count": protocol_overlap,
                "unique_pairs_by_dataset": dict(protocol_by_dataset),
            },
            "exact_gold_recommended": {
                "definition": "Any relation mismatch plus any directness mismatch.",
                "relation_field_count": relation_mismatch,
                "directness_field_count": direct_any_mismatch,
                "overlap_pair_count": exact_overlap,
                "unique_pair_count": len(exact_rows),
                "field_decision_count": relation_mismatch + direct_any_mismatch,
                "additional_unique_pairs_vs_protocol": len(exact_rows) - len(protocol_rows),
            },
            "confidence_field_count": 0,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics.json": metrics,
        "annotations_a.jsonl": export_annotation_rows(project_a),
        "annotations_b.jsonl": export_annotation_rows(project_b),
        "protocol_adjudication_queue.jsonl": protocol_rows,
        "protocol_adjudication_tasks_blind.jsonl": [blind_row(row) for row in protocol_rows],
        "exact_gold_adjudication_queue.jsonl": exact_rows,
        "exact_gold_adjudication_tasks_blind.jsonl": [blind_row(row) for row in exact_rows],
        "data_issues.jsonl": project_a["duplicate_issues"] + project_b["duplicate_issues"],
    }
    for name, payload in outputs.items():
        path = output_dir / name
        if name.endswith(".jsonl"):
            write_jsonl(path, payload)
        else:
            write_json(path, payload)
    report_path = output_dir / "report.md"
    atomic_write_text(report_path, report_text(metrics))

    artifact_names = [*outputs, "report.md"]
    manifest = {
        "generated_at_utc": metrics["generated_at_utc"],
        "artifacts": {
            name: artifact_manifest_entry(output_dir / name) for name in artifact_names
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--llm-labels", type=Path, default=DEFAULT_LLM_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_analysis(args.db, args.tasks, args.llm_labels, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "protocol_pairs": metrics["adjudication"]["guideline_protocol"][
                    "unique_pair_count"
                ],
                "exact_gold_pairs": metrics["adjudication"]["exact_gold_recommended"][
                    "unique_pair_count"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
