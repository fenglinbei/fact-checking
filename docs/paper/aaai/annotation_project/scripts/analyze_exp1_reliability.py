#!/usr/bin/env python3
"""Export and analyze the completed Exp1 double annotations.

The script deliberately produces a *pre-adjudication* analysis.  It keeps both
annotators' labels, reports agreement and prevalence-robust agreement, and
creates an adjudication queue instead of turning two-rater disagreements into
gold labels.

Atom-level keys are ``(dataset, event_id, atom_id)``.  Claim-level fields are
collapsed on ``(dataset, event_id)`` because Label Studio repeats those fields
on every atom task.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DB = PROJECT_ROOT / "label_studio_data" / "label_studio.sqlite3"
DEFAULT_TASK_UNIVERSE = PROJECT_ROOT / "data" / "exp1_tasks_flat_zh.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp1_reliability_pre_adjudication"
WRITING_ANCHOR = PROJECT_ROOT.parent / "writing_outline_v0.4.2_structure_only.md"
EXPECTED_ATOMS = 257
EXPECTED_CLAIMS = 200
EXPECTED_CLAIMS_BY_DATASET = {"liar_raw": 100, "rawfc": 100}
GOLD_RESOLUTION_PROTOCOL_VERSION = "exp1-exact-gold-resolution-v1-20260717"

REQUIRED_FIELDS = (
    "claim_complexity",
    "completeness_missed",
    "faithfulness",
    "atomicity",
)
ALLOWED_VALUES = {
    "claim_complexity": ("simple", "compound"),
    "completeness_missed": ("0", "1", "2", "3+"),
    "faithfulness": ("yes", "no"),
    "atomicity": ("yes", "no"),
}
COMPLETENESS_ORDER = {"0": 0, "1": 1, "2": 2, "3+": 3}


@dataclass(frozen=True)
class ProjectSpec:
    project_id: int
    label: str
    expected_email: str | None = None


def _json_load(raw: Any, context: str) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {context}: {exc}") from exc
    return raw


def _parse_controls(raw: Any, require_all: bool) -> tuple[dict[str, Any], list[str]]:
    """Parse Label Studio controls by ``from_name``, never by array order."""

    payload = _json_load(raw, "annotation.result")
    if not isinstance(payload, list):
        raise ValueError("annotation.result must be a JSON list")

    parsed: dict[str, Any] = {}
    notes: list[str] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("annotation.result entries must be JSON objects")
        field = entry.get("from_name")
        value = entry.get("value") or {}
        if field == "notes":
            text = value.get("text", [])
            if isinstance(text, str):
                text = [text]
            if not isinstance(text, list) or not all(isinstance(item, str) for item in text):
                raise ValueError("notes.value.text must be a string list")
            notes.extend(item for item in text if item)
            continue
        if field not in ALLOWED_VALUES:
            continue
        if field in parsed:
            raise ValueError(f"Duplicate control in annotation.result: {field}")
        choices = value.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"{field} must contain exactly one choice")
        choice = choices[0]
        if choice not in ALLOWED_VALUES[field]:
            raise ValueError(f"Invalid {field} choice: {choice!r}")
        parsed[field] = choice

    missing = [field for field in REQUIRED_FIELDS if field not in parsed]
    if require_all and missing:
        raise ValueError(f"Missing required controls: {missing}")
    parsed["notes"] = notes
    return parsed, missing


def parse_annotation_result(raw: Any) -> dict[str, Any]:
    parsed, _ = _parse_controls(raw, require_all=True)
    return parsed


def parse_draft_result(raw: Any) -> tuple[dict[str, Any], list[str]]:
    return _parse_controls(raw, require_all=False)


def semantic_atom_key(data: dict[str, Any]) -> tuple[str, str, str]:
    raw_values = (data.get("dataset"), data.get("event_id"), data.get("atom_id"))
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in raw_values):
        raise ValueError(f"Invalid semantic atom key values: {raw_values}")
    key = raw_values
    if key[2] == "-":
        raise ValueError(f"Invalid semantic atom key: {key}")
    return key


def semantic_claim_key(data: dict[str, Any]) -> tuple[str, str]:
    atom_key = semantic_atom_key(data)
    return atom_key[0], atom_key[1]


def load_authoritative_universe(path: Path) -> dict[str, Any]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_content = path.read_bytes()
    for line_number, raw_line in enumerate(raw_content.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        data = _json_load(raw_line, f"{path}:{line_number}")
        if not isinstance(data, dict):
            raise ValueError(f"Authoritative task row {line_number} must be an object")
        key = semantic_atom_key(data)
        if key in records:
            raise ValueError(f"Duplicate authoritative task key: {key}")
        reference = {
            "claim": data.get("claim"),
            "proposition": data.get("proposition"),
            "atom_type": data.get("type"),
            "all_atoms_text": data.get("all_atoms_text"),
        }
        invalid_fields = [
            field
            for field, value in reference.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_fields:
            raise ValueError(
                f"Authoritative task {key} has invalid text fields: {invalid_fields}"
            )
        records[key] = reference
    claims_by_dataset: Counter[str] = Counter(
        dataset for dataset, _event_id in {(key[0], key[1]) for key in records}
    )
    claim_count = sum(claims_by_dataset.values())
    if len(records) != EXPECTED_ATOMS or claim_count != EXPECTED_CLAIMS:
        raise ValueError(
            f"Unexpected authoritative universe size: atoms={len(records)}, claims={claim_count}"
        )
    if dict(claims_by_dataset) != EXPECTED_CLAIMS_BY_DATASET:
        raise ValueError(f"Unexpected authoritative dataset counts: {dict(claims_by_dataset)}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_content).hexdigest(),
        "records": records,
        "atom_count": len(records),
        "claim_count": claim_count,
        "claims_by_dataset": dict(claims_by_dataset),
    }


def validate_project_universe(project: dict[str, Any], universe: dict[str, Any]) -> None:
    expected = universe["records"]
    actual = project["records"]
    if set(actual) != set(expected):
        raise ValueError(f"Project {project['project_id']} does not match the authoritative task keys")
    for key, reference in expected.items():
        for field, expected_value in reference.items():
            if actual[key][field] != expected_value:
                raise ValueError(
                    f"Project {project['project_id']} differs from authoritative task {key}, field={field}"
                )


def validate_distinct_project_pair(project_a: dict[str, Any], project_b: dict[str, Any]) -> None:
    if project_a["project_id"] == project_b["project_id"]:
        raise ValueError("Formal double annotation requires two distinct projects")
    if project_a["annotator_email"] == project_b["annotator_email"]:
        raise ValueError("Formal double annotation requires two distinct annotators")


def load_project_annotations(connection: sqlite3.Connection, spec: ProjectSpec) -> dict[str, Any]:
    project_row = connection.execute(
        "SELECT id, title, deleted_at FROM project WHERE id = ?", (spec.project_id,)
    ).fetchone()
    if project_row is None:
        raise ValueError(f"Project {spec.project_id} does not exist")
    if project_row["deleted_at"] is not None:
        raise ValueError(f"Project {spec.project_id} is archived")

    rows = connection.execute(
        """
        SELECT
            t.id AS task_id,
            t.inner_id,
            t.data,
            tc.id AS annotation_id,
            tc.result,
            tc.created_at,
            tc.updated_at,
            u.email AS annotator_email,
            TRIM(u.first_name || ' ' || u.last_name) AS annotator_name
        FROM task AS t
        JOIN task_completion AS tc
          ON tc.task_id = t.id AND tc.project_id = t.project_id
        JOIN htx_user AS u ON u.id = tc.completed_by_id
        WHERE t.project_id = ? AND tc.was_cancelled = 0
        ORDER BY t.inner_id, t.id, tc.id
        """,
        (spec.project_id,),
    ).fetchall()

    by_task: Counter[int] = Counter(row["task_id"] for row in rows)
    duplicates = sorted(task_id for task_id, count in by_task.items() if count != 1)
    if duplicates:
        raise ValueError(f"Project {spec.project_id} has multiple active annotations for tasks: {duplicates[:10]}")

    annotators = {(row["annotator_email"], row["annotator_name"]) for row in rows}
    if len(annotators) != 1:
        raise ValueError(f"Project {spec.project_id} has unexpected annotators: {sorted(annotators)}")

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        data = _json_load(row["data"], f"task {row['task_id']}.data")
        if not isinstance(data, dict):
            raise ValueError(f"Task {row['task_id']} data must be a JSON object")
        key = semantic_atom_key(data)
        if key in records:
            raise ValueError(f"Duplicate semantic key in project {spec.project_id}: {key}")
        labels = parse_annotation_result(row["result"])
        records[key] = {
            "dataset": key[0],
            "event_id": key[1],
            "atom_id": key[2],
            "claim": data.get("claim", ""),
            "proposition": data.get("proposition", ""),
            "atom_type": data.get("type", ""),
            "all_atoms_text": data.get("all_atoms_text", ""),
            "task_id": row["task_id"],
            "inner_id": row["inner_id"],
            "annotation_id": row["annotation_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "labels": labels,
        }

    total_tasks = connection.execute(
        "SELECT COUNT(*) FROM task WHERE project_id = ?", (spec.project_id,)
    ).fetchone()[0]
    if len(records) != total_tasks:
        raise ValueError(
            f"Project {spec.project_id} is incomplete: annotations={len(records)}, tasks={total_tasks}"
        )

    email, database_name = next(iter(annotators)) if annotators else (None, None)
    if spec.expected_email is not None and email != spec.expected_email:
        raise ValueError(
            f"Project {spec.project_id} owner mismatch: expected={spec.expected_email}, actual={email}"
        )
    return {
        "project_id": spec.project_id,
        "project_title": project_row["title"],
        "label": spec.label,
        "annotator_email": email,
        "annotator_name": database_name or spec.label,
        "records": records,
    }


def load_draft_issues(
    connection: sqlite3.Connection,
    project_ids: Sequence[int],
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in project_ids)
    rows = connection.execute(
        f"""
        SELECT
            t.project_id,
            t.inner_id,
            t.data,
            d.id AS draft_id,
            d.updated_at AS draft_updated_at,
            d.result AS draft_result,
            tc.id AS annotation_id,
            tc.updated_at AS annotation_updated_at,
            tc.result AS annotation_result
        FROM tasks_annotationdraft AS d
        JOIN task AS t ON t.id = d.task_id
        LEFT JOIN task_completion AS tc ON tc.id = d.annotation_id
        WHERE t.project_id IN ({placeholders})
        ORDER BY t.project_id, t.inner_id, d.id
        """,
        tuple(project_ids),
    ).fetchall()
    issues: list[dict[str, Any]] = []
    for row in rows:
        data = _json_load(row["data"], f"draft task {row['inner_id']}.data")
        key = semantic_atom_key(data)
        draft_parse_error: str | None = None
        try:
            draft, missing_fields = parse_draft_result(row["draft_result"])
        except ValueError as exc:
            draft = {"notes": []}
            missing_fields = list(REQUIRED_FIELDS)
            draft_parse_error = str(exc)
        submitted = parse_annotation_result(row["annotation_result"]) if row["annotation_result"] else None
        changed = {
            field: {"submitted": submitted[field], "draft": draft[field]}
            for field in REQUIRED_FIELDS
            if submitted is not None and field in draft and submitted[field] != draft[field]
        }
        issues.append(
            {
                "issue_type": "unsubmitted_draft",
                "project_id": row["project_id"],
                "inner_id": row["inner_id"],
                "dataset": key[0],
                "event_id": key[1],
                "atom_id": key[2],
                "draft_id": row["draft_id"],
                "annotation_id": row["annotation_id"],
                "annotation_updated_at": row["annotation_updated_at"],
                "draft_updated_at": row["draft_updated_at"],
                "draft_present_fields": sorted(field for field in REQUIRED_FIELDS if field in draft),
                "draft_missing_fields": missing_fields,
                "draft_parse_error": draft_parse_error,
                "changed_fields": changed,
            }
        )
    return issues


def cohen_kappa(a_values: Sequence[str], b_values: Sequence[str], categories: Sequence[str]) -> float | None:
    if len(a_values) != len(b_values) or not a_values:
        return None
    n = len(a_values)
    observed = sum(a == b for a, b in zip(a_values, b_values)) / n
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    expected = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return None
    return (observed - expected) / denominator


def linear_weighted_kappa(
    a_values: Sequence[str],
    b_values: Sequence[str],
    categories: Sequence[str],
) -> float | None:
    if len(a_values) != len(b_values) or not a_values:
        return None
    n = len(a_values)
    index = {category: position for position, category in enumerate(categories)}
    scale = max(len(categories) - 1, 1)
    observed = sum(1.0 - abs(index[a] - index[b]) / scale for a, b in zip(a_values, b_values)) / n
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    expected = 0.0
    for a in categories:
        for b in categories:
            weight = 1.0 - abs(index[a] - index[b]) / scale
            expected += weight * (a_counts[a] / n) * (b_counts[b] / n)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return None
    return (observed - expected) / denominator


def gwet_ac1(a_values: Sequence[str], b_values: Sequence[str], categories: Sequence[str]) -> float | None:
    """Return Gwet's AC1 for two raters and nominal categories."""

    if len(a_values) != len(b_values) or not a_values:
        return None
    n = len(a_values)
    q = len(categories)
    if q < 2:
        return None
    observed = sum(a == b for a, b in zip(a_values, b_values)) / n
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    mean_marginals = {
        category: (a_counts[category] + b_counts[category]) / (2.0 * n)
        for category in categories
    }
    expected = sum(p * (1.0 - p) for p in mean_marginals.values()) / (q - 1)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return None
    return (observed - expected) / denominator


def agreement_metrics(
    a_values: Sequence[str],
    b_values: Sequence[str],
    categories: Sequence[str],
    positive_label: str | None = None,
) -> dict[str, Any]:
    if len(a_values) != len(b_values):
        raise ValueError("Paired value arrays have different lengths")
    n = len(a_values)
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    matrix = {
        a: {b: sum(x == a and y == b for x, y in zip(a_values, b_values)) for b in categories}
        for a in categories
    }
    agreements = sum(a == b for a, b in zip(a_values, b_values))
    metrics: dict[str, Any] = {
        "n": n,
        "annotator_a_distribution": dict(a_counts),
        "annotator_b_distribution": dict(b_counts),
        "confusion_matrix": matrix,
        "exact_agreement_count": agreements,
        "exact_agreement": agreements / n if n else None,
        "disagreement_count": n - agreements,
        "cohen_kappa": cohen_kappa(a_values, b_values, categories),
        "gwet_ac1": gwet_ac1(a_values, b_values, categories),
    }
    if len(categories) > 2:
        metrics["linear_weighted_kappa"] = linear_weighted_kappa(a_values, b_values, categories)

    if positive_label is not None and n:
        a_positive = sum(value == positive_label for value in a_values)
        b_positive = sum(value == positive_label for value in b_values)
        both_positive = sum(a == positive_label and b == positive_label for a, b in zip(a_values, b_values))
        both_negative = sum(a != positive_label and b != positive_label for a, b in zip(a_values, b_values))
        discordant = n - both_positive - both_negative
        positive_denominator = 2 * both_positive + discordant
        negative_denominator = 2 * both_negative + discordant
        metrics["positive_label"] = positive_label
        metrics["annotator_a_positive_count"] = a_positive
        metrics["annotator_a_positive_rate"] = a_positive / n
        metrics["annotator_b_positive_count"] = b_positive
        metrics["annotator_b_positive_rate"] = b_positive / n
        metrics["both_positive_count"] = both_positive
        metrics["both_negative_count"] = both_negative
        metrics["binary_disagreement_count"] = discordant
        metrics["positive_agreement"] = (
            2 * both_positive / positive_denominator if positive_denominator else None
        )
        metrics["negative_agreement"] = (
            2 * both_negative / negative_denominator if negative_denominator else None
        )
        metrics["pre_adjudication_positive_lower_bound"] = both_positive / n
        metrics["pre_adjudication_positive_upper_bound"] = (n - both_negative) / n
        if len(categories) == 2:
            combined = {
                category: a_counts[category] + b_counts[category]
                for category in categories
            }
            minority_label = min(
                categories,
                key=lambda category: (combined[category], categories.index(category)),
            )
            both_minority = sum(
                a == minority_label and b == minority_label
                for a, b in zip(a_values, b_values)
            )
            minority_denominator = a_counts[minority_label] + b_counts[minority_label]
            metrics["minority_label"] = minority_label
            metrics["minority_agreement"] = (
                2 * both_minority / minority_denominator if minority_denominator else None
            )
    return metrics


def percentile(values: Sequence[float], probability: float) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def stratified_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    metric_fn: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    metric_names: Sequence[str],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if reps <= 0:
        return {}
    by_dataset_claim: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_dataset_claim[row["dataset"]][(row["dataset"], row["event_id"])].append(row)
    if not by_dataset_claim:
        return {}

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(reps):
        sampled_rows: list[dict[str, Any]] = []
        for claims in by_dataset_claim.values():
            keys = list(claims)
            for _draw in range(len(keys)):
                sampled_rows.extend(claims[rng.choice(keys)])
        result = metric_fn(sampled_rows)
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


def pair_atom_records(project_a: dict[str, Any], project_b: dict[str, Any]) -> list[dict[str, Any]]:
    a_records = project_a["records"]
    b_records = project_b["records"]
    if set(a_records) != set(b_records):
        only_a = sorted(set(a_records) - set(b_records))
        only_b = sorted(set(b_records) - set(a_records))
        raise ValueError(f"Project task universes differ: only_a={only_a[:5]}, only_b={only_b[:5]}")

    paired: list[dict[str, Any]] = []
    for key in sorted(a_records):
        a = a_records[key]
        b = b_records[key]
        for field in ("claim", "proposition", "atom_type", "all_atoms_text"):
            if a[field] != b[field]:
                raise ValueError(f"Task data mismatch for {key}, field={field}")
        paired.append(
            {
                "dataset": key[0],
                "event_id": key[1],
                "atom_id": key[2],
                "claim": a["claim"],
                "proposition": a["proposition"],
                "atom_type": a["atom_type"],
                "a": a,
                "b": b,
            }
        )
    return paired


def collapse_claim_annotations(
    atom_pairs: Sequence[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in atom_pairs:
        grouped[(row["dataset"], row["event_id"])].append(row)

    claim_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for key in sorted(grouped):
        atoms = sorted(grouped[key], key=lambda row: row["atom_id"])
        record: dict[str, Any] = {
            "dataset": key[0],
            "event_id": key[1],
            "claim": atoms[0]["claim"],
            "atom_count": len(atoms),
            "atom_ids": [row["atom_id"] for row in atoms],
            "a": {},
            "b": {},
        }
        for side, label in (("a", label_a), ("b", label_b)):
            for field in ("claim_complexity", "completeness_missed"):
                by_atom = {row["atom_id"]: row[side]["labels"][field] for row in atoms}
                unique_values = sorted(set(by_atom.values()))
                value = unique_values[0] if len(unique_values) == 1 else None
                record[side][field] = value
                record[side][f"{field}_by_atom"] = by_atom
                if value is None:
                    issues.append(
                        {
                            "issue_type": "within_annotator_claim_conflict",
                            "annotator": label,
                            "dataset": key[0],
                            "event_id": key[1],
                            "field": field,
                            "values_by_atom": by_atom,
                        }
                    )
        claim_rows.append(record)
    return claim_rows, issues


def atom_field_metrics(
    rows: Sequence[dict[str, Any]],
    field: str,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    categories = ALLOWED_VALUES[field]

    def calculate(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return agreement_metrics(
            [row["a"]["labels"][field] for row in sample],
            [row["b"]["labels"][field] for row in sample],
            categories,
            positive_label="yes",
        )

    result = calculate(rows)
    result["claim_cluster_bootstrap_ci95"] = stratified_cluster_bootstrap(
        rows,
        calculate,
        (
            "exact_agreement",
            "cohen_kappa",
            "gwet_ac1",
            "annotator_a_positive_rate",
            "annotator_b_positive_rate",
        ),
        bootstrap_reps,
        seed,
    )
    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = calculate(subset)
    result["by_dataset"] = by_dataset
    return result


def completeness_metrics(
    claim_rows: Sequence[dict[str, Any]],
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clean = [
        row
        for row in claim_rows
        if row["a"]["completeness_missed"] is not None
        and row["b"]["completeness_missed"] is not None
    ]

    def calculate_ordinal(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        a_values = [row["a"]["completeness_missed"] for row in sample]
        b_values = [row["b"]["completeness_missed"] for row in sample]
        result = agreement_metrics(
            a_values,
            b_values,
            ALLOWED_VALUES["completeness_missed"],
        )
        result["within_one_count"] = sum(
            abs(COMPLETENESS_ORDER[a] - COMPLETENESS_ORDER[b]) <= 1
            for a, b in zip(a_values, b_values)
        )
        result["within_one"] = result["within_one_count"] / len(sample) if sample else None
        return result

    def calculate_binary(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return agreement_metrics(
            ["complete" if row["a"]["completeness_missed"] == "0" else "incomplete" for row in sample],
            ["complete" if row["b"]["completeness_missed"] == "0" else "incomplete" for row in sample],
            ("complete", "incomplete"),
            positive_label="complete",
        )

    ordinal = calculate_ordinal(clean)
    ordinal["excluded_internal_conflict_claims"] = len(claim_rows) - len(clean)
    ordinal["claim_cluster_bootstrap_ci95"] = stratified_cluster_bootstrap(
        clean,
        calculate_ordinal,
        (
            "exact_agreement",
            "within_one",
            "cohen_kappa",
            "linear_weighted_kappa",
            "gwet_ac1",
        ),
        bootstrap_reps,
        seed,
    )
    ordinal["by_dataset"] = {
        dataset: calculate_ordinal([row for row in clean if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in clean})
    }

    coverage = calculate_binary(clean)
    coverage["excluded_internal_conflict_claims"] = len(claim_rows) - len(clean)
    coverage["definition"] = "complete iff completeness_missed=0; incomplete otherwise"
    coverage["claim_cluster_bootstrap_ci95"] = stratified_cluster_bootstrap(
        clean,
        calculate_binary,
        (
            "exact_agreement",
            "cohen_kappa",
            "gwet_ac1",
            "annotator_a_positive_rate",
            "annotator_b_positive_rate",
        ),
        bootstrap_reps,
        seed,
    )
    coverage["by_dataset"] = {
        dataset: calculate_binary([row for row in clean if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in clean})
    }
    return {"ordinal": ordinal, "coverage_binary": coverage}, clean


def complexity_metrics(claim_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clean = [
        row
        for row in claim_rows
        if row["a"]["claim_complexity"] is not None and row["b"]["claim_complexity"] is not None
    ]

    def calculate(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return agreement_metrics(
            [row["a"]["claim_complexity"] for row in sample],
            [row["b"]["claim_complexity"] for row in sample],
            ALLOWED_VALUES["claim_complexity"],
        )

    result = calculate(clean)
    result["excluded_internal_conflict_claims"] = len(claim_rows) - len(clean)
    result["by_dataset"] = {
        dataset: calculate([row for row in clean if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in clean})
    }
    return result


def strict_claim_pass_metrics(
    atom_pairs: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claims = {(row["dataset"], row["event_id"]): row for row in claim_rows}
    atoms_by_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in atom_pairs:
        atoms_by_claim[(row["dataset"], row["event_id"])].append(row)

    strict_rows: list[dict[str, Any]] = []
    for key in sorted(atoms_by_claim):
        claim = claims[key]
        if claim["a"]["completeness_missed"] is None or claim["b"]["completeness_missed"] is None:
            continue
        atoms = atoms_by_claim[key]
        components: list[tuple[bool, bool]] = [
            (
                claim["a"]["completeness_missed"] == "0",
                claim["b"]["completeness_missed"] == "0",
            )
        ]
        for atom in atoms:
            components.extend(
                (
                    (
                        atom["a"]["labels"][field] == "yes",
                        atom["b"]["labels"][field] == "yes",
                    )
                    for field in ("faithfulness", "atomicity")
                )
            )
        a_pass = all(a_component for a_component, _ in components)
        b_pass = all(b_component for _, b_component in components)
        confirmed_pass = all(a_component and b_component for a_component, b_component in components)
        possible_pass = all(a_component or b_component for a_component, b_component in components)
        strict_rows.append(
            {
                "dataset": key[0],
                "event_id": key[1],
                "a_value": "pass" if a_pass else "fail",
                "b_value": "pass" if b_pass else "fail",
                "componentwise_confirmed_pass": confirmed_pass,
                "componentwise_possible_pass": possible_pass,
            }
        )

    def calculate(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result = agreement_metrics(
            [row["a_value"] for row in sample],
            [row["b_value"] for row in sample],
            ("pass", "fail"),
            positive_label="pass",
        )
        n = len(sample)
        confirmed = sum(row["componentwise_confirmed_pass"] for row in sample)
        possible = sum(row["componentwise_possible_pass"] for row in sample)
        result["componentwise_confirmed_pass_count"] = confirmed
        result["componentwise_possible_pass_count"] = possible
        result["pre_adjudication_positive_lower_bound"] = confirmed / n if n else None
        result["pre_adjudication_positive_upper_bound"] = possible / n if n else None
        return result

    result = calculate(strict_rows)
    result["unit"] = "claim"
    result["definition"] = "completeness=0 and every atom has faithfulness=yes and atomicity=yes"
    result["excluded_internal_conflict_claims"] = len(claim_rows) - len(strict_rows)
    result["claim_cluster_bootstrap_ci95"] = stratified_cluster_bootstrap(
        strict_rows,
        calculate,
        (
            "exact_agreement",
            "cohen_kappa",
            "gwet_ac1",
            "annotator_a_positive_rate",
            "annotator_b_positive_rate",
            "pre_adjudication_positive_lower_bound",
            "pre_adjudication_positive_upper_bound",
        ),
        bootstrap_reps,
        seed,
    )
    result["by_dataset"] = {
        dataset: calculate([row for row in strict_rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in strict_rows})
    }
    return result, strict_rows


def build_disagreements(
    atom_pairs: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    disagreements: list[dict[str, Any]] = []
    adjudication: list[dict[str, Any]] = []
    for row in atom_pairs:
        differing = {
            field: {"annotator_a": row["a"]["labels"][field], "annotator_b": row["b"]["labels"][field]}
            for field in ("faithfulness", "atomicity")
            if row["a"]["labels"][field] != row["b"]["labels"][field]
        }
        if not differing:
            continue
        item = {
            "unit": "atom",
            "dataset": row["dataset"],
            "event_id": row["event_id"],
            "atom_id": row["atom_id"],
            "claim": row["claim"],
            "proposition": row["proposition"],
            "disagreements": differing,
            "needs_adjudication": True,
        }
        disagreements.append(item)
        adjudication.append(item)

    for row in claim_rows:
        differing: dict[str, Any] = {}
        for field in ("claim_complexity", "completeness_missed"):
            a = row["a"][field]
            b = row["b"][field]
            if a is not None and b is not None and a != b:
                differing[field] = {"annotator_a": a, "annotator_b": b}
        if not differing:
            continue
        item = {
            "unit": "claim",
            "dataset": row["dataset"],
            "event_id": row["event_id"],
            "claim": row["claim"],
            "atom_ids": row["atom_ids"],
            "disagreements": differing,
            "needs_adjudication": "completeness_missed" in differing,
        }
        disagreements.append(item)
        if item["needs_adjudication"]:
            adjudication.append(item)
    return disagreements, adjudication


def build_blind_adjudication_tasks(
    queue: Sequence[dict[str, Any]],
    atom_pairs: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove A/B labels while retaining everything needed for adjudication."""

    atoms = {
        (row["dataset"], row["event_id"], row["atom_id"]): row
        for row in atom_pairs
    }
    claims = {(row["dataset"], row["event_id"]): row for row in claim_rows}
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in queue:
        dataset = item["dataset"]
        event_id = item["event_id"]
        if item.get("unit") == "atom":
            atom_id = item["atom_id"]
            source = atoms[(dataset, event_id, atom_id)]
            fields = sorted(item.get("disagreements", {}))
            task = {
                "unit": "atom",
                "dataset": dataset,
                "event_id": event_id,
                "atom_id": atom_id,
                "claim": source["claim"],
                "proposition": source["proposition"],
                "atom_type": source["atom_type"],
                "fields_to_adjudicate": fields,
            }
        else:
            source = claims[(dataset, event_id)]
            if (
                item.get("issue_type") == "within_annotator_claim_conflict"
                and item.get("field") in REQUIRED_FIELDS
            ):
                fields = [item["field"]]
            else:
                fields = [
                    field
                    for field in ("completeness_missed",)
                    if field in item.get("disagreements", {})
                ]
            if not fields:
                raise ValueError(f"Claim adjudication item has no field: {item}")
            task = {
                "unit": "claim",
                "dataset": dataset,
                "event_id": event_id,
                "claim": source["claim"],
                "atoms": [
                    {
                        "atom_id": atom_id,
                        "proposition": atoms[(dataset, event_id, atom_id)]["proposition"],
                    }
                    for atom_id in source["atom_ids"]
                ],
                "fields_to_adjudicate": fields,
            }
        identity = {
            "protocol_version": GOLD_RESOLUTION_PROTOCOL_VERSION,
            "unit": task["unit"],
            "dataset": dataset,
            "event_id": event_id,
            "atom_id": task.get("atom_id"),
            "fields_to_adjudicate": task["fields_to_adjudicate"],
        }
        suffix = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        task["adjudication_id"] = f"exp1-adj-{suffix}"
        if task["adjudication_id"] in seen_ids:
            raise ValueError(f"Duplicate blind adjudication task identity: {identity}")
        seen_ids.add(task["adjudication_id"])
        tasks.append(task)
    return tasks


def _rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _ci(metrics: dict[str, Any], name: str, *, rate: bool = True) -> str:
    interval = metrics.get("claim_cluster_bootstrap_ci95", {}).get(name, {})
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return "NA"
    formatter = _rate if rate else _number
    return f"{formatter(low)}–{formatter(high)}"


def render_report(metrics: dict[str, Any]) -> str:
    faith = metrics["atom_level"]["faithfulness"]
    atomic = metrics["atom_level"]["atomicity"]
    complete = metrics["claim_level"]["complete_coverage"]
    complete_ordinal = metrics["claim_level"]["completeness_ordinal"]
    complexity = metrics["claim_level"]["claim_complexity"]
    strict = metrics["claim_level"]["strict_all_criteria_pass"]
    label_a = metrics["annotators"]["a"]["label"]
    label_b = metrics["annotators"]["b"]["label"]
    issues = metrics["quality_control"]
    draft_changed_fields = sorted(
        {
            field
            for issue in issues.get("unsubmitted_draft_summaries", [])
            for field in issue.get("changed_fields", {})
        }
    )
    if draft_changed_fields:
        draft_scope_note = (
            "当前草稿中的未提交修改字段为 "
            + ", ".join(f"`{field}`" for field in draft_changed_fields)
            + ("；不改变三项主要结果。" if draft_changed_fields == ["claim_complexity"] else "。")
        )
    else:
        draft_scope_note = "当前未提交草稿没有可与正式提交比较的字段变化。"

    lines = [
        "# Exp1 Claim Atomization 人工可靠性分析（预仲裁）",
        "",
        f"- 生成时间：`{metrics['snapshot']['generated_at_utc']}`",
        "- 写作锚点：`writing_outline_v0.4.2_structure_only.md`。",
        f"- 正式标注者：**{label_a}**、**{label_b}**",
        f"- 样本：**{metrics['snapshot']['paired_atoms']} atoms / {metrics['snapshot']['paired_claims']} claims**",
        "- 抽样设计：LIAR-RAW 与 RAWFC 各 100 claims，70% 随机、30% 困难优先；这是设计样本，不是候选总体的自然分布估计。",
        "- 状态：两位标注者的正式双标已齐；双人分歧和一个 claim 内部冲突保持未决，因此本报告不是最终 gold error rate。",
        "- 结论边界：仅审计 claim atomization；不外推到 Evidence Map 的 relation/directness/confidence，也不建立与下游 F1 的因果关系。",
        "",
        "## 主要结果",
        "",
        f"| 维度 | 单位/N | {label_a} 通过率 | {label_b} 通过率 | Exact | Cohen κ | Gwet AC1 | 未仲裁通过率界 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, unit, result in (
        ("忠实性", "atom", faith),
        ("原子性", "atom", atomic),
        ("完整覆盖（`missed=0`）", "claim", complete),
    ):
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{unit}/{result['n']}",
                    _rate(result.get("annotator_a_positive_rate")),
                    _rate(result.get("annotator_b_positive_rate")),
                    _rate(result.get("exact_agreement")),
                    _number(result.get("cohen_kappa")),
                    _number(result.get("gwet_ac1")),
                    f"{_rate(result.get('pre_adjudication_positive_lower_bound'))}–{_rate(result.get('pre_adjudication_positive_upper_bound'))}",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "完整覆盖的分母为 199：一个 claim 存在同一标注者在不同 atoms 上填写不一致，已暂时排除。未仲裁通过率界不是置信区间；它只表示在逐项独立仲裁后，当前标签允许的通过率范围。",
            "",
            "三个维度均由多数通过类主导。类别偏斜会影响 Cohen κ，但少数失败类的一致性也确实有限，因此 Exact、κ、AC1 与少数类 agreement 必须共同解释，不能用 AC1 抵消低 κ。",
            "",
            "### 少数失败类一致性",
            "",
            "| 维度 / 少数类 | 少数类 agreement | 两人共同判失败 | 二值分歧 |",
            "|---|---:|---:|---:|",
            f"| 忠实性 / no | {_rate(faith.get('minority_agreement'))} | {faith['both_negative_count']} | {faith['binary_disagreement_count']} |",
            f"| 原子性 / no | {_rate(atomic.get('minority_agreement'))} | {atomic['both_negative_count']} | {atomic['binary_disagreement_count']} |",
            f"| 完整覆盖 / incomplete | {_rate(complete.get('minority_agreement'))} | {complete['both_negative_count']} | {complete['binary_disagreement_count']} |",
            "",
            "### 95% claim-cluster bootstrap 区间",
            "",
            f"| 维度 | {label_a} 通过率 CI | {label_b} 通过率 CI | Exact CI | κ CI | AC1 CI |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 忠实性 | {_ci(faith, 'annotator_a_positive_rate')} | {_ci(faith, 'annotator_b_positive_rate')} | {_ci(faith, 'exact_agreement')} | {_ci(faith, 'cohen_kappa', rate=False)} | {_ci(faith, 'gwet_ac1', rate=False)} |",
            f"| 原子性 | {_ci(atomic, 'annotator_a_positive_rate')} | {_ci(atomic, 'annotator_b_positive_rate')} | {_ci(atomic, 'exact_agreement')} | {_ci(atomic, 'cohen_kappa', rate=False)} | {_ci(atomic, 'gwet_ac1', rate=False)} |",
            f"| 完整覆盖 | {_ci(complete, 'annotator_a_positive_rate')} | {_ci(complete, 'annotator_b_positive_rate')} | {_ci(complete, 'exact_agreement')} | {_ci(complete, 'cohen_kappa', rate=False)} | {_ci(complete, 'gwet_ac1', rate=False)} |",
            "",
            f"区间使用 percentile bootstrap（{metrics['snapshot']['bootstrap_replicates']} 次）；在 LIAR-RAW 与 RAWFC 内分别按 claim 有放回抽样，并携带该 claim 的全部 atoms。各指标使用 snapshot 中记录的派生 seed。",
            "",
            "## 序数完整性与辅助字段",
            "",
            f"- 原始 `completeness_missed`（0/1/2/3+）Exact：**{complete_ordinal['exact_agreement_count']}/{complete_ordinal['n']}（{_rate(complete_ordinal['exact_agreement'])}）**。",
            f"- Within-1：**{complete_ordinal['within_one_count']}/{complete_ordinal['n']}（{_rate(complete_ordinal['within_one'])}）**。",
            f"- 线性加权 κ：**{_number(complete_ordinal.get('linear_weighted_kappa'))}**；四类别 nominal AC1={_number(complete_ordinal.get('gwet_ac1'))}。正文的“完整覆盖”使用二值 `0` vs `>0` AC1，不混用这一四类别 AC1。",
            f"- Claim complexity Exact：**{complexity['exact_agreement_count']}/{complexity['n']}（{_rate(complexity['exact_agreement'])}）**；κ={_number(complexity.get('cohen_kappa'))}。该字段仅作辅助分层，不是 LLM 质量维度。",
            "",
            "## 数据集分层",
            "",
            "| 数据集 | 忠实性 Exact / AC1 | 原子性 Exact / AC1 | 完整覆盖 Exact / AC1 |",
            "|---|---:|---:|---:|",
        ]
    )
    datasets = sorted(faith["by_dataset"])
    for dataset in datasets:
        f = faith["by_dataset"][dataset]
        a = atomic["by_dataset"][dataset]
        c = complete["by_dataset"][dataset]
        lines.append(
            f"| {dataset} | {_rate(f['exact_agreement'])} / {_number(f['gwet_ac1'])} "
            f"| {_rate(a['exact_agreement'])} / {_number(a['gwet_ac1'])} "
            f"| {_rate(c['exact_agreement'])} / {_number(c['gwet_ac1'])} |"
        )

    lines.extend(
        [
            "",
            "总体值是该等量 claim 设计样本上的 atom-micro / claim rate，不是按原始候选池规模自然加权的总体估计。LIAR-RAW 含 137 atoms、RAWFC 含 120 atoms，因此 atom-level micro 指标仍受每个 claim 的 atom 数影响。",
            "",
            "## 次要派生分析：严格三维全通过",
            "",
            f"严格 claim pass 定义为 `completeness=0` 且该 claim 的每个 atom 均同时满足 faithfulness=yes、atomicity=yes。{label_a} 为 {_rate(strict['annotator_a_positive_rate'])}，{label_b} 为 {_rate(strict['annotator_b_positive_rate'])}；Exact={_rate(strict['exact_agreement'])}，κ={_number(strict['cohen_kappa'])}，AC1={_number(strict['gwet_ac1'])}。",
            f"逐组件仲裁允许的预仲裁界为 **{_rate(strict['pre_adjudication_positive_lower_bound'])}–{_rate(strict['pre_adjudication_positive_upper_bound'])}**（{strict['componentwise_confirmed_pass_count']}–{strict['componentwise_possible_pass_count']} / {strict['n']}）。该逻辑合取对 atom 数敏感，只作附录诊断，不作为正文综合质量分。",
            "",
            "## 质量控制与待仲裁项",
            "",
            f"- 标注者内部 claim 冲突：**{issues['within_annotator_claim_conflict_count']}**。",
            f"- 未提交草稿：**{issues['unsubmitted_draft_count']}**。",
            f"- Atom 分歧：**{issues['atom_disagreement_count']} 条，涉及 {issues['atom_disagreement_claim_count']} claims**。",
            f"- 完整性 Exact 分歧：**{issues['completeness_exact_resolution_count']}**；按现指导书 `差值 >= 2` 的完整性仲裁项为 **{issues['protocol_completeness_adjudication_count']}**。",
            f"- 按现指导书的协议仲裁量：**{issues['protocol_adjudication_records']} 条 atom 记录 / {issues['protocol_adjudication_unique_claims']} claims**。",
            f"- 为形成唯一 gold 的扩展 resolution 队列：**{issues['extended_gold_resolution_records']} 条记录 / {issues['extended_gold_resolution_unique_claims']} claims**。",
            f"- Gold resolution 协议版本：`{metrics['methodology']['gold_resolution_protocol_version']}`。现指导书仅要求仲裁完整性 `差值 >= 2`；扩展协议为形成唯一 gold，另纳入所有主维度 Exact mismatch 与内部冲突。",
            f"- 未提交草稿不改变 {metrics['snapshot']['paired_atoms']}/{metrics['snapshot']['paired_atoms']} 正式完成数；{draft_scope_note}",
            "",
            "## 流程证据边界",
            "",
            "本次分析输入不含 20 条 calibration 或每 50 条 running-checkpoint 的独立产物，因此不能追溯验证指导书中的过程门槛是否按时执行。这里的 full-sample κ 只描述最终双标结果，不能替代那些过程检查，也不据此声称门槛已通过或未通过。",
            "",
            "## 结论",
            "",
            "在该预仲裁设计样本上，两位标注者均给出较高的多数类通过率；原子性的通过率和一致性相对最低，三个维度的失败类判定稳定性仍有限，最终 gold 质量率待独立仲裁后确定。该结果缩小了 v0.4.2 对 atomization 风险“尚未量化”的空白，但不支持“claim decomposition 已被普遍验证可靠”或“能改善下游事实核查”的更强结论。",
            "",
            "与 v0.4.2 对齐的正文候选段落和 Limitations 替换稿见 `paper_insert_v0.4.2.md`；在完成独立仲裁前，不建议把该诊断加入 Abstract 或贡献列表。",
            "",
            "## 生成文件",
            "",
            "- `metrics.json`：完整指标与 bootstrap 区间。",
            "- `atom_annotations_a.jsonl`、`atom_annotations_b.jsonl`：两位正式标注者导出。",
            "- `claim_annotations.jsonl`：按 claim 折叠后的标签。",
            "- `disagreements.jsonl`：全部 Exact 分歧。",
            "- `adjudication_queue.jsonl`：含 A/B 标签的仲裁审计队列。",
            "- `adjudication_tasks_blind.jsonl`：不含 A/B 标签、可交给独立仲裁者的任务。",
            "- `data_issues.jsonl`：claim 内部冲突与未提交草稿。",
            "- `gold_resolution_protocol.md`：原指导书与扩展 exact-gold 协议的边界。",
            "- `paper_insert_v0.4.2.md`：正文候选小节与 Limitations 替换稿。",
            "- `manifest.json`：输入指纹、文件哈希与完成标记；最后发布。",
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_insert(metrics: dict[str, Any]) -> str:
    faith = metrics["atom_level"]["faithfulness"]
    atomic = metrics["atom_level"]["atomicity"]
    complete = metrics["claim_level"]["complete_coverage"]
    label_a = "Annotator A"
    label_b = "Annotator B"
    return "\n".join(
        [
            "## Claim Atomization Reliability Study",
            "",
            "为审计 claim atomization 这一上游输入，我们从 LIAR-RAW 与 RAWFC validation data 各抽取 100 条 claims（70% 随机、30% 困难优先），得到 257 个 atoms，并由两位标注者独立评估 faithfulness、atomicity 与 completeness。表中的结果均为 pre-adjudication IAA；一个 claim 存在同一标注者的 claim-level 内部冲突，因此 completeness 暂按 199 条 clean claims 统计。这些人工标签仅用于事后可靠性审计，不参与 atomization 生成、Evidence Map 构建、selector preference 构造、verifier 训练或 checkpoint selection。",
            "",
            f"| Dimension | Unit / N | {label_a} pass | {label_b} pass | Exact | Cohen's $\\kappa$ | Gwet AC1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Faithfulness | atom / {faith['n']} | {_rate(faith['annotator_a_positive_rate'])} | {_rate(faith['annotator_b_positive_rate'])} | {_rate(faith['exact_agreement'])} | {_number(faith['cohen_kappa'])} | {_number(faith['gwet_ac1'])} |",
            f"| Atomicity | atom / {atomic['n']} | {_rate(atomic['annotator_a_positive_rate'])} | {_rate(atomic['annotator_b_positive_rate'])} | {_rate(atomic['exact_agreement'])} | {_number(atomic['cohen_kappa'])} | {_number(atomic['gwet_ac1'])} |",
            f"| Complete coverage | claim / {complete['n']} | {_rate(complete['annotator_a_positive_rate'])} | {_rate(complete['annotator_b_positive_rate'])} | {_rate(complete['exact_agreement'])} | {_number(complete['cohen_kappa'])} | {_number(complete['gwet_ac1'])} |",
            "",
            "两位标注者在 faithfulness 与 complete coverage 上均给出较高的多数类通过率和 raw exact agreement；atomicity 的通过率差异与分歧更集中。三个维度的类别分布均明显偏斜，且少数失败类别的一致性有限，因此我们并列报告 Exact、Cohen's $\\kappa$ 与 Gwet AC1，而不以任一单项系数替代其他证据。该结果只刻画这一等量数据集、困难样本过采样设计下的 pre-adjudication reliability；双人分歧仍待独立仲裁，因而不能视为最终 gold error rate，也不外推到 Evidence Map 标注或 downstream performance。",
            "",
            "表注：Faithfulness 与 atomicity 为 atom-level micro statistics；complete coverage 将 `completeness_missed=0` 视为 complete、`>0` 视为 incomplete。完整的 bootstrap intervals、dataset-stratified results、minority-class agreement、strict all-criteria diagnostic 与仲裁队列放入附录。",
            "",
            "### Limitations replacement (v0.4.2 paragraphs 1--2)",
            "",
            "首先，claim decomposition 并不总能稳定提高事实核查表现，错误拆分、遗漏限定条件或过度细分仍可能向 retrieval 与 Evidence Map 传播 \\citep{Hu2025DecompositionDilemmas}。为审计并初步量化这一风险，我们在 200 条 claims、257 个 atoms 上开展了两位标注者的独立双标，分别评估 faithfulness、atomicity 与 completeness。预仲裁结果显示多数通过类别上的 raw agreement 较高，但类别分布明显偏斜，少数失败类别的一致性有限，其中 atomicity 是分歧最集中的维度。由于 claim-level 冲突与双人分歧仍待独立仲裁，这些结果只刻画本研究设计样本上的 pre-adjudication reliability，不能作为最终 gold error rate，也不能证明 claim decomposition 普遍改善 downstream verification。",
            "",
            "第二，Evidence Map 仍依赖 LLM API，且上述人工审计只覆盖 claim atomization，不能外推为对 relation、directness 或 confidence 标注的验证。Evidence Map 的人工双标与仲裁尚未完成，self-reported confidence 也未经校准；冻结缓存、prompt/schema hash、调用日期与调用元数据提高了可复现性和 artifact-level 可审计性，但不能将这些结构标注等同于人工 gold supervision。",
            "",
        ]
    )


def render_gold_resolution_protocol(metrics: dict[str, Any]) -> str:
    issues = metrics["quality_control"]
    return "\n".join(
        [
            "# Exp1 Gold Resolution Protocol",
            "",
            f"Version: `{GOLD_RESOLUTION_PROTOCOL_VERSION}`",
            "",
            "本文件区分原标注指导书中的仲裁条件与为形成唯一 exact gold 所需的扩展规则。扩展规则不回写两位标注者的原始标签，也不改变 pre-adjudication IAA。",
            "",
            "## 原指导书协议",
            "",
            "- faithfulness：两位标注者不一致时，由独立第三人仲裁。",
            "- atomicity：两位标注者不一致时，由独立第三人仲裁。",
            "- completeness_missed：两位标注者的等级差值至少为 2 时仲裁。",
            "- 第三人不查看 A/B 原始标签，先独立作答。",
            "",
            f"当前对应 **{issues['protocol_adjudication_records']} 条记录 / {issues['protocol_adjudication_unique_claims']} claims**。其中满足 `completeness_missed` 差值至少 2 的记录为 {issues['protocol_completeness_adjudication_count']}。",
            "",
            "## Exact-gold 扩展规则",
            "",
            "为给后续 gold-based 分析提供唯一标签，扩展队列还纳入：",
            "",
            "- 所有 completeness_missed exact mismatches，包括 0 与 1 的分歧；",
            "- 同一标注者在同一 claim 的重复 claim-level 字段中产生的内部冲突；",
            "- 若将来出现辅助 claim_complexity 的内部冲突，可单独校正，但标注者间 complexity 分歧不属于三项质量主指标。",
            "",
            f"当前扩展队列为 **{issues['extended_gold_resolution_records']} 条记录 / {issues['extended_gold_resolution_unique_claims']} claims**。盲化任务使用语义键与待仲裁字段生成稳定 ID，不包含 A/B 标签。",
            "",
            "## 输出使用约束",
            "",
            "- `metrics.json` 中的 IAA 始终来自原双标，不混入仲裁者。",
            "- 仲裁完成前只报告 pre-adjudication rates/bounds，不称为最终 gold error rate。",
            "- Exp1 只覆盖 claim atomization，不能用来验证 Evidence Map relation/directness/confidence。",
            "",
        ]
    )


def atomic_write_text(path: Path, content: str) -> None:
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
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, content)


def artifact_manifest_entry(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "line_count": content.count(b"\n"),
    }


def exported_atom_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(project["records"]):
        record = project["records"][key]
        rows.append(
            {
                "dataset": record["dataset"],
                "event_id": record["event_id"],
                "atom_id": record["atom_id"],
                "claim": record["claim"],
                "proposition": record["proposition"],
                "atom_type": record["atom_type"],
                "annotator": project["label"],
                "project_id": project["project_id"],
                "task_inner_id": record["inner_id"],
                "annotation_id": record["annotation_id"],
                "labels": record["labels"],
            }
        )
    return rows


def analysis_input_sha256(
    projects: Sequence[dict[str, Any]],
    draft_issues: Sequence[dict[str, Any]],
    universe: dict[str, Any],
    bootstrap_reps: int,
    seed: int,
) -> str:
    annotations: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for project in projects:
        identities.append(
            {
                "project_id": project["project_id"],
                "project_title": project["project_title"],
                "label": project["label"],
                "annotator_email": project["annotator_email"],
                "annotator_name": project["annotator_name"],
            }
        )
        for key in sorted(project["records"]):
            record = project["records"][key]
            annotations.append(
                {
                    "project_id": project["project_id"],
                    "semantic_key": key,
                    "claim": record["claim"],
                    "proposition": record["proposition"],
                    "atom_type": record["atom_type"],
                    "all_atoms_text": record["all_atoms_text"],
                    "labels": record["labels"],
                }
            )
    semantic_drafts = [
        {
            key: issue.get(key)
            for key in (
                "issue_type",
                "project_id",
                "dataset",
                "event_id",
                "atom_id",
                "draft_present_fields",
                "draft_missing_fields",
                "draft_parse_error",
                "changed_fields",
            )
        }
        for issue in draft_issues
    ]
    payload = {
        "schema_version": 2,
        "gold_resolution_protocol_version": GOLD_RESOLUTION_PROTOCOL_VERSION,
        "authoritative_universe_sha256": universe["sha256"],
        "project_identities": identities,
        "annotations": annotations,
        "draft_issues": semantic_drafts,
        "analysis_config": {
            "bootstrap_method": "dataset-stratified claim-cluster percentile bootstrap",
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_seed": seed,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def database_provenance_sha256(
    projects: Sequence[dict[str, Any]],
    draft_issues: Sequence[dict[str, Any]],
) -> str:
    provenance: list[dict[str, Any]] = []
    for project in projects:
        for key in sorted(project["records"]):
            record = project["records"][key]
            provenance.append(
                {
                    "project_id": project["project_id"],
                    "semantic_key": key,
                    "task_id": record["task_id"],
                    "task_inner_id": record["inner_id"],
                    "annotation_id": record["annotation_id"],
                    "created_at": record["created_at"],
                    "updated_at": record["updated_at"],
                }
            )
    payload = {"annotations": provenance, "draft_issues": list(draft_issues)}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def analyze(
    db_path: Path,
    task_universe_path: Path,
    output_dir: Path,
    project_a: ProjectSpec,
    project_b: ProjectSpec,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    if project_a.project_id == project_b.project_id:
        raise ValueError("Formal double annotation requires two distinct projects")
    db_path = db_path.resolve()
    task_universe_path = task_universe_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if not task_universe_path.exists():
        raise FileNotFoundError(task_universe_path)
    if not WRITING_ANCHOR.exists():
        raise FileNotFoundError(WRITING_ANCHOR)
    universe = load_authoritative_universe(task_universe_path)
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick_check}")
        a = load_project_annotations(connection, project_a)
        b = load_project_annotations(connection, project_b)
        validate_distinct_project_pair(a, b)
        validate_project_universe(a, universe)
        validate_project_universe(b, universe)
        draft_issues = load_draft_issues(connection, (project_a.project_id, project_b.project_id))
    finally:
        connection.close()

    atom_pairs = pair_atom_records(a, b)
    claim_rows, internal_issues = collapse_claim_annotations(atom_pairs, project_a.label, project_b.label)
    faith = atom_field_metrics(atom_pairs, "faithfulness", bootstrap_reps, seed + 1)
    atomic = atom_field_metrics(atom_pairs, "atomicity", bootstrap_reps, seed + 2)
    completeness, _ = completeness_metrics(claim_rows, bootstrap_reps, seed + 3)
    complete = completeness["coverage_binary"]
    completeness_ordinal = completeness["ordinal"]
    complexity = complexity_metrics(claim_rows)
    strict, strict_rows = strict_claim_pass_metrics(atom_pairs, claim_rows, bootstrap_reps, seed + 4)
    disagreements, adjudication = build_disagreements(atom_pairs, claim_rows)

    atom_disagreement_count = sum(row["unit"] == "atom" for row in disagreements)
    completeness_exact_resolution_count = sum(
        row["unit"] == "claim" and "completeness_missed" in row["disagreements"]
        for row in adjudication
    )
    protocol_completeness_adjudication = [
        row
        for row in adjudication
        if row["unit"] == "claim"
        and "completeness_missed" in row["disagreements"]
        and abs(
            COMPLETENESS_ORDER[row["disagreements"]["completeness_missed"]["annotator_a"]]
            - COMPLETENESS_ORDER[row["disagreements"]["completeness_missed"]["annotator_b"]]
        )
        >= 2
    ]
    data_issues = internal_issues + draft_issues
    full_adjudication_queue = internal_issues + adjudication
    blind_adjudication_tasks = build_blind_adjudication_tasks(
        full_adjudication_queue,
        atom_pairs,
        claim_rows,
    )
    adjudication_claims = {
        (row["dataset"], row["event_id"])
        for row in full_adjudication_queue
    }
    atom_disagreement_claims = {
        (row["dataset"], row["event_id"])
        for row in disagreements
        if row["unit"] == "atom"
    }
    protocol_adjudication_claims = set(atom_disagreement_claims)
    protocol_adjudication_claims.update(
        (row["dataset"], row["event_id"])
        for row in protocol_completeness_adjudication
    )

    metrics: dict[str, Any] = {
        "schema_version": 2,
        "analysis_status": "pre_adjudication",
        "snapshot": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_path": str(db_path),
            "analysis_input_sha256": analysis_input_sha256(
                (a, b), draft_issues, universe, bootstrap_reps, seed
            ),
            "database_provenance_sha256": database_provenance_sha256((a, b), draft_issues),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sqlite_quick_check": quick_check,
            "writing_anchor_path": str(WRITING_ANCHOR.resolve()),
            "writing_anchor_sha256": hashlib.sha256(WRITING_ANCHOR.read_bytes()).hexdigest(),
            "authoritative_task_universe_path": universe["path"],
            "authoritative_task_universe_sha256": universe["sha256"],
            "authoritative_task_universe_atoms": universe["atom_count"],
            "authoritative_task_universe_claims": universe["claim_count"],
            "authoritative_claims_by_dataset": universe["claims_by_dataset"],
            "paired_atoms": len(atom_pairs),
            "paired_claims": len(claim_rows),
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_seed": seed,
            "bootstrap_derived_seeds": {
                "faithfulness": seed + 1,
                "atomicity": seed + 2,
                "completeness": seed + 3,
                "strict_all_criteria_pass": seed + 4,
            },
        },
        "annotators": {
            "a": {
                "label": project_a.label,
                "project_id": a["project_id"],
                "project_title": a["project_title"],
                "database_name": a["annotator_name"],
            },
            "b": {
                "label": project_b.label,
                "project_id": b["project_id"],
                "project_title": b["project_title"],
                "database_name": b["annotator_name"],
            },
        },
        "methodology": {
            "writing_anchor": "writing_outline_v0.4.2_structure_only.md",
            "study_scope": "claim atomization only; no inference to Evidence Map labels or downstream causality",
            "atom_key": ["dataset", "event_id", "atom_id"],
            "claim_key": ["dataset", "event_id"],
            "claim_conflict_policy": "exclude field-specific within-annotator conflicts; never majority vote",
            "gold_policy": "two-rater exact disagreements remain unresolved until independent adjudication",
            "bootstrap": "95% percentile interval; dataset-stratified claim-cluster resampling",
            "bootstrap_checkpoint_note": "completed-sample IAA is descriptive and does not substitute for calibration or running-checkpoint evidence",
            "gold_resolution_protocol_version": GOLD_RESOLUTION_PROTOCOL_VERSION,
        },
        "atom_level": {"faithfulness": faith, "atomicity": atomic},
        "claim_level": {
            "complete_coverage": complete,
            "completeness_ordinal": completeness_ordinal,
            "claim_complexity": complexity,
            "strict_all_criteria_pass": strict,
        },
        "quality_control": {
            "within_annotator_claim_conflict_count": len(internal_issues),
            "within_annotator_claim_conflicts": [
                {
                    key: issue.get(key)
                    for key in ("annotator", "dataset", "event_id", "field", "values_by_atom")
                }
                for issue in internal_issues
            ],
            "unsubmitted_draft_count": len(draft_issues),
            "unsubmitted_draft_summaries": [
                {
                    key: issue.get(key)
                    for key in (
                        "project_id",
                        "dataset",
                        "event_id",
                        "atom_id",
                        "draft_present_fields",
                        "draft_missing_fields",
                        "draft_parse_error",
                        "changed_fields",
                    )
                }
                for issue in draft_issues
            ],
            "atom_disagreement_count": atom_disagreement_count,
            "claim_exact_disagreement_count": sum(row["unit"] == "claim" for row in disagreements),
            "completeness_exact_resolution_count": completeness_exact_resolution_count,
            "protocol_completeness_adjudication_count": len(protocol_completeness_adjudication),
            "atom_disagreement_claim_count": len(atom_disagreement_claims),
            "protocol_adjudication_records": atom_disagreement_count + len(protocol_completeness_adjudication),
            "protocol_adjudication_unique_claims": len(protocol_adjudication_claims),
            "extended_gold_resolution_records": len(full_adjudication_queue),
            "extended_gold_resolution_unique_claims": len(adjudication_claims),
            "calibration_or_running_checkpoint_artifacts_in_analysis_input": False,
        },
    }

    strict_by_key = {(row["dataset"], row["event_id"]): row for row in strict_rows}
    claim_export: list[dict[str, Any]] = []
    for row in claim_rows:
        key = (row["dataset"], row["event_id"])
        exported = dict(row)
        strict_row = strict_by_key.get(key)
        exported["strict_pass"] = (
            {
                "annotator_a": strict_row["a_value"],
                "annotator_b": strict_row["b_value"],
                "componentwise_confirmed_pass": strict_row["componentwise_confirmed_pass"],
                "componentwise_possible_pass": strict_row["componentwise_possible_pass"],
            }
            if strict_row
            else None
        )
        claim_export.append(exported)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = [
        "metrics.json",
        "atom_annotations_a.jsonl",
        "atom_annotations_b.jsonl",
        "claim_annotations.jsonl",
        "disagreements.jsonl",
        "adjudication_queue.jsonl",
        "adjudication_tasks_blind.jsonl",
        "data_issues.jsonl",
        "report.md",
        "paper_insert_v0.4.2.md",
        "gold_resolution_protocol.md",
    ]
    manifest_path = output_dir / "manifest.json"
    generation_id = hashlib.sha256(
        (
            metrics["snapshot"]["analysis_input_sha256"]
            + metrics["snapshot"]["analysis_script_sha256"]
            + metrics["snapshot"]["writing_anchor_sha256"]
        ).encode("ascii")
    ).hexdigest()[:16]
    manifest_base = {
        "schema_version": 1,
        "generation_id": generation_id,
        "analysis_input_sha256": metrics["snapshot"]["analysis_input_sha256"],
        "authoritative_task_universe_sha256": metrics["snapshot"][
            "authoritative_task_universe_sha256"
        ],
        "analysis_script_sha256": metrics["snapshot"]["analysis_script_sha256"],
        "writing_anchor_sha256": metrics["snapshot"]["writing_anchor_sha256"],
        "gold_resolution_protocol_version": GOLD_RESOLUTION_PROTOCOL_VERSION,
        "expected_artifacts": artifact_names,
    }
    write_json(manifest_path, {**manifest_base, "complete": False, "artifacts": {}})

    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "atom_annotations_a.jsonl", exported_atom_rows(a))
    write_jsonl(output_dir / "atom_annotations_b.jsonl", exported_atom_rows(b))
    write_jsonl(output_dir / "claim_annotations.jsonl", claim_export)
    write_jsonl(output_dir / "disagreements.jsonl", disagreements)
    write_jsonl(output_dir / "adjudication_queue.jsonl", full_adjudication_queue)
    write_jsonl(output_dir / "adjudication_tasks_blind.jsonl", blind_adjudication_tasks)
    write_jsonl(output_dir / "data_issues.jsonl", data_issues)
    atomic_write_text(output_dir / "report.md", render_report(metrics))
    atomic_write_text(output_dir / "paper_insert_v0.4.2.md", render_paper_insert(metrics))
    atomic_write_text(
        output_dir / "gold_resolution_protocol.md",
        render_gold_resolution_protocol(metrics),
    )
    artifact_entries = {
        name: artifact_manifest_entry(output_dir / name)
        for name in artifact_names
    }
    write_json(
        manifest_path,
        {
            **manifest_base,
            "complete": True,
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifact_entries,
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--task-universe", type=Path, default=DEFAULT_TASK_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-a", type=int, default=14)
    parser.add_argument("--project-b", type=int, default=15)
    parser.add_argument("--label-a", default="Yulin")
    parser.add_argument("--label-b", default="Zhiqiang")
    parser.add_argument("--expected-email-a", default="1849812973@qq.com")
    parser.add_argument("--expected-email-b", default="3180643570@qq.com")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze(
        args.db,
        args.task_universe,
        args.output_dir,
        ProjectSpec(args.project_a, args.label_a, args.expected_email_a),
        ProjectSpec(args.project_b, args.label_b, args.expected_email_b),
        args.bootstrap_reps,
        args.seed,
    )
    faith = metrics["atom_level"]["faithfulness"]
    atomic = metrics["atom_level"]["atomicity"]
    complete = metrics["claim_level"]["complete_coverage"]
    print(f"Wrote Exp1 pre-adjudication analysis to: {args.output_dir.resolve()}")
    print(
        "faithfulness_exact="
        f"{faith['exact_agreement']:.4f} atomicity_exact={atomic['exact_agreement']:.4f} "
        f"completeness_exact={complete['exact_agreement']:.4f}"
    )


if __name__ == "__main__":
    main()
