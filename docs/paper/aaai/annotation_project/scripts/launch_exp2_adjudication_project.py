#!/usr/bin/env python3
"""Create or validate the blinded Exp2 exact-gold adjudication project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "label_studio_data"
DEFAULT_DATABASE = DEFAULT_DATA_DIR / "label_studio.sqlite3"
DEFAULT_PREPARED_DIR = ROOT / "results" / "exp2_adjudication_v1"
DEFAULT_REPORT = DEFAULT_PREPARED_DIR / "launch_report.json"
CONFIG_PATH = ROOT / "config" / "exp2_adjudication_exact_gold.xml"

TITLE = "[ZIJIE ONLY] Exp2-Evidence-Map-Adjudication-v1"
MARKER = "exp2-exact-gold-adjudication-v1-20260719"
PROTOCOL_VERSION = "exp2-exact-gold-resolution-v1-20260719"
ADJUDICATOR_EMAIL = "1349410043@qq.com"
ADMIN_EMAIL = "admin@annotation.local"
SOURCE_PROJECT_ID = 19
EXPECTED_TASKS = 125
FORBIDDEN_KEYS = {
    "annotator_a",
    "annotator_b",
    "llm_relation",
    "llm_directness",
    "llm_confidence",
    "llm_evidence_role",
}


def bootstrap_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args, _ = parser.parse_known_args()
    return args


_bootstrap = bootstrap_arguments()
os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = str(_bootstrap.data_dir.resolve())
os.environ["LABEL_STUDIO_DATABASE_NAME"] = str(_bootstrap.database.resolve())
os.environ.setdefault("LABEL_STUDIO_LATEST_VERSION_CHECK", "false")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")

import label_studio  # noqa: E402

LABEL_STUDIO_PACKAGE_DIR = Path(label_studio.__file__).resolve().parent
if str(LABEL_STUDIO_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_STUDIO_PACKAGE_DIR))

import django  # noqa: E402

django.setup()

from core.label_config import validate_label_config  # noqa: E402
from django.conf import settings  # noqa: E402
from django.db import connection, transaction  # noqa: E402
from projects.models import Project, ProjectMember  # noqa: E402
from tasks.models import Task  # noqa: E402
from tasks.validation import TaskValidator  # noqa: E402
from users.models import User  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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


def database_path() -> Path:
    return Path(settings.DATABASES["default"]["NAME"]).resolve()


def quick_check() -> str:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA quick_check")
        return str(cursor.fetchone()[0])


def foreign_key_violation_count() -> int:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA foreign_key_check")
        return len(cursor.fetchall())


def task_fingerprints(rows: list[dict[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()
        for row in rows
    ]


def project_description(manifest: dict[str, Any]) -> str:
    source_hash = manifest["source"]["source_blind_queue_sha256"]
    return (
        "Zijie 专用 Exp2 Evidence Map 独立盲仲裁项目。"
        "这是 exact-gold 扩展口径：裁决全部 relation 和 directness exact mismatch；"
        "confidence 不仲裁。不得查看 Yulin/Zhiqiang 或 LLM 原标签。"
        f" marker={MARKER}; protocol={PROTOCOL_VERSION}; source_blind={source_hash}."
    )


def project_instruction() -> str:
    return (
        "这是 Exp2 Evidence Map 的独立盲仲裁。每页只显示实际存在分歧的 "
        "Relation、Directness 或两者，显示的问题均须作答。只依据本页英文 "
        "evidence 与 atom，不结合其他证据、不上网查证；中文仅作辅助。"
        "不要打开含 A/B 标签的审计文件，也不要标 Confidence。"
    )


def load_contract(prepared_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = prepared_dir / "task_manifest.json"
    task_path = prepared_dir / "tasks.jsonl"
    manifest = load_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("Prepared manifest is not complete")
    if manifest.get("adjudication_protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Prepared protocol version mismatch")
    expected_counts = {
        "relation_only_pairs": 39,
        "directness_only_pairs": 4,
        "both_fields_pairs": 82,
        "total_pairs": 125,
        "relation_decisions": 121,
        "directness_decisions": 86,
        "total_field_decisions": 207,
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError(f"Prepared count contract mismatch: {manifest.get('counts')}")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("Prepared config hash mismatch")
    artifact = manifest.get("artifacts", {}).get("tasks.jsonl", {})
    if artifact.get("rows") != EXPECTED_TASKS:
        raise ValueError("Prepared task row count mismatch")
    if artifact.get("sha256") != sha256_file(task_path):
        raise ValueError("Prepared task file hash mismatch")

    config = CONFIG_PATH.read_text(encoding="utf-8")
    validate_label_config(config)
    rows = load_jsonl(task_path)
    if len(rows) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} tasks, found {len(rows)}")
    ids: list[str] = []
    relation_count = 0
    directness_count = 0
    for row in rows:
        assert_blinded(row, f"prepared task {row.get('adjudication_id')}")
        if row.get("adjudication_protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Task protocol mismatch: {row.get('adjudication_id')}")
        fields = row.get("fields_to_adjudicate")
        relation_needed = "gold_relation" in fields
        directness_needed = "gold_directness" in fields
        if bool(row.get("relation_questions")) != relation_needed:
            raise ValueError(f"Relation question mismatch: {row.get('adjudication_id')}")
        if bool(row.get("directness_questions")) != directness_needed:
            raise ValueError(f"Directness question mismatch: {row.get('adjudication_id')}")
        if len(row.get("relation_questions", [])) > 1 or len(
            row.get("directness_questions", [])
        ) > 1:
            raise ValueError(f"Unexpected repeated question count: {row.get('adjudication_id')}")
        relation_count += relation_needed
        directness_count += directness_needed
        ids.append(str(row.get("adjudication_id", "")))
    if len(ids) != EXPECTED_TASKS or len(set(ids)) != EXPECTED_TASKS or any(not item for item in ids):
        raise ValueError("Prepared adjudication IDs must be 125 unique non-empty values")
    if (relation_count, directness_count) != (121, 86):
        raise ValueError("Prepared field decision counts changed")
    return manifest, rows


def validate_existing_project(
    project: Project,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    expected_member_ids: set[int],
) -> dict[str, Any]:
    if project.deleted_at is not None or not project.is_published:
        raise RuntimeError(f"Existing project is archived or unpublished: {project.id}")
    if project.maximum_annotations != 1 or project.sampling != Project.SEQUENCE:
        raise RuntimeError(f"Existing project queue contract mismatch: {project.id}")
    if project.show_skip_button or project.enable_empty_annotation:
        raise RuntimeError(f"Existing project permits skip or empty annotations: {project.id}")
    if project.show_annotation_history or project.show_collab_predictions:
        raise RuntimeError(f"Existing project exposes annotation history or collaboration: {project.id}")
    if project.description != project_description(manifest) or MARKER not in project.description:
        raise RuntimeError(f"Existing project description mismatch: {project.id}")
    if project.label_config != CONFIG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError(f"Existing project config mismatch: {project.id}")
    database_tasks = list(project.tasks.order_by("inner_id", "id"))
    if len(database_tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"Existing project task count mismatch: {project.id}")
    if [task.inner_id for task in database_tasks] != list(range(1, EXPECTED_TASKS + 1)):
        raise RuntimeError(f"Existing project task sequence mismatch: {project.id}")
    for task in database_tasks:
        TaskValidator.check_data(project, dict(task.data))
        assert_blinded(task.data, f"database task {task.id}")
        if task.meta.get("adjudication_marker") != MARKER:
            raise RuntimeError(f"Task marker mismatch: {task.id}")
    if task_fingerprints([task.data for task in database_tasks]) != task_fingerprints(rows):
        raise RuntimeError(f"Existing project task data mismatch: {project.id}")
    memberships = list(
        ProjectMember.objects.filter(project=project).values_list("user_id", "enabled")
    )
    enabled_member_ids = {user_id for user_id, enabled in memberships if enabled}
    if len(memberships) != len(expected_member_ids) or enabled_member_ids != expected_member_ids:
        raise RuntimeError(f"Existing project members mismatch: {project.id} {memberships}")
    annotation_count = project.annotations.filter(was_cancelled=False).count()
    cancelled_count = project.annotations.filter(was_cancelled=True).count()
    labeled_count = project.tasks.filter(is_labeled=True).count()
    draft_count = sum(task.drafts.count() for task in database_tasks)
    return {
        "id": project.id,
        "title": project.title,
        "tasks": len(database_tasks),
        "annotations": annotation_count,
        "cancelled_annotations": cancelled_count,
        "labeled_tasks": labeled_count,
        "drafts": draft_count,
        "config_sha256": sha256_file(CONFIG_PATH),
        "task_data_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
        "members": sorted(enabled_member_ids),
        "path": f"/projects/{project.id}/data",
    }


def inspect_or_plan(
    prepared_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], User, User, Project, dict[str, Any]]:
    expected_database = _bootstrap.database.resolve()
    actual_database = database_path()
    if actual_database != expected_database:
        raise RuntimeError(
            f"Refusing wrong database: expected={expected_database}, actual={actual_database}"
        )
    if not actual_database.exists():
        raise FileNotFoundError(actual_database)
    if quick_check() != "ok" or foreign_key_violation_count() != 0:
        raise RuntimeError("Live database integrity checks failed")
    manifest, rows = load_contract(prepared_dir)
    adjudicator = User.objects.get(email=ADJUDICATOR_EMAIL)
    admin = User.objects.get(email=ADMIN_EMAIL)
    if not adjudicator.is_active or adjudicator.active_organization_id is None:
        raise RuntimeError("Adjudicator is inactive or has no active organization")
    source_project = Project.all_objects.get(id=SOURCE_PROJECT_ID)
    if source_project.organization_id != adjudicator.active_organization_id:
        raise RuntimeError("Source project and adjudicator organization mismatch")
    expected_member_ids = {admin.id, adjudicator.id}
    existing = list(Project.all_objects.filter(title=TITLE))
    marker_matches = list(Project.all_objects.filter(description__contains=MARKER))
    if len(existing) > 1 or len(marker_matches) > 1:
        raise RuntimeError("Duplicate Exp2 adjudication project state")
    if existing and marker_matches and existing[0].id != marker_matches[0].id:
        raise RuntimeError("Title and marker identify different projects")
    project = existing[0] if existing else (marker_matches[0] if marker_matches else None)
    if project is None:
        project_result = {
            "title": TITLE,
            "tasks": EXPECTED_TASKS,
            "config_sha256": sha256_file(CONFIG_PATH),
            "task_data_sha256": hashlib.sha256(
                canonical_json(rows).encode("utf-8")
            ).hexdigest(),
        }
        state = "would_create"
    else:
        if project.title != TITLE:
            raise RuntimeError(f"Marker found under unexpected title: {project.title}")
        project_result = validate_existing_project(
            project, rows, manifest, expected_member_ids
        )
        state = "already_applied"
    result = {
        "marker": MARKER,
        "state": state,
        "database": str(actual_database),
        "database_quick_check": "ok",
        "database_foreign_key_violations": 0,
        "adjudicator": {"id": adjudicator.id, "email": adjudicator.email},
        "admin": {"id": admin.id, "email": admin.email},
        "prepared_manifest_sha256": sha256_file(prepared_dir / "task_manifest.json"),
        "protocol_version": PROTOCOL_VERSION,
        "source_blind_queue_sha256": manifest["source"]["source_blind_queue_sha256"],
        "project": project_result,
    }
    return manifest, rows, adjudicator, admin, source_project, result


def create_project(prepared_dir: Path) -> dict[str, Any]:
    manifest, rows, adjudicator, admin, source_project, plan = inspect_or_plan(
        prepared_dir
    )
    if plan["state"] == "already_applied":
        plan["applied_at_utc"] = None
        return plan
    expected_member_ids = {admin.id, adjudicator.id}
    with transaction.atomic():
        if Project.all_objects.filter(title=TITLE).exists() or Project.all_objects.filter(
            description__contains=MARKER
        ).exists():
            raise RuntimeError("Exp2 adjudication project appeared after dry-run")
        project = Project.objects.create(
            title=TITLE,
            description=project_description(manifest),
            organization=adjudicator.active_organization,
            label_config=CONFIG_PATH.read_text(encoding="utf-8"),
            expert_instruction=project_instruction(),
            show_instruction=True,
            show_skip_button=False,
            reveal_preannotations_interactively=False,
            show_annotation_history=False,
            show_collab_predictions=False,
            evaluate_predictions_automatically=False,
            color=source_project.color,
            created_by=admin,
            maximum_annotations=1,
            min_annotations_to_start_training=0,
            is_draft=False,
            is_published=True,
            sampling=Project.SEQUENCE,
            skip_queue=Project.SkipQueue.REQUEUE_FOR_ME,
            overlap_cohort_percentage=100,
            show_overlap_first=False,
            enable_empty_annotation=False,
        )
        ProjectMember.objects.get_or_create(
            project=project, user=adjudicator, defaults={"enabled": True}
        )
        ProjectMember.objects.get_or_create(
            project=project, user=admin, defaults={"enabled": True}
        )
        Task.objects.bulk_create(
            [
                Task(
                    data=row,
                    meta={
                        "adjudication_marker": MARKER,
                        "adjudication_id": row["adjudication_id"],
                        "adjudication_protocol_version": PROTOCOL_VERSION,
                        "source_blind_queue_sha256": manifest["source"][
                            "source_blind_queue_sha256"
                        ],
                    },
                    project=project,
                    is_labeled=False,
                    allow_skip=False,
                    overlap=1,
                    inner_id=index,
                    total_annotations=0,
                    cancelled_annotations=0,
                    total_predictions=0,
                )
                for index, row in enumerate(rows, start=1)
            ],
            batch_size=100,
        )
        project_result = validate_existing_project(
            project, rows, manifest, expected_member_ids
        )
    if quick_check() != "ok" or foreign_key_violation_count() != 0:
        raise RuntimeError("Database integrity checks failed after creation")
    plan["state"] = "created"
    plan["project"] = project_result
    plan["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["database_quick_check_after"] = "ok"
    plan["database_foreign_key_violations_after"] = 0
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_dir.resolve() != _bootstrap.data_dir.resolve():
        raise RuntimeError("--data-dir changed after Django bootstrap")
    if args.database.resolve() != _bootstrap.database.resolve():
        raise RuntimeError("--database changed after Django bootstrap")
    if args.dry_run:
        *_, result = inspect_or_plan(args.prepared_dir)
    else:
        result = create_project(args.prepared_dir)
    atomic_write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
