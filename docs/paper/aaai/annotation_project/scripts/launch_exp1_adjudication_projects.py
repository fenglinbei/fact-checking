#!/usr/bin/env python3
"""Create or validate the two blinded Exp1 adjudication projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "label_studio_data"
DEFAULT_DATABASE = DEFAULT_DATA_DIR / "label_studio.sqlite3"
DEFAULT_PREPARED_DIR = ROOT / "results" / "exp1_adjudication_v1"
DEFAULT_REPORT = DEFAULT_PREPARED_DIR / "launch_report.json"
MARKER = "exp1-exact-gold-adjudication-v1-20260717"
ADJUDICATOR_EMAIL = "1349410043@qq.com"
ADMIN_EMAIL = "admin@annotation.local"
SOURCE_PROJECT_ID = 18


def _bootstrap_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args, _ = parser.parse_known_args()
    return args


_bootstrap = _bootstrap_arguments()
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


@dataclass(frozen=True)
class AdjudicationProjectSpec:
    key: str
    title: str
    task_file: str
    config_path: Path
    expected_tasks: int


PROJECT_SPECS = (
    AdjudicationProjectSpec(
        "atom",
        "[ZIJIE ONLY] Exp1-Atom-Adjudication-v1",
        "atom_tasks.jsonl",
        ROOT / "config" / "exp1_adjudication_atom.xml",
        37,
    ),
    AdjudicationProjectSpec(
        "completeness",
        "[ZIJIE ONLY] Exp1-Completeness-Adjudication-v1",
        "completeness_tasks.jsonl",
        ROOT / "config" / "exp1_adjudication_completeness.xml",
        10,
    ),
)


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
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def database_path() -> Path:
    return Path(settings.DATABASES["default"]["NAME"]).resolve()


def quick_check() -> str:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA quick_check")
        return str(cursor.fetchone()[0])


def load_contract(prepared_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = prepared_dir / "task_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("Prepared task manifest is not complete")
    if manifest.get("counts") != {"atom": 37, "completeness": 10, "total": 47}:
        raise ValueError(f"Unexpected prepared counts: {manifest.get('counts')}")

    tasks_by_key: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for spec in PROJECT_SPECS:
        config = spec.config_path.read_text(encoding="utf-8")
        validate_label_config(config)
        expected_config_hash = manifest.get("config_sha256", {}).get(spec.key)
        actual_config_hash = sha256_file(spec.config_path)
        if actual_config_hash != expected_config_hash:
            raise ValueError(
                f"Config hash mismatch for {spec.key}: expected={expected_config_hash}, actual={actual_config_hash}"
            )
        task_path = prepared_dir / spec.task_file
        artifact = manifest.get("artifacts", {}).get(spec.task_file, {})
        if sha256_file(task_path) != artifact.get("sha256"):
            raise ValueError(f"Task artifact hash mismatch: {task_path}")
        rows = load_jsonl(task_path)
        if len(rows) != spec.expected_tasks or artifact.get("rows") != spec.expected_tasks:
            raise ValueError(f"Unexpected task count for {spec.key}: {len(rows)}")
        for row in rows:
            if row.get("source_analysis_input_sha256") != manifest.get("source_analysis_input_sha256"):
                raise ValueError(f"Task source hash mismatch: {row.get('adjudication_id')}")
            if row.get("unit") != ("atom" if spec.key == "atom" else "claim"):
                raise ValueError(f"Task unit mismatch: {row.get('adjudication_id')}")
            if any(key in canonical_json(row) for key in ('"annotator_a"', '"annotator_b"', '"disagreements"')):
                raise ValueError(f"Blinding violation: {row.get('adjudication_id')}")
            all_ids.append(str(row.get("adjudication_id", "")))
        tasks_by_key[spec.key] = rows
    if len(all_ids) != 47 or len(set(all_ids)) != 47 or any(not value for value in all_ids):
        raise ValueError("Prepared adjudication IDs must contain 47 unique non-empty values")
    return manifest, tasks_by_key


def project_description(spec: AdjudicationProjectSpec, manifest: dict[str, Any]) -> str:
    return (
        "Zijie 专用 Exp1 独立盲仲裁项目。不得查看 Yulin/Zhiqiang 原标签；"
        "只按页面显示的待仲裁字段独立判断。"
        f" marker={MARKER}; protocol={manifest['adjudication_protocol_version']}; "
        f"source_analysis={manifest['source_analysis_input_sha256']}; "
        f"source_generation={manifest['source_reliability_generation_id']}; kind={spec.key}."
    )


def project_instruction(spec: AdjudicationProjectSpec) -> str:
    if spec.key == "atom":
        return (
            "这是 Exp1 atom 盲仲裁。每页只出现实际存在分歧的 1–2 个问题，均须作答。"
            "请只依据页面中的英文 claim、全部 atoms 与当前 atom 独立判断；中文仅辅助。"
            "不要打开其他标注项目或含 A/B 标签的审计文件。"
        )
    return (
        "这是 Exp1 claim completeness 盲仲裁。请对照英文 claim 与全部 atoms，"
        "独立判断遗漏断言数 0/1/2/3+。不要判断 claim complexity，不要查看 A/B 原标签。"
    )


def task_fingerprints(rows: list[dict[str, Any]]) -> list[str]:
    return [hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest() for row in rows]


def validate_existing_project(
    project: Project,
    spec: AdjudicationProjectSpec,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    expected_member_ids: set[int],
) -> dict[str, Any]:
    if project.deleted_at is not None or not project.is_published:
        raise RuntimeError(f"Existing project is archived or unpublished: {project.id}")
    if project.maximum_annotations != 1 or project.sampling != Project.SEQUENCE:
        raise RuntimeError(f"Existing project queue contract mismatch: {project.id}")
    if project.show_skip_button or project.enable_empty_annotation:
        raise RuntimeError(f"Existing project allows skip or empty annotations: {project.id}")
    if MARKER not in (project.description or ""):
        raise RuntimeError(f"Existing project marker mismatch: {project.id}")
    if project.description != project_description(spec, manifest):
        raise RuntimeError(f"Existing project description mismatch: {project.id}")
    if sha256_file(spec.config_path) != hashlib.sha256(project.label_config.encode("utf-8")).hexdigest():
        raise RuntimeError(f"Existing project config mismatch: {project.id}")
    database_tasks = list(project.tasks.order_by("inner_id", "id"))
    if len(database_tasks) != spec.expected_tasks:
        raise RuntimeError(f"Existing project task count mismatch: {project.id}")
    if [task.inner_id for task in database_tasks] != list(range(1, spec.expected_tasks + 1)):
        raise RuntimeError(f"Existing project inner_id sequence mismatch: {project.id}")
    for task in database_tasks:
        TaskValidator.check_data(project, dict(task.data))
    if task_fingerprints([task.data for task in database_tasks]) != task_fingerprints(rows):
        raise RuntimeError(f"Existing project task data mismatch: {project.id}")
    memberships = list(
        ProjectMember.objects.filter(project=project).values_list("user_id", "enabled")
    )
    member_ids = {user_id for user_id, enabled in memberships if enabled}
    if len(memberships) != len(expected_member_ids) or member_ids != expected_member_ids:
        raise RuntimeError(f"Existing project members mismatch: {project.id} {memberships}")
    annotation_count = project.annotations.filter(was_cancelled=False).count()
    labeled_count = project.tasks.filter(is_labeled=True).count()
    return {
        "key": spec.key,
        "id": project.id,
        "title": project.title,
        "tasks": len(database_tasks),
        "annotations": annotation_count,
        "labeled_tasks": labeled_count,
        "config_sha256": sha256_file(spec.config_path),
        "task_data_sha256": hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest(),
        "path": f"/projects/{project.id}/data",
    }


def inspect_or_plan(
    prepared_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], User, User, Project, dict[str, Any]]:
    expected_database = _bootstrap.database.resolve()
    actual_database = database_path()
    if actual_database != expected_database:
        raise RuntimeError(f"Refusing wrong database: expected={expected_database}, actual={actual_database}")
    if not actual_database.exists():
        raise FileNotFoundError(actual_database)
    if quick_check() != "ok":
        raise RuntimeError("Live database quick_check failed")
    manifest, tasks_by_key = load_contract(prepared_dir)
    adjudicator = User.objects.get(email=ADJUDICATOR_EMAIL)
    admin = User.objects.get(email=ADMIN_EMAIL)
    if not adjudicator.is_active or adjudicator.active_organization_id is None:
        raise RuntimeError("Adjudicator account is inactive or has no active organization")
    source_project = Project.all_objects.get(id=SOURCE_PROJECT_ID)
    if source_project.organization_id != adjudicator.active_organization_id:
        raise RuntimeError("Source project/adjudicator organization mismatch")
    titles = [spec.title for spec in PROJECT_SPECS]
    existing = {
        project.title: project
        for project in Project.all_objects.filter(title__in=titles)
    }
    expected_member_ids = {admin.id, adjudicator.id}
    if existing and len(existing) != len(PROJECT_SPECS):
        raise RuntimeError(f"Partial adjudication project state: {sorted(existing)}")
    if existing:
        projects = [
            validate_existing_project(
                existing[spec.title], spec, tasks_by_key[spec.key], manifest, expected_member_ids
            )
            for spec in PROJECT_SPECS
        ]
        state = "already_applied"
    else:
        projects = [
            {
                "key": spec.key,
                "title": spec.title,
                "tasks": spec.expected_tasks,
                "config_sha256": sha256_file(spec.config_path),
                "task_data_sha256": hashlib.sha256(
                    canonical_json(tasks_by_key[spec.key]).encode("utf-8")
                ).hexdigest(),
            }
            for spec in PROJECT_SPECS
        ]
        state = "would_create"
    result = {
        "marker": MARKER,
        "state": state,
        "database": str(actual_database),
        "database_quick_check": "ok",
        "adjudicator": {"id": adjudicator.id, "email": adjudicator.email},
        "admin": {"id": admin.id, "email": admin.email},
        "prepared_manifest_sha256": sha256_file(prepared_dir / "task_manifest.json"),
        "protocol_version": manifest["adjudication_protocol_version"],
        "source_analysis_input_sha256": manifest["source_analysis_input_sha256"],
        "projects": projects,
    }
    return manifest, tasks_by_key, adjudicator, admin, source_project, result


def create_projects(prepared_dir: Path) -> dict[str, Any]:
    manifest, tasks_by_key, adjudicator, admin, source_project, plan = inspect_or_plan(prepared_dir)
    if plan["state"] == "already_applied":
        plan["applied_at_utc"] = None
        return plan

    expected_member_ids = {admin.id, adjudicator.id}
    created: list[dict[str, Any]] = []
    with transaction.atomic():
        if Project.all_objects.filter(title__in=[spec.title for spec in PROJECT_SPECS]).exists():
            raise RuntimeError("Adjudication project titles appeared after dry-run; retry inspection")
        for spec in PROJECT_SPECS:
            config = spec.config_path.read_text(encoding="utf-8")
            project = Project.objects.create(
                title=spec.title,
                description=project_description(spec, manifest),
                organization=adjudicator.active_organization,
                label_config=config,
                expert_instruction=project_instruction(spec),
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
            rows = tasks_by_key[spec.key]
            Task.objects.bulk_create(
                [
                    Task(
                        data=row,
                        meta={
                            "adjudication_marker": MARKER,
                            "adjudication_id": row["adjudication_id"],
                            "source_analysis_input_sha256": manifest["source_analysis_input_sha256"],
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
            created.append(
                validate_existing_project(
                    project, spec, rows, manifest, expected_member_ids
                )
            )
    if quick_check() != "ok":
        raise RuntimeError("Database quick_check failed after project creation")
    plan["state"] = "created"
    plan["projects"] = created
    plan["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["database_quick_check_after"] = "ok"
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
        result = create_projects(args.prepared_dir)
    atomic_write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
