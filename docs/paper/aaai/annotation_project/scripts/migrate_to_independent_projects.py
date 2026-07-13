#!/usr/bin/env python3
"""Split mixed Label Studio projects into one complete project per annotator.

Run this script with the Label Studio environment.  It clones every source
task into four independent projects and copies each target annotator's own
submitted annotations into their project.  Source projects are soft-archived
only after all validation checks pass.

The migration is atomic and intentionally refuses to overwrite projects with
the destination titles.  Use ``--dry-run`` first, then ``--apply``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import label_studio  # noqa: E402


LABEL_STUDIO_PACKAGE_DIR = Path(label_studio.__file__).resolve().parent
if str(LABEL_STUDIO_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_STUDIO_PACKAGE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")

import django  # noqa: E402

django.setup()

from django.db import transaction  # noqa: E402
from django.db.models import Count  # noqa: E402
from django.utils import timezone  # noqa: E402
from projects.models import Project, ProjectMember  # noqa: E402
from tasks.models import Annotation, Task  # noqa: E402
from users.models import User  # noqa: E402


MIGRATION_MARKER = "independent-iaa-v1-20260711"


@dataclass(frozen=True)
class Destination:
    source_project_id: int
    source_short_name: str
    title: str
    annotator_email: str
    annotator_short_name: str


DESTINATIONS = (
    Destination(12, "Exp1", "[YULIN ONLY] Exp1-Atom-Quality", "1849812973@qq.com", "Yulin"),
    Destination(12, "Exp1", "[ZHIQIANG ONLY] Exp1-Atom-Quality", "3180643570@qq.com", "Zhiqiang"),
    Destination(13, "Exp2", "[YULIN ONLY] Exp2-Evidence-Map", "1849812973@qq.com", "Yulin"),
    Destination(13, "Exp2", "[ZHIQIANG ONLY] Exp2-Evidence-Map", "3180643570@qq.com", "Zhiqiang"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_fingerprint(task: Task) -> str:
    payload = {
        "inner_id": task.inner_id,
        "data": task.data,
        "meta": task.meta,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def annotation_fingerprint(annotation: Annotation) -> str:
    payload = {
        "task_inner_id": annotation.task.inner_id,
        "completed_by": annotation.completed_by.email if annotation.completed_by else None,
        "was_cancelled": annotation.was_cancelled,
        "result": annotation.result,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def get_sources() -> dict[int, Project]:
    sources = {
        project.id: project
        for project in Project.all_objects.filter(id__in={destination.source_project_id for destination in DESTINATIONS})
    }
    missing = sorted({destination.source_project_id for destination in DESTINATIONS} - set(sources))
    if missing:
        raise RuntimeError(f"Missing source projects: {missing}")
    return sources


def get_annotators() -> dict[str, User]:
    emails = {destination.annotator_email for destination in DESTINATIONS}
    users = {user.email: user for user in User.objects.filter(email__in=emails)}
    missing = sorted(emails - set(users))
    if missing:
        raise RuntimeError(f"Missing annotator accounts: {missing}")
    return users


def inspect_plan() -> dict[str, Any]:
    sources = get_sources()
    annotators = get_annotators()
    plan: dict[str, Any] = {
        "marker": MIGRATION_MARKER,
        "destinations": [],
        "source_projects": {},
    }

    for source_id, source in sorted(sources.items()):
        task_count = source.tasks.count()
        annotation_counts = {
            row["completed_by__email"]: row["count"]
            for row in source.annotations.filter(was_cancelled=False)
            .values("completed_by__email")
            .annotate(count=Count("id"))
        }
        plan["source_projects"][str(source_id)] = {
            "title": source.title,
            "tasks": task_count,
            "annotations_by_user": annotation_counts,
            "already_archived": source.deleted_at is not None,
        }

    existing_titles = set(
        Project.all_objects.filter(title__in=[destination.title for destination in DESTINATIONS]).values_list(
            "title", flat=True
        )
    )
    if existing_titles:
        raise RuntimeError(f"Destination projects already exist: {sorted(existing_titles)}")

    for destination in DESTINATIONS:
        source = sources[destination.source_project_id]
        annotator = annotators[destination.annotator_email]
        annotation_count = source.annotations.filter(completed_by=annotator, was_cancelled=False).count()
        plan["destinations"].append(
            {
                "title": destination.title,
                "source_project_id": source.id,
                "annotator": destination.annotator_email,
                "tasks_to_clone": source.tasks.count(),
                "annotations_to_migrate": annotation_count,
                "remaining_after_migration": source.tasks.count() - annotation_count,
            }
        )
    return plan


def clone_project(source: Project, destination: Destination, annotator: User, admin: User) -> Project:
    description = (
        f"{destination.annotator_short_name} 专用独立双盲标注项目。"
        f"请勿进入另一位标注者的 ONLY 项目。迁移标记: {MIGRATION_MARKER}; "
        f"source_project={source.id}; annotator={annotator.email}"
    )
    project = Project.objects.create(
        title=destination.title,
        description=description,
        organization=source.organization,
        label_config=source.label_config,
        expert_instruction=source.expert_instruction,
        show_instruction=source.show_instruction,
        show_skip_button=source.show_skip_button,
        reveal_preannotations_interactively=source.reveal_preannotations_interactively,
        show_annotation_history=False,
        show_collab_predictions=False,
        evaluate_predictions_automatically=False,
        color=source.color,
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
    ProjectMember.objects.get_or_create(project=project, user=annotator, defaults={"enabled": True})
    ProjectMember.objects.get_or_create(project=project, user=admin, defaults={"enabled": True})
    return project


def clone_tasks(source: Project, destination: Project) -> dict[int, Task]:
    source_tasks = list(source.tasks.order_by("inner_id", "id"))
    cloned_tasks = [
        Task(
            data=copy.deepcopy(source_task.data),
            meta=copy.deepcopy(source_task.meta),
            project=destination,
            is_labeled=False,
            allow_skip=source_task.allow_skip,
            overlap=1,
            inner_id=source_task.inner_id,
            total_annotations=0,
            cancelled_annotations=0,
            total_predictions=0,
        )
        for source_task in source_tasks
    ]
    Task.objects.bulk_create(cloned_tasks, batch_size=500)
    created_by_inner_id = {
        task.inner_id: task for task in destination.tasks.order_by("inner_id", "id")
    }
    if len(created_by_inner_id) != len(source_tasks):
        raise RuntimeError(
            f"Task clone count mismatch for {destination.title}: "
            f"source={len(source_tasks)}, destination={len(created_by_inner_id)}"
        )
    task_map = {source_task.id: created_by_inner_id[source_task.inner_id] for source_task in source_tasks}
    source_fingerprints = [task_fingerprint(task) for task in source_tasks]
    destination_fingerprints = [task_fingerprint(created_by_inner_id[task.inner_id]) for task in source_tasks]
    if source_fingerprints != destination_fingerprints:
        raise RuntimeError(f"Task fingerprints differ for {destination.title}")
    return task_map


def clone_annotations(
    source: Project,
    destination: Project,
    annotator: User,
    task_map: dict[int, Task],
) -> int:
    source_annotations = list(
        source.annotations.filter(completed_by=annotator, was_cancelled=False)
        .select_related("task", "completed_by")
        .order_by("id")
    )
    source_fingerprints = [annotation_fingerprint(annotation) for annotation in source_annotations]
    for source_annotation in source_annotations:
        cloned = Annotation.objects.create(
            result=copy.deepcopy(source_annotation.result),
            was_cancelled=False,
            task=task_map[source_annotation.task_id],
            prediction=copy.deepcopy(source_annotation.prediction),
            lead_time=source_annotation.lead_time,
            completed_by=annotator,
            ground_truth=source_annotation.ground_truth,
            project=destination,
            updated_by=source_annotation.updated_by or annotator,
            last_action=source_annotation.last_action,
            last_created_by=source_annotation.last_created_by,
            draft_created_at=source_annotation.draft_created_at,
            import_id=source_annotation.id,
            bulk_created=True,
        )
        Annotation.objects.filter(pk=cloned.pk).update(
            created_at=source_annotation.created_at,
            updated_at=source_annotation.updated_at,
        )

    destination_annotations = list(
        destination.annotations.filter(completed_by=annotator, was_cancelled=False)
        .select_related("task", "completed_by")
        .order_by("import_id")
    )
    destination_fingerprints = [annotation_fingerprint(annotation) for annotation in destination_annotations]
    if source_fingerprints != destination_fingerprints:
        raise RuntimeError(f"Annotation fingerprints differ for {destination.title}")
    return len(source_annotations)


def export_source_snapshot(sources: dict[int, Project], export_dir: Path) -> list[str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for source_id, source in sorted(sources.items()):
        path = export_dir / f"project_{source_id}_pre_split.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task in source.tasks.prefetch_related("annotations__completed_by").order_by("inner_id", "id"):
                record = {
                    "source_project_id": source.id,
                    "source_project_title": source.title,
                    "task_id": task.id,
                    "inner_id": task.inner_id,
                    "data": task.data,
                    "meta": task.meta,
                    "annotations": [
                        {
                            "annotation_id": annotation.id,
                            "completed_by": annotation.completed_by.email if annotation.completed_by else None,
                            "created_at": annotation.created_at.isoformat(),
                            "updated_at": annotation.updated_at.isoformat(),
                            "lead_time": annotation.lead_time,
                            "was_cancelled": annotation.was_cancelled,
                            "result": annotation.result,
                        }
                        for annotation in task.annotations.all().order_by("id")
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written.append(str(path))
    return written


def apply_migration(export_dir: Path) -> dict[str, Any]:
    sources = get_sources()
    annotators = get_annotators()
    admin = User.objects.get(email="admin@annotation.local")
    existing_titles = set(
        Project.all_objects.filter(title__in=[destination.title for destination in DESTINATIONS]).values_list(
            "title", flat=True
        )
    )
    if existing_titles:
        raise RuntimeError(f"Destination projects already exist: {sorted(existing_titles)}")

    result: dict[str, Any] = {"marker": MIGRATION_MARKER, "projects": [], "exports": []}
    with transaction.atomic():
        for destination in DESTINATIONS:
            source = sources[destination.source_project_id]
            annotator = annotators[destination.annotator_email]
            project = clone_project(source, destination, annotator, admin)
            task_map = clone_tasks(source, project)
            migrated = clone_annotations(source, project, annotator, task_map)
            project.refresh_from_db()
            labeled = project.tasks.filter(is_labeled=True).count()
            if labeled != migrated:
                raise RuntimeError(
                    f"Labeled count mismatch for {project.title}: labeled={labeled}, migrated={migrated}"
                )
            result["projects"].append(
                {
                    "id": project.id,
                    "title": project.title,
                    "annotator": annotator.email,
                    "tasks": project.tasks.count(),
                    "migrated_annotations": migrated,
                    "remaining": project.tasks.count() - migrated,
                }
            )

        for source in sources.values():
            original_title = source.title
            archive_title = f"[ARCHIVE] {original_title}"
            source.title = archive_title[:50]
            source.is_published = False
            source.deleted_at = timezone.now()
            source.deleted_by = admin
            source.purge_at = None
            source.description = (
                f"Soft-archived by {MIGRATION_MARKER}. Original title: {original_title}. "
                "Mixed sequential single-label pilot; retain for audit only."
            )
            source.save(
                update_fields=[
                    "title",
                    "is_published",
                    "deleted_at",
                    "deleted_by",
                    "purge_at",
                    "description",
                ],
                recalc=False,
            )

    result["exports"] = export_source_snapshot(sources, export_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = inspect_plan() if args.dry_run else apply_migration(args.export_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
