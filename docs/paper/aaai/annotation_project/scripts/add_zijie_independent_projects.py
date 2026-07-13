#!/usr/bin/env python3
"""Add Zijie's independent Exp1/Exp2 projects after the initial split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from django.db import transaction

from migrate_to_independent_projects import (
    Destination,
    Project,
    User,
    clone_annotations,
    clone_project,
    clone_tasks,
)


ZIJIE_EMAIL = "1349410043@qq.com"
DESTINATIONS = (
    Destination(12, "Exp1", "[ZIJIE ONLY] Exp1-Atom-Quality", ZIJIE_EMAIL, "Zijie"),
    Destination(13, "Exp2", "[ZIJIE ONLY] Exp2-Evidence-Map", ZIJIE_EMAIL, "Zijie"),
)


def inspect_plan() -> dict:
    user = User.objects.get(email=ZIJIE_EMAIL)
    existing = list(
        Project.all_objects.filter(title__in=[item.title for item in DESTINATIONS]).values_list("title", flat=True)
    )
    if existing:
        raise RuntimeError(f"Destination projects already exist: {existing}")
    projects = []
    for item in DESTINATIONS:
        source = Project.all_objects.get(id=item.source_project_id)
        annotations = source.annotations.filter(completed_by=user, was_cancelled=False).count()
        projects.append(
            {
                "title": item.title,
                "source_project_id": source.id,
                "tasks": source.tasks.count(),
                "annotations_to_migrate": annotations,
                "remaining": source.tasks.count() - annotations,
            }
        )
    return {"annotator": ZIJIE_EMAIL, "projects": projects}


def apply() -> dict:
    plan = inspect_plan()
    user = User.objects.get(email=ZIJIE_EMAIL)
    admin = User.objects.get(email="admin@annotation.local")
    created = []
    with transaction.atomic():
        if not user.first_name:
            user.first_name = "ZIJIE"
        if not user.last_name:
            user.last_name = "LIAO"
        user.save(update_fields=["first_name", "last_name"])

        for item in DESTINATIONS:
            source = Project.all_objects.get(id=item.source_project_id)
            project = clone_project(source, item, user, admin)
            task_map = clone_tasks(source, project)
            migrated = clone_annotations(source, project, user, task_map)
            labeled = project.tasks.filter(is_labeled=True).count()
            if labeled != migrated:
                raise RuntimeError(
                    f"Labeled count mismatch for {project.title}: labeled={labeled}, migrated={migrated}"
                )
            created.append(
                {
                    "id": project.id,
                    "title": project.title,
                    "tasks": project.tasks.count(),
                    "migrated_annotations": migrated,
                    "remaining": project.tasks.count() - migrated,
                }
            )
    return {"annotator": ZIJIE_EMAIL, "profile_name": "ZIJIE LIAO", "projects": created, "plan": plan}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = inspect_plan() if args.dry_run else apply()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
