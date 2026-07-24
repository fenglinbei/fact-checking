#!/usr/bin/env python3
"""Safely create or validate the staged blinded trace-alignment projects.

The default invocation is a transactional dry-run for the two per-person
pilot preference projects.  Database changes require ``--apply``.  Later
cohorts/stages are gated by completion of the preceding stage and an explicit
operator confirmation that the relevant interface/exclusion rules are frozen.
Transition projects are unpublished unless ``--publish-transition`` is given.
Every project belongs to an explicit revision namespace.  A formal stage can
only consume completed prerequisites from the same revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "label_studio_data"
DEFAULT_DATABASE = DEFAULT_DATA_DIR / "label_studio.sqlite3"
DEFAULT_PREPARED_DIR = ROOT / "results" / "exp3_trace_alignment_v1"
DEFAULT_BACKUP_DIR = DEFAULT_DATA_DIR / "backups"
PREFERENCE_CONFIG = ROOT / "config" / "exp3_trace_preference.xml"
TRANSITION_CONFIG = ROOT / "config" / "exp4_transition_audit.xml"

MARKER = "trace-alignment-human-eval-20260724"
ADMIN_EMAIL = "admin@annotation.local"
DEFAULT_REVISION = "v1"
REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,4}$")

PREFERENCE_FIELDS = frozenset(
    {
        "blind_task_id",
        "claim_en",
        "claim_zh",
        "sequence_a_html",
        "sequence_b_html",
    }
)
TRANSITION_FIELDS = frozenset(
    {
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
)
ALLOWED_HTML_TAGS = frozenset({"div", "p", "ol", "li", "span", "strong", "br"})
VOID_HTML_TAGS = frozenset({"br"})
METHOD_LEAK_RE = re.compile(
    r"(?:\bevitrace\b|\bsource[-_ ]?score\b|\bs4\b|learned[_ -]?marginal|"
    r"atom[_ -]?anchor|method[_ -]?to[_ -]?side)",
    re.IGNORECASE,
)
CHECK_CUE_RE = re.compile(r"\bcheck\s*:", re.IGNORECASE)
TRANSITION_RE = re.compile(r"^[USRQC]\s*(?:→|->)\s*[USRQC]$")


@dataclass(frozen=True)
class Annotator:
    short_name: str
    email: str


ANNOTATORS = (
    Annotator("YULIN", "1849812973@qq.com"),
    Annotator("ZHIQIANG", "3180643570@qq.com"),
)


def validate_revision(value: str) -> str:
    revision = str(value or "").strip()
    if not REVISION_RE.fullmatch(revision):
        raise argparse.ArgumentTypeError(
            "revision must be 1-5 characters using only letters, digits, '_' or '-', "
            "and must start with a letter or digit"
        )
    return revision


@dataclass(frozen=True)
class TaskArtifact:
    cohort: str
    stage: str
    filename: str
    expected_rows: int
    public_fields: frozenset[str]
    config_path: Path


TASK_ARTIFACTS = (
    TaskArtifact(
        "pilot",
        "preference",
        "pilot_preference_tasks.jsonl",
        30,
        PREFERENCE_FIELDS,
        PREFERENCE_CONFIG,
    ),
    TaskArtifact(
        "formal",
        "preference",
        "preference_tasks.jsonl",
        200,
        PREFERENCE_FIELDS,
        PREFERENCE_CONFIG,
    ),
    TaskArtifact(
        "pilot",
        "transition",
        "pilot_transition_tasks.jsonl",
        15,
        TRANSITION_FIELDS,
        TRANSITION_CONFIG,
    ),
    TaskArtifact(
        "formal",
        "transition",
        "transition_tasks.jsonl",
        100,
        TRANSITION_FIELDS,
        TRANSITION_CONFIG,
    ),
)
ARTIFACT_BY_STAGE = {
    (artifact.cohort, artifact.stage): artifact for artifact in TASK_ARTIFACTS
}


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
os.environ.setdefault("COLLECT_ANALYTICS", "false")
os.environ.setdefault("SENTRY_DSN", "")
os.environ.setdefault("FRONTEND_SENTRY_DSN", "")
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
class PreparedContract:
    prepared_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    order_namespace: str
    rows: dict[tuple[str, str], list[dict[str, Any]]]
    artifact_sha256: dict[tuple[str, str], str]


class NeutralHTMLValidator(HTMLParser):
    """Accept the exporter's small neutral HTML subset and reject active markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.tag_counts: dict[str, int] = {}
        self.class_counts: dict[str, int] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        normalized = tag.lower()
        if normalized not in ALLOWED_HTML_TAGS:
            raise ValueError(f"Disallowed HTML tag: {tag}")
        self.tag_counts[normalized] = self.tag_counts.get(normalized, 0) + 1
        for name, value in attrs:
            if name.lower() != "class":
                raise ValueError(f"Disallowed HTML attribute: {name}")
            if value and METHOD_LEAK_RE.search(value):
                raise ValueError("Method-specific HTML class leaked")
            for class_name in (value or "").split():
                self.class_counts[class_name] = (
                    self.class_counts.get(class_name, 0) + 1
                )
        if normalized not in VOID_HTML_TAGS:
            self.stack.append(normalized)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_HTML_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in VOID_HTML_TAGS:
            return
        if not self.stack or self.stack[-1] != normalized:
            raise ValueError(f"Unbalanced HTML closing tag: {tag}")
        self.stack.pop()

    def close(self) -> None:
        super().close()
        if self.stack:
            raise ValueError(f"Unclosed HTML tag(s): {self.stack}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


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
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def database_path() -> Path:
    return Path(settings.DATABASES["default"]["NAME"]).resolve()


def quick_check() -> str:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA quick_check")
        row = cursor.fetchone()
    return str(row[0]) if row else ""


def foreign_key_violation_count() -> int:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA foreign_key_check")
        return len(cursor.fetchall())


def assert_database_contract() -> Path:
    expected = _bootstrap.database.resolve()
    actual = database_path()
    if actual != expected:
        raise RuntimeError(
            f"Refusing unexpected database: expected={expected}, actual={actual}"
        )
    if not actual.is_file():
        raise FileNotFoundError(actual)
    if quick_check() != "ok":
        raise RuntimeError("Database PRAGMA quick_check failed")
    violations = foreign_key_violation_count()
    if violations:
        raise RuntimeError(f"Database has {violations} foreign-key violation(s)")
    return actual


def validate_neutral_html(value: str, context: str) -> NeutralHTMLValidator:
    if METHOD_LEAK_RE.search(value):
        raise ValueError(f"Method name leaked in {context}")
    parser = NeutralHTMLValidator()
    try:
        parser.feed(value)
        parser.close()
    except ValueError as error:
        raise ValueError(f"Unsafe or malformed HTML in {context}: {error}") from error
    return parser


def validate_evidence_translations(
    parser: NeutralHTMLValidator, context: str
) -> None:
    evidence_count = parser.tag_counts.get("li", 0)
    translation_count = parser.class_counts.get("evidence-zh", 0)
    if evidence_count <= 0 or translation_count != evidence_count:
        raise ValueError(
            f"Every displayed evidence item must contain its cached Chinese "
            f"translation in {context}: evidence={evidence_count}, "
            f"translations={translation_count}"
        )


def validate_public_task(
    row: dict[str, Any],
    artifact: TaskArtifact,
    context: str,
) -> str:
    if set(row) != artifact.public_fields:
        missing = sorted(artifact.public_fields - set(row))
        extra = sorted(set(row) - artifact.public_fields)
        raise ValueError(
            f"Public task schema mismatch in {context}: missing={missing}, extra={extra}"
        )
    for key, value in row.items():
        if not isinstance(value, str):
            raise ValueError(f"{context}.{key} must be a string")
    blind_task_id = row["blind_task_id"].strip()
    if not blind_task_id:
        raise ValueError(f"Empty blind_task_id in {context}")
    if not row["claim_en"].strip() or not row["claim_zh"].strip():
        raise ValueError(f"Missing claim text or cached translation in {context}")
    serialized = canonical_json(row)
    if METHOD_LEAK_RE.search(serialized):
        raise ValueError(f"Method identity leaked in {context}")

    if artifact.stage == "preference":
        if CHECK_CUE_RE.search(serialized):
            raise ValueError(f"Check cue leaked in preference task {context}")
        a_parser = validate_neutral_html(
            row["sequence_a_html"], f"{context}.sequence_a_html"
        )
        b_parser = validate_neutral_html(
            row["sequence_b_html"], f"{context}.sequence_b_html"
        )
        validate_evidence_translations(a_parser, f"{context}.sequence_a_html")
        validate_evidence_translations(b_parser, f"{context}.sequence_b_html")
    else:
        if not row["focal_atom_en"].strip() or not row["focal_atom_zh"].strip():
            raise ValueError(
                f"Missing focal proposition or cached translation in {context}"
            )
        validate_neutral_html(
            row["state_legend_html"], f"{context}.state_legend_html"
        )
        prior_parser = validate_neutral_html(
            row["prior_evidence_html"], f"{context}.prior_evidence_html"
        )
        if prior_parser.tag_counts.get("li", 0):
            validate_evidence_translations(
                prior_parser, f"{context}.prior_evidence_html"
            )
        current_parser = validate_neutral_html(
            row["current_evidence_html"], f"{context}.current_evidence_html"
        )
        validate_evidence_translations(
            current_parser, f"{context}.current_evidence_html"
        )
        if not TRANSITION_RE.fullmatch(row["proposed_transition"].strip()):
            raise ValueError(
                f"Invalid proposed_transition in {context}: "
                f"{row['proposed_transition']!r}"
            )
    return blind_task_id


def resolve_artifact_entry(
    artifacts: dict[str, Any], filename: str
) -> tuple[str, dict[str, Any]]:
    candidates = [
        (str(value.get("path", name)), value)
        for name, value in artifacts.items()
        if isinstance(value, dict)
        and Path(str(value.get("path", name))).name == filename
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Manifest must contain exactly one artifact named {filename}: "
            f"found={len(candidates)}"
        )
    return candidates[0]


def artifact_path(prepared_dir: Path, relative_name: str) -> Path:
    candidate = (prepared_dir / relative_name).resolve()
    root = prepared_dir.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Artifact escapes prepared directory: {relative_name}")
    return candidate


def manifest_config_hash(
    manifest: dict[str, Any], stage: str
) -> str | None:
    declared = manifest.get("config_sha256")
    if not isinstance(declared, dict):
        return None
    config_path = PREFERENCE_CONFIG if stage == "preference" else TRANSITION_CONFIG
    matches = [
        value
        for key, value in declared.items()
        if key == stage or Path(str(key)).name == config_path.name
    ]
    if len(matches) > 1 and len(set(matches)) != 1:
        raise ValueError(f"Conflicting config hashes for {stage}")
    return str(matches[0]) if matches else None


def load_prepared_contract(prepared_dir: Path) -> PreparedContract:
    prepared_dir = prepared_dir.resolve()
    manifest_path = prepared_dir / "task_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("Prepared task_manifest.json must have complete=true")
    if manifest.get("annotation_complete") is True:
        raise ValueError(
            "Prepared task manifest unexpectedly claims annotation_complete=true"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("task_manifest.json must contain an artifacts mapping")

    for stage, config_path in (
        ("preference", PREFERENCE_CONFIG),
        ("transition", TRANSITION_CONFIG),
    ):
        config_text = config_path.read_text(encoding="utf-8")
        validate_label_config(config_text)
        declared_hash = manifest_config_hash(manifest, stage)
        if declared_hash is not None and declared_hash != sha256_file(config_path):
            raise ValueError(f"{stage} config hash differs from prepared manifest")

    rows_by_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    artifact_hashes: dict[tuple[str, str], str] = {}
    all_blind_ids: set[str] = set()
    for artifact in TASK_ARTIFACTS:
        relative_name, record = resolve_artifact_entry(artifacts, artifact.filename)
        path = artifact_path(prepared_dir, relative_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = record.get("sha256")
        if expected_hash != sha256_file(path):
            raise ValueError(f"Artifact hash mismatch: {path}")
        if record.get("rows") != artifact.expected_rows:
            raise ValueError(
                f"Manifest row count mismatch for {path}: "
                f"expected={artifact.expected_rows}, manifest={record.get('rows')}"
            )
        if record.get("visibility") not in (None, "public"):
            raise ValueError(f"Task artifact must be public: {path}")
        rows = load_jsonl(path)
        if len(rows) != artifact.expected_rows:
            raise ValueError(
                f"Task row count mismatch for {path}: "
                f"expected={artifact.expected_rows}, actual={len(rows)}"
            )
        for index, row in enumerate(rows, start=1):
            blind_id = validate_public_task(
                row, artifact, f"{artifact.filename}:{index}"
            )
            if blind_id in all_blind_ids:
                raise ValueError(f"Duplicate blind_task_id across artifacts: {blind_id}")
            all_blind_ids.add(blind_id)
        rows_by_stage[(artifact.cohort, artifact.stage)] = rows
        artifact_hashes[(artifact.cohort, artifact.stage)] = str(expected_hash)

    manifest_sha256 = sha256_file(manifest_path)
    private_commitments = manifest.get("private_key_commitments")
    if not isinstance(private_commitments, dict):
        private_commitments = {}
    order_namespace = str(
        private_commitments.get("blind_seed_sha256")
        or manifest.get("blind_seed_sha256")
        or manifest.get("private_key_commitment_sha256")
        or manifest.get("blinding_commitment_sha256")
        or manifest_sha256
    )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", order_namespace):
        order_namespace = sha256_bytes(order_namespace.encode("utf-8"))
    return PreparedContract(
        prepared_dir=prepared_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        order_namespace=order_namespace.lower(),
        rows=rows_by_stage,
        artifact_sha256=artifact_hashes,
    )


def project_title(
    annotator: Annotator, cohort: str, stage: str, revision: str
) -> str:
    cohort_name = "Pilot" if cohort == "pilot" else "Formal"
    experiment = (
        "Exp3-Trace-Preference"
        if stage == "preference"
        else "Exp4-Transition-Audit"
    )
    return f"[{annotator.short_name} ONLY] {cohort_name}-{experiment}-{revision}"


def stage_marker(cohort: str, stage: str, revision: str) -> str:
    return f"{MARKER}:{revision}:{cohort}:{stage}"


def project_description(
    annotator: Annotator,
    artifact: TaskArtifact,
    contract: PreparedContract,
    revision: str,
) -> str:
    if artifact.stage == "preference":
        instruction = (
            "Blinded evidence-sequence preference. Use only this personal ONLY "
            "project; do not discuss labels with the other annotator."
        )
    else:
        instruction = (
            "Blinded state-transition audit. This project stays unpublished "
            "until the preceding preference stage and freeze gate are complete."
        )
    return (
        f"{instruction} "
        f"marker={stage_marker(artifact.cohort, artifact.stage, revision)}; "
        f"revision={revision}; "
        f"annotator={annotator.email}; "
        f"task_manifest_sha256={contract.manifest_sha256}."
    )


def project_instruction(stage: str) -> str:
    if stage == "preference":
        return (
            "English is authoritative and Chinese is auxiliary. Choose the ordered "
            "evidence sequence that better supports accurate, well-grounded "
            "fact-checking. Do not choose by writing style or persuasiveness. "
            "Do not try to infer which system produced either side."
        )
    return (
        "English is authoritative and Chinese is auxiliary. Judge whether the "
        "current evidence warrants the displayed state update and how much new "
        "decision-relevant information it adds beyond the earlier prefix. Do not "
        "open any private key or another annotator's project."
    )


def ordered_rows(
    rows: list[dict[str, Any]],
    annotator: Annotator,
    artifact: TaskArtifact,
    contract: PreparedContract,
    revision: str,
) -> list[dict[str, Any]]:
    namespace = (
        f"{contract.order_namespace}\0{revision}\0"
        f"{artifact.cohort}\0{artifact.stage}\0"
        f"{annotator.email}\0"
    )
    return sorted(
        rows,
        key=lambda row: sha256_bytes(
            (namespace + row["blind_task_id"]).encode("utf-8")
        ),
    )


def task_order_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    return sha256_json([row["blind_task_id"] for row in rows])


def expected_member_ids(admin: User, annotator: User) -> set[int]:
    return {admin.id, annotator.id}


def load_users() -> tuple[User, dict[str, User]]:
    emails = {ADMIN_EMAIL, *(annotator.email for annotator in ANNOTATORS)}
    users = {user.email: user for user in User.objects.filter(email__in=emails)}
    missing = sorted(emails - set(users))
    if missing:
        raise RuntimeError(f"Missing Label Studio account(s): {missing}")
    admin = users[ADMIN_EMAIL]
    annotator_users = {
        annotator.email: users[annotator.email] for annotator in ANNOTATORS
    }
    organization_ids = {
        user.active_organization_id for user in [admin, *annotator_users.values()]
    }
    if None in organization_ids or len(organization_ids) != 1:
        raise RuntimeError(
            "Admin and both annotators must have the same active organization"
        )
    inactive = sorted(user.email for user in users.values() if not user.is_active)
    if inactive:
        raise RuntimeError(f"Inactive Label Studio account(s): {inactive}")
    return admin, annotator_users


def locate_project_pair(
    artifact: TaskArtifact,
    revision: str,
) -> dict[str, Project] | None:
    titles = {
        annotator.email: project_title(
            annotator, artifact.cohort, artifact.stage, revision
        )
        for annotator in ANNOTATORS
    }
    title_matches = list(Project.all_objects.filter(title__in=titles.values()))
    marker_matches = list(
        Project.all_objects.filter(
            description__contains=stage_marker(
                artifact.cohort, artifact.stage, revision
            )
        )
    )
    if not title_matches and not marker_matches:
        return None
    title_by_id = {project.id: project for project in title_matches}
    marker_by_id = {project.id: project for project in marker_matches}
    if set(title_by_id) != set(marker_by_id):
        raise RuntimeError(
            "Project title and marker identify different or partial project sets"
        )
    by_email: dict[str, Project] = {}
    for annotator in ANNOTATORS:
        matches = [
            project
            for project in title_matches
            if project.title == titles[annotator.email]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one project titled {titles[annotator.email]!r}"
            )
        by_email[annotator.email] = matches[0]
    if len(set(project.id for project in by_email.values())) != len(ANNOTATORS):
        raise RuntimeError("The two annotators resolved to the same project")
    return by_email


def active_annotation_summary(
    project: Project, annotator: Annotator
) -> dict[str, Any]:
    active = list(
        project.annotations.filter(was_cancelled=False).values_list(
            "task_id", "completed_by__email"
        )
    )
    unexpected = sorted(
        {
            "<missing>" if email is None else str(email)
            for _, email in active
            if email != annotator.email
        }
    )
    if unexpected:
        raise RuntimeError(
            f"Cross-account annotation(s) in project {project.id}: {unexpected}"
        )
    task_ids = [task_id for task_id, _ in active]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(f"Duplicate active completion in project {project.id}")
    all_unexpected = sorted(
        {
            "<missing>" if email is None else str(email)
            for email in project.annotations.values_list(
                "completed_by__email", flat=True
            )
            if email != annotator.email
        }
    )
    if all_unexpected:
        raise RuntimeError(
            f"Unexpected annotator has completion rows in project {project.id}: "
            f"{all_unexpected}"
        )
    return {
        "active_annotations": len(active),
        "unique_completed_tasks": len(set(task_ids)),
        "labeled_tasks": project.tasks.filter(is_labeled=True).count(),
        "drafts": sum(task.drafts.count() for task in project.tasks.all()),
    }


def validate_project(
    project: Project,
    annotator: Annotator,
    annotator_user: User,
    admin: User,
    artifact: TaskArtifact,
    contract: PreparedContract,
    revision: str,
    *,
    expected_published: bool | None,
) -> dict[str, Any]:
    if project.deleted_at is not None:
        raise RuntimeError(f"Project is archived: {project.id}")
    if project.title != project_title(
        annotator, artifact.cohort, artifact.stage, revision
    ):
        raise RuntimeError(f"Project title mismatch: {project.id}")
    if project.description != project_description(
        annotator, artifact, contract, revision
    ):
        raise RuntimeError(f"Project description mismatch: {project.id}")
    if expected_published is not None and project.is_published != expected_published:
        raise RuntimeError(
            f"Project publication state mismatch: {project.id} "
            f"expected={expected_published}, actual={project.is_published}"
        )
    if artifact.stage == "preference" and not project.is_published:
        raise RuntimeError(f"Preference project must be published: {project.id}")
    if project.maximum_annotations != 1 or project.sampling != Project.SEQUENCE:
        raise RuntimeError(f"Queue contract mismatch: {project.id}")
    if (
        project.show_skip_button
        or project.enable_empty_annotation
        or project.show_annotation_history
        or project.show_collab_predictions
        or project.reveal_preannotations_interactively
    ):
        raise RuntimeError(f"Blinding/skip contract mismatch: {project.id}")
    expected_config = artifact.config_path.read_text(encoding="utf-8")
    if project.label_config != expected_config:
        raise RuntimeError(f"Label config mismatch: {project.id}")
    if project.expert_instruction != project_instruction(artifact.stage):
        raise RuntimeError(f"Expert instruction mismatch: {project.id}")

    rows = ordered_rows(
        contract.rows[(artifact.cohort, artifact.stage)],
        annotator,
        artifact,
        contract,
        revision,
    )
    database_tasks = list(project.tasks.order_by("inner_id", "id"))
    if len(database_tasks) != artifact.expected_rows:
        raise RuntimeError(f"Task count mismatch: {project.id}")
    if [task.inner_id for task in database_tasks] != list(
        range(1, artifact.expected_rows + 1)
    ):
        raise RuntimeError(f"Task inner_id sequence mismatch: {project.id}")
    for expected_row, task in zip(rows, database_tasks, strict=True):
        TaskValidator.check_data(project, dict(task.data))
        if task.allow_skip or task.overlap != 1:
            raise RuntimeError(f"Task queue flags mismatch: {task.id}")
        expected_meta = {
            "trace_alignment_marker": stage_marker(
                artifact.cohort, artifact.stage, revision
            ),
            "blind_task_id": expected_row["blind_task_id"],
            "source_task_sha256": sha256_json(expected_row),
            "task_manifest_sha256": contract.manifest_sha256,
        }
        if task.meta != expected_meta:
            raise RuntimeError(f"Task metadata mismatch: {task.id}")
    if sha256_json([task.data for task in database_tasks]) != sha256_json(rows):
        raise RuntimeError(f"Task data/order fingerprint mismatch: {project.id}")

    memberships = list(
        ProjectMember.objects.filter(project=project).values_list(
            "user_id", "enabled"
        )
    )
    enabled_ids = {user_id for user_id, enabled in memberships if enabled}
    expected_ids = expected_member_ids(admin, annotator_user)
    if len(memberships) != len(expected_ids) or enabled_ids != expected_ids:
        raise RuntimeError(
            f"Project membership mismatch: {project.id} {memberships}"
        )
    annotation_summary = active_annotation_summary(project, annotator)
    return {
        "id": project.id,
        "title": project.title,
        "annotator": annotator.email,
        "cohort": artifact.cohort,
        "stage": artifact.stage,
        "revision": revision,
        "published": project.is_published,
        "maximum_annotations": project.maximum_annotations,
        "tasks": len(database_tasks),
        "task_order_sha256": task_order_fingerprint(rows),
        "task_data_sha256": sha256_json(rows),
        "config_sha256": sha256_file(artifact.config_path),
        **annotation_summary,
        "path": f"/projects/{project.id}/data",
    }


def create_project(
    annotator: Annotator,
    annotator_user: User,
    admin: User,
    artifact: TaskArtifact,
    contract: PreparedContract,
    revision: str,
    *,
    published: bool,
) -> Project:
    project = Project.objects.create(
        title=project_title(
            annotator, artifact.cohort, artifact.stage, revision
        ),
        description=project_description(
            annotator, artifact, contract, revision
        ),
        organization=annotator_user.active_organization,
        label_config=artifact.config_path.read_text(encoding="utf-8"),
        expert_instruction=project_instruction(artifact.stage),
        show_instruction=True,
        show_skip_button=False,
        reveal_preannotations_interactively=False,
        show_annotation_history=False,
        show_collab_predictions=False,
        evaluate_predictions_automatically=False,
        created_by=admin,
        maximum_annotations=1,
        min_annotations_to_start_training=0,
        is_draft=False,
        is_published=published,
        sampling=Project.SEQUENCE,
        skip_queue=Project.SkipQueue.REQUEUE_FOR_ME,
        overlap_cohort_percentage=100,
        show_overlap_first=False,
        enable_empty_annotation=False,
    )
    ProjectMember.objects.get_or_create(
        project=project, user=annotator_user, defaults={"enabled": True}
    )
    ProjectMember.objects.get_or_create(
        project=project, user=admin, defaults={"enabled": True}
    )
    rows = ordered_rows(
        contract.rows[(artifact.cohort, artifact.stage)],
        annotator,
        artifact,
        contract,
        revision,
    )
    Task.objects.bulk_create(
        [
            Task(
                data=row,
                meta={
                    "trace_alignment_marker": stage_marker(
                        artifact.cohort, artifact.stage, revision
                    ),
                    "blind_task_id": row["blind_task_id"],
                    "source_task_sha256": sha256_json(row),
                    "task_manifest_sha256": contract.manifest_sha256,
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
        batch_size=200,
    )
    return project


def project_pair_complete(
    artifact: TaskArtifact,
    contract: PreparedContract,
    admin: User,
    annotator_users: dict[str, User],
    revision: str,
) -> tuple[bool, list[dict[str, Any]]]:
    projects = locate_project_pair(artifact, revision)
    if projects is None:
        return False, []
    reports: list[dict[str, Any]] = []
    complete = True
    for annotator in ANNOTATORS:
        report = validate_project(
            projects[annotator.email],
            annotator,
            annotator_users[annotator.email],
            admin,
            artifact,
            contract,
            revision,
            expected_published=True,
        )
        reports.append(report)
        if (
            report["active_annotations"] != artifact.expected_rows
            or report["unique_completed_tasks"] != artifact.expected_rows
            or report["labeled_tasks"] != artifact.expected_rows
            or report["drafts"] != 0
        ):
            complete = False
    return complete, reports


def prerequisite_artifacts(
    artifact: TaskArtifact,
) -> tuple[TaskArtifact, ...]:
    if (artifact.cohort, artifact.stage) == ("pilot", "preference"):
        return ()
    if (artifact.cohort, artifact.stage) == ("formal", "preference"):
        return (ARTIFACT_BY_STAGE[("pilot", "preference")],)
    if (artifact.cohort, artifact.stage) == ("pilot", "transition"):
        return (
            ARTIFACT_BY_STAGE[("pilot", "preference")],
            ARTIFACT_BY_STAGE[("formal", "preference")],
        )
    return (
        ARTIFACT_BY_STAGE[("pilot", "preference")],
        ARTIFACT_BY_STAGE[("formal", "preference")],
        ARTIFACT_BY_STAGE[("pilot", "transition")],
    )


def gate_report(
    artifact: TaskArtifact,
    contract: PreparedContract,
    admin: User,
    annotator_users: dict[str, User],
    revision: str,
    *,
    confirmation: bool,
) -> dict[str, Any]:
    prerequisites = prerequisite_artifacts(artifact)
    if not prerequisites:
        return {
            "required": False,
            "satisfied": True,
            "revision": revision,
            "confirmation_received": confirmation,
            "prerequisite": None,
            "prerequisites": [],
        }
    prerequisite_reports = []
    all_complete = True
    for prerequisite in prerequisites:
        complete, reports = project_pair_complete(
            prerequisite, contract, admin, annotator_users, revision
        )
        all_complete = all_complete and complete
        prerequisite_reports.append(
            {
                "cohort": prerequisite.cohort,
                "stage": prerequisite.stage,
                "revision": revision,
                "complete": complete,
                "projects": reports,
            }
        )
    return {
        "required": True,
        "satisfied": all_complete and confirmation,
        "revision": revision,
        "confirmation_received": confirmation,
        # Keep the immediate predecessor at this stable key for concise runbooks.
        "prerequisite": prerequisite_reports[-1],
        "prerequisites": prerequisite_reports,
    }


def validate_pair(
    projects: dict[str, Project],
    artifact: TaskArtifact,
    contract: PreparedContract,
    admin: User,
    annotator_users: dict[str, User],
    revision: str,
    *,
    expected_published: bool | None,
) -> list[dict[str, Any]]:
    reports = [
        validate_project(
            projects[annotator.email],
            annotator,
            annotator_users[annotator.email],
            admin,
            artifact,
            contract,
            revision,
            expected_published=expected_published,
        )
        for annotator in ANNOTATORS
    ]
    orders = {report["task_order_sha256"] for report in reports}
    if len(orders) != len(ANNOTATORS):
        raise RuntimeError("Annotator projects unexpectedly have identical task order")
    if len({report["published"] for report in reports}) != 1:
        raise RuntimeError("The paired projects have inconsistent publication states")
    return reports


def backup_database(database: Path, backup_dir: Path) -> dict[str, Any]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = backup_dir / f"pre_trace_alignment_{timestamp}.sqlite3"
    temporary = backup_dir / f".{destination.name}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    source_connection = sqlite3.connect(
        f"file:{database.as_posix()}?mode=ro", uri=True
    )
    destination_connection = sqlite3.connect(temporary)
    try:
        source_connection.backup(destination_connection)
        check = destination_connection.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError("Backup database quick_check failed")
        foreign_keys = destination_connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_keys:
            raise RuntimeError(
                f"Backup database has {len(foreign_keys)} foreign-key violation(s)"
            )
        destination_connection.commit()
    except BaseException:
        destination_connection.close()
        source_connection.close()
        temporary.unlink(missing_ok=True)
        raise
    destination_connection.close()
    source_connection.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "quick_check": "ok",
        "foreign_key_violations": 0,
    }


def base_report(
    database: Path,
    artifact: TaskArtifact,
    contract: PreparedContract,
    gate: dict[str, Any],
    revision: str,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "marker": stage_marker(artifact.cohort, artifact.stage, revision),
        "revision": revision,
        "mode": mode,
        "cohort": artifact.cohort,
        "stage": artifact.stage,
        "database": str(database),
        "database_quick_check": "ok",
        "database_foreign_key_violations": 0,
        "prepared_manifest": str(contract.manifest_path),
        "prepared_manifest_sha256": contract.manifest_sha256,
        "task_artifact": artifact.filename,
        "task_artifact_rows": artifact.expected_rows,
        "task_artifact_sha256": contract.artifact_sha256[
            (artifact.cohort, artifact.stage)
        ],
        "config": str(artifact.config_path),
        "config_sha256": sha256_file(artifact.config_path),
        "order_namespace_sha256": contract.order_namespace,
        "gate": gate,
    }


def dry_run(
    database: Path,
    artifact: TaskArtifact,
    contract: PreparedContract,
    admin: User,
    annotator_users: dict[str, User],
    revision: str,
    *,
    publish_transition: bool,
    gate: dict[str, Any],
) -> dict[str, Any]:
    report = base_report(
        database, artifact, contract, gate, revision, mode="dry_run"
    )
    existing = locate_project_pair(artifact, revision)
    if gate["required"] and not gate["satisfied"]:
        report.update(
            {
                "state": "blocked_by_stage_gate",
                "transactional_dry_run": False,
                "rolled_back": True,
                "projects": [],
            }
        )
        return report

    desired_published = artifact.stage == "preference" or publish_transition
    if existing is not None:
        projects = validate_pair(
            existing,
            artifact,
            contract,
            admin,
            annotator_users,
            revision,
            expected_published=None,
        )
        currently_published = bool(projects[0]["published"])
        if (
            artifact.stage == "transition"
            and publish_transition
            and not currently_published
        ):
            with transaction.atomic():
                for project in existing.values():
                    project.is_published = True
                    project.save(update_fields=["is_published"])
                projects = validate_pair(
                    existing,
                    artifact,
                    contract,
                    admin,
                    annotator_users,
                    revision,
                    expected_published=True,
                )
                transaction.set_rollback(True)
            state = "would_publish_transition"
        else:
            state = "already_applied"
        report.update(
            {
                "state": state,
                "transactional_dry_run": True,
                "rolled_back": True,
                "projects": projects,
            }
        )
    else:
        before_count = Project.all_objects.filter(
            description__contains=stage_marker(
                artifact.cohort, artifact.stage, revision
            )
        ).count()
        with transaction.atomic():
            created = {
                annotator.email: create_project(
                    annotator,
                    annotator_users[annotator.email],
                    admin,
                    artifact,
                    contract,
                    revision,
                    published=desired_published,
                )
                for annotator in ANNOTATORS
            }
            projects = validate_pair(
                created,
                artifact,
                contract,
                admin,
                annotator_users,
                revision,
                expected_published=desired_published,
            )
            transaction.set_rollback(True)
        after_count = Project.all_objects.filter(
            description__contains=stage_marker(
                artifact.cohort, artifact.stage, revision
            )
        ).count()
        if before_count != after_count:
            raise RuntimeError("Transactional dry-run left project rows behind")
        report.update(
            {
                "state": "would_create",
                "transactional_dry_run": True,
                "rolled_back": True,
                "projects": projects,
            }
        )
    if quick_check() != "ok" or foreign_key_violation_count() != 0:
        raise RuntimeError("Database integrity changed during transactional dry-run")
    report["database_quick_check_after"] = "ok"
    report["database_foreign_key_violations_after"] = 0
    return report


def apply(
    database: Path,
    backup_dir: Path,
    artifact: TaskArtifact,
    contract: PreparedContract,
    admin: User,
    annotator_users: dict[str, User],
    revision: str,
    *,
    publish_transition: bool,
    gate: dict[str, Any],
) -> dict[str, Any]:
    if gate["required"] and not gate["satisfied"]:
        raise RuntimeError(
            "Stage gate is not satisfied: complete the preceding paired projects "
            "and pass --confirm-gate-frozen"
        )
    report = base_report(
        database, artifact, contract, gate, revision, mode="apply"
    )
    existing = locate_project_pair(artifact, revision)
    desired_published = artifact.stage == "preference" or publish_transition
    if existing is not None:
        projects = validate_pair(
            existing,
            artifact,
            contract,
            admin,
            annotator_users,
            revision,
            expected_published=None,
        )
        currently_published = bool(projects[0]["published"])
        if (
            artifact.stage == "transition"
            and publish_transition
            and not currently_published
        ):
            backup = backup_database(database, backup_dir)
            with transaction.atomic():
                current = locate_project_pair(artifact, revision)
                if current is None:
                    raise RuntimeError("Transition project pair disappeared")
                for project in current.values():
                    project.is_published = True
                    project.save(update_fields=["is_published"])
                projects = validate_pair(
                    current,
                    artifact,
                    contract,
                    admin,
                    annotator_users,
                    revision,
                    expected_published=True,
                )
            state = "published_transition"
            report["backup"] = backup
        else:
            state = "already_applied"
        report.update({"state": state, "projects": projects})
    else:
        backup = backup_database(database, backup_dir)
        with transaction.atomic():
            if locate_project_pair(artifact, revision) is not None:
                raise RuntimeError(
                    "Project pair appeared after inspection; retry the operation"
                )
            created = {
                annotator.email: create_project(
                    annotator,
                    annotator_users[annotator.email],
                    admin,
                    artifact,
                    contract,
                    revision,
                    published=desired_published,
                )
                for annotator in ANNOTATORS
            }
            projects = validate_pair(
                created,
                artifact,
                contract,
                admin,
                annotator_users,
                revision,
                expected_published=desired_published,
            )
        report.update(
            {
                "state": "created",
                "projects": projects,
                "backup": backup,
            }
        )
    if quick_check() != "ok" or foreign_key_violation_count() != 0:
        raise RuntimeError("Database integrity checks failed after apply")
    report["database_quick_check_after"] = "ok"
    report["database_foreign_key_violations_after"] = 0
    report["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and transactionally simulate without persisting (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Back up the SQLite database, then persist the selected stage.",
    )
    parser.add_argument(
        "--cohort", choices=("pilot", "formal"), default="pilot"
    )
    parser.add_argument(
        "--stage", choices=("preference", "transition"), default="preference"
    )
    parser.add_argument(
        "--revision",
        type=validate_revision,
        default=DEFAULT_REVISION,
        help=(
            "Protocol/interface revision namespace (default: v1). Increment it "
            "after any pilot-driven task, translation, XML, or instruction change."
        ),
    )
    parser.add_argument(
        "--confirm-gate-frozen",
        action="store_true",
        help=(
            "For any stage after pilot preference, attest that the preceding "
            "stage is complete and the relevant interface/exclusion rules are frozen."
        ),
    )
    parser.add_argument(
        "--publish-transition",
        action="store_true",
        help=(
            "Publish the selected transition pair. This is an explicit assertion "
            "that the preference-first state-exposure gate has been satisfied."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR
    )
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Defaults to "
            "PREPARED_DIR/launch_<revision>_<cohort>_<stage>_"
            "<dry_run|apply>.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_dir.resolve() != _bootstrap.data_dir.resolve():
        raise RuntimeError("--data-dir changed after Django bootstrap")
    if args.database.resolve() != _bootstrap.database.resolve():
        raise RuntimeError("--database changed after Django bootstrap")
    if args.publish_transition and args.stage != "transition":
        raise ValueError("--publish-transition is only valid with --stage transition")

    database = assert_database_contract()
    contract = load_prepared_contract(args.prepared_dir)
    artifact = ARTIFACT_BY_STAGE[(args.cohort, args.stage)]
    admin, annotator_users = load_users()
    gate = gate_report(
        artifact,
        contract,
        admin,
        annotator_users,
        args.revision,
        confirmation=args.confirm_gate_frozen,
    )
    if args.apply:
        result = apply(
            database,
            args.backup_dir,
            artifact,
            contract,
            admin,
            annotator_users,
            args.revision,
            publish_transition=args.publish_transition,
            gate=gate,
        )
    else:
        result = dry_run(
            database,
            artifact,
            contract,
            admin,
            annotator_users,
            args.revision,
            publish_transition=args.publish_transition,
            gate=gate,
        )
    report_path = args.report or (
        args.prepared_dir
        / (
            f"launch_{args.revision}_{args.cohort}_{args.stage}_"
            f"{result['mode']}.json"
        )
    )
    atomic_write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
