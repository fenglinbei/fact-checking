from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LAUNCHER = SCRIPT_DIR / "launch_exp3_trace_alignment_projects.py"
LIVE_DATABASE = PROJECT_ROOT / "label_studio_data" / "label_studio.sqlite3"
PREFERENCE_CONFIG = PROJECT_ROOT / "config" / "exp3_trace_preference.xml"
TRANSITION_CONFIG = PROJECT_ROOT / "config" / "exp4_transition_audit.xml"
MARKER = "trace-alignment-human-eval-20260724"

PREFERENCE_FIELDS = {
    "blind_task_id",
    "claim_en",
    "claim_zh",
    "sequence_a_html",
    "sequence_b_html",
}
TRANSITION_FIELDS = {
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preference_rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "blind_task_id": f"{prefix}-P-{index:03d}",
            "claim_en": f"English claim {index}.",
            "claim_zh": f"中文声明 {index}。",
            "sequence_a_html": (
                '<ol class="evidence-sequence"><li>'
                '<div class="evidence-source"><strong>Source 1</strong>'
                "<span> · example.org</span></div>"
                f'<p class="evidence-en">Evidence A {index}.</p>'
                f'<p class="evidence-zh"><strong>中文辅助：</strong>证据 A {index}。</p>'
                "</li></ol>"
            ),
            "sequence_b_html": (
                '<ol class="evidence-sequence"><li>'
                '<div class="evidence-source"><strong>Source 2</strong>'
                "<span> · example.net</span></div>"
                f'<p class="evidence-en">Evidence B {index}.</p>'
                f'<p class="evidence-zh"><strong>中文辅助：</strong>证据 B {index}。</p>'
                "</li></ol>"
            ),
        }
        for index in range(1, count + 1)
    ]


def transition_rows(prefix: str, count: int) -> list[dict[str, str]]:
    legend = (
        '<div class="state-legend"><span><strong>U</strong> · Unresolved</span>'
        "<br><span><strong>S</strong> · Supported</span></div>"
    )
    return [
        {
            "blind_task_id": f"{prefix}-T-{index:03d}",
            "claim_en": f"English transition claim {index}.",
            "claim_zh": f"中文转移声明 {index}。",
            "focal_atom_en": f"Focal proposition {index}.",
            "focal_atom_zh": f"原子命题 {index}。",
            "state_legend_html": legend,
            "prior_evidence_html": (
                '<p class="empty-prefix">No earlier evidence for this atom.</p>'
            ),
            "current_evidence_html": (
                '<ol class="evidence-sequence"><li>'
                '<div class="evidence-source"><strong>Source 1</strong>'
                "<span> · example.org</span></div>"
                f'<p class="evidence-en">Current evidence {index}.</p>'
                f'<p class="evidence-zh"><strong>中文辅助：</strong>'
                f"当前证据 {index}。</p>"
                "</li></ol>"
            ),
            "proposed_transition": "U → S",
        }
        for index in range(1, count + 1)
    ]


def prepare_contract(output_dir: Path) -> Path:
    rows_by_logical = {
        "pilot_preference_tasks": preference_rows("PILOT", 30),
        "preference_tasks": preference_rows("FORMAL", 200),
        "pilot_transition_tasks": transition_rows("PILOT", 15),
        "transition_tasks": transition_rows("FORMAL", 100),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for logical, rows in rows_by_logical.items():
        path = output_dir / f"{logical}.jsonl"
        write_jsonl(path, rows)
        artifacts[logical] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
            "visibility": "public",
        }
    manifest = {
        "schema_version": "evitrace_human_alignment_task_manifest_v1",
        "complete": True,
        "annotation_complete": False,
        "artifacts": artifacts,
        "private_key_commitments": {
            "blind_seed_sha256": hashlib.sha256(
                b"launcher-test-order-secret"
            ).hexdigest()
        },
    }
    manifest_path = output_dir / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def sqlite_backup(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(
        f"file:{source.resolve().as_posix()}?mode=ro", uri=True
    )
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def choices(root: ET.Element, name: str) -> list[str]:
    element = root.find(f".//Choices[@name='{name}']")
    if element is None:
        raise AssertionError(f"Choices {name!r} not found")
    return [choice.attrib["value"] for choice in element.findall("./Choice")]


class TraceAlignmentConfigTest(unittest.TestCase):
    def test_preference_config_exact_contract(self) -> None:
        text = PREFERENCE_CONFIG.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        variables = set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", text))
        self.assertEqual(variables, PREFERENCE_FIELDS)
        self.assertEqual(
            choices(root, "overall_preference"),
            [
                "strongly_prefer_a",
                "prefer_a",
                "tie",
                "prefer_b",
                "strongly_prefer_b",
            ],
        )
        self.assertEqual(
            choices(root, "data_issue"),
            [
                "translation",
                "missing_or_malformed_evidence",
                "duplicate_evidence",
                "source_or_format",
                "other",
            ],
        )
        self.assertIsNotNone(root.find(".//TextArea[@name='notes']"))
        self.assertIn(
            "哪一个有序证据序列更能帮助独立事实核查者形成准确且有依据的判断",
            text,
        )
        for forbidden in (
            "method_to_side",
            "candidate_uid",
            "evidence_count",
            "confidence",
            "Check:",
            "EviTrace",
        ):
            self.assertNotIn(forbidden, text)

    def test_transition_config_exact_contract(self) -> None:
        text = TRANSITION_CONFIG.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        variables = set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", text))
        self.assertEqual(variables, TRANSITION_FIELDS)
        self.assertEqual(
            choices(root, "transition_validity"),
            ["valid", "partially_valid", "invalid"],
        )
        self.assertEqual(
            choices(root, "marginal_contribution"),
            ["clear", "limited", "none"],
        )
        for forbidden in (
            "relation",
            "directness",
            "confidence",
            "gold_label",
            "verifier_prediction",
            "method_name",
            "EviTrace",
        ):
            self.assertNotIn(forbidden, text)


@unittest.skipUnless(
    importlib.util.find_spec("label_studio") is not None,
    "Label Studio environment is required for launcher integration tests",
)
class TraceAlignmentLauncherIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        if not LIVE_DATABASE.is_file():
            self.skipTest(f"Live schema template is unavailable: {LIVE_DATABASE}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "label_studio_data"
        self.data_dir.mkdir()
        self.database = self.data_dir / "sandbox.sqlite3"
        sqlite_backup(LIVE_DATABASE, self.database)
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                """
                UPDATE project
                SET title = 'sandbox-template-' || id,
                    description = 'sandbox template isolated from launcher tests'
                WHERE description LIKE ?
                """,
                (f"%{MARKER}:%",),
            )
            connection.commit()
        finally:
            connection.close()
        self.prepared_dir = self.root / "prepared"
        self.prepared_dir.mkdir()
        prepare_contract(self.prepared_dir)
        self.backup_dir = self.root / "backups"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_launcher(
        self, *extra: str, report_name: str
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
        report = self.root / report_name
        command = [
            sys.executable,
            str(LAUNCHER),
            "--data-dir",
            str(self.data_dir),
            "--database",
            str(self.database),
            "--prepared-dir",
            str(self.prepared_dir),
            "--backup-dir",
            str(self.backup_dir),
            "--report",
            str(report),
            *extra,
        ]
        environment = os.environ.copy()
        environment["LABEL_STUDIO_LATEST_VERSION_CHECK"] = "false"
        environment["COLLECT_ANALYTICS"] = "false"
        environment["SENTRY_DSN"] = ""
        environment["FRONTEND_SENTRY_DSN"] = ""
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "Launcher failed\n"
                f"command={command}\n"
                f"stdout={completed.stdout[-4000:]}\n"
                f"stderr={completed.stderr[-4000:]}"
            )
        return completed, json.loads(report.read_text(encoding="utf-8"))

    def run_launcher_failure(
        self, *extra: str, report_name: str
    ) -> subprocess.CompletedProcess[str]:
        report = self.root / report_name
        command = [
            sys.executable,
            str(LAUNCHER),
            "--data-dir",
            str(self.data_dir),
            "--database",
            str(self.database),
            "--prepared-dir",
            str(self.prepared_dir),
            "--backup-dir",
            str(self.backup_dir),
            "--report",
            str(report),
            *extra,
        ]
        environment = os.environ.copy()
        environment["LABEL_STUDIO_LATEST_VERSION_CHECK"] = "false"
        environment["COLLECT_ANALYTICS"] = "false"
        environment["SENTRY_DSN"] = ""
        environment["FRONTEND_SENTRY_DSN"] = ""
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(report.exists())
        return completed

    def run_launcher_with_default_report(
        self, *extra: str
    ) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Any]]:
        command = [
            sys.executable,
            str(LAUNCHER),
            "--data-dir",
            str(self.data_dir),
            "--database",
            str(self.database),
            "--prepared-dir",
            str(self.prepared_dir),
            "--backup-dir",
            str(self.backup_dir),
            *extra,
        ]
        environment = os.environ.copy()
        environment["LABEL_STUDIO_LATEST_VERSION_CHECK"] = "false"
        environment["COLLECT_ANALYTICS"] = "false"
        environment["SENTRY_DSN"] = ""
        environment["FRONTEND_SENTRY_DSN"] = ""
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "Launcher failed\n"
                f"command={command}\n"
                f"stdout={completed.stdout[-4000:]}\n"
                f"stderr={completed.stderr[-4000:]}"
            )
        mode = "apply" if "--apply" in extra else "dry_run"
        report = (
            self.prepared_dir
            / f"launch_v1_pilot_preference_{mode}.json"
        )
        self.assertTrue(report.is_file())
        return completed, report, json.loads(report.read_text(encoding="utf-8"))

    def project_count(self, marker_suffix: str, revision: str = "v1") -> int:
        connection = sqlite3.connect(self.database)
        try:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM project WHERE description LIKE ?",
                    (f"%{MARKER}:{revision}:{marker_suffix}%",),
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def test_default_is_transactional_pilot_preference_dry_run(self) -> None:
        _, report = self.run_launcher(report_name="pilot_dry_run.json")
        self.assertEqual(report["mode"], "dry_run")
        self.assertEqual(report["state"], "would_create")
        self.assertTrue(report["transactional_dry_run"])
        self.assertTrue(report["rolled_back"])
        self.assertEqual(report["cohort"], "pilot")
        self.assertEqual(report["stage"], "preference")
        self.assertEqual(report["revision"], "v1")
        self.assertEqual(report["gate"]["revision"], "v1")
        self.assertEqual(len(report["projects"]), 2)
        self.assertTrue(all(project["published"] for project in report["projects"]))
        self.assertEqual(
            {project["tasks"] for project in report["projects"]}, {30}
        )
        self.assertEqual(
            len(
                {
                    project["task_order_sha256"]
                    for project in report["projects"]
                }
            ),
            2,
        )
        self.assertEqual(self.project_count("pilot:preference"), 0)
        self.assertFalse(self.backup_dir.exists())
        connection = sqlite3.connect(self.database)
        try:
            self.assertEqual(
                connection.execute("PRAGMA quick_check").fetchone()[0], "ok"
            )
        finally:
            connection.close()

    def test_default_dry_run_and_apply_reports_do_not_overwrite(self) -> None:
        _, dry_run_path, dry_run_report = self.run_launcher_with_default_report()
        dry_run_receipt = dry_run_path.read_bytes()
        self.assertEqual(
            dry_run_path.name,
            "launch_v1_pilot_preference_dry_run.json",
        )
        self.assertEqual(dry_run_report["mode"], "dry_run")

        _, apply_path, apply_report = self.run_launcher_with_default_report(
            "--apply"
        )
        self.assertEqual(
            apply_path.name,
            "launch_v1_pilot_preference_apply.json",
        )
        self.assertNotEqual(dry_run_path, apply_path)
        self.assertEqual(apply_report["mode"], "apply")
        self.assertIn("backup", apply_report)
        self.assertEqual(dry_run_path.read_bytes(), dry_run_receipt)
        self.assertEqual(
            json.loads(dry_run_path.read_text(encoding="utf-8"))["mode"],
            "dry_run",
        )

    def test_sandbox_apply_backs_up_and_later_stage_is_blocked(self) -> None:
        _, report = self.run_launcher(
            "--apply", report_name="pilot_apply.json"
        )
        self.assertEqual(report["state"], "created")
        self.assertEqual(self.project_count("pilot:preference"), 2)
        self.assertEqual(len(report["projects"]), 2)
        self.assertTrue(all(project["published"] for project in report["projects"]))
        backup = Path(report["backup"]["path"])
        self.assertTrue(backup.is_file())
        self.assertEqual(sha256_file(backup), report["backup"]["sha256"])
        backup_connection = sqlite3.connect(
            f"file:{backup.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            self.assertEqual(
                backup_connection.execute("PRAGMA quick_check").fetchone()[0],
                "ok",
            )
        finally:
            backup_connection.close()

        connection = sqlite3.connect(self.database)
        try:
            project_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM project WHERE description LIKE ?",
                    (f"%{MARKER}:v1:pilot:preference%",),
                )
            ]
            placeholders = ",".join("?" for _ in project_ids)
            task_count = connection.execute(
                f"SELECT COUNT(*) FROM task WHERE project_id IN ({placeholders})",
                project_ids,
            ).fetchone()[0]
            self.assertEqual(task_count, 60)
        finally:
            connection.close()

        _, blocked = self.run_launcher(
            "--cohort",
            "formal",
            "--stage",
            "preference",
            "--confirm-gate-frozen",
            report_name="formal_blocked.json",
        )
        self.assertEqual(blocked["state"], "blocked_by_stage_gate")
        self.assertFalse(
            blocked["gate"]["prerequisite"]["complete"]
        )
        self.assertEqual(self.project_count("formal:preference"), 0)

    def test_revision_namespaces_projects_orders_and_gate(self) -> None:
        _, v1_report = self.run_launcher(
            "--apply",
            "--revision",
            "v1",
            report_name="pilot_v1_apply.json",
        )
        _, v2_report = self.run_launcher(
            "--apply",
            "--revision",
            "v2",
            report_name="pilot_v2_apply.json",
        )
        self.assertEqual(v1_report["revision"], "v1")
        self.assertEqual(v2_report["revision"], "v2")
        self.assertEqual(self.project_count("pilot:preference", "v1"), 2)
        self.assertEqual(self.project_count("pilot:preference", "v2"), 2)
        self.assertTrue(
            all(project["title"].endswith("-v1") for project in v1_report["projects"])
        )
        self.assertTrue(
            all(project["title"].endswith("-v2") for project in v2_report["projects"])
        )
        orders_v1 = {
            project["annotator"]: project["task_order_sha256"]
            for project in v1_report["projects"]
        }
        orders_v2 = {
            project["annotator"]: project["task_order_sha256"]
            for project in v2_report["projects"]
        }
        self.assertEqual(set(orders_v1), set(orders_v2))
        self.assertTrue(
            all(orders_v1[annotator] != orders_v2[annotator] for annotator in orders_v1)
        )

        _, blocked = self.run_launcher(
            "--cohort",
            "formal",
            "--stage",
            "preference",
            "--revision",
            "v2",
            "--confirm-gate-frozen",
            report_name="formal_v2_blocked.json",
        )
        self.assertEqual(blocked["state"], "blocked_by_stage_gate")
        self.assertEqual(blocked["revision"], "v2")
        self.assertEqual(blocked["gate"]["revision"], "v2")
        self.assertEqual(
            blocked["gate"]["prerequisite"]["revision"], "v2"
        )
        self.assertEqual(self.project_count("formal:preference", "v2"), 0)

    def test_revision_rejects_unsafe_or_overlong_values(self) -> None:
        for index, revision in enumerate(("../v2", "v2.1", "abcdef")):
            completed = self.run_launcher_failure(
                "--revision",
                revision,
                report_name=f"invalid_revision_{index}.json",
            )
            self.assertIn("revision must be 1-5 characters", completed.stderr)


if __name__ == "__main__":
    unittest.main()
