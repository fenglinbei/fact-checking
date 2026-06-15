from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "phase9_utils" / "sync_results.sh"


class SyncResultsScriptTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["REMOTE"] = "example:/tmp/fact-checking/"
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_core_mode_prints_lightweight_rules_without_predictions(self) -> None:
        result = self._run("--mode", "core", "--dry-run", "--show-rules")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODE=core", result.stdout)
        self.assertIn("--include=outputs/**/metrics.json", result.stdout)
        self.assertIn("--include=outputs/**/training_complete.json", result.stdout)
        self.assertIn("--exclude=outputs/cache/**", result.stdout)
        self.assertNotIn("*predictions*.jsonl", result.stdout)
        self.assertNotIn("confusion_matrix.png", result.stdout)

    def test_audit_mode_adds_predictions_and_confusion_png(self) -> None:
        result = self._run("--mode", "audit", "--dry-run", "--show-rules")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODE=audit", result.stdout)
        self.assertIn("--include=outputs/**/*predictions*.jsonl", result.stdout)
        self.assertIn("--include=outputs/**/confusion_matrix.png", result.stdout)

    def test_resume_state_mode_requires_explicit_allow_and_path(self) -> None:
        result = self._run("--mode", "resume-state", "--dry-run", "--show-rules")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resume-state", result.stderr)
        self.assertIn("--allow-resume-state", result.stderr)


if __name__ == "__main__":
    unittest.main()
