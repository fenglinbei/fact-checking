from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_case(case_root: Path) -> None:
    for step, macro_f1, macro_f1_se in (
        (100, 0.610, 0.020),
        (200, 0.625, 0.015),
        (300, 0.630, 0.020),
    ):
        checkpoint_dir = case_root / "train" / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapter_config.json").write_text(
            json.dumps({"step": step}),
            encoding="utf-8",
        )
        metrics_dir = case_root / "eval" / f"step-{step}"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "macro_f1": macro_f1,
                    "macro_f1_se": macro_f1_se,
                    "checkpoint_selection_score": macro_f1,
                }
            ),
            encoding="utf-8",
        )


def test_sync_best_materializes_one_se_alias_and_metadata(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    _write_case(case_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/select_checkpoint_for_eval.py"),
            "sync-best",
            "--case-root",
            str(case_root),
            "--policy",
            "one_standard_error",
            "--alias",
            "one_se_best",
            "--link-mode",
            "copy",
            "--print-json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["selected_checkpoint"] == "checkpoint-100"
    assert payload["fmax_checkpoint"] == "checkpoint-300"
    assert payload["threshold"] == 0.610
    assert (case_root / "train" / "one_se_best" / "adapter_config.json").exists()

    selection_path = case_root / "train" / "one_se_best_checkpoint_selection.json"
    assert selection_path.exists()
    saved = json.loads(selection_path.read_text(encoding="utf-8"))
    assert saved["alias"] == "one_se_best"
    assert saved["selected_step"] == 100


def test_sync_best_dry_run_does_not_create_alias(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    _write_case(case_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/select_checkpoint_for_eval.py"),
            "sync-best",
            "--case-root",
            str(case_root),
            "--dry-run",
            "--print-json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["selected_checkpoint"] == "checkpoint-100"
    assert not (case_root / "train" / "one_se_best").exists()
    assert not (case_root / "train" / "one_se_best_checkpoint_selection.json").exists()
