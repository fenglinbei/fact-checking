from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_audit_logit_adjust_alignment_reports_config_train_and_eval_tau(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    (case_root / "train").mkdir(parents=True)
    (case_root / "eval" / "val" / "best" / "checkpoint_gap_E0_current_composite_tau1").mkdir(parents=True)
    (case_root / "train.resolved.yaml").write_text(
        "sft_train:\n  logit_adjust:\n    enabled: false\n    tau: 1.0\n",
        encoding="utf-8",
    )
    (case_root / "train" / "logit_adjust.json").write_text(
        json.dumps({"enabled": True, "tau": 0.75}),
        encoding="utf-8",
    )
    metrics_path = case_root / "eval" / "val" / "best" / "checkpoint_gap_E0_current_composite_tau1" / "metrics.json"
    metrics_path.write_text(
        json.dumps({"logit_adjust": {"enabled": True, "tau": 1.0}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/audit_logit_adjust_alignment.py"),
            "--case-root",
            str(case_root),
            "--print-json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(result.stdout)

    assert report["config"]["enabled"] is False
    assert report["config"]["tau"] == 1.0
    assert report["train_saved"]["enabled"] is True
    assert report["train_saved"]["tau"] == 0.75
    assert report["eval_metrics"][0]["enabled"] is True
    assert report["eval_metrics"][0]["tau"] == 1.0
    assert report["eval_metrics"][0]["matches_train_saved"] is False
