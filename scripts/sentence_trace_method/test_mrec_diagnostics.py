from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _source_row() -> dict[str, object]:
    step = {
        "step": 1,
        "operation": "OPEN",
        "atom_id": "A1",
        "atom_text": "The city approved the project.",
        "state_before": "U",
        "state_after": "S",
        "cue_text": "Did the city approve the project?",
        "cue_source": "qd_question",
        "evidence_id": "E1",
        "candidate_idx": 0,
        "selector_candidate_idx": 0,
        "evidence_text": "The city council approved the project.",
        "covered_atom_ids": ["A1"],
        "relation": "support",
        "directness": "direct",
        "map_confidence": 0.9,
        "evidence_map_quality_score": 0.8,
        "token_cost": 7,
        "transition_reason": "A1 changes from unresolved to S",
    }
    return {
        "event_id": "evt-1",
        "mrec_trace_version": "mrec_trace_v0_1",
        "mrec_selector_name": "mrec_test",
        "selector_name": "mrec_test",
        "atom_states_initial": {"A1": "U"},
        "atom_states_final": {"A1": "S"},
        "mrec_steps": [step],
        "mrec_diagnostics": {
            "stop_reason": "target_resolution_reached",
            "resolved_atom_rate": 1.0,
            "duplicate_rejected_count": 0,
            "background_rejected_count": 0,
            "no_transition_rejected_count": 0,
        },
        "compat_chain_steps": [
            {
                "step": 1,
                "candidate_idx": 0,
                "selector_candidate_idx": 0,
                "evidence_id": "E1",
                "cue_text": "Did the city approve the project?",
                "covered_atom_ids": ["A1"],
            }
        ],
    }


def _build_row(*, prompt: str = "System\nUser\nCheck: Did the city approve the project?\nEvidence text.") -> dict[str, object]:
    return {
        "event_id": "evt-1",
        "trace_prompt_style": "mrec_min",
        "evidence_text_mode": "anchor_only",
        "prompt": prompt,
        "prompt_token_count": 32,
        "evidence_count": 1,
        "was_truncated": False,
        "evidence_text_truncated": False,
        "candidates": [{"text": "Check: Did the city approve the project?\nEvidence text."}],
        "mrec_steps": _source_row()["mrec_steps"],
        "mrec_diagnostics": _source_row()["mrec_diagnostics"],
        "atom_states_final": {"A1": "S"},
        "mrec_prompt_steps": [
            {
                "step": 1,
                "source": "mrec_steps",
                "candidate_idx": 0,
                "evidence_id": "E1",
                "cue_type": "mrec_step",
                "check": "Did the city approve the project?",
                "operation": "OPEN",
                "state_before": "U",
                "state_after": "S",
                "covered_atom_ids": ["A1"],
            }
        ],
        "mrec_prompt_diagnostics": {
            "source": "mrec_steps",
            "cue_type_counts": {"mrec_step": 1},
            "operation_counts": {"OPEN": 1},
            "mean_check_token_count": 6.0,
            "fallback_rate": 0.0,
        },
    }


def test_mrec_diagnostics_passes_clean_build(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    case_root = output_root / "rawfc__ministral3_8b__mrec_min_anchor_only"
    _write_jsonl(
        output_root / "_sources" / "rawfc" / "mrec_test" / "train" / "selection_trace_train.jsonl",
        [_source_row()],
    )
    _write_jsonl(case_root / "build" / "build_train.jsonl", [_build_row()])
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/check_mrec_diagnostics.py"),
            "--dataset",
            "rawfc",
            "--output-root",
            str(output_root),
            "--case-root",
            str(case_root),
            "--source-selector-name",
            "mrec_test",
            "--splits",
            "train",
            "--expected-evidence-text-mode",
            "anchor_only",
            "--report-path",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[mrec-diagnostics] PASSED" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["splits"]["train"]["source"]["resolved_atom_rate_mean"] == 1.0
    assert report["splits"]["train"]["prompt"]["prompt_leak_rate"] == 0.0


def test_mrec_diagnostics_fails_when_prompt_exposes_transition_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    case_root = output_root / "rawfc__ministral3_8b__mrec_min_anchor_only"
    _write_jsonl(
        output_root / "_sources" / "rawfc" / "mrec_test" / "train" / "selection_trace_train.jsonl",
        [_source_row()],
    )
    _write_jsonl(
        case_root / "build" / "build_train.jsonl",
        [_build_row(prompt="Check: cue\nrelation=support\nOPEN")],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/check_mrec_diagnostics.py"),
            "--dataset",
            "rawfc",
            "--output-root",
            str(output_root),
            "--case-root",
            str(case_root),
            "--source-selector-name",
            "mrec_test",
            "--splits",
            "train",
            "--expected-evidence-text-mode",
            "anchor_only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "prompt_leak_rate=1.000000" in result.stdout
