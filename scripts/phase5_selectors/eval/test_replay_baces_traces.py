from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from fact_checking.selectors.baces_trace import build_exact_trace
from scripts.phase5_selectors.eval.replay_baces_traces import main


def test_file_replay_reports_valid_and_tampered_traces(tmp_path: Path) -> None:
    trace = build_exact_trace(_feature_row(), k_min=2, k_max=2)
    valid_path = tmp_path / "valid.jsonl"
    _write_jsonl(valid_path, [trace])

    assert main(
        [
            "--trace",
            str(valid_path),
            "--output-jsonl",
            str(tmp_path / "valid_audit.jsonl"),
            "--summary-json",
            str(tmp_path / "valid_summary.json"),
        ]
    ) == 0
    valid_summary = json.loads(
        (tmp_path / "valid_summary.json").read_text(encoding="utf-8")
    )
    assert valid_summary["all_replays_ok"] is True

    tampered = deepcopy(trace)
    tampered["baces_steps"][0]["display_marginal_coverage_units"] = 0
    tampered_path = tmp_path / "tampered.jsonl"
    _write_jsonl(tampered_path, [tampered])

    assert main(
        [
            "--trace",
            str(tampered_path),
            "--output-jsonl",
            str(tmp_path / "tampered_audit.jsonl"),
            "--summary-json",
            str(tmp_path / "tampered_summary.json"),
        ]
    ) == 1
    audit = _read_jsonl(tmp_path / "tampered_audit.jsonl")[0]
    assert audit["status"] == "replay_error"
    assert any("display_marginal_coverage_units" in error for error in audit["errors"])


def _feature_row() -> dict:
    return {
        "event_id": "event-1",
        "claim_atoms": [{"atom_id": "A1", "proposition": "One fact"}],
        "candidates": [
            _candidate("a", "E-a", "direct"),
            _candidate("b", "E-b", "partial"),
        ],
    }


def _candidate(uid: str, evidence_id: str, directness: str) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_key": f"key-{uid}",
        "evidence_id": evidence_id,
        "num_tokens": 1,
        "candidate_atom_alignments": [
            {
                "atom_id": "A1",
                "evidence_id": evidence_id,
                "relation": "support",
                "directness": directness,
                "confidence": 1.0,
                "key_spans": ["span"],
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
