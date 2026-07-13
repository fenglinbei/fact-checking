from __future__ import annotations

import json
from pathlib import Path

from scripts.phase5_selectors.build.build_baces_traces import main


def test_cli_uses_uid_aligned_mrec_costs_and_replay_audits(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    costs = tmp_path / "costs.jsonl"
    output = tmp_path / "out"
    _write_jsonl(features, [_feature_row()])
    _write_jsonl(
        costs,
        [
            {
                "event_id": "event-1",
                "candidate_pool": [
                    {"candidate_uid": "b", "mrec_token_cost": 2},
                    {"candidate_uid": "a", "mrec_token_cost": 7},
                ],
            }
        ],
    )

    assert main(
        [
            "--features",
            str(features),
            "--cost-trace",
            str(costs),
            "--cost-tokenizer-id",
            "fixture-tokenizer",
            "--output-dir",
            str(output),
            "--k-min",
            "1",
            "--k-max",
            "2",
        ]
    ) == 0

    trace = _read_jsonl(output / "baces_trace.jsonl")[0]
    by_uid = {
        candidate["candidate_uid"]: candidate for candidate in trace["candidate_pool"]
    }
    assert by_uid["a"]["token_cost"] == 7
    assert by_uid["b"]["token_cost"] == 2
    assert trace["cost_tokenizer_id"] == "fixture-tokenizer"
    assert trace["cost_source"] == "cost_trace.candidate_pool.mrec_token_cost"
    audit = _read_jsonl(output / "baces_replay_audit.jsonl")[0]
    assert audit["ok"] is True
    summary = json.loads((output / "baces_summary.json").read_text(encoding="utf-8"))
    assert summary["all_replays_ok"] is True
    assert summary["params"]["cost_source"] == "candidate_pool.mrec_token_cost"


def test_cli_fails_closed_on_feature_cost_uid_mismatch(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    costs = tmp_path / "costs.jsonl"
    output = tmp_path / "out"
    _write_jsonl(features, [_feature_row()])
    _write_jsonl(
        costs,
        [
            {
                "event_id": "event-1",
                "candidate_pool": [
                    {"candidate_uid": "a", "mrec_token_cost": 7},
                    {"candidate_uid": "not-b", "mrec_token_cost": 2},
                ],
            }
        ],
    )

    assert main(
        [
            "--features",
            str(features),
            "--cost-trace",
            str(costs),
            "--output-dir",
            str(output),
            "--k-min",
            "1",
            "--k-max",
            "2",
        ]
    ) == 1
    summary = json.loads((output / "baces_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_count"] == 1
    assert summary["trace_rows_written"] == 0
    assert "UID sets differ" in summary["failures"][0]["errors"][0]


def _feature_row() -> dict:
    return {
        "event_id": "event-1",
        "claim_atoms": [{"atom_id": "A1", "proposition": "One fact"}],
        "candidates": [
            _candidate("a", "E-a", directness="direct", num_tokens=1),
            _candidate("b", "E-b", directness="partial", num_tokens=99),
        ],
    }


def _candidate(
    uid: str, evidence_id: str, *, directness: str, num_tokens: int
) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_key": f"key-{uid}",
        "evidence_id": evidence_id,
        "num_tokens": num_tokens,
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
