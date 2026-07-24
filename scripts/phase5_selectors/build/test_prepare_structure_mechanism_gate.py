from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.prepare_structure_mechanism_gate import (
    MechanismGateError,
    _sha256_file,
    prepare_structure_mechanism_gate,
)
from sft.label_token_matrix_infer import prepare_matrix


BASE_IDS = ("R", "H", "O", "S")
WEIGHTS_FP = "frozen-weights-fp"


def test_prepare_gate_projects_common_support_and_is_matrix_compatible(
    tmp_path: Path,
) -> None:
    cell_paths, trace_paths = _artifacts(tmp_path, include_shuffles=True)
    output_dir = tmp_path / "gate"

    manifest = prepare_structure_mechanism_gate(
        cell_paths=cell_paths,
        trace_paths=trace_paths,
        split="val",
        expected_k=2,
        retrieval_id="R",
        hard_id="H",
        one_shot_id="O",
        stateful_id="S",
        expected_weights_fingerprint=WEIGHTS_FP,
        output_dir=output_dir,
    )

    assert manifest["all_ready"] is True
    assert manifest["event_count"] == 2
    assert "selector_levels" not in manifest
    assert "controller_levels" not in manifest
    assert [cell["cell_id"] for cell in manifest["cells"]] == [
        "R",
        "H",
        "O",
        "S",
        "shuffle_seed0",
        "shuffle_seed1",
    ]
    for cell in manifest["cells"]:
        build_path = output_dir / cell["cell_id"] / "build" / "build_val.jsonl"
        assert build_path.is_file()
        assert cell["build_sha256"] == _sha256_file(build_path)
        assert [row["event_id"] for row in _read_jsonl(build_path)] == ["e0", "e1"]

    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["trace_audit"]["row_count"] == 3
    assert audit["common_support"]["dropped_reference_event_count"] == 1
    assert audit["comparisons"]["S_vs_O"]["visible_sequence_difference_rate"] == 0.5
    assert audit["activity_gate"]["passed"] is True
    assert audit["shuffle_audits"]["shuffle_seed0"]["order_change_rate"] == 1.0
    assert audit["shuffle_order_gate"]["passed"] is True

    prepared = prepare_matrix(
        matrix_manifest_path=output_dir / "manifest.json",
        build_root=output_dir,
        output_dir=tmp_path / "prepared",
        split="val",
        label_prefix="Label:",
    )
    assert prepared["cell_count"] == 6
    assert prepared["reference_count"] == 12


def test_prepare_gate_records_inactive_stateful_mechanism_as_no_go(
    tmp_path: Path,
) -> None:
    cell_paths, trace_paths = _artifacts(tmp_path, active=False)
    output_dir = tmp_path / "gate"

    manifest = prepare_structure_mechanism_gate(
        cell_paths=cell_paths,
        trace_paths=trace_paths,
        split="val",
        expected_k=2,
        retrieval_id="R",
        hard_id="H",
        one_shot_id="O",
        stateful_id="S",
        expected_weights_fingerprint=WEIGHTS_FP,
        output_dir=output_dir,
    )

    assert manifest["all_ready"] is False
    assert manifest["activity_gate_passed"] is False
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["activity_gate"]["observed_rate"] == 0.0
    assert audit["activity_gate"]["threshold"] == 0.1


def test_prepare_gate_rejects_trace_candidate_pool_drift_atomically(
    tmp_path: Path,
) -> None:
    cell_paths, trace_paths = _artifacts(tmp_path)
    rows = _read_jsonl(trace_paths["H"])
    rows[1]["candidate_pool"][1]["candidate_uid"] = "drifted"
    _write_jsonl(trace_paths["H"], rows)
    output_dir = tmp_path / "gate"

    with pytest.raises(MechanismGateError, match="candidate-pool UID sequence differs"):
        prepare_structure_mechanism_gate(
            cell_paths=cell_paths,
            trace_paths=trace_paths,
            split="val",
            expected_k=2,
            retrieval_id="R",
            hard_id="H",
            one_shot_id="O",
            stateful_id="S",
            expected_weights_fingerprint=WEIGHTS_FP,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_prepare_gate_rejects_shuffle_prompt_token_multiset_drift(
    tmp_path: Path,
) -> None:
    cell_paths, trace_paths = _artifacts(tmp_path, include_shuffles=True)
    shuffle_path = cell_paths["shuffle_seed0"]
    rows = _read_jsonl(shuffle_path)
    rows[0]["prompt_input_ids"][-1] = 999
    _write_jsonl(shuffle_path, rows)
    output_dir = tmp_path / "gate"

    with pytest.raises(MechanismGateError, match="changed prompt token multiset"):
        prepare_structure_mechanism_gate(
            cell_paths=cell_paths,
            trace_paths=trace_paths,
            split="val",
            expected_k=2,
            retrieval_id="R",
            hard_id="H",
            one_shot_id="O",
            stateful_id="S",
            expected_weights_fingerprint=WEIGHTS_FP,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_prepare_gate_rejects_weight_fingerprint_drift(tmp_path: Path) -> None:
    cell_paths, trace_paths = _artifacts(tmp_path)
    rows = _read_jsonl(trace_paths["O"])
    rows[0]["params"]["weight_fingerprint"] = "wrong"
    rows[0]["mrec_diagnostics"]["weight_fingerprint"] = "wrong"
    _write_jsonl(trace_paths["O"], rows)

    with pytest.raises(MechanismGateError, match="weight fingerprint 'wrong'"):
        prepare_structure_mechanism_gate(
            cell_paths=cell_paths,
            trace_paths=trace_paths,
            split="val",
            expected_k=2,
            retrieval_id="R",
            hard_id="H",
            one_shot_id="O",
            stateful_id="S",
            expected_weights_fingerprint=WEIGHTS_FP,
            output_dir=tmp_path / "gate",
        )


def _artifacts(
    tmp_path: Path,
    *,
    active: bool = True,
    include_shuffles: bool = False,
) -> tuple[dict[str, Path], dict[str, Path]]:
    trace_root = tmp_path / "traces"
    build_root = tmp_path / "builds"
    trace_paths: dict[str, Path] = {}
    cell_paths: dict[str, Path] = {}
    events = ("e0", "e1", "e2")
    for cell_id in BASE_IDS:
        trace_path = trace_root / f"{cell_id}.jsonl"
        rows = [_trace_row(event_id, weighted=cell_id in {"O", "S"}) for event_id in events]
        _write_jsonl(trace_path, rows)
        trace_paths[cell_id] = trace_path

    orders = {
        "R": {"e0": ("u1", "u2"), "e1": ("u1", "u2"), "e2": ("u1", "u2")},
        "H": {"e0": ("u1", "u0"), "e1": ("u0", "u1"), "e2": ("u0", "u1")},
        "O": {
            "e0": ("u0", "u1"),
            "e1": ("u1", "u0") if active else ("u0", "u1"),
            "e2": ("u0", "u1"),
        },
        "S": {"e0": ("u0", "u1"), "e1": ("u0", "u1"), "e2": ("u0", "u1")},
    }
    for cell_id in BASE_IDS:
        path = build_root / cell_id / "build_val.jsonl"
        event_order = ("e1", "e0", "e2") if cell_id == "R" else events
        rows = [
            _build_row(
                event_id,
                orders[cell_id][event_id],
                truncated=cell_id == "H" and event_id == "e2",
            )
            for event_id in event_order
        ]
        _write_jsonl(path, rows)
        cell_paths[cell_id] = path

    if include_shuffles:
        for seed in (0, 1):
            cell_id = f"shuffle_seed{seed}"
            path = build_root / cell_id / "build_val.jsonl"
            rows = [
                _build_row(event_id, ("u1", "u0"), truncated=False)
                for event_id in events
            ]
            _write_jsonl(path, rows)
            cell_paths[cell_id] = path
    return cell_paths, trace_paths


def _trace_row(event_id: str, *, weighted: bool) -> dict:
    row = {
        "event_id": event_id,
        "candidate_pool": [_candidate("u0"), _candidate("u1"), _candidate("u2")],
    }
    if weighted:
        row["params"] = {"weight_fingerprint": WEIGHTS_FP}
        row["mrec_diagnostics"] = {"weight_fingerprint": WEIGHTS_FP}
    return row


def _build_row(
    event_id: str, order: tuple[str, str], *, truncated: bool
) -> dict:
    token_by_uid = {"u0": 10, "u1": 11, "u2": 12}
    event_token = {"e0": 100, "e1": 101, "e2": 102}[event_id]
    prompt_ids = [90, event_token, *(token_by_uid[uid] for uid in order)]
    return {
        "event_id": event_id,
        "claim": f"claim:{event_id}",
        "prompt": "prompt:" + ",".join(str(token) for token in prompt_ids),
        "target": "Label: A",
        "target_token_count": 3,
        "gold_id": 0,
        "gold_label": "pants-fire",
        "gold_explain": f"explain:{event_id}",
        "label": "pants-fire",
        "label_schema": "liar6",
        "candidates": [_candidate(uid) for uid in order],
        "evidence_count": 2,
        "prompt_input_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "was_truncated": truncated,
        "evidence_text_truncated": False,
    }


def _candidate(uid: str) -> dict:
    return {"candidate_uid": uid, "text": f"text:{uid}", "evidence_id": f"E-{uid}"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
