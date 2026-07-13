from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.expand_capacity_prefix_plans import (
    materialize_prefix_lattice,
)
from scripts.phase5_selectors.build.materialize_capacity_prefix_matrix import (
    CapacityMatrixError,
    materialize_capacity_matrix,
)


def test_materialize_capacity_matrix_freezes_integrity_gate(tmp_path: Path) -> None:
    plan_root, plan_manifest, build_root = _artifacts(tmp_path)
    output = tmp_path / "matrix"
    output.mkdir()

    manifest = materialize_capacity_matrix(
        plan_manifest_path=plan_manifest,
        build_root=build_root,
        output_dir=output,
        logical_output_dir=output,
        split="val",
    )

    assert manifest["schema_version"] == "baces_capacity_prefix_matrix_v0_1"
    assert manifest["cell_count"] == 3
    assert manifest["event_count"] == 1
    assert manifest["all_ready"] is True
    gate = json.loads((output / "prefix_integrity_gate.json").read_text())
    assert gate["passed"] is True
    assert gate["adjacent_prefix_check_count"] == 2
    assert gate["plateau_check_count"] == 1
    assert manifest["cells"][2]["mean_realized_k"] == 2.0
    assert Path(manifest["cells"][0]["plan_file"]).is_file()
    assert plan_root.is_dir()


def test_materialize_capacity_matrix_rejects_final_truncation(tmp_path: Path) -> None:
    _, plan_manifest, build_root = _artifacts(tmp_path)
    bad_path = build_root / "baces_exact__prefix_k02" / "build" / "build_val.jsonl"
    row = json.loads(bad_path.read_text())
    row["was_truncated"] = True
    bad_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "matrix"
    output.mkdir()

    with pytest.raises(CapacityMatrixError, match="final build is truncated"):
        materialize_capacity_matrix(
            plan_manifest_path=plan_manifest,
            build_root=build_root,
            output_dir=output,
            logical_output_dir=output,
            split="val",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selector_level", "wrong_selector"),
        ("controller_level", "prefix_k99"),
    ],
)
def test_materialize_capacity_matrix_rejects_plan_row_factor_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    plan_root, plan_manifest, build_root = _artifacts(tmp_path)
    manifest = json.loads(plan_manifest.read_text(encoding="utf-8"))
    cell = next(
        cell for cell in manifest["cells"] if cell["cell_id"] == "baces_exact__prefix_k02"
    )
    plan_path = plan_root / cell["plan_file"]
    row = json.loads(plan_path.read_text(encoding="utf-8"))
    row[field] = value
    plan_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cell["plan_sha256"] = _sha256(plan_path)
    plan_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    output = tmp_path / "matrix"
    output.mkdir()

    with pytest.raises(CapacityMatrixError, match="plan row metadata"):
        materialize_capacity_matrix(
            plan_manifest_path=plan_manifest,
            build_root=build_root,
            output_dir=output,
            logical_output_dir=output,
            split="val",
        )


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    maximal = tmp_path / "maximal.jsonl"
    maximal.write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "trace_order_field": "selector_full_ordered_indices",
                "requested_prefix_k": 3,
                "available_prefix_k": 3,
                "prompt_feasible_prefix_k": 2,
                "selected_indices": [2, 0],
                "selected_candidate_uids": ["u2", "u0"],
                "selected_evidence_ids": ["E2", "E0"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_root = tmp_path / "plans"
    plan_root.mkdir()
    materialize_prefix_lattice(
        selector_plans={"baces_exact": maximal},
        output_dir=plan_root,
        logical_output_dir=plan_root,
        split="val",
        min_k=1,
        max_k=3,
        source_controller="ordinal_replay_minmax5_10",
    )
    plan_manifest = plan_root / "manifest.json"
    plan_payload = json.loads(plan_manifest.read_text())
    build_root = tmp_path / "builds"
    for cell in plan_payload["cells"]:
        plan_path = plan_root / cell["plan_file"]
        plan = json.loads(plan_path.read_text())
        cell_id = cell["cell_id"]
        build_dir = build_root / cell_id / "build"
        build_dir.mkdir(parents=True)
        candidates = [
            {"candidate_uid": uid, "evidence_id": evidence_id, "text": uid}
            for uid, evidence_id in zip(
                plan["selected_candidate_uids"], plan["selected_evidence_ids"]
            )
        ]
        prompt_ids = [1, 2, 3, *range(len(candidates))]
        build = {
            "event_id": "event-1",
            "label_schema": "liar6",
            "candidates": candidates,
            "evidence_count": len(candidates),
            "was_truncated": False,
            "evidence_text_truncated": False,
            "prompt_add_special_tokens": False,
            "preserve_prompt_prefix": True,
            "prompt_evidence_policy": "selected_set",
            "prompt_token_count": len(prompt_ids),
            "prompt_input_ids": prompt_ids,
            "selector_trace": {
                "selected_indices": plan["selected_indices"],
                "selection_plan": plan,
            },
        }
        (build_dir / "build_val.jsonl").write_text(
            json.dumps(build) + "\n", encoding="utf-8"
        )
    return plan_root, plan_manifest, build_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
