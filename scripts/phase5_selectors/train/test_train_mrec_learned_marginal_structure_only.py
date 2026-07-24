from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from scripts.phase5_selectors.train import train_mrec_learned_marginal_structure_only as module


ROOT = Path(__file__).resolve().parents[3]


def _row(*, oracle_winner: str) -> dict[str, object]:
    return {
        "event_id": "event-1",
        "label": "false",
        "gold_label": "false",
        "oracle_ordered_keys": [oracle_winner],
        "oracle_selected_count": 1,
        "claim_atoms": [{"atom_id": "A1", "text": "claim atom"}],
        "candidates": [
            {
                "candidate_key": "weak",
                "evidence_id": "E01",
                "text": "background",
                "evidence_map_quality_score": 0.1,
                "hybrid_score": 0.1,
                "oracle_selected": oracle_winner == "weak",
                "oracle_step": 0,
                "candidate_atom_alignments": [
                    {
                        "evidence_id": "E01",
                        "atom_id": "A1",
                        "relation": "background",
                        "directness": "none",
                        "confidence": 0.1,
                    }
                ],
            },
            {
                "candidate_key": "strong",
                "evidence_id": "E02",
                "text": "direct support",
                "evidence_map_quality_score": 0.9,
                "hybrid_score": 0.9,
                "oracle_selected": oracle_winner == "strong",
                "oracle_step": 0,
                "candidate_atom_alignments": [
                    {
                        "evidence_id": "E02",
                        "atom_id": "A1",
                        "relation": "support",
                        "directness": "direct",
                        "confidence": 0.9,
                    }
                ],
            },
        ],
    }


def test_supervision_audit_preserves_poison_fields_for_core_mode_enforcement() -> None:
    row = _row(oracle_winner="weak")
    observed: Counter[str] = Counter()

    module._audit_supervision_fields(row, observed_fields=observed, is_row_root=True)

    assert row["label"] == "false"
    assert row["oracle_ordered_keys"] == ["weak"]
    assert all("oracle_selected" in candidate for candidate in row["candidates"])
    assert observed["oracle_selected"] == 2
    assert observed["oracle_step"] == 2


def test_cli_writes_structure_only_contract_and_refuses_overwrite(tmp_path: Path) -> None:
    train_input = tmp_path / "train.jsonl"
    val_input = tmp_path / "val.jsonl"
    output_dir = tmp_path / "weights"
    train_input.write_text(json.dumps(_row(oracle_winner="weak")) + "\n", encoding="utf-8")
    val_input.write_text(json.dumps(_row(oracle_winner="strong")) + "\n", encoding="utf-8")
    args = [
        "--train-input",
        str(train_input),
        "--val-input",
        str(val_input),
        "--output-dir",
        str(output_dir),
        "--candidate-top-n",
        "2",
        "--rollout-steps",
        "1",
        "--epochs",
        "2",
    ]

    assert module.main(args) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((output_dir / "val_metrics.json").read_text(encoding="utf-8"))
    weights = json.loads((output_dir / "weights.json").read_text(encoding="utf-8"))
    assert manifest["training_supervision"] == "structure_only"
    assert manifest["compute_device"] == "cpu"
    assert manifest["supervision_contract"]["core_supervision_mode"] == "structure_only"
    assert manifest["train_input_supervision_audit"]["fields_preserved_for_core_mode_enforcement"] is True
    assert manifest["train_input_supervision_audit"]["rows_with_oracle_fields"] == 1
    assert val_metrics["evaluation_target"] == "structure_winner_vs_rest"
    assert val_metrics["oracle_read_row_count"] == 0
    assert val_metrics["teacher_read_count"] == 0
    assert val_metrics["utility_read_count"] == 0
    assert val_metrics["reward_read_count"] == 0
    assert weights["metadata"]["supervision_mode"] == "structure_only"
    assert weights["metadata"]["initialized_from"] == "equal_weight_neutral_v0_1"

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        module.main(args)


def test_liar_raw_wrapper_is_cpu_only_full_data_and_has_isolated_output() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_mrec_structure_only_weights.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "TRAIN_SAMPLE_LIMIT=\"${TRAIN_SAMPLE_LIMIT:-0}\"" in text
    assert "VAL_SAMPLE_LIMIT=\"${VAL_SAMPLE_LIMIT:-0}\"" in text
    assert "05_mrec_v0_2_learned_marginal_structure_only/weights" in text
    assert "train_mrec_learned_marginal_structure_only.py" in text
    assert "Refusing to overwrite non-empty output directory" in text
