from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.phase5_selectors.build.build_map_selector_ablation_traces import main


def test_build_map_selector_ablation_traces_writes_standard_trace(tmp_path: Path) -> None:
    input_path = tmp_path / "candidate_evidence_map_features_val.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        input_path,
        [
            {
                "event_id": "evt-1",
                "claim": "The city approved the project.",
                "gold_label": "true",
                "oracle_ordered_keys": ["map-quality"],
                "evidence_map": {
                    "claim_atoms": [
                        {
                            "atom_id": "A1",
                            "text": "The city approved the project.",
                            "importance": 1.0,
                        }
                    ]
                },
                "candidates": [
                    _candidate("retrieval-only", hybrid=0.99, map_quality=0.10, base=0.99, union_rank=1),
                    _candidate("map-quality", hybrid=0.10, map_quality=0.80, base=0.10, union_rank=2),
                ],
            }
        ],
    )

    exit_code = main(
        Namespace(
            input=str(input_path),
            output_dir=str(output_dir),
            split="val",
            sample_limit=0,
            selector_name="map_selector_s2_map_quality_top5",
            top_k=1,
            candidate_top_n=20,
            chunk_mmr_fingerprint="fp123",
        )
    )

    assert exit_code == 0
    traces = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert len(traces) == 1
    assert traces[0]["selector_name"] == "map_selector_s2_map_quality_top5"
    assert traces[0]["selected_keys"] == ["map-quality"]
    assert traces[0]["candidate_pool"][0]["candidate_idx"] == 0
    assert traces[0]["candidate_scores"][1]["selector_selected_step"] == 0
    assert traces[0]["fingerprint"] == "fp123"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selector_name"] == "map_selector_s2_map_quality_top5"
    assert manifest["n_trace_rows"] == 1

    diagnostics = json.loads((output_dir / "selector_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["selector_names"] == {"map_selector_s2_map_quality_top5": 1}
    assert diagnostics["selected_count"]["mean"] == 1.0


def test_build_map_selector_ablation_traces_writes_s3_selection_steps(tmp_path: Path) -> None:
    input_path = tmp_path / "candidate_evidence_map_features_val.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        input_path,
        [
            {
                "event_id": "evt-1",
                "claim": "The city approved the project.",
                "gold_label": "true",
                "oracle_ordered_keys": [],
                "evidence_map": {
                    "claim_atoms": [
                        {"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0},
                        {"atom_id": "A2", "text": "The approval was official.", "importance": 5.0},
                    ]
                },
                "candidates": [
                    _candidate("quality-a1", hybrid=0.99, map_quality=0.99, base=0.99, union_rank=1, covered_atom_ids=["A1"]),
                    _candidate("covers-a2", hybrid=0.10, map_quality=0.10, base=0.10, union_rank=2, covered_atom_ids=["A2"]),
                ],
            }
        ],
    )

    exit_code = main(
        Namespace(
            input=str(input_path),
            output_dir=str(output_dir),
            split="val",
            sample_limit=0,
            selector_name="map_selector_s3_weighted_set_cover_top5",
            top_k=2,
            candidate_top_n=20,
            chunk_mmr_fingerprint="fp123",
        )
    )

    assert exit_code == 0
    traces = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert traces[0]["selector_name"] == "map_selector_s3_weighted_set_cover_top5"
    assert traces[0]["selected_keys"] == ["covers-a2", "quality-a1"]
    assert traces[0]["selection_steps"][0]["weighted_new_atom_gain"] == 5.0


def _candidate(
    key: str,
    *,
    hybrid: float,
    map_quality: float,
    base: float,
    union_rank: int,
    covered_atom_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "candidate_uid": key,
        "text": f"Evidence sentence for {key}.",
        "hybrid_score": hybrid,
        "evidence_map_quality_score": map_quality,
        "evidence_map_base_score": base,
        "union_pool_rank": union_rank,
        "source_group": "report:1",
        "chunk_sent_indices": [0],
        "covered_atom_ids": covered_atom_ids if covered_atom_ids is not None else ["A1"],
        "map_relation": "support",
        "map_directness": "direct",
        "map_confidence": 0.9,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
