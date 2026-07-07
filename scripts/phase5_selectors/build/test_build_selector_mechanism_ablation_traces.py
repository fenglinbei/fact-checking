from __future__ import annotations

import json
import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from scripts.phase5_selectors.build.build_selector_mechanism_ablation_traces import main


def test_builder_writes_no_evidence_trace(tmp_path: Path) -> None:
    cache_path = _write_cache(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        Namespace(
            chunk_cache_path=str(cache_path),
            atom_union_jsonl=None,
            output_dir=str(output_dir),
            split="val",
            selector_name="selector_mech_s0_no_evidence",
            top_k=5,
            claim_pool_top_n=20,
            random_seed=0,
            merge_mmr_lambda=0.70,
            chunk_mmr_fingerprint="fp",
            sample_limit=0,
        )
    )

    assert exit_code == 0
    rows = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert rows[0]["selector_name"] == "selector_mech_s0_no_evidence"
    assert rows[0]["candidate_pool"] == []
    assert rows[0]["selected_indices"] == []


def test_builder_writes_claim_pool_hybrid_trace(tmp_path: Path) -> None:
    cache_path = _write_cache(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        Namespace(
            chunk_cache_path=str(cache_path),
            atom_union_jsonl=None,
            output_dir=str(output_dir),
            split="val",
            selector_name="selector_mech_s2_claim_pool_hybrid_top5",
            top_k=2,
            claim_pool_top_n=4,
            random_seed=0,
            merge_mmr_lambda=0.70,
            chunk_mmr_fingerprint="fp",
            sample_limit=0,
        )
    )

    assert exit_code == 0
    rows = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert len(rows[0]["candidate_pool"]) == 4
    assert rows[0]["selected_indices"] == [0, 1]
    assert rows[0]["candidate_scores"][0]["selector_selected_step"] == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selector_name"] == "selector_mech_s2_claim_pool_hybrid_top5"


def test_builder_writes_atom_union_source_score_trace(tmp_path: Path) -> None:
    cache_path = _write_cache(tmp_path)
    union_path = tmp_path / "union.jsonl"
    _write_jsonl(
        union_path,
        [
            {
                "event_id": "event0",
                "claim": "claim token",
                "label": "true",
                "gold_label": "true",
                "claim_atoms": [{"atom_id": "A1", "text": "atom"}],
                "candidates": [
                    {
                        "text": "baseline",
                        "canonical_text": "baseline",
                        "from_baseline": True,
                        "from_atom_route": False,
                        "baseline_rank": 1,
                        "atom_rrf_score": 0.0,
                        "atom_route_hit_count": 0,
                        "atom_max_route_hybrid": 0.0,
                        "union_pool_rank": 1,
                        "chunk_sent_indices": [0],
                    },
                    {
                        "text": "atom",
                        "canonical_text": "atom",
                        "from_baseline": False,
                        "from_atom_route": True,
                        "atom_pool_rank": 1,
                        "atom_rrf_score": 0.2,
                        "atom_route_hit_count": 2,
                        "atom_max_route_hybrid": 0.9,
                        "union_pool_rank": 2,
                        "chunk_sent_indices": [1],
                    },
                ],
            }
        ],
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        Namespace(
            chunk_cache_path=str(cache_path),
            atom_union_jsonl=str(union_path),
            output_dir=str(output_dir),
            split="val",
            selector_name="selector_mech_s4_atom_union_source_score_top5",
            top_k=2,
            claim_pool_top_n=4,
            random_seed=0,
            merge_mmr_lambda=0.70,
            chunk_mmr_fingerprint="fp",
            sample_limit=0,
        )
    )

    assert exit_code == 0
    rows = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert rows[0]["selected_indices"] == [1, 0]
    assert rows[0]["selected_candidates"][0]["text"] == "atom"


def test_builder_writes_full_atom_union_source_score_order(tmp_path: Path) -> None:
    cache_path = _write_cache(tmp_path)
    union_path = tmp_path / "union.jsonl"
    _write_jsonl(
        union_path,
        [
            {
                "event_id": "event0",
                "claim": "claim token",
                "label": "true",
                "gold_label": "true",
                "claim_atoms": [{"atom_id": "A1", "text": "atom"}],
                "candidates": [
                    {
                        "text": "baseline",
                        "canonical_text": "baseline",
                        "from_baseline": True,
                        "from_atom_route": False,
                        "baseline_rank": 1,
                        "atom_rrf_score": 0.0,
                        "atom_route_hit_count": 0,
                        "atom_max_route_hybrid": 0.0,
                        "union_pool_rank": 1,
                        "chunk_sent_indices": [0],
                    },
                    {
                        "text": "atom",
                        "canonical_text": "atom",
                        "from_baseline": False,
                        "from_atom_route": True,
                        "atom_pool_rank": 1,
                        "atom_rrf_score": 0.2,
                        "atom_route_hit_count": 2,
                        "atom_max_route_hybrid": 0.9,
                        "union_pool_rank": 2,
                        "chunk_sent_indices": [1],
                    },
                    {
                        "text": "weak",
                        "canonical_text": "weak",
                        "from_baseline": False,
                        "from_atom_route": True,
                        "atom_pool_rank": 2,
                        "atom_rrf_score": 0.01,
                        "atom_route_hit_count": 0,
                        "atom_max_route_hybrid": 0.0,
                        "union_pool_rank": 3,
                        "chunk_sent_indices": [2],
                    },
                ],
            }
        ],
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        Namespace(
            chunk_cache_path=str(cache_path),
            atom_union_jsonl=str(union_path),
            output_dir=str(output_dir),
            split="val",
            selector_name="selector_mech_s4_atom_union_source_score_ordered",
            top_k=2,
            claim_pool_top_n=4,
            random_seed=0,
            merge_mmr_lambda=0.70,
            chunk_mmr_fingerprint="fp",
            sample_limit=0,
        )
    )

    assert exit_code == 0
    rows = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert rows[0]["selector_ordered_indices"] == [1, 0, 2]
    assert rows[0]["selected_candidates"][0]["text"] == "atom"
    assert rows[0]["selected_candidates"][-1]["text"] == "weak"
    assert rows[0]["adaptive_policy"] == "source_score_ordered"


def _write_cache(tmp_path: Path) -> Path:
    path = tmp_path / "val.pkl"
    sample = ChunkMMRSample(
        event_id="event0",
        claim="claim token",
        label="true",
        explain="",
        candidates=[
            {"text": f"Evidence {idx} claim token", "chunk_sent_indices": [idx]}
            for idx in range(6)
        ],
        chunk_emb=np.asarray([[float(6 - idx), 0.0] for idx in range(6)], dtype=np.float32),
        claim_emb=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    with path.open("wb") as handle:
        pickle.dump([sample], handle)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
