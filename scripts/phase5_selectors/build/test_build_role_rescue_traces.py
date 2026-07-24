from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.phase5_selectors.build.build_role_rescue_traces import (
    CELLS,
    build_role_rescue_rows,
    materialize_role_rescue_traces,
)


def test_cells_share_resolving_core_and_promote_only_valid_roles() -> None:
    source = _source_row()
    rows = build_role_rescue_rows(source, k=5, seed=17)

    assert tuple(rows) == CELLS
    assert _source_indices(rows["r_only"]) == [0]
    assert len(_source_indices(rows["r_only"])) == 1
    for cell in CELLS:
        assert _source_indices(rows[cell])[:1] == [0]
    for cell in CELLS[1:]:
        assert len(_source_indices(rows[cell])) == 5

    cor_meta = rows["cor"]["role_rescue_metadata"]
    assert cor_meta["available_role_source_indices"]["cor"] == [1]
    assert cor_meta["promoted_role_source_indices"] == {"cor": [1]}
    assert _source_indices(rows["cor"])[1] == 1

    opp_meta = rows["opp"]["role_rescue_metadata"]
    assert set(opp_meta["available_role_source_indices"]["opp"]) == {3, 4}
    assert opp_meta["promoted_role_source_indices"]["opp"] == [3]
    assert _source_indices(rows["opp"])[1] == 3

    ctx_meta = rows["ctx"]["role_rescue_metadata"]
    assert ctx_meta["available_role_source_indices"]["ctx"] == [5, 8]
    assert 6 not in ctx_meta["available_role_source_indices"]["ctx"]
    assert 7 not in ctx_meta["available_role_source_indices"]["ctx"]
    assert ctx_meta["promoted_role_source_indices"]["ctx"] == [5]

    full_meta = rows["full"]["role_rescue_metadata"]
    assert set(full_meta["promoted_role_source_indices"]) == {"cor", "opp", "ctx"}
    promoted_indices = [
        idx
        for indices in full_meta["promoted_role_source_indices"].values()
        for idx in indices
    ]
    assert len(set(promoted_indices)) == 3
    assert _source_indices(rows["full"])[1:4] == [
        full_meta["promoted_role_source_indices"][role][0]
        for role in ("cor", "opp", "ctx")
    ]


def test_random_and_retrieval_controls_have_explicit_distinct_fill_contracts() -> None:
    rows = build_role_rescue_rows(_source_row(), k=5, seed=19)
    random_meta = rows["random"]["role_rescue_metadata"]
    retr_meta = rows["retr"]["role_rescue_metadata"]

    assert _source_indices(rows["random"])[1:] == random_meta[
        "stable_random_source_indices"
    ][:4]
    # Candidate 8 has the highest retrieval score, followed by 7, 6, and 5.
    assert _source_indices(rows["retr"]) == [0, 8, 7, 6, 5]
    assert retr_meta["fill_policy"] == "retrieval_fill"
    assert random_meta["fill_policy"] == "stable_random_fill"


def test_no_resolution_uses_first_k_as_core_and_leaves_no_rescue_capacity() -> None:
    source = _source_row()
    for step in source["mrec_steps"]:
        step["trace_state"]["target_resolved"] = False
    rows = build_role_rescue_rows(source, k=5, seed=0)

    for cell in CELLS:
        assert _source_indices(rows[cell]) == [0, 1, 2, 3, 4]
        assert rows[cell]["role_rescue_metadata"]["core_stop_reason"] == "k_cap"
        assert rows[cell]["role_rescue_metadata"]["promoted_role_source_indices"] == {}


def test_learned_fixed5_preserves_history_while_all_controls_avoid_duplicates() -> None:
    source = _source_row()
    source["candidate_pool"][0]["duplicate_group"] = "dup-core"
    source["candidate_pool"][1]["duplicate_group"] = "dup-core"
    rows = build_role_rescue_rows(source, k=5, seed=7)

    assert _source_indices(rows["learned_fixed5"]) == [0, 1, 2, 3, 4]
    for cell in ("random", "retr", "cor", "opp", "ctx", "full"):
        assert 1 not in _source_indices(rows[cell])


def test_role_remainder_can_select_unpromoted_eligible_candidates() -> None:
    source = _source_row()
    source["candidate_pool"] = source["candidate_pool"][:5]
    source["mrec_steps"] = source["mrec_steps"][:5]
    source["selector_ordered_indices"] = list(range(5))
    # Candidate 2 becomes a second candidate for the already-filled cor slot.
    source["candidate_pool"][2]["source_group"] = "s-new"
    rows = build_role_rescue_rows(source, k=5, seed=0)

    assert set(_source_indices(rows["full"])) == set(range(5))
    assert 2 in rows["full"]["role_rescue_metadata"][
        "random_or_retrieval_fill_source_indices"
    ]


def test_cor_uses_separate_atom_direction_slots() -> None:
    source = _source_row()
    source["mrec_steps"][0]["trace_state"]["target_resolved"] = False
    source["mrec_steps"][1]["trace_state"]["target_resolved"] = True
    source["candidate_pool"][1]["source_group"] = "s1"
    source["candidate_pool"][1]["candidate_atom_alignments"] = [
        _alignment("E01", "A1", relation="refute", directness="direct", confidence=0.9)
    ]
    source["candidate_pool"][2]["source_group"] = "support-new"
    source["candidate_pool"][3]["source_group"] = "refute-new"
    rows = build_role_rescue_rows(source, k=4, seed=0)

    assert set(rows["cor"]["role_rescue_metadata"]["realized_atom_role_slots"]) == {
        "cor:A1|support",
        "cor:A1|refute",
    }


def test_context_rejects_insufficient_marked_irrelevant_even_with_confidence_and_span() -> None:
    source = _source_row()
    pair = source["candidate_pool"][6]["candidate_atom_alignments"][0]
    pair.update({"confidence": 0.99, "key_spans": ["nonempty"], "evidence_role": "irrelevant"})
    rows = build_role_rescue_rows(source, k=5, seed=0)

    assert 6 not in rows["ctx"]["role_rescue_metadata"]["available_role_source_indices"]["ctx"]


def test_atom_role_slots_follow_learned_suffix_and_full_interleaves_roles() -> None:
    source = _source_row()
    source["claim_atoms"].append({"atom_id": "A2", "text": "The second atom"})
    source["candidate_pool"][0]["candidate_atom_alignments"].append(
        _alignment("E00", "A2", relation="support", directness="direct", confidence=0.9)
    )
    # Two independent corroboration slots, deliberately encountered A2 then A1.
    source["candidate_pool"][1]["candidate_atom_alignments"] = [
        _alignment("E01", "A2", relation="support", directness="direct", confidence=0.9)
    ]
    source["candidate_pool"][2]["source_group"] = "s3"
    source["candidate_pool"][2]["candidate_atom_alignments"] = [
        _alignment("E02", "A1", relation="support", directness="direct", confidence=0.9)
    ]
    source["candidate_pool"][3]["candidate_atom_alignments"] = [
        _alignment("E03", "A2", relation="refute", directness="direct", confidence=0.9)
    ]
    source["candidate_pool"][5]["candidate_atom_alignments"] = [
        _alignment("E05", "A2", relation="background", directness="context", confidence=0.9)
    ]
    source["mrec_steps"][3]["atom_id"] = "A2"
    source["mrec_steps"][5]["atom_id"] = "A2"
    # Full must see ctx, opp, and cor in this learned order; the SHA order is
    # used only after role-slot acquisition.
    source["selector_ordered_indices"] = [0, 5, 3, 1, 4, 2, 6, 7, 8]
    rows = build_role_rescue_rows(source, k=7, seed=999)

    cor_meta = rows["cor"]["role_rescue_metadata"]
    assert cor_meta["realized_atom_role_slots"] == {
        "cor:A1|support": {"candidate_uid": "u2", "source_candidate_idx": 2},
        "cor:A2|support": {"candidate_uid": "u1", "source_candidate_idx": 1},
    }
    assert _source_indices(rows["cor"])[1:3] == [1, 2]

    full_meta = rows["full"]["role_rescue_metadata"]
    promoted = set(full_meta["realized_atom_role_slots"])
    assert {
        "ctx:A2",
        "opp:A2",
        "cor:A2|support",
        "opp:A1",
        "cor:A1|support",
    } <= promoted
    assert _source_indices(rows["full"])[1:6] == [5, 3, 1, 4, 2]


def test_selection_is_deterministic_and_does_not_consult_gold_label() -> None:
    source = _source_row()
    first = build_role_rescue_rows(source, k=5, seed=23)
    second = build_role_rescue_rows(deepcopy(source), k=5, seed=23)
    poisoned = deepcopy(source)
    poisoned["gold_label"] = "POISON"
    third = build_role_rescue_rows(poisoned, k=5, seed=23)

    assert first == second
    for cell in CELLS:
        assert _source_indices(first[cell]) == _source_indices(third[cell])
        assert first[cell]["role_rescue_metadata"]["selection_uses_gold_label"] is False


def test_projected_trace_is_selected_set_compatible_and_cue_only() -> None:
    rows = build_role_rescue_rows(_source_row(), k=5, seed=3)
    for cell, row in rows.items():
        count = len(row["candidate_pool"])
        assert row["selector_ordered_indices"] == list(range(count))
        assert row["display_ordered_indices"] == list(range(count))
        assert row["selected_indices"] == list(range(count))
        assert len(row["mrec_steps"]) == count
        assert [candidate["candidate_idx"] for candidate in row["candidate_pool"]] == list(
            range(count)
        )
        assert [
            candidate["source_candidate_idx"] for candidate in row["candidate_pool"]
        ] == _source_indices(row)
        for step, candidate in zip(row["mrec_steps"], row["candidate_pool"]):
            assert step["selector_candidate_idx"] == candidate["selector_candidate_idx"]
            assert step["candidate_uid"] == candidate["candidate_uid"]
            assert step["cue_text"] == "The claim atom"
            assert "operation" not in step
            assert "state_before" not in step
            assert "state_after" not in step
            assert "role" not in step
        assert row["params"]["prompt_evidence_policy"] == "selected_set"
        assert row["selector_name"] == f"role_rescue_{cell}_v0_1"


def test_streaming_materializer_writes_all_cells_and_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(_source_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "role_rescue"

    manifest = materialize_role_rescue_traces(
        input_path=source_path,
        output_dir=output_dir,
        split="val",
        k=5,
        seed=11,
    )

    assert manifest["row_count"] == 1
    assert manifest["core_count"]["mean"] == 1.0
    assert manifest["core_stop_reasons"] == {"target_resolved": 1}
    assert set(manifest["cells"]) == set(CELLS)
    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    for cell in CELLS:
        path = output_dir / cell / "selection_trace_val.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["role_rescue_metadata"]["cell"] == cell
    assert manifest["cells"]["cor"]["role_promoted_count"]["cor"] == 1
    assert manifest["cells"]["full"]["role_promoted_count"] == {
        "cor": 1,
        "ctx": 1,
        "opp": 1,
    }


def _source_indices(row: dict) -> list[int]:
    return list(row["role_rescue_metadata"]["selected_source_indices"])


def _source_row() -> dict:
    candidates = [
        _candidate(0, relation="support", directness="direct", source="s1", retrieval=0.10),
        _candidate(1, relation="support", directness="direct", source="s2", retrieval=0.20),
        _candidate(2, relation="support", directness="direct", source="s1", retrieval=0.30),
        _candidate(3, relation="refute", directness="direct", source="s1", retrieval=0.40),
        _candidate(4, relation="qualify", directness="partial", source="s3", retrieval=0.50),
        _candidate(5, relation="background", directness="context", source="s4", retrieval=0.60),
        _candidate(
            6,
            relation="insufficient",
            directness="none",
            source="s5",
            retrieval=0.70,
            confidence=0.0,
            key_spans=[],
        ),
        _candidate(
            7,
            relation="irrelevant",
            directness="none",
            source="s6",
            retrieval=0.80,
            confidence=0.0,
            key_spans=[],
        ),
        _candidate(
            8,
            relation="background",
            directness="none",
            source="s7",
            retrieval=0.90,
            confidence=0.0,
            key_spans=[],
        ),
    ]
    steps = []
    for idx, candidate in enumerate(candidates):
        relation = candidate["candidate_atom_alignments"][0]["relation"]
        steps.append(
            {
                "selector_candidate_idx": idx,
                "candidate_idx": idx,
                "atom_id": "A1",
                "relation": relation,
                "operation": "CONTRAST" if idx in {3, 4} else "FALLBACK",
                "trace_state": {"target_resolved": idx == 0},
            }
        )
    return {
        "event_id": "event-1",
        "claim": "A claim",
        "gold_label": "true",
        "selector_name": "learned_fullpool",
        "fingerprint": "abc123",
        "candidate_pool_metadata": {"chunk_mmr_fingerprint": "abc123"},
        "claim_atoms": [{"atom_id": "A1", "text": "The claim atom"}],
        "candidate_pool": candidates,
        "selector_ordered_indices": list(range(len(candidates))),
        "mrec_steps": steps,
    }


def _candidate(
    idx: int,
    *,
    relation: str,
    directness: str,
    source: str,
    retrieval: float,
    confidence: float = 0.9,
    key_spans: list[str] | None = None,
) -> dict:
    evidence_id = f"E{idx:02d}"
    return {
        "candidate_uid": f"u{idx}",
        "candidate_key": f"key-{idx}",
        "evidence_id": evidence_id,
        "text": f"Evidence text {idx}",
        "source_group": source,
        "baseline_hybrid_score": retrieval,
        "union_pool_rank": idx + 1,
        "mrec_token_cost": idx + 2,
        "covered_atom_ids": ["A1"],
        "candidate_atom_alignments": [
            {
                "evidence_id": evidence_id,
                "atom_id": "A1",
                "relation": relation,
                "directness": directness,
                "confidence": confidence,
                "key_spans": ["span"] if key_spans is None else key_spans,
            }
        ],
    }


def _alignment(
    evidence_id: str,
    atom_id: str,
    *,
    relation: str,
    directness: str,
    confidence: float,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "atom_id": atom_id,
        "relation": relation,
        "directness": directness,
        "confidence": confidence,
        "key_spans": ["span"],
    }
