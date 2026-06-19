from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_trace_verifier_data.py")
SPEC = importlib.util.spec_from_file_location("build_trace_verifier_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
build_trace_verifier_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_trace_verifier_data)


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del truncation
        ids = list(range(1, len(str(text).split()) + 1))
        if add_special_tokens:
            ids.insert(0, 0)
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        del tokenize
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return prompt


def test_trace_lite_prompt_fields_are_rendered_without_forbidden_trace_metadata(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="trace_lite",
        expected_selector_name="test_selector",
        top_k=2,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    assert report["trace_prompt_style"] == "trace_lite"
    assert report["evidence_count_before"]["mean"] == 2.0
    assert rows[0]["trace_prompt_style"] == "trace_lite"
    assert rows[0]["claim"].endswith("Claim atoms:\nA1: First atom\nA2: Second atom")

    prompt = rows[0]["prompt"]
    assert "Claim atoms:" in prompt
    assert "[covers=A1,A2; relation=support; directness=direct]" in prompt
    assert "[covers=none; relation=unknown; directness=unknown]" in prompt
    for forbidden in (
        "gold_label",
        "oracle_ordered_indices",
        "selector_score",
        "selection_steps",
        "adaptive_stop_reason",
        "sufficiency_state",
        "P1",
        "secret_uid",
        "secret_key",
    ):
        assert forbidden not in prompt

    assert rows[0]["candidates"][0]["candidate_uid"] == "secret_uid"
    assert rows[0]["candidates"][0]["text"].endswith("Evidence one.")


def test_plain_prompt_style_preserves_existing_prompt_shape(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="plain",
        expected_selector_name="test_selector",
        top_k=2,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    assert report["trace_prompt_style"] == "plain"
    assert rows[0]["trace_prompt_style"] == "plain"
    assert rows[0]["claim"] == "Original claim"
    assert "Claim atoms:" not in rows[0]["prompt"]
    assert "covers=A1,A2" not in rows[0]["prompt"]
    assert rows[0]["candidates"][0]["text"] == "Evidence one."


def test_qec_min_prompt_uses_question_then_atom_then_fallback_cues(tmp_path: Path) -> None:
    raw_path, trace_path = _write_qec_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="qec_min",
        expected_selector_name="test_selector",
        top_k=3,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    assert report["trace_prompt_style"] == "qec_min"
    assert row["trace_prompt_style"] == "qec_min"
    assert row["qec_diagnostics"]["cue_type_counts"] == {
        "qd_question": 1,
        "claim_atom": 1,
        "fallback": 1,
    }
    assert row["qec_diagnostics"]["qd_cue_rate"] == 1 / 3
    assert row["qec_diagnostics"]["atom_fallback_rate"] == 1 / 3
    assert row["qec_diagnostics"]["fallback_rate"] == 1 / 3
    assert [step["cue_type"] for step in row["qec_steps"]] == [
        "qd_question",
        "claim_atom",
        "fallback",
    ]
    assert row["qec_steps"][0]["question_id"] == "q1"
    assert row["qec_steps"][0]["question_focus"] == "quantity"
    assert row["qec_steps"][1]["check"] == "Second atom"
    assert row["qec_steps"][2]["check"] == "Verify the main factual claim."

    prompt = row["prompt"]
    assert "Check: Did the amount increase?" in prompt
    assert "Check: Second atom" in prompt
    assert "Check: Verify the main factual claim." in prompt
    assert "[covers=" not in prompt


def test_qec_min_prefers_trace_chain_steps_when_present(tmp_path: Path) -> None:
    raw_path, trace_path = _write_qec_chain_step_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="qec_min",
        expected_selector_name="test_selector",
        top_k=2,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    prompt = row["prompt"]
    assert report["trace_prompt_style"] == "qec_min"
    assert "Check: Atom-step cue" in prompt
    assert "Check: Second chain cue" in prompt
    assert "Chain-step evidence override." in prompt
    assert "Did route question win?" not in prompt
    assert "role=primary" not in prompt
    assert "map_confidence" not in prompt
    assert row["qec_steps"][0]["check"] == "Atom-step cue"
    assert row["qec_steps"][0]["cue_type"] == "chain_step"


def test_qec_min_anchor_only_uses_anchor_text_before_chain_step_evidence(tmp_path: Path) -> None:
    raw_path, trace_path = _write_qec_anchor_only_chain_step_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="qec_min",
        evidence_text_mode="anchor_only",
        expected_selector_name="test_selector",
        top_k=1,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    prompt = row["prompt"]
    candidate = row["candidates"][0]
    assert report["evidence_text_mode"] == "anchor_only"
    assert row["evidence_text_mode"] == "anchor_only"
    assert "Check: Atom-step cue" in prompt
    assert "Anchor-only winning sentence." in prompt
    assert "Full chunk first sentence. Full chunk second sentence." not in prompt
    assert "Chain-step evidence override." not in prompt
    assert candidate["text"] == "Check: Atom-step cue\nAnchor-only winning sentence."
    assert candidate["full_chunk_text"] == "Full chunk first sentence. Full chunk second sentence."
    assert candidate["evidence_text_mode"] == "anchor_only"
    assert candidate["evidence_text_source"] == "anchor_text"


def test_qec_map_prompt_adds_compact_map_tags(tmp_path: Path) -> None:
    raw_path, trace_path = _write_qec_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="qec_map",
        expected_selector_name="test_selector",
        top_k=3,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    assert report["trace_prompt_style"] == "qec_map"
    assert row["qec_diagnostics"]["map_relation_counts"] == {
        "support": 1,
        "refute": 1,
        "unknown": 1,
    }

    prompt = row["prompt"]
    assert "Check: Did the amount increase? [covers=A1; relation=support; directness=direct]" in prompt
    assert "Check: Second atom [covers=A2; relation=refute; directness=partial]" in prompt
    assert (
        "Check: Verify the main factual claim. [covers=none; relation=unknown; directness=unknown]"
        in prompt
    )


def test_mrec_min_prompt_uses_mrec_steps_without_visible_transition_metadata(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="mrec_min",
        expected_selector_name="test_selector",
        top_k=2,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    prompt = row["prompt"]
    assert report["trace_prompt_style"] == "mrec_min"
    assert row["trace_prompt_style"] == "mrec_min"
    assert "Check: MREC atom cue" in prompt
    assert "Check: MREC contrast cue" in prompt
    assert "MREC evidence override." in prompt
    assert "MREC contrast evidence override." in prompt
    assert "Did route question win?" not in prompt
    for forbidden in (
        "state_before",
        "state_after",
        "OPEN",
        "CONTRAST",
        "relation=",
        "directness=",
        "covers=",
    ):
        assert forbidden not in prompt

    assert row["mrec_trace_version"] == "mrec_trace_v0_1"
    assert row["mrec_selector_name"] == "mrec_greedy_transition_v0_1"
    assert row["atom_states_final"] == {"A1": "C"}
    assert row["mrec_steps"][0]["operation"] == "OPEN"
    assert row["mrec_diagnostics"]["stop_reason"] == "target_resolution_reached"
    assert [step["cue_type"] for step in row["mrec_prompt_steps"]] == ["mrec_step", "mrec_step"]
    assert row["mrec_prompt_steps"][0]["operation"] == "OPEN"
    assert row["mrec_prompt_steps"][1]["state_after"] == "C"
    assert row["mrec_prompt_diagnostics"]["operation_counts"] == {"OPEN": 1, "CONTRAST": 1}
    assert "qec_steps" not in row


def test_mrec_min_anchor_only_uses_anchor_text_before_mrec_step_evidence(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_inputs(tmp_path, with_anchor=True)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="mrec_min",
        evidence_text_mode="anchor_only",
        expected_selector_name="test_selector",
        top_k=1,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    prompt = row["prompt"]
    candidate = row["candidates"][0]
    assert report["evidence_text_mode"] == "anchor_only"
    assert "Check: MREC atom cue" in prompt
    assert "MREC anchor text." in prompt
    assert "MREC evidence override." not in prompt
    assert "Full MREC chunk text." not in prompt
    assert candidate["text"] == "Check: MREC atom cue\nMREC anchor text."
    assert candidate["full_chunk_text"] == "Full MREC chunk text."
    assert candidate["evidence_text_source"] == "anchor_text"


def test_mrec_min_falls_back_to_compat_chain_steps_when_mrec_steps_are_absent(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_inputs(tmp_path, use_compat_only=True)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="mrec_min",
        expected_selector_name="test_selector",
        top_k=1,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    row = rows[0]
    prompt = row["prompt"]
    assert report["trace_prompt_style"] == "mrec_min"
    assert "Check: Compat cue" in prompt
    assert "Compat evidence override." in prompt
    assert row["mrec_prompt_steps"][0]["source"] == "compat_chain_steps"
    assert row["mrec_prompt_steps"][0]["cue_type"] == "compat_chain_step"


def test_rawfc_boundaries_prompt_style_uses_rawfc_three_label_boundaries() -> None:
    prompt_cfg = build_trace_verifier_data._prompt_cfg_for_trace_style(
        {"auto_length": False, "output_mode": "label_only", "label_format": "letter"},
        trace_prompt_style="rawfc_boundaries",
        label_schema="rawfc3",
    )

    system_prompt = prompt_cfg["system_prompt"]

    assert "RAWFC claims" in system_prompt
    assert "false means the evidence contradicts or refutes the main claim" in system_prompt
    assert "half means the evidence supports part of the claim" in system_prompt
    assert "true means the evidence supports the main claim" in system_prompt
    assert "pants-fire" not in system_prompt
    assert "barely-true" not in system_prompt
    assert "half-true" not in system_prompt
    assert "mostly-true" not in system_prompt


def test_coverage_label_from_processed_raw_is_preserved_in_build_rows(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path, coverage_label="weak_covered")

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={"auto_length": False, "output_mode": "label_with_coverage", "label_format": "letter"},
        selection_mode="trace",
        trace_prompt_style="plain",
        expected_selector_name="test_selector",
        top_k=2,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
    )

    assert report["coverage_labels"] == {"weak_covered": 1}
    assert rows[0]["coverage_label"] == "weak_covered"
    assert "Coverage: B" in rows[0]["target"]
    assert "Evidence coverage labels:" in rows[0]["prompt"]


def _write_minimal_inputs(tmp_path: Path, coverage_label: str | None = None) -> tuple[Path, Path]:
    raw_row = {
        "event_id": "event-1",
        "claim": "Original claim",
        "label": "true",
        "explain": "",
        "reports": [],
    }
    if coverage_label is not None:
        raw_row["coverage_label"] = coverage_label
        raw_row["coverage_score"] = 0.42
    raw_path = tmp_path / "val.json"
    raw_path.write_text(
        json.dumps([raw_row]),
        encoding="utf-8",
    )

    trace_path = tmp_path / "trace.jsonl"
    trace = {
        "event_id": "event-1",
        "selector_name": "test_selector",
        "fingerprint": "fp",
        "candidate_pool": [
            {
                "text": "Evidence one.",
                "covered_atom_ids": ["A1", "A2"],
                "map_relation": "support",
                "map_directness": "direct",
                "candidate_uid": "secret_uid",
                "candidate_key": "secret_key",
            },
            {
                "text": "Evidence two.",
                "covered_atom_ids": [],
                "map_relation": "",
                "map_directness": "",
            },
        ],
        "candidate_scores": [
            {"candidate_idx": 0, "selector_score": 0.9},
            {"candidate_idx": 1},
        ],
        "selector_ordered_indices": [0, 1],
        "oracle_ordered_indices": [0],
        "oracle_ordered_keys": ["secret_key"],
        "claim_atoms": [
            {"atom_id": "A1", "text": "First\natom"},
            {"node_id": "A2", "text": "Second   atom"},
        ],
        "selection_steps": [{"rule": "P1"}],
        "adaptive_stop_reason": "sufficiency",
        "sufficiency_state": {"complete": True},
    }
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path


def _write_qec_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_row = {
        "event_id": "event-qec",
        "claim": "Original claim",
        "label": "true",
        "explain": "",
        "reports": [],
    }
    raw_path = tmp_path / "val.json"
    raw_path.write_text(json.dumps([raw_row]), encoding="utf-8")

    trace = {
        "event_id": "event-qec",
        "selector_name": "test_selector",
        "fingerprint": "fp",
        "candidate_pool": [
            {
                "text": "Evidence with a QD route.",
                "candidate_idx": 10,
                "evidence_id": "E10",
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
                "qd_question_routes": [
                    {
                        "question_id": "q2",
                        "question": "Did the less relevant question match?",
                        "focus": "overall",
                        "rank": 2,
                        "hybrid_score": 0.99,
                    },
                    {
                        "question_id": "q1",
                        "question": "Did the amount increase?",
                        "focus": "quantity",
                        "rank": 1,
                        "hybrid_score": 0.50,
                    },
                ],
            },
            {
                "text": "Evidence with only an atom.",
                "candidate_idx": 11,
                "evidence_id": "E11",
                "covered_atom_ids": ["A2"],
                "map_relation": "refute",
                "map_directness": "partial",
            },
            {
                "text": "Evidence with no cue metadata.",
                "candidate_idx": 12,
                "evidence_id": "E12",
                "covered_atom_ids": [],
                "map_relation": "",
                "map_directness": "",
            },
        ],
        "candidate_scores": [
            {"candidate_idx": 0},
            {"candidate_idx": 1},
            {"candidate_idx": 2},
        ],
        "selector_ordered_indices": [0, 1, 2],
        "oracle_ordered_indices": [0],
        "claim_atoms": [
            {"atom_id": "A1", "text": "First atom", "importance": 0.4},
            {"atom_id": "A2", "text": "Second atom", "importance": 0.9},
        ],
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path


def _write_qec_chain_step_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_row = {
        "event_id": "event-chain-qec",
        "claim": "Original claim",
        "label": "true",
        "explain": "",
        "reports": [],
    }
    raw_path = tmp_path / "val.json"
    raw_path.write_text(json.dumps([raw_row]), encoding="utf-8")

    trace = {
        "event_id": "event-chain-qec",
        "selector_name": "test_selector",
        "fingerprint": "fp",
        "candidate_pool": [
            {
                "text": "Evidence with route question.",
                "candidate_idx": 0,
                "evidence_id": "E20",
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
                "map_confidence": 0.91,
                "qd_question_routes": [
                    {
                        "question_id": "q-route",
                        "question": "Did route question win?",
                        "focus": "quantity",
                        "rank": 1,
                        "hybrid_score": 0.99,
                    }
                ],
            },
            {
                "text": "Second candidate evidence.",
                "candidate_idx": 1,
                "evidence_id": "E21",
                "covered_atom_ids": ["A2"],
                "map_relation": "refute",
                "map_directness": "partial",
            },
        ],
        "candidate_scores": [
            {"candidate_idx": 0},
            {"candidate_idx": 1},
        ],
        "selector_ordered_indices": [0, 1],
        "oracle_ordered_indices": [0],
        "claim_atoms": [
            {"atom_id": "A1", "text": "First atom", "importance": 0.4},
            {"atom_id": "A2", "text": "Second atom", "importance": 0.9},
        ],
        "chain_steps": [
            {
                "step": 1,
                "candidate_idx": 0,
                "selector_candidate_idx": 0,
                "evidence_id": "E20",
                "cue_text": "Atom-step cue",
                "cue_source": "qd_question",
                "evidence_text": "Chain-step evidence override.",
                "role": "primary",
                "relation": "support",
                "directness": "direct",
                "map_confidence": 0.91,
                "covered_atom_ids": ["A1"],
            },
            {
                "step": 2,
                "candidate_idx": 1,
                "selector_candidate_idx": 1,
                "evidence_id": "E21",
                "cue_text": "Second chain cue",
                "cue_source": "claim_atom",
                "evidence_text": "Second chain evidence override.",
                "role": "primary",
                "relation": "refute",
                "directness": "partial",
                "covered_atom_ids": ["A2"],
            },
        ],
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path


def _write_qec_anchor_only_chain_step_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_path, trace_path = _write_qec_chain_step_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["candidate_pool"][0]["text"] = "Full chunk first sentence. Full chunk second sentence."
    trace["candidate_pool"][0]["anchor_text"] = "Anchor-only winning sentence."
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path


def _write_mrec_inputs(
    tmp_path: Path,
    *,
    with_anchor: bool = False,
    use_compat_only: bool = False,
) -> tuple[Path, Path]:
    raw_row = {
        "event_id": "event-mrec",
        "claim": "Original claim",
        "label": "true",
        "explain": "",
        "reports": [],
    }
    raw_path = tmp_path / "val.json"
    raw_path.write_text(json.dumps([raw_row]), encoding="utf-8")

    first_text = "Full MREC chunk text." if with_anchor else "Evidence with route question."
    first_candidate = {
        "text": first_text,
        "candidate_idx": 0,
        "evidence_id": "E30",
        "covered_atom_ids": ["A1"],
        "map_relation": "support",
        "map_directness": "direct",
        "qd_question_routes": [
            {
                "question_id": "q-route",
                "question": "Did route question win?",
                "focus": "quantity",
                "rank": 1,
                "hybrid_score": 0.99,
            }
        ],
    }
    if with_anchor:
        first_candidate["anchor_text"] = "MREC anchor text."

    trace = {
        "event_id": "event-mrec",
        "selector_name": "test_selector",
        "fingerprint": "fp",
        "candidate_pool": [
            first_candidate,
            {
                "text": "Second MREC candidate evidence.",
                "candidate_idx": 1,
                "evidence_id": "E31",
                "covered_atom_ids": ["A1"],
                "map_relation": "refute",
                "map_directness": "partial",
            },
        ],
        "candidate_scores": [
            {"candidate_idx": 0},
            {"candidate_idx": 1},
        ],
        "selector_ordered_indices": [0, 1],
        "oracle_ordered_indices": [0],
        "claim_atoms": [
            {"atom_id": "A1", "text": "First atom", "importance": 1.0},
        ],
        "mrec_trace_version": "mrec_trace_v0_1",
        "mrec_selector_name": "mrec_greedy_transition_v0_1",
        "atom_states_initial": {"A1": "U"},
        "atom_states_final": {"A1": "C"},
        "mrec_diagnostics": {
            "stop_reason": "target_resolution_reached",
            "resolved_atom_rate": 1.0,
        },
        "mrec_steps": [
            {
                "step": 1,
                "operation": "OPEN",
                "atom_id": "A1",
                "atom_text": "First atom",
                "state_before": "U",
                "state_after": "S",
                "cue_text": "MREC atom cue",
                "cue_source": "claim_atom",
                "candidate_idx": 0,
                "selector_candidate_idx": 0,
                "evidence_id": "E30",
                "evidence_text": "MREC evidence override.",
                "covered_atom_ids": ["A1"],
                "relation": "support",
                "directness": "direct",
                "token_cost": 4,
                "transition_reason": "A1 changes from unresolved to S",
            },
            {
                "step": 2,
                "operation": "CONTRAST",
                "atom_id": "A1",
                "atom_text": "First atom",
                "state_before": "S",
                "state_after": "C",
                "cue_text": "MREC contrast cue",
                "cue_source": "claim_atom",
                "candidate_idx": 1,
                "selector_candidate_idx": 1,
                "evidence_id": "E31",
                "evidence_text": "MREC contrast evidence override.",
                "covered_atom_ids": ["A1"],
                "relation": "refute",
                "directness": "partial",
                "token_cost": 5,
                "transition_reason": "A1 changes from S to C",
            },
        ],
        "compat_chain_steps": [
            {
                "step": 1,
                "candidate_idx": 0,
                "selector_candidate_idx": 0,
                "evidence_id": "E30",
                "cue_text": "Compat cue",
                "cue_source": "claim_atom",
                "evidence_text": "Compat evidence override.",
                "role": "open",
                "covered_atom_ids": ["A1"],
            }
        ],
    }
    if use_compat_only:
        trace.pop("mrec_steps")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path
