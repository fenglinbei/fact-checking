from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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


class MistralCommonFakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in str(text).split():
            token_id = self._token_to_id.get(token)
            if token_id is None:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(token_id)
        return ids

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del truncation
        ids = self._encode(text)
        if add_special_tokens:
            ids.insert(0, self.eos_token_id)
        return {"input_ids": ids}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(
            self._id_to_token[token_id]
            for token_id in token_ids
            if not skip_special_tokens or token_id != self.eos_token_id
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> str | list[int]:
        del kwargs
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return self._encode(prompt) if tokenize else prompt


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


def test_allow_empty_evidence_builds_claim_only_prompt(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["selector_name"] = "selector_mech_s0_no_evidence"
    trace["candidate_pool"] = []
    trace["candidate_scores"] = []
    trace["selector_ordered_indices"] = []
    trace["selected_indices"] = []
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

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
        expected_selector_name="selector_mech_s0_no_evidence",
        top_k=5,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        allow_empty_evidence=True,
    )

    assert report["evidence_count"]["mean"] == 0.0
    assert rows[0]["evidence_count"] == 0
    assert rows[0]["candidates"] == []
    assert "(no evidence available)" in rows[0]["prompt"]


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


def test_mrec_prompt_evidence_minmax_policy_stops_after_target_state(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)

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
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "minmax",
            "min_evidence_count": 2,
            "max_evidence_count": 4,
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0, 1]
    assert [step["evidence_id"] for step in row["mrec_prompt_steps"]] == ["E40", "E41"]
    assert row["prompt_evidence_policy"] == "minmax"
    assert row["prompt_evidence_min_count"] == 2
    assert row["prompt_evidence_max_count"] == 4
    assert row["prompt_evidence_selected_count_before_prompt_truncation"] == 2
    assert row["prompt_evidence_stop_reason"] == "target_resolved"
    assert report["prompt_evidence"]["policy"] == "minmax"
    assert report["prompt_evidence"]["stop_reasons"] == {"target_resolved": 1}


def test_selected_set_policy_consumes_exact_variable_length_order() -> None:
    trace = {
        "candidate_pool": [
            {"mrec_token_cost": 2},
            {"mrec_token_cost": 3},
            {"mrec_token_cost": 4},
        ]
    }

    decision = build_trace_verifier_data._select_prompt_evidence_indices(
        trace,
        ordered_indices=[2, 0],
        config={
            "policy": "selected_set",
            "min_evidence_count": 0,
            "max_evidence_count": 0,
            "evidence_token_budget": None,
        },
    )

    assert decision["selected_indices"] == [2, 0]
    assert decision["selected_count_before_prompt_truncation"] == 2
    assert decision["min_evidence_count"] == 2
    assert decision["max_evidence_count"] == 2
    assert decision["selected_token_cost"] == 6
    assert decision["stop_reason"] == "selected_set_exhausted"

    assert build_trace_verifier_data._ordered_trace_indices(
        {
            "display_ordered_indices": [2, 0],
            "selector_ordered_indices": [1],
        }
    ) == [2, 0]


def test_trace_order_field_can_use_frozen_full_selector_order() -> None:
    trace = {
        "display_ordered_indices": [0],
        "selector_ordered_indices": [0],
        "selector_full_ordered_indices": [2, 0, 1],
    }

    assert build_trace_verifier_data._ordered_trace_indices(
        trace,
        field="selector_full_ordered_indices",
    ) == [2, 0, 1]

    with pytest.raises(ValueError, match="trace has no selector_available_ordered_indices"):
        build_trace_verifier_data._ordered_trace_indices(
            trace,
            field="selector_available_ordered_indices",
        )


def test_selection_plan_replays_prompt_feasible_full_order_prefix(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["selector_full_ordered_indices"] = [1, 0]
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    plan_path = tmp_path / "prefix_plan.jsonl"
    plan_path.write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "requested_prefix_k": 2,
                "prompt_feasible_prefix_k": 1,
                "selected_indices": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
        trace_order_field="selector_full_ordered_indices",
        selection_plan_path=plan_path,
        trace_prompt_style="plain",
        expected_selector_name="test_selector",
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={"policy": "selected_set"},
        forbid_prompt_truncation=True,
    )

    assert rows[0]["selector_trace"]["selected_indices"] == [1]
    assert rows[0]["selector_trace"]["trace_order_field"] == "selector_full_ordered_indices"
    assert rows[0]["selector_trace"]["selection_plan"]["requested_prefix_k"] == 2
    assert rows[0]["candidates"][0]["text"] == "Evidence two."
    assert report["selection_plan"]["enabled"] is True
    assert report["selection_plan"]["row_count"] == 1


def test_selection_plan_rejects_nonprefix_indices() -> None:
    trace = {
        "candidate_pool": [
            {"candidate_uid": "u0"},
            {"candidate_uid": "u1"},
        ]
    }

    with pytest.raises(ValueError, match="exact prefix"):
        build_trace_verifier_data._selected_indices_from_plan(
            trace,
            ordered_indices=[0, 1],
            plan={
                "selected_indices": [1],
                "requested_prefix_k": 1,
                "prompt_feasible_prefix_k": 1,
            },
            event_id="event-1",
        )


def test_forbid_prompt_truncation_rejects_auto_length_tail_deletion(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)
    common = {
        "split": "val",
        "source_type": "trace",
        "source_path": trace_path,
        "raw_path": raw_path,
        "dataset": None,
        "label_schema": "liar6",
        "tokenizer": _FakeTokenizer(),
        "prompt_cfg": {
            "auto_length": True,
            "max_length": 132,
            "output_mode": "label_only",
            "label_format": "letter",
        },
        "selection_mode": "trace",
        "trace_prompt_style": "plain",
        "expected_selector_name": "test_selector",
        "top_k": 2,
        "random_seed": 0,
        "expected_chunk_mmr_fingerprint": "fp",
        "sample_limit": None,
        "show_progress": False,
    }

    rows, report = build_trace_verifier_data._build_split(**common)

    assert rows[0]["evidence_count_before"] == 2
    assert rows[0]["evidence_count"] == 1
    assert rows[0]["was_truncated"] is True
    assert rows[0]["evidence_text_truncated"] is False
    assert report["forbid_prompt_truncation"] is False

    with pytest.raises(
        ValueError,
        match=(
            r"val:event-1: prompt truncation is forbidden: "
            r"was_truncated=True, evidence_text_truncated=False"
        ),
    ):
        build_trace_verifier_data._build_split(
            **common,
            forbid_prompt_truncation=True,
        )


def test_forbid_prompt_truncation_accepts_clean_selected_set(tmp_path: Path) -> None:
    raw_path, trace_path = _write_minimal_inputs(tmp_path)

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=_FakeTokenizer(),
        prompt_cfg={
            "auto_length": True,
            "max_length": 160,
            "output_mode": "label_only",
            "label_format": "letter",
        },
        selection_mode="trace",
        trace_prompt_style="plain",
        expected_selector_name="test_selector",
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={"policy": "selected_set"},
        forbid_prompt_truncation=True,
    )

    assert rows[0]["prompt_evidence_policy"] == "selected_set"
    assert rows[0]["was_truncated"] is False
    assert rows[0]["evidence_text_truncated"] is False
    assert report["forbid_prompt_truncation"] is True


def test_mrec_prompt_evidence_budget_policy_records_max_length_guard(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)

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
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "budget",
            "min_evidence_count": 1,
            "max_evidence_count": 4,
            "evidence_token_budget": 9,
            "max_length_guard": {
                "enabled": True,
                "build_prompt_max_length": 50,
                "sft_train_max_length": 10,
                "reserve_tokens": 0,
                "on_violation": "warn",
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0, 1]
    assert row["prompt_evidence_policy"] == "budget"
    assert row["prompt_evidence_token_budget"] == 9
    assert row["prompt_evidence_selected_count_before_prompt_truncation"] == 2
    assert row["prompt_evidence_stop_reason"] == "token_budget_exhausted"
    assert report["prompt_evidence"]["policy"] == "budget"
    assert report["prompt_evidence"]["stop_reasons"] == {"token_budget_exhausted": 1}
    assert report["max_length_guard"]["enabled"] is True
    assert report["max_length_guard"]["on_violation"] == "warn"
    assert report["max_length_guard"]["config_conflict"] is True
    assert report["max_length_guard"]["violation_count"] == 1


def test_prompt_budget_requires_explicit_final_prompt_budget() -> None:
    with pytest.raises(ValueError, match="requires a positive prompt_token_budget"):
        build_trace_verifier_data._normalize_prompt_evidence_config(
            {"policy": "prompt_budget", "min_evidence_count": 1},
            fallback_top_k=10,
            prompt_cfg={"max_length": 1024},
        )


@pytest.mark.parametrize("prompt_budget", [512, 768, 1024])
def test_prompt_budget_selects_largest_exact_verifier_prefix(
    tmp_path: Path,
    prompt_budget: int,
) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for idx, (candidate, step) in enumerate(zip(trace["candidate_pool"], trace["mrec_steps"])):
        long_text = " ".join([f"evidence{idx}"] * 190)
        candidate["text"] = long_text
        step["evidence_text"] = long_text
        step["token_cost"] = 1
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    tokenizer = MistralCommonFakeTokenizer()
    prompt_cfg = {
        "auto_length": False,
        "max_length": 1400,
        "output_mode": "label_only",
        "label_format": "letter",
    }
    base_candidates = build_trace_verifier_data._selected_candidates(
        trace,
        list(range(len(trace["candidate_pool"]))),
        selection_mode="trace",
    )
    prefix_counts: list[int] = []
    for count in range(1, len(base_candidates) + 1):
        claim, rendered, _, _ = build_trace_verifier_data._apply_trace_prompt_style(
            claim="Original claim",
            candidates=base_candidates[:count],
            trace=trace,
            trace_prompt_style="mrec_min",
        )
        rendered_row = build_trace_verifier_data.build_training_row(
            {
                "event_id": "event-mrec-policy",
                "claim": claim,
                "label": "true",
                "label_schema": "liar6",
                "explain": "",
                "candidates": rendered,
            },
            tokenizer,
            prompt_cfg,
        )
        prefix_counts.append(int(rendered_row["prompt_token_count"]))
    target_reserve = int(rendered_row["target_token_count"])
    effective_budget = min(prompt_budget, 1400 - target_reserve)
    expected_count = max(
        count
        for count, token_count in enumerate(prefix_counts, start=1)
        if token_count <= effective_budget
    )

    rows, report = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        selection_mode="trace",
        trace_prompt_style="mrec_min",
        expected_selector_name="test_selector",
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "prompt_budget",
            "min_evidence_count": 1,
            "max_evidence_count": 4,
            "prompt_token_budget": prompt_budget,
            "max_length_guard": {
                "enabled": True,
                "build_prompt_max_length": 1400,
                "sft_train_max_length": 1400,
                "reserve_tokens": 0,
                "on_violation": "error",
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == list(range(expected_count))
    assert row["evidence_count"] == expected_count
    assert len(row["candidates"]) == expected_count
    assert len(row["mrec_prompt_steps"]) == expected_count
    assert len(row["prompt_input_ids"]) == row["prompt_token_count"] <= prompt_budget
    assert row["prompt_evidence_prompt_token_budget"] == prompt_budget
    assert row["prompt_evidence_effective_prompt_token_budget"] == effective_budget
    assert row["prompt_evidence_considered_count"] == 4
    assert row["prompt_evidence_selected_token_cost"] == expected_count
    assert row["prompt_evidence_partial_evidence"] is False
    assert row["was_truncated"] is False
    assert row.get("evidence_text_truncated", False) is False
    if expected_count < len(prefix_counts):
        assert prefix_counts[expected_count] > effective_budget
        assert row["prompt_evidence_stop_reason"] == "prompt_token_budget_exhausted"
    assert report["prompt_evidence"]["prompt_token_budget"] == prompt_budget
    assert report["max_length_guard"]["violation_count"] == 0


def test_prompt_budget_explicitly_truncates_oversized_first_evidence(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    oversized = " ".join(["oversized"] * 1200)
    trace["candidate_pool"][0]["text"] = oversized
    trace["mrec_steps"][0]["evidence_text"] = oversized
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    rows, _ = build_trace_verifier_data._build_split(
        split="val",
        source_type="trace",
        source_path=trace_path,
        raw_path=raw_path,
        dataset=None,
        label_schema="liar6",
        tokenizer=MistralCommonFakeTokenizer(),
        prompt_cfg={
            "auto_length": False,
            "max_length": 1024,
            "output_mode": "label_only",
            "label_format": "letter",
        },
        selection_mode="trace",
        trace_prompt_style="mrec_min",
        expected_selector_name="test_selector",
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "prompt_budget",
            "min_evidence_count": 1,
            "max_evidence_count": 4,
            "prompt_token_budget": 512,
            "max_length_guard": {
                "enabled": True,
                "build_prompt_max_length": 1024,
                "sft_train_max_length": 1024,
                "on_violation": "error",
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0]
    assert row["evidence_count"] == len(row["candidates"]) == len(row["mrec_prompt_steps"]) == 1
    assert row["candidates"][0]["text"].startswith("Check: First cue")
    assert len(row["candidates"][0]["text"].split()) < len(oversized.split())
    assert len(row["prompt_input_ids"]) == row["prompt_token_count"] <= 512
    assert row["prompt_evidence_partial_evidence"] is True
    assert row["prompt_evidence_stop_reason"] == "prompt_token_budget_single_evidence_truncated"
    assert row["was_truncated"] is True
    assert row["evidence_text_truncated"] is True


def test_prompt_budget_fails_when_fixed_prompt_cannot_retain_first_evidence(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)
    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_rows[0]["claim"] = " ".join(["claim"] * 500)
    raw_path.write_text(json.dumps(raw_rows), encoding="utf-8")

    with pytest.raises(ValueError, match="too small to retain a non-empty first evidence"):
        build_trace_verifier_data._build_split(
            split="val",
            source_type="trace",
            source_path=trace_path,
            raw_path=raw_path,
            dataset=None,
            label_schema="liar6",
            tokenizer=MistralCommonFakeTokenizer(),
            prompt_cfg={
                "auto_length": False,
                "max_length": 1024,
                "output_mode": "label_only",
                "label_format": "letter",
            },
            selection_mode="trace",
            trace_prompt_style="mrec_min",
            expected_selector_name="test_selector",
            top_k=99,
            random_seed=0,
            expected_chunk_mmr_fingerprint="fp",
            sample_limit=None,
            show_progress=False,
            prompt_evidence_config={
                "policy": "prompt_budget",
                "min_evidence_count": 1,
                "max_evidence_count": 4,
                "prompt_token_budget": 100,
                "max_length_guard": {
                    "enabled": True,
                    "build_prompt_max_length": 1024,
                    "sft_train_max_length": 1024,
                    "on_violation": "error",
                },
            },
        )


def test_mrec_prompt_evidence_state_budget_keeps_state_changing_lookahead(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["mrec_steps"][2]["operation"] = "CONTRAST"
    trace["mrec_steps"][2]["state_after"] = "C"
    trace["mrec_steps"][2]["trace_state"]["target_resolved"] = True
    trace["mrec_steps"][2]["trace_state"]["resolved_atom_rate"] = 1.0
    trace["mrec_steps"][2]["trace_state"]["conflicted_atom_ids"] = ["A1"]
    trace["mrec_steps"][2]["trace_state"]["atom_states_after"] = {"A1": "C"}
    trace["mrec_steps"][3]["trace_state"]["conflicted_atom_ids"] = ["A1"]
    trace["mrec_steps"][3]["trace_state"]["atom_states_after"] = {"A1": "C"}
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

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
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "state_budget",
            "min_evidence_count": 1,
            "max_evidence_count": 0,
            "state_budget": {
                "lookahead_on_target_resolved": True,
                "unresolved_patience": 1,
                "budget_ratio": 1.0,
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0, 1, 2]
    assert [step["evidence_id"] for step in row["mrec_prompt_steps"]] == ["E40", "E41", "E42"]
    assert row["prompt_evidence_policy"] == "state_budget"
    assert row["prompt_evidence_selected_count_before_prompt_truncation"] == 3
    assert row["prompt_evidence_stop_reason"] == "target_resolved_stable"
    assert report["prompt_evidence"]["policy"] == "state_budget"
    assert report["prompt_evidence"]["state_budget"]["lookahead_on_target_resolved"] is True
    assert report["prompt_evidence"]["stop_reasons"] == {"target_resolved_stable": 1}


def test_mrec_prompt_evidence_state_budget_respects_soft_token_budget(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)

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
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "state_budget",
            "min_evidence_count": 1,
            "max_evidence_count": 0,
            "evidence_token_budget": 8,
            "state_budget": {
                "lookahead_on_target_resolved": True,
                "unresolved_patience": 1,
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0]
    assert row["prompt_evidence_policy"] == "state_budget"
    assert row["prompt_evidence_token_budget"] == 8
    assert row["prompt_evidence_selected_token_cost"] == 4
    assert row["prompt_evidence_stop_reason"] == "token_budget_exhausted"
    assert report["prompt_evidence"]["stop_reasons"] == {"token_budget_exhausted": 1}


def test_mrec_prompt_evidence_two_pass_uncertainty_reads_decision_cache(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)
    decision_dir = tmp_path / "two_pass_decisions"
    decision_dir.mkdir()
    (decision_dir / "two_pass_uncertainty_decisions_val.jsonl").write_text(
        json.dumps(
            {
                "event_id": "event-mrec-policy",
                "split": "val",
                "initial_indices": [0],
                "selected_indices": [0, 1, 2],
                "threshold": 0.42,
                "uncertainty_margin": 0.51,
                "prompt_evidence_expanded": True,
                "score_trace": [
                    {
                        "role": "initial",
                        "selected_indices": [0],
                        "pred_margin": 0.10,
                        "pred_label": "true",
                        "prompt_token_count": 100,
                        "was_truncated": False,
                    },
                    {
                        "role": "expanded",
                        "selected_indices": [0, 1, 2],
                        "pred_margin": 0.51,
                        "pred_label": "true",
                        "prompt_token_count": 220,
                        "was_truncated": False,
                    },
                ],
                "stop_reason": "expanded_confident",
                "prompt_token_count_before_final_build": 220,
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
        top_k=99,
        random_seed=0,
        expected_chunk_mmr_fingerprint="fp",
        sample_limit=None,
        show_progress=False,
        prompt_evidence_config={
            "policy": "two_pass_uncertainty",
            "two_pass_uncertainty": {
                "decision_dir": str(decision_dir),
                "calibration_file": str(decision_dir / "two_pass_uncertainty_calibration.json"),
            },
        },
    )

    row = rows[0]
    assert row["selector_trace"]["selected_indices"] == [0, 1, 2]
    assert row["prompt_evidence_policy"] == "two_pass_uncertainty"
    assert row["prompt_evidence_two_pass_initial_count"] == 1
    assert row["prompt_evidence_uncertainty_margin"] == 0.51
    assert row["prompt_evidence_expanded"] is True
    assert row["prompt_evidence_decision_source"].endswith("two_pass_uncertainty_decisions_val.jsonl")
    assert row["prompt_evidence_stop_reason"] == "expanded_confident"
    assert len(row["prompt_evidence_score_trace"]) == 2
    assert report["prompt_evidence"]["two_pass_uncertainty"]["expanded_rate"] == 1.0
    assert report["prompt_evidence"]["stop_reasons"] == {"expanded_confident": 1}


def test_mrec_prompt_evidence_two_pass_uncertainty_requires_decision_cache(tmp_path: Path) -> None:
    raw_path, trace_path = _write_mrec_policy_inputs(tmp_path)

    try:
        build_trace_verifier_data._build_split(
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
            top_k=99,
            random_seed=0,
            expected_chunk_mmr_fingerprint="fp",
            sample_limit=None,
            show_progress=False,
            prompt_evidence_config={
                "policy": "two_pass_uncertainty",
                "two_pass_uncertainty": {
                    "decision_dir": str(tmp_path / "missing_decisions"),
                },
            },
        )
    except ValueError as exc:
        assert "missing two-pass uncertainty decision cache" in str(exc)
    else:
        raise AssertionError("two_pass_uncertainty should fail when decision cache is missing")


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


def _write_mrec_policy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_row = {
        "event_id": "event-mrec-policy",
        "claim": "Original claim",
        "label": "true",
        "explain": "",
        "reports": [],
    }
    raw_path = tmp_path / "val.json"
    raw_path.write_text(json.dumps([raw_row]), encoding="utf-8")

    candidate_pool = []
    mrec_steps = []
    for idx, (evidence_id, cue, token_cost, target_resolved) in enumerate(
        (
            ("E40", "First cue", 4, False),
            ("E41", "Second cue", 5, True),
            ("E42", "Third cue", 7, True),
            ("E43", "Fourth cue", 3, True),
        )
    ):
        candidate_pool.append(
            {
                "text": f"Candidate {idx} evidence text.",
                "candidate_idx": idx,
                "evidence_id": evidence_id,
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
            }
        )
        mrec_steps.append(
            {
                "step": idx + 1,
                "operation": "OPEN" if idx == 0 else "CORROBORATE",
                "atom_id": "A1",
                "atom_text": "First atom",
                "state_before": "U" if idx == 0 else "S",
                "state_after": "S",
                "cue_text": cue,
                "cue_source": "claim_atom",
                "candidate_idx": idx,
                "selector_candidate_idx": idx,
                "evidence_id": evidence_id,
                "evidence_text": f"{evidence_id} evidence override.",
                "covered_atom_ids": ["A1"],
                "relation": "support",
                "directness": "direct",
                "token_cost": token_cost,
                "trace_state": {
                    "selected_count": idx + 1,
                    "cumulative_token_cost": sum(step["token_cost"] for step in mrec_steps) + token_cost,
                    "resolved_atom_rate": 1.0 if target_resolved else 0.0,
                    "target_resolved": target_resolved,
                    "unresolved_atom_ids": [] if target_resolved else ["A1"],
                    "conflicted_atom_ids": [],
                    "atom_states_after": {"A1": "S" if target_resolved else "U"},
                },
            }
        )

    trace = {
        "event_id": "event-mrec-policy",
        "selector_name": "test_selector",
        "fingerprint": "fp",
        "candidate_pool": candidate_pool,
        "candidate_scores": [{"candidate_idx": idx} for idx in range(len(candidate_pool))],
        "selector_ordered_indices": list(range(len(candidate_pool))),
        "oracle_ordered_indices": [0],
        "claim_atoms": [{"atom_id": "A1", "text": "First atom", "importance": 1.0}],
        "mrec_trace_version": "mrec_trace_v0_1",
        "mrec_selector_name": "mrec_greedy_transition_v0_1",
        "atom_states_initial": {"A1": "U"},
        "atom_states_final": {"A1": "S"},
        "mrec_diagnostics": {
            "stop_reason": "reached_max_steps",
            "resolved_atom_rate": 1.0,
        },
        "mrec_steps": mrec_steps,
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return raw_path, trace_path
