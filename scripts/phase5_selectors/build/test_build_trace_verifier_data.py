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
