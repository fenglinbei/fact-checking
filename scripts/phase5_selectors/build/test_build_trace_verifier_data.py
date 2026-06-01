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


def _write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw_path = tmp_path / "val.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "event-1",
                    "claim": "Original claim",
                    "label": "true",
                    "explain": "",
                    "reports": [],
                }
            ]
        ),
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
