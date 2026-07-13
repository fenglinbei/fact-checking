from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_strict_same_set_controls.py")
SPEC = importlib.util.spec_from_file_location("build_strict_same_set_controls", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
strict_controls = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = strict_controls
SPEC.loader.exec_module(strict_controls)


class MistralCommonFakeTokenizer:
    """Small fake that exercises the prompt_input_ids-preserving code path."""

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
        **_: object,
    ) -> str | list[int]:
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if tokenize:
            return list(range(10, 10 + len(rendered.split())))
        return rendered


PROMPT_CFG = {
    "auto_length": False,
    "max_length": 512,
    "output_mode": "label_only",
    "label_format": "letter",
    "label_schema": "liar6",
    "chat_template": {
        "mode": "tokenizer_default",
        "add_generation_prompt": True,
        "template_kwargs": {},
    },
}


def test_every_order_arm_is_exactly_the_frozen_uid_set() -> None:
    feature_row = _feature_row()
    original = ("b", "a", "c")
    exact = ("a", "b", "c")
    orders = strict_controls._build_arm_orders(
        event_id="event-1",
        original_uids=original,
        exact_uids=exact,
        feature_row=feature_row,
        random_seeds=(0, 1, 2, 3, 4),
    )

    assert orders["original"] == original
    assert orders["baces_exact"] == exact
    assert orders["retrieval_score"] == ("b", "a", "c")
    assert orders["candidate_pool"] == ("c", "a", "b")
    assert orders["reverse"] == ("c", "a", "b")
    assert set(orders) == {
        "original",
        "baces_exact",
        "retrieval_score",
        "candidate_pool",
        "reverse",
        "random_seed0",
        "random_seed1",
        "random_seed2",
        "random_seed3",
        "random_seed4",
    }
    for order in orders.values():
        assert len(order) == len(original)
        assert set(order) == set(original)


def test_event_hashed_random_order_is_process_independent_and_deterministic() -> None:
    kwargs = {
        "event_id": "event-with-stable-id",
        "original_uids": ("d", "c", "b", "a"),
        "exact_uids": ("a", "b", "c", "d"),
        "feature_row": _feature_row(include_d=True),
        "random_seeds": (0, 7),
    }
    first = strict_controls._build_arm_orders(**kwargs)
    second = strict_controls._build_arm_orders(**kwargs)

    assert strict_controls._event_random_seed(0, "event-with-stable-id") == (
        strict_controls._event_random_seed(0, "event-with-stable-id")
    )
    assert strict_controls._event_random_seed(0, "event-with-stable-id") != (
        strict_controls._event_random_seed(7, "event-with-stable-id")
    )
    assert first["random_seed0"] == second["random_seed0"]
    assert first["random_seed7"] == second["random_seed7"]
    assert set(first["random_seed0"]) == {"a", "b", "c", "d"}


def test_original_prompt_and_nonempty_prompt_input_ids_rebuild_exactly() -> None:
    tokenizer = MistralCommonFakeTokenizer()
    source = _source_build_row(tokenizer)
    frozen = strict_controls._freeze_source_candidates(source)
    rebuilt = strict_controls._rebuild_arm_row(
        source_row=source,
        ordered_candidates=frozen,
        tokenizer=tokenizer,
        prompt_cfg={**PROMPT_CFG, "auto_length": True},
    )

    assert source["prompt_input_ids"]
    assert rebuilt["prompt"] == source["prompt"]
    assert rebuilt["prompt_input_ids"] == source["prompt_input_ids"]
    strict_controls._validate_original_rebuild(source_row=source, rebuilt=rebuilt)

    poisoned = dict(source)
    poisoned["prompt"] = str(source["prompt"]) + " changed"
    with pytest.raises(strict_controls.StrictControlError, match="prompt differs"):
        strict_controls._validate_original_rebuild(source_row=poisoned, rebuilt=rebuilt)


def test_single_visible_text_truncation_is_a_hard_failure() -> None:
    source = _source_build_row(MistralCommonFakeTokenizer())
    source["candidates"] = source["candidates"][:1]
    source["evidence_count"] = 1
    source["evidence_count_before"] = 1
    source["evidence_text_truncated"] = True

    with pytest.raises(
        strict_controls.StrictControlError,
        match="evidence_text_truncated must be exactly false",
    ):
        strict_controls._freeze_source_candidates(source)


def test_event_builder_freezes_solver_role_but_replays_display_marginal() -> None:
    tokenizer = MistralCommonFakeTokenizer()
    source = _source_build_row(tokenizer)
    feature = _feature_row()
    problem = strict_controls.compile_feature_problem(
        feature,
        k_max=3,
        token_budget=None,
        weights=[1],
        cost_overrides={"a": 1, "b": 1, "c": 1},
    )
    exact = strict_controls.solve_fixed_set_order(problem, ("b", "a", "c"))
    audit = {
        "event_id": "event-1",
        "status": "ok",
        "weight_policy": "unit",
        "K_final": 3,
        "k_max": 3,
        "token_budget": None,
        "build_evidence_text_truncated": False,
        "final_keys": ["b", "a", "c"],
        "final_same_set_optimal_keys": list(exact.keys),
        "final_same_set_T_opt": exact.acquisition_time,
    }

    payloads = strict_controls._build_event_controls(
        source_row=source,
        feature_row=feature,
        audit_row=audit,
        split="val",
        random_seeds=(0,),
        tokenizer=tokenizer,
        prompt_cfg=PROMPT_CFG,
        max_length=512,
    )

    assert set(payloads) == {
        "original",
        "baces_exact",
        "retrieval_score",
        "candidate_pool",
        "reverse",
        "random_seed0",
    }
    original_steps = payloads["original"]["sidecar"]["steps"]
    assert original_steps[0]["candidate_uid"] == "b"
    assert original_steps[0]["solver_role"] == "FILL"
    assert original_steps[0]["display_operation"] == "ORDINAL_UPGRADE"
    assert original_steps[0]["display_marginal_coverage_units"] == 1

    reference_fingerprints = {
        key: payloads["original"]["sidecar"][key]
        for key in (
            "uid_set_fingerprint",
            "uid_text_fingerprint",
            "candidate_block_fingerprint",
        )
    }
    for payload in payloads.values():
        sidecar = payload["sidecar"]
        assert {key: sidecar[key] for key in reference_fingerprints} == reference_fingerprints
        assert payload["build_row"]["evidence_count"] == 3
        assert payload["build_row"]["evidence_text_truncated"] is False
        assert payload["build_row"]["was_truncated"] is False

    original_order = payloads["original"]["sidecar"]["ordered_candidate_uids"]
    original_display_fp = payloads["original"]["sidecar"]["display_order_fingerprint"]
    for payload in payloads.values():
        sidecar = payload["sidecar"]
        if sidecar["ordered_candidate_uids"] == original_order:
            assert sidecar["display_order_fingerprint"] == original_display_fp
        else:
            assert sidecar["display_order_fingerprint"] != original_display_fp


def _source_build_row(tokenizer: MistralCommonFakeTokenizer) -> dict:
    candidates = [
        _source_candidate("b", "Partial evidence text."),
        _source_candidate("a", "Direct evidence text."),
        _source_candidate("c", "Redundant evidence text."),
    ]
    retrieval_row = {
        "event_id": "event-1",
        "claim": "A compact factual claim.",
        "label": "false",
        "label_schema": "liar6",
        "explain": "",
        "candidates": candidates,
    }
    row = strict_controls.build_training_row(
        retrieval_row,
        tokenizer,
        PROMPT_CFG,
    )
    row["evidence_count_before"] = 3
    row["evidence_text_truncated"] = False
    return row


def _source_candidate(uid: str, text: str) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_key": f"key-{uid}",
        "evidence_id": f"E-{uid}",
        "mrec_token_cost": 1,
        "text": text,
        "frozen_nested_metadata": {"uid": uid, "values": [1, 2]},
    }


def _feature_row(*, include_d: bool = False) -> dict:
    candidates = [
        _feature_candidate("c", score=None, directness=None),
        _feature_candidate("a", score=0.1, directness="direct"),
        _feature_candidate("b", score=0.9, directness="partial"),
    ]
    if include_d:
        candidates.insert(1, _feature_candidate("d", score=0.5, directness=None))
    return {
        "event_id": "event-1",
        "claim_atoms": [{"atom_id": "A1", "proposition": "One atom"}],
        "candidates": candidates,
    }


def _feature_candidate(
    uid: str, *, score: float | None, directness: str | None
) -> dict:
    alignments = []
    if directness is not None:
        alignments.append(
            {
                "atom_id": "A1",
                "evidence_id": f"E-{uid}",
                "relation": "support",
                "directness": directness,
                "confidence": 1.0,
                "key_spans": [f"span-{uid}"],
            }
        )
    return {
        "candidate_uid": uid,
        "candidate_key": f"key-{uid}",
        "evidence_id": f"E-{uid}",
        "num_tokens": 1,
        "hybrid_score": score,
        "candidate_atom_alignments": alignments,
    }
