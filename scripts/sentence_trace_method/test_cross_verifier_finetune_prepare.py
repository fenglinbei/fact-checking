from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


MODULE_PATH = Path(__file__).with_name("cross_verifier_finetune_prepare.py")
SPEC = importlib.util.spec_from_file_location(
    "cross_verifier_finetune_prepare",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [1000 + ord(character) for character in text]

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        if text in {f" {letter}" for letter in prepare.LETTERS}:
            return {"input_ids": [2000 + ord(text[-1])]}
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str | list[int]:
        assert add_generation_prompt is True
        self.template_calls.append(dict(kwargs))
        rendered = (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user>"
            "<assistant>\n"
        )
        return self._encode(rendered) if tokenize else rendered


def _event(
    event_id: str,
    *,
    label: str = "false",
    complexity: str = "single",
    evi_uids: Sequence[str] = ("a", "b", "c"),
    s4_top_uids: Sequence[str] | None = None,
    reordered_uids: Sequence[str] = ("b", "a", "c"),
) -> SimpleNamespace:
    all_uids = set(evi_uids) | set(reordered_uids) | set(s4_top_uids or ())
    candidates = {
        uid: SimpleNamespace(text=f"clean evidence {event_id} {uid}")
        for uid in all_uids
    }
    return SimpleNamespace(
        event_id=event_id,
        split="test",
        claim=f"Synthetic claim {event_id}.",
        gold_label=label,
        complexity=complexity,
        k_visible=len(evi_uids),
        evi_visible_uids=tuple(evi_uids),
        s4_order_uids=tuple(s4_top_uids or reordered_uids),
        s4_reordered_evi_uids=tuple(reordered_uids),
        candidates_by_uid=candidates,
        order_is_identical=tuple(evi_uids) == tuple(reordered_uids),
    )


def test_complementary_assignments_are_exactly_balanced_and_stratified() -> None:
    events = []
    for label in prepare.LABELS:
        for complexity in ("single", "multi"):
            for index in range(3):
                events.append(
                    _event(
                        f"{label}-{complexity}-{index}",
                        label=label,
                        complexity=complexity,
                    )
                )

    assignment_a, assignment_b, audit = prepare.complementary_assignments(
        events,
        seed=prepare.DEFAULT_SEED,
    )
    repeated_a, repeated_b, repeated_audit = prepare.complementary_assignments(
        list(reversed(events)),
        seed=prepare.DEFAULT_SEED,
    )

    assert assignment_a == repeated_a
    assert assignment_b == repeated_b
    assert audit == repeated_audit
    assert len(assignment_a) == 36
    assert list(assignment_a.values()).count("evitrace") == 18
    assert list(assignment_a.values()).count("s4") == 18
    assert all(
        assignment_a[event_id] != assignment_b[event_id]
        for event_id in assignment_a
    )
    assert audit["pointwise_complementary"] is True
    for cell in audit["cells"].values():
        counts = cell["assignment_a"]
        assert abs(counts["evitrace"] - counts["s4"]) <= 1


def test_prefix_builder_keeps_every_k_and_marks_only_strict_positional_pairs() -> None:
    event = _event(
        "event-1",
        evi_uids=("a", "b", "c"),
        reordered_uids=("b", "a", "c"),
    )

    rows = prepare.build_prefix_comparisons([event])

    assert [row["k"] for row in rows] == [1, 2, 3]
    assert [row["prefix_relation"] for row in rows] == [
        "different_set",
        "same_set_different_order",
        "same_set_different_order",
    ]
    assert [row["positional_only"] for row in rows] == [False, True, True]
    assert rows[0]["arms"]["evitrace"]["candidate_uids"] == ["a"]
    assert rows[0]["arms"]["s4"]["candidate_uids"] == ["b"]
    for row in rows[1:]:
        assert set(row["arms"]["evitrace"]["candidate_uids"]) == set(
            row["arms"]["s4"]["candidate_uids"]
        )
    prepare._assert_gold_free(rows, context="synthetic-prefix")


def test_order_only_uses_the_full_same_uid_set_but_main_may_not() -> None:
    event = _event(
        "event-2",
        evi_uids=("a", "b", "c"),
        s4_top_uids=("d", "b", "a"),
        reordered_uids=("b", "a", "c"),
    )

    main = prepare._main_comparison(event)
    order = prepare._order_comparison(event)

    assert set(main["arms"]["evitrace"]["candidate_uids"]) != set(
        main["arms"]["s4"]["candidate_uids"]
    )
    assert set(order["arms"]["evitrace"]["candidate_uids"]) == set(
        order["arms"]["s4"]["candidate_uids"]
    )
    assert order["arms"]["evitrace"]["candidate_uids"] != order["arms"]["s4"][
        "candidate_uids"
    ]


def test_prompt_renderer_stores_chat_only_ids_and_disables_qwen_thinking() -> None:
    tokenizer = FakeTokenizer()
    renderer = prepare.PromptRenderer(tokenizer, model_key="qwen3")

    payload = renderer.render("A neutral claim.", ["First.", "Second."])

    assert payload["prompt_input_ids"]
    assert payload["prompt_token_count"] == len(payload["prompt_input_ids"])
    assert payload["prompt_input_ids"][-len(renderer.label_prefix_ids) :] != (
        renderer.label_prefix_ids
    )
    assert all(
        call.get("enable_thinking") is False for call in tokenizer.template_calls
    )
    assert set(renderer.label_token_ids) == set(prepare.LETTERS)
    assert len(set(renderer.label_token_ids.values())) == 6
    target, target_count = renderer.supervised_target("false")
    assert target == "Label: B"
    assert target_count > 0


def test_prompt_renderer_fails_instead_of_truncating_overflow() -> None:
    renderer = prepare.PromptRenderer(
        FakeTokenizer(),
        model_key="tiny",
        max_model_len=32,
    )

    with pytest.raises(prepare.FinetunePrepareError, match="truncation is forbidden"):
        renderer.render("A long enough claim.", ["Some evidence."])


def test_eval_registry_has_required_logical_fields_and_no_gold() -> None:
    event = _event(
        "event-3",
        evi_uids=("a", "b"),
        reordered_uids=("b", "a"),
    )
    comparisons = [
        prepare._main_comparison(event),
        prepare._order_comparison(event),
    ]
    prefixes = prepare.build_prefix_comparisons([event])
    renderer = prepare.PromptRenderer(FakeTokenizer(), model_key="fake")

    rows = prepare.build_eval_registry(comparisons, prefixes, renderer)

    assert len(rows) == 2 * (len(comparisons) + len(prefixes))
    required = {
        "logical_id",
        "event_id",
        "comparison_type",
        "evidence_arm",
        "k",
        "k_visible",
        "prefix_relation",
        "prompt_input_ids",
        "prompt_input_ids_sha256",
    }
    assert all(required <= set(row) for row in rows)
    assert all(
        row["evidence_arm"] in {"evitrace", "s4"} for row in rows
    )
    assert all(row["label_prefix_in_prompt_input_ids"] is False for row in rows)
    prepare._assert_gold_free(rows, context="synthetic-registry")
    with pytest.raises(prepare.FinetunePrepareError, match="gold keys leaked"):
        prepare._assert_gold_free(
            {**rows[0], "gold_label": "false"},
            context="leaked-registry",
        )


def test_mismatch_derangement_stays_within_label_complexity_strata() -> None:
    events = []
    for label in prepare.LABELS:
        for complexity in ("single", "multi"):
            for index in range(3):
                event = _event(
                    f"val-{label}-{complexity}-{index}",
                    label=label,
                    complexity=complexity,
                    evi_uids=tuple("abc"[: index + 1]),
                    reordered_uids=tuple(reversed("abc"[: index + 1])),
                )
                event.split = "val"
                events.append(event)

    donors, audit = prepare._mismatch_donors(
        events,
        seed=prepare.DEFAULT_SEED,
    )

    assert set(donors) == {event.event_id for event in events}
    assert len({donor.event_id for donor in donors.values()}) == len(events)
    by_id = {event.event_id: event for event in events}
    for event_id, donor in donors.items():
        event = by_id[event_id]
        assert donor.event_id != event_id
        assert donor.gold_label == event.gold_label
        assert donor.complexity == event.complexity
    assert audit["one_to_one"] is True
    assert audit["same_gold_label"] is True
    assert audit["absolute_k_difference"]["count"] == len(events)


def test_val_registry_projects_all_diagnostics_without_gold() -> None:
    def source(
        comparison_type: str,
        evidence_arm: str,
        *,
        suffix: str,
    ) -> dict[str, Any]:
        row = {
            "event_id": f"val-{suffix}",
            "comparison_type": comparison_type,
            "evidence_arm": evidence_arm,
            "candidate_uids": [] if evidence_arm == "claim_only" else ["u1"],
            "evidence_sequence_sha256": f"sequence-{suffix}",
            "evidence_snippet_sha256s": [] if evidence_arm == "claim_only" else ["s1"],
            "evidence_count": 0 if evidence_arm == "claim_only" else 1,
            "k_visible": 3,
            "prompt_input_ids": [1, 2, 3],
            "prompt_input_ids_sha256": f"ids-{suffix}",
            "prompt_text_sha256": f"text-{suffix}",
            "prompt_token_count": 3,
            "gold_label": "false",
            "gold_id": 1,
            "target": "Label: B",
        }
        if evidence_arm == "mismatched":
            row.update(
                {
                    "donor_event_id": "donor",
                    "donor_k_visible": 1,
                    "mismatch_absolute_k_difference": 2,
                    "mismatch_absolute_character_difference": 10,
                }
            )
        return row

    rows = prepare.build_val_eval_registry(
        [
            source("val_paired", "evitrace", suffix="paired-evi"),
            source("val_paired", "s4", suffix="paired-s4"),
        ],
        [source("val_claim_only", "claim_only", suffix="claim")],
        [source("val_mismatched", "mismatched", suffix="mismatch")],
    )

    assert [row["comparison_type"] for row in rows] == [
        "val_paired",
        "val_paired",
        "val_claim_only",
        "val_mismatched",
    ]
    assert rows[-1]["donor_event_id"] == "donor"
    assert all("evidence_texts" not in row for row in rows)
    prepare._assert_gold_free(rows, context="val-registry")


def test_frozen_counts_capture_full_order_and_all_prefix_contracts() -> None:
    assert prepare.EXPECTED_COUNTS == {
        "train": 10_050,
        "val": 1_274,
        "test": 1_250,
    }
    assert prepare.EXPECTED_ORDER == 1_152
    assert prepare.EXPECTED_PREFIX == 6_996
    assert (
        prepare.EXPECTED_PREFIX_RELATIONS[
            "same_set_different_order"
        ]
        == 2_020
    )
    assert sum(prepare.EXPECTED_PREFIX_RELATIONS.values()) == 6_996
    assert prepare.PREAUDIT_PLANNED_PREFIX == 7_448
    assert prepare.IDENTICAL_ORDER_EXCLUDED_PREFIX_POSITIONS == 452
    assert prepare.EXPECTED_TEST_LOGICAL_ROWS == 18_796
    assert prepare.EXPECTED_VAL_LOGICAL_ROWS == 5_096
    assert prepare.EXPECTED_EVAL_LOGICAL_ROWS == 23_892
