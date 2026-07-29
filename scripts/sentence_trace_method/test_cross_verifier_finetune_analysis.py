from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


MODULE_PATH = Path(__file__).with_name(
    "cross_verifier_finetune_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "cross_verifier_finetune_analysis", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


RUNS = tuple(
    (
        f"{backbone}-{assignment}-{seed}",
        backbone,
        assignment,
        seed,
    )
    for backbone in analysis.BACKBONES
    for assignment in analysis.ASSIGNMENTS
    for seed in (11, 22, 33)
)


def _paired_rows(
    *,
    evitrace_predictions: Sequence[str],
    s4_predictions: Sequence[str],
    token_differences: Mapping[tuple[str, str], int] | None = None,
) -> list[dict[str, Any]]:
    gold = list(analysis.LIAR6_LABELS)
    assert len(evitrace_predictions) == len(gold)
    assert len(s4_predictions) == len(gold)
    rows: list[dict[str, Any]] = []
    for run_id, backbone, assignment, seed in RUNS:
        for index, gold_label in enumerate(gold):
            event_id = f"event-{index}"
            difference = (
                token_differences.get((backbone, event_id), 0)
                if token_differences
                else 0
            )
            rows.append(
                {
                    "run_id": run_id,
                    "backbone": backbone,
                    "assignment_id": assignment,
                    "seed": seed,
                    "event_id": event_id,
                    "gold_label": gold_label,
                    "evitrace_pred_label": evitrace_predictions[index],
                    "s4_pred_label": s4_predictions[index],
                    "evitrace_gold_logprob": -0.1 - index / 100,
                    "s4_gold_logprob": -0.4 - index / 100,
                    "token_difference_evi_minus_s4": difference,
                }
            )
    return rows


def _reverse(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reversed_rows: list[dict[str, Any]] = []
    for row in rows:
        reversed_rows.append(
            {
                **row,
                "evitrace_pred_label": row["s4_pred_label"],
                "s4_pred_label": row["evitrace_pred_label"],
                "evitrace_gold_logprob": row["s4_gold_logprob"],
                "s4_gold_logprob": row["evitrace_gold_logprob"],
                "token_difference_evi_minus_s4": -int(
                    row["token_difference_evi_minus_s4"]
                ),
            }
        )
    return reversed_rows


def _logits(primary: str, secondary: str, margin: float = 0.2) -> dict[str, float]:
    values = {letter: -2.0 for letter in analysis.LETTERS}
    values[primary] = 1.0
    values[secondary] = 1.0 - margin
    return values


def _predicted_label(logits: Mapping[str, float]) -> str:
    letter = max(analysis.LETTERS, key=lambda item: float(logits[item]))
    return analysis.LETTER_TO_LABEL[letter]


def test_arm_reversal_negates_deltas_and_preserves_correctness_ties() -> None:
    gold = list(analysis.LIAR6_LABELS)
    evitrace = [gold[0], gold[1], gold[2], gold[3], gold[0], gold[0]]
    s4 = [gold[0], gold[2], gold[3], gold[3], gold[0], gold[0]]
    rows = _paired_rows(
        evitrace_predictions=evitrace,
        s4_predictions=s4,
    )

    point = analysis.hierarchical_point(rows)["panel"]
    reversed_point = analysis.hierarchical_point(_reverse(rows))["panel"]

    assert reversed_point["delta"]["macro_f1"] == pytest.approx(
        -point["delta"]["macro_f1"]
    )
    assert reversed_point["delta"]["accuracy"] == pytest.approx(
        -point["delta"]["accuracy"]
    )
    assert reversed_point["delta"]["gold_logprob_mean"] == pytest.approx(
        -point["delta"]["gold_logprob_mean"]
    )
    assert point["wlt_pooled_descriptive"]["tie"] == (
        reversed_point["wlt_pooled_descriptive"]["tie"]
    )
    assert point["wlt_pooled_descriptive"]["evitrace_win"] == (
        reversed_point["wlt_pooled_descriptive"]["s4_win"]
    )


def test_label_stratified_bootstrap_is_deterministic_with_one_claim_per_label() -> None:
    gold = list(analysis.LIAR6_LABELS)
    shifted = gold[1:] + gold[:1]
    rows = _paired_rows(
        evitrace_predictions=gold,
        s4_predictions=shifted,
    )

    result = analysis.stratified_claim_bootstrap(
        rows,
        iterations=32,
        seed=7,
    )

    assert result["cluster"].startswith("event_id")
    assert result["run_count_per_claim"] == 12
    assert result["point"]["macro_f1_delta"] == pytest.approx(1.0)
    assert result["ci95"]["macro_f1_delta"] == pytest.approx([1.0, 1.0])


def test_shared_claim_swap_randomization_is_reversal_symmetric() -> None:
    gold = list(analysis.LIAR6_LABELS)
    shifted = gold[1:] + gold[:1]
    rows = _paired_rows(
        evitrace_predictions=gold,
        s4_predictions=shifted,
    )

    forward = analysis.shared_claim_swap_randomization(
        rows,
        iterations=128,
        seed=19,
    )
    reverse = analysis.shared_claim_swap_randomization(
        _reverse(rows),
        iterations=128,
        seed=19,
    )

    assert forward["same_swap_bit_across_all_runs_for_each_claim"] is True
    assert forward["observed_macro_f1_delta"] == pytest.approx(
        -reverse["observed_macro_f1_delta"]
    )
    assert forward["two_sided_pvalue"] == pytest.approx(
        reverse["two_sided_pvalue"]
    )


def test_holm_and_sesoi_decisions_are_locked() -> None:
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    assert analysis.assess_sesoi(
        0.02, [0.011, 0.03]
    )["category"] == "beneficial_beyond_sesoi"
    assert analysis.assess_sesoi(
        0.0, [-0.009, 0.008]
    )["category"] == "practically_equivalent_within_sesoi"
    assert analysis.assess_sesoi(
        0.005, [-0.02, 0.03]
    )["category"] == "inconclusive_relative_to_sesoi"


def test_tau_grid_is_validation_only_and_uses_negative_log_prior_sign() -> None:
    prior = {
        "A": 0.70,
        "B": 0.06,
        "C": 0.06,
        "D": 0.06,
        "E": 0.06,
        "F": 0.06,
    }
    rows: list[dict[str, Any]] = []
    for run_id, backbone, assignment, seed in RUNS:
        for index, gold_label in enumerate(analysis.LIAR6_LABELS):
            gold_letter = analysis.LABEL_TO_LETTER[gold_label]
            secondary = "B" if gold_letter != "B" else "A"
            evi_logits = _logits(gold_letter, secondary)
            s4_logits = _logits(secondary, gold_letter)
            rows.append(
                {
                    "run_id": run_id,
                    "backbone": backbone,
                    "assignment_id": assignment,
                    "seed": seed,
                    "event_id": f"val-{index}",
                    "gold_label": gold_label,
                    "evitrace_logits": evi_logits,
                    "s4_logits": s4_logits,
                    "evitrace_raw_pred_label": _predicted_label(evi_logits),
                    "s4_raw_pred_label": _predicted_label(s4_logits),
                }
            )

    result = analysis.compute_logit_adjustment_tau_grid(rows, prior)

    assert result["scope"] == "validation_only"
    assert result["does_not_replace_raw_test_primary"] is True
    assert list(result["grid"]) == ["0", "0.25", "0.5", "0.75", "1"]
    assert result["formula"] == (
        "adjusted_logits = raw_logits - tau * log(label_prior)"
    )
    # Subtracting a negative log prior gives rarer labels a larger boost.
    assert (
        -math.log(prior["B"])
        > -math.log(prior["A"])
    )
    assert result["grid"]["0"]["point"]["evitrace"]["accuracy"] == pytest.approx(
        1.0
    )
    assert (
        result["grid"]["1"]["point"]["evitrace"]["accuracy"]
        != result["grid"]["0"]["point"]["evitrace"]["accuracy"]
    )


def test_main_token_sensitivity_uses_both_tokenizer_intersection() -> None:
    gold = list(analysis.LIAR6_LABELS)
    differences: dict[tuple[str, str], int] = {}
    for index in range(6):
        event_id = f"event-{index}"
        differences[("qwen3", event_id)] = 70 if index == 1 else 10
        differences[("llama31", event_id)] = 70 if index == 2 else 10
    rows = _paired_rows(
        evitrace_predictions=gold,
        s4_predictions=gold,
        token_differences=differences,
    )

    result = analysis.compute_main_token_sensitivity(rows)

    assert result["eligible_claims_per_backbone_tokenizer"] == {
        "llama31": 5,
        "qwen3": 5,
    }
    assert result["intersection_claim_count"] == 4
    assert result["intersection_point"]["delta"]["accuracy"] == pytest.approx(
        0.0
    )


def test_prefix_auc_stable_correct_and_strict_per_position_subset() -> None:
    curves: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    gold_label = analysis.LIAR6_LABELS[0]
    for run_id, backbone, assignment, seed in RUNS:
        for event_index in range(2):
            event_positions: list[dict[str, Any]] = []
            evi_predictions = [
                analysis.LIAR6_LABELS[1],
                gold_label,
                gold_label,
            ]
            s4_predictions = [
                analysis.LIAR6_LABELS[1],
                gold_label,
                analysis.LIAR6_LABELS[1],
            ]
            relations = (
                "different_set",
                "same_set_different_order",
                "same_order",
            )
            for index, relation in enumerate(relations, start=1):
                row = {
                    "run_id": run_id,
                    "backbone": backbone,
                    "assignment_id": assignment,
                    "seed": seed,
                    "event_id": f"prefix-{event_index}",
                    "gold_label": gold_label,
                    "prefix_relation": relation,
                    "k": index,
                    "k_visible": 3,
                    "evitrace_pred_label": evi_predictions[index - 1],
                    "s4_pred_label": s4_predictions[index - 1],
                    "evitrace_gold_logprob": -0.3 + index / 10,
                    "s4_gold_logprob": -0.5 + index / 20,
                    "evitrace_token_count": 10 * index,
                    "s4_token_count": 10 * index + (index == 2),
                }
                event_positions.append(row)
                positions.append(row)
            curve = analysis._prefix_curve_summary(event_positions)
            curves.append(
                {
                    "run_id": run_id,
                    "backbone": backbone,
                    "assignment_id": assignment,
                    "seed": seed,
                    "event_id": f"prefix-{event_index}",
                    "gold_label": gold_label,
                    **curve,
                }
            )

    result = analysis.summarize_prefix_curves(
        curves,
        positions,
        bootstrap=32,
        seed=23,
    )

    overall = result["overall"]["panel"]
    assert overall["evitrace"]["normalized_accuracy_auc"] == pytest.approx(
        2 / 3
    )
    assert overall["s4"]["normalized_accuracy_auc"] == pytest.approx(1 / 3)
    assert overall["evitrace"]["stable_correct"] == pytest.approx(1.0)
    assert overall["s4"]["stable_correct"] == pytest.approx(0.0)
    strict = result["strict_positional_subset"]
    assert strict["paired_position_count_unique"] == 2
    assert strict["claim_count"] == 2
    assert "need not" in strict["definition"]
    assert (
        result["paired_positions_by_prefix_relation"][
            "same_set_different_order"
        ]["panel"]["delta"]["accuracy"]
        == pytest.approx(0.0)
    )


def test_exact_train_selected_snippet_sensitivity_is_assignment_specific() -> None:
    gold = list(analysis.LIAR6_LABELS)
    rows = _paired_rows(
        evitrace_predictions=gold,
        s4_predictions=gold,
    )
    hash_a = "a" * 64
    hash_b = "b" * 64
    clean_hash = "c" * 64
    for row in rows:
        row["evitrace_evidence_snippet_sha256s"] = [clean_hash]
        row["s4_evidence_snippet_sha256s"] = [clean_hash]
        if row["assignment_id"] == "a" and row["event_id"] == "event-0":
            row["evitrace_evidence_snippet_sha256s"] = [hash_a]
        if row["assignment_id"] == "b" and row["event_id"] == "event-1":
            row["s4_evidence_snippet_sha256s"] = [hash_b]

    result = analysis.compute_exact_snippet_sensitivity(
        rows,
        {"a": {hash_a}, "b": {hash_b}},
    )

    assert {
        item["retained_claim_count"]
        for item in result["claim_counts_by_run"].values()
    } == {5}
    assert {
        item["excluded_claim_count"]
        for item in result["claim_counts_by_run"].values()
    } == {1}
    assert result["panel_macro_f1_direction"] == "tie"


def test_validation_diagnostics_include_recall_nll_ece_and_logp_contrasts() -> None:
    conditions: dict[str, list[dict[str, Any]]] = {
        "correct_evitrace": [],
        "correct_s4": [],
        "correct_pooled": [],
        "claim_only": [],
        "mismatched": [],
    }
    for run_id, backbone, assignment, seed in RUNS:
        for index, gold_label in enumerate(analysis.LIAR6_LABELS):
            gold_letter = analysis.LABEL_TO_LETTER[gold_label]
            shifted_letter = analysis.LETTERS[(index + 1) % 6]
            correct_probabilities = {
                letter: (0.9 if letter == gold_letter else 0.02)
                for letter in analysis.LETTERS
            }
            wrong_probabilities = {
                letter: (0.9 if letter == shifted_letter else 0.02)
                for letter in analysis.LETTERS
            }
            common = {
                "run_id": run_id,
                "backbone": backbone,
                "assignment_id": assignment,
                "seed": seed,
                "event_id": f"val-{index}",
                "gold_label": gold_label,
            }
            evi = {
                **common,
                "pred_label": gold_label,
                "gold_logprob": math.log(0.9),
                "probabilities": correct_probabilities,
            }
            s4 = dict(evi)
            claim_only = {
                **common,
                "pred_label": analysis.LETTER_TO_LABEL[shifted_letter],
                "gold_logprob": math.log(0.02),
                "probabilities": wrong_probabilities,
            }
            mismatched = {
                **claim_only,
                "gold_logprob": math.log(0.01),
            }
            conditions["correct_evitrace"].append(evi)
            conditions["correct_s4"].append(s4)
            conditions["correct_pooled"].extend((evi, s4))
            conditions["claim_only"].append(claim_only)
            conditions["mismatched"].append(mismatched)

    result = analysis.summarize_validation_conditions(conditions)

    correct = result["conditions"]["correct_pooled"]["panel"]
    assert correct["macro_f1"] == pytest.approx(1.0)
    assert correct["accuracy"] == pytest.approx(1.0)
    assert correct["nll"] == pytest.approx(-math.log(0.9))
    assert correct["ece_15_equal_width"] == pytest.approx(0.1)
    assert all(
        recall == pytest.approx(1.0)
        for recall in correct["per_class_recall"].values()
    )
    contrast = result["gold_logprob_contrasts"][
        "correct_pooled_minus_claim_only"
    ]["panel"]["mean"]
    assert contrast == pytest.approx(math.log(0.9) - math.log(0.02))


def test_run_grid_requires_the_three_frozen_calendar_seeds() -> None:
    def materialize(
        seeds: Sequence[int],
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        runs: list[dict[str, Any]] = []
        rows: dict[str, list[dict[str, Any]]] = {}
        for backbone in analysis.BACKBONES:
            for assignment in analysis.ASSIGNMENTS:
                for seed in seeds:
                    run_id = f"{backbone}-{assignment}-{seed}"
                    runs.append(
                        {
                            "run_id": run_id,
                            "backbone": backbone,
                            "assignment_id": assignment,
                            "seed": seed,
                        }
                    )
                    rows[run_id] = [
                        {
                            "event_id": "event",
                            "comparison_type": "main",
                            "evidence_arm": "evitrace",
                            "k": 1,
                            "prefix_relation": "not_applicable",
                            "k_visible": 1,
                            "evidence_snippet_sha256s": ["a" * 64],
                        }
                    ]
        return runs, rows

    formal_runs, formal_rows = materialize(analysis.FORMAL_SEEDS)
    assert analysis._validate_run_grid(formal_runs, formal_rows) == list(
        analysis.FORMAL_SEEDS
    )

    wrong_runs, wrong_rows = materialize((11, 22, 33))
    with pytest.raises(
        analysis.FineTuneAnalysisError,
        match="exactly",
    ):
        analysis._validate_run_grid(wrong_runs, wrong_rows)


def test_duplicate_clean_text_hashes_within_one_sequence_are_allowed() -> None:
    logits = {letter: (1.0 if letter == "A" else 0.0) for letter in analysis.LETTERS}
    probabilities = analysis._softmax(logits)
    runtime = {
        "run_id": "run",
        "backbone": "qwen3",
        "assignment_id": "a",
        "seed": analysis.FORMAL_SEEDS[0],
    }
    row = {
        **runtime,
        "logical_id": "event::main::evitrace",
        "event_id": "event",
        "comparison_type": "main",
        "evidence_arm": "evitrace",
        "k": 2,
        "k_visible": 2,
        "prefix_relation": "not_applicable",
        "input_ids_sha256": "0" * 64,
        "token_count": 10,
        "evidence_snippet_sha256s": ["a" * 64, "a" * 64],
        "logits": logits,
        "log_probs": {
            letter: math.log(probability)
            for letter, probability in probabilities.items()
        },
        "probabilities": probabilities,
        "pred_label": "pants-fire",
    }

    validated = analysis._validate_logical_row(
        row,
        runtime=runtime,
        row_number=1,
        result_path=Path("synthetic.jsonl"),
    )

    assert validated["evidence_snippet_sha256s"] == [
        "a" * 64,
        "a" * 64,
    ]
