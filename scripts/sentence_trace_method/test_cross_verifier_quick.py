from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


MODULE_PATH = Path(__file__).with_name("cross_verifier_quick.py")
SPEC = importlib.util.spec_from_file_location("cross_verifier_quick", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cross_verifier_quick = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cross_verifier_quick
SPEC.loader.exec_module(cross_verifier_quick)

LABELS = ("A", "B", "C", "D", "E", "F")


def _token_sha(input_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in input_ids],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _logical_prompt(
    *,
    logical_id: str,
    comparison_id: str,
    event_id: str,
    arm: str,
    input_ids: Sequence[int],
) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in input_ids]
    input_ids_sha256 = _token_sha(token_ids)
    return {
        "logical_id": logical_id,
        "comparison_id": comparison_id,
        "event_id": event_id,
        "comparison_type": "main",
        "arm": arm,
        "prompt_text": f"synthetic prompt {input_ids_sha256}",
        "input_ids": token_ids,
        "input_ids_sha256": input_ids_sha256,
    }


def _score(
    *,
    input_ids: Sequence[int],
    predicted_label: str = "A",
    model_sha256: str = "model-sha",
) -> dict[str, Any]:
    raw_logprobs = {
        label: (-0.1 if label == predicted_label else -4.0)
        for label in LABELS
    }
    logprobs = cross_verifier_quick.normalize_label_logprobs(raw_logprobs)
    probabilities = {
        label: math.exp(value) for label, value in logprobs.items()
    }
    return {
        "model_sha256": model_sha256,
        "input_ids_sha256": _token_sha(input_ids),
        "predicted_label": predicted_label,
        "label_logprobs": logprobs,
        "label_probabilities": probabilities,
    }


def _registry_entries(registry: Any) -> list[Mapping[str, Any]]:
    """Accept the two natural public shapes while testing registry behavior."""

    if isinstance(registry, tuple):
        registry = registry[0]
    if isinstance(registry, Mapping):
        for key in ("unique_prompts", "registry", "prompts"):
            if key in registry:
                return _registry_entries(registry[key])
        if all(isinstance(value, Mapping) for value in registry.values()):
            return list(registry.values())
    if isinstance(registry, Sequence) and not isinstance(registry, (str, bytes)):
        return list(registry)
    raise AssertionError(f"Unsupported prompt-registry result: {type(registry)!r}")


def _metric_value(metrics: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = metrics
        found = True
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found:
            return value
    raise AssertionError(f"None of the metric paths exist: {paths}")


def test_sha256_file_hashes_bytes_not_path(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"EviTrace\n")
    second.write_bytes(b"EviTrace\n")

    expected = hashlib.sha256(b"EviTrace\n").hexdigest()
    assert cross_verifier_quick.sha256_file(first) == expected
    assert cross_verifier_quick.sha256_file(second) == expected

    second.write_bytes(b"EviTrace changed\n")
    assert cross_verifier_quick.sha256_file(second) != expected


def test_normalize_label_logprobs_is_a_six_way_conditional_distribution() -> None:
    raw = {
        "A": -4.0,
        "B": -3.0,
        "C": -2.0,
        "D": -1.0,
        "E": -0.5,
        "F": -0.25,
    }

    normalized = cross_verifier_quick.normalize_label_logprobs(raw)

    assert tuple(normalized) == LABELS
    assert sum(math.exp(normalized[label]) for label in LABELS) == pytest.approx(1.0)
    assert max(normalized, key=normalized.get) == "F"
    for left, right in zip(LABELS, LABELS[1:]):
        assert normalized[right] - normalized[left] == pytest.approx(
            raw[right] - raw[left]
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"A": -1.0},
        {**{label: -1.0 for label in LABELS}, "G": -1.0},
        {**{label: -1.0 for label in LABELS}, "F": math.nan},
    ],
)
def test_normalize_label_logprobs_rejects_incomplete_or_nonfinite_scores(
    invalid: Mapping[str, float],
) -> None:
    with pytest.raises((cross_verifier_quick.QuickEvalError, TypeError)):
        cross_verifier_quick.normalize_label_logprobs(invalid)


def test_canonical_prompt_logprob_reads_the_appended_label_position() -> None:
    output = SimpleNamespace(
        prompt_token_ids=[10, 11, 42],
        prompt_logprobs=[
            None,
            {11: SimpleNamespace(logprob=-0.3)},
            {42: SimpleNamespace(logprob=-0.7)},
        ],
    )

    assert cross_verifier_quick.extract_canonical_prompt_logprob(
        output, 42
    ) == pytest.approx(-0.7)

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="end"):
        cross_verifier_quick.extract_canonical_prompt_logprob(output, 41)
    missing = SimpleNamespace(
        prompt_token_ids=[10, 42],
        prompt_logprobs=[None, {99: SimpleNamespace(logprob=-1.0)}],
    )
    with pytest.raises(cross_verifier_quick.QuickEvalError, match="missing"):
        cross_verifier_quick.extract_canonical_prompt_logprob(missing, 42)


def test_direct_scoring_rejects_vllm_top_k_that_omits_label_tokens() -> None:
    label_token_ids = {
        letter: 100 + index for index, letter in enumerate(LABELS)
    }
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                logprobs=[
                    {
                        label_token_ids["A"]: SimpleNamespace(logprob=-0.1),
                        label_token_ids["B"]: SimpleNamespace(logprob=-0.2),
                    }
                ]
            )
        ]
    )

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="missing"):
        cross_verifier_quick.extract_direct_label_logprobs(
            output, label_token_ids
        )


def test_exact_mcnemar_uses_only_discordant_pairs_and_is_symmetric() -> None:
    assert cross_verifier_quick.exact_mcnemar_pvalue(3, 0) == pytest.approx(0.25)
    assert cross_verifier_quick.exact_mcnemar_pvalue(0, 3) == pytest.approx(0.25)
    assert cross_verifier_quick.exact_mcnemar_pvalue(4, 4) == pytest.approx(1.0)
    assert cross_verifier_quick.exact_mcnemar_pvalue(0, 0) == pytest.approx(1.0)


def test_holm_adjust_preserves_input_order_and_step_down_monotonicity() -> None:
    adjusted = cross_verifier_quick.holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(0.0 <= value <= 1.0 for value in adjusted)

    with pytest.raises(ValueError):
        cross_verifier_quick.holm_adjust([0.1, 1.1])


def test_recompute_s4_order_uses_all_frozen_tie_breaks_stably() -> None:
    common = {
        "from_baseline": False,
        "atom_rrf_score": 0.2,
        "atom_route_hit_count": 0,
        "atom_max_route_hybrid": 0.0,
    }
    candidates = [
        {
            **common,
            "candidate_uid": "missing",
            "baseline_rank": None,
            "atom_pool_rank": None,
            "union_pool_rank": None,
        },
        {
            **common,
            "candidate_uid": "union-late",
            "baseline_rank": 2,
            "atom_pool_rank": 3,
            "union_pool_rank": 4,
        },
        {
            **common,
            "candidate_uid": "atom-early",
            "baseline_rank": 2,
            "atom_pool_rank": 1,
            "union_pool_rank": 9,
        },
        {
            **common,
            "candidate_uid": "union-early",
            "baseline_rank": 2,
            "atom_pool_rank": 3,
            "union_pool_rank": 1,
        },
        {
            **common,
            "candidate_uid": "baseline-early",
            "baseline_rank": 1,
            "atom_pool_rank": 9,
            "union_pool_rank": 9,
        },
    ]

    ranked = cross_verifier_quick.recompute_s4_source_score_order(candidates)

    assert [row["candidate_uid"] for row in ranked] == [
        "baseline-early",
        "atom-early",
        "union-early",
        "union-late",
        "missing",
    ]
    assert [row["source_score_rank"] for row in ranked] == [1, 2, 3, 4, 5]
    assert "atom_union_source_score" not in candidates[0]


def test_prompt_registry_deduplicates_repeated_final_token_ids() -> None:
    duplicate_left = _logical_prompt(
        logical_id="main:e1:evitrace",
        comparison_id="main:e1",
        event_id="e1",
        arm="evitrace",
        input_ids=[1, 2, 3],
    )
    duplicate_right = {
        **_logical_prompt(
            logical_id="order:e1:evitrace",
            comparison_id="order:e1",
            event_id="e1",
            arm="evitrace",
            input_ids=[1, 2, 3],
        ),
        "prompt_text": duplicate_left["prompt_text"],
    }
    distinct = _logical_prompt(
        logical_id="main:e1:control",
        comparison_id="main:e1",
        event_id="e1",
        arm="control",
        input_ids=[1, 2, 4],
    )

    registry, refs = cross_verifier_quick.make_prompt_registry(
        [duplicate_left, duplicate_right, distinct]
    )
    entries = _registry_entries(registry)

    assert len(entries) == 2
    assert len(refs) == 3
    assert {row["input_ids_sha256"] for row in entries} == {
        _token_sha([1, 2, 3]),
        _token_sha([1, 2, 4]),
    }
    duplicate_entry = next(
        row for row in entries if row["input_ids_sha256"] == _token_sha([1, 2, 3])
    )
    assert duplicate_entry["logical_ids"] == [
        "main:e1:evitrace",
        "order:e1:evitrace",
    ]


def test_prompt_rendering_preserves_assistant_header_separator_and_disables_thinking() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.template_kwargs: list[dict[str, Any]] = []
            self.encoded_texts: list[str] = []

        def apply_chat_template(
            self,
            messages: Sequence[Mapping[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            assert add_generation_prompt is True
            self.template_kwargs.append(dict(kwargs))
            rendered = (
                f"<system>{messages[0]['content']}</system>"
                f"<user>{messages[1]['content']}</user>"
                "<assistant>\n\n"
            )
            return list(rendered.encode("utf-8")) if tokenize else rendered

        def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
            self.encoded_texts.append(text)
            return {"input_ids": list(text.encode("utf-8"))}

    tokenizer = FakeTokenizer()
    comparison = {
        "comparison_id": "event-1::main",
        "event_id": "event-1",
        "comparison_type": "main",
        "claim": "A neutral synthetic claim.",
        "complexity": "single",
        "k_visible": 1,
        "arms": {
            arm: {
                "method": arm,
                "candidate_uids": [f"{arm}-uid"],
                "evidence_texts": ["Shared neutral evidence."],
                "evidence_multiset_sha256": f"{arm}-set-sha",
                "character_count": 24,
            }
            for arm in ("evitrace", "control")
        },
    }

    logical_rows = cross_verifier_quick._model_prompt_rows(
        [comparison],
        tokenizer,
        model_name="qwen3",
    )

    assert len(logical_rows) == 2
    full_prompts = [
        text for text in tokenizer.encoded_texts if text != "Label:"
    ]
    assert len(full_prompts) == 2
    assert all(text.endswith("<assistant>\n\nLabel:") for text in full_prompts)
    assert tokenizer.template_kwargs
    assert all(
        kwargs.get("enable_thinking") is False
        for kwargs in tokenizer.template_kwargs
    )


def test_expand_logical_results_reuses_score_without_aliasing_logical_identity() -> None:
    logical_rows = [
        _logical_prompt(
            logical_id="main:e1:evitrace",
            comparison_id="main:e1",
            event_id="e1",
            arm="evitrace",
            input_ids=[1, 2, 3],
        ),
        _logical_prompt(
            logical_id="order:e1:evitrace",
            comparison_id="order:e1",
            event_id="e1",
            arm="evitrace",
            input_ids=[1, 2, 3],
        ),
    ]
    score = _score(input_ids=[1, 2, 3])

    expanded = cross_verifier_quick.expand_logical_results(
        logical_rows,
        {score["input_ids_sha256"]: score},
    )

    assert len(expanded) == 2
    assert [row["logical_id"] for row in expanded] == [
        "main:e1:evitrace",
        "order:e1:evitrace",
    ]
    assert all(row["predicted_label"] == "A" for row in expanded)
    assert all(row["input_ids_sha256"] == score["input_ids_sha256"] for row in expanded)


def test_resume_validation_rejects_wrong_model_hash() -> None:
    stale_score = _score(input_ids=[1, 2, 3], model_sha256="old-model")

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="model"):
        cross_verifier_quick.validate_resume_row(
            stale_score,
            model_sha="new-model",
        )


def test_resume_loader_rejects_duplicate_completion_and_config_drift(
    tmp_path: Path,
) -> None:
    score = {
        **_score(input_ids=[1, 2, 3]),
        "scoring_config_sha256": "config-a",
    }
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        "\n".join(
            json.dumps(score, sort_keys=True)
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="Duplicate"):
        cross_verifier_quick.load_resume_scores(
            duplicate_path,
            model_sha="model-sha",
            scoring_config_sha="config-a",
        )
    with pytest.raises(cross_verifier_quick.QuickEvalError, match="configuration"):
        cross_verifier_quick.load_resume_scores(
            duplicate_path,
            model_sha="model-sha",
            scoring_config_sha="config-b",
        )


def test_prepared_manifest_must_be_complete_and_content_addressed(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "comparisons.jsonl"
    prepared.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = {
        "experiment": "evitrace_cross_verifier_quick_v1",
        "complete": False,
        "prepared_files": {
            "comparisons_test": {
                "path": str(prepared),
                "sha256": cross_verifier_quick.sha256_file(prepared),
                "bytes": prepared.stat().st_size,
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="not complete"):
        cross_verifier_quick._verify_prepared_manifest(manifest_path)

    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cross_verifier_quick._verify_prepared_manifest(manifest_path)["complete"]

    prepared.write_text('{"drift":true}\n', encoding="utf-8")
    with pytest.raises(cross_verifier_quick.QuickEvalError, match="SHA"):
        cross_verifier_quick._verify_prepared_manifest(manifest_path)


def test_expand_logical_results_rejects_missing_or_wrong_prompt_hash() -> None:
    logical_rows = [
        _logical_prompt(
            logical_id="main:e1:evitrace",
            comparison_id="main:e1",
            event_id="e1",
            arm="evitrace",
            input_ids=[1, 2, 3],
        )
    ]
    wrong_score = _score(input_ids=[9, 9, 9])

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="score|hash|prompt"):
        cross_verifier_quick.expand_logical_results(
            logical_rows,
            {wrong_score["input_ids_sha256"]: wrong_score},
        )


def test_comparison_metrics_counts_wins_losses_and_both_kinds_of_ties() -> None:
    pairs = [
        {
            "event_id": "win",
            "gold_label": "pants-fire",
            "complexity": "single",
            "evitrace_pred_label": "pants-fire",
            "control_pred_label": "false",
            "evitrace_gold_logprob": -0.1,
            "control_gold_logprob": -1.1,
        },
        {
            "event_id": "loss",
            "gold_label": "false",
            "complexity": "single",
            "evitrace_pred_label": "pants-fire",
            "control_pred_label": "false",
            "evitrace_gold_logprob": -1.3,
            "control_gold_logprob": -0.3,
        },
        {
            "event_id": "both-correct",
            "gold_label": "barely-true",
            "complexity": "multi",
            "evitrace_pred_label": "barely-true",
            "control_pred_label": "barely-true",
            "evitrace_gold_logprob": -0.2,
            "control_gold_logprob": -0.2,
        },
        {
            "event_id": "both-wrong",
            "gold_label": "half-true",
            "complexity": "multi",
            "evitrace_pred_label": "mostly-true",
            "control_pred_label": "true",
            "evitrace_gold_logprob": -2.0,
            "control_gold_logprob": -2.0,
        },
    ]

    metrics = cross_verifier_quick.compute_comparison_metrics(pairs)

    assert _metric_value(
        metrics,
        "correctness_pairs.evitrace_only_correct_wins",
    ) == 1
    assert _metric_value(
        metrics,
        "correctness_pairs.control_only_correct_wins",
    ) == 1
    assert _metric_value(metrics, "correctness_pairs.ties") == 2
    assert _metric_value(
        metrics,
        "correctness_pairs.both_correct",
    ) == 1
    assert _metric_value(
        metrics,
        "correctness_pairs.both_wrong",
    ) == 1
    assert _metric_value(
        metrics,
        "correctness_pairs.conditional_evitrace_win_rate",
    ) == pytest.approx(0.5)
    assert _metric_value(
        metrics,
        "evitrace.accuracy",
        "accuracy.evitrace",
        "evitrace_accuracy",
    ) == pytest.approx(0.5)
    assert _metric_value(
        metrics,
        "control.accuracy",
        "accuracy.control",
        "control_accuracy",
    ) == pytest.approx(0.5)
    assert _metric_value(
        metrics,
        "gold_logprob_delta.mean",
        "mean_gold_logprob_delta",
    ) == pytest.approx(0.0)


def test_claim_swap_randomization_keeps_models_clustered_by_claim() -> None:
    rows: list[dict[str, Any]] = []
    for event_index in range(6):
        for model_name in ("qwen3", "llama31"):
            rows.append(
                {
                    "event_id": f"event-{event_index}",
                    "model_name": model_name,
                    "gold_label": "true",
                    "evitrace_pred_label": "true",
                    "control_pred_label": "false",
                }
            )

    first = cross_verifier_quick.claim_swap_randomization(
        rows,
        iterations=2_000,
        seed=20260724,
    )
    second = cross_verifier_quick.claim_swap_randomization(
        rows,
        iterations=2_000,
        seed=20260724,
    )

    assert first == second
    assert first["observed_accuracy_delta"] == pytest.approx(1.0)
    assert 0.0 < first["two_sided_pvalue"] < 0.1
    assert first["claim_count"] == 6
    assert first["model_count_per_claim"] == 2
    assert first["same_swap_bit_across_models"] is True


def test_stratified_cluster_bootstrap_is_deterministic_and_claim_level() -> None:
    rows: list[dict[str, Any]] = []
    labels = cross_verifier_quick.LIAR6_LABELS
    for label_index, label in enumerate(labels):
        for claim_index in range(2):
            event_id = f"{label}-{claim_index}"
            delta = 0.1 + label_index * 0.01 + claim_index * 0.001
            for model_name in ("qwen3", "llama31"):
                rows.append(
                    {
                        "event_id": event_id,
                        "gold_label": label,
                        "model_name": model_name,
                        "complexity": (
                            "single" if claim_index == 0 else "multi"
                        ),
                        "evitrace_pred_label": label,
                        "control_pred_label": labels[
                            (label_index + 1) % len(labels)
                        ],
                        "evitrace_gold_logprob": -0.1,
                        "control_gold_logprob": -0.1 - delta,
                    }
                )

    first = cross_verifier_quick.stratified_cluster_bootstrap(
        rows,
        iterations=500,
        seed=20260724,
    )
    second = cross_verifier_quick.stratified_cluster_bootstrap(
        rows,
        iterations=500,
        seed=20260724,
    )

    assert first == second
    assert first["point"]["accuracy_delta"] == pytest.approx(1.0)
    assert first["point"]["gold_logprob_delta"] == pytest.approx(0.1255)
    assert first["ci95"]["accuracy_delta"] == pytest.approx([1.0, 1.0])
    lower, upper = first["ci95"]["gold_logprob_delta"]
    assert 0.125 <= lower <= upper <= 0.126
    assert first["claim_count"] == 12
    assert first["model_count_per_claim"] == 2


def test_order_only_contract_and_prompt_leakage_helpers() -> None:
    clean_pair = {
        "comparison_type": "order_only",
        "event_id": "event-1",
        "arms": {
            "evitrace": {
                "candidate_uids": ["u1", "u2", "u3"],
                "evidence_texts": ["one", "two", "three"],
            },
            "control": {
                "candidate_uids": ["u3", "u1", "u2"],
                "evidence_texts": ["three", "one", "two"],
            },
        },
    }
    cross_verifier_quick.validate_order_only_pair(clean_pair)
    cross_verifier_quick.validate_prompt_text(
        "Claim: a neutral claim\nEvidence 1: neutral evidence"
    )

    with pytest.raises(cross_verifier_quick.QuickEvalError, match="set|UID|evidence"):
        cross_verifier_quick.validate_order_only_pair(
            {
                **clean_pair,
                "arms": {
                    **clean_pair["arms"],
                    "control": {
                        "candidate_uids": ["u3", "u1", "different"],
                        "evidence_texts": ["three", "one", "different"],
                    },
                },
            }
        )
    with pytest.raises(cross_verifier_quick.QuickEvalError, match="order|identical"):
        cross_verifier_quick.validate_order_only_pair(
            {
                **clean_pair,
                "arms": {
                    **clean_pair["arms"],
                    "control": {
                        "candidate_uids": ["u1", "u2", "u3"],
                        "evidence_texts": ["one", "two", "three"],
                    },
                },
            }
        )
    for leaked in (
        "Check: hidden atom",
        "method=EviTrace",
        "source_score=0.9",
        "state_before=U",
        "candidate_uid=u1",
    ):
        with pytest.raises(
            cross_verifier_quick.QuickEvalError,
            match="forbidden|leak|Check",
        ):
            cross_verifier_quick.validate_prompt_text(leaked)
