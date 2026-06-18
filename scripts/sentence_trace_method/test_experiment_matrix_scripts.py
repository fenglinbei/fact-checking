from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_script(path: str, extra_env: dict[str, str] | None = None) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "DRY_RUN": "true",
        "PREPARE_V0_7_SOURCES": "false",
        "PREPARE_SELECTOR_SOURCES": "false",
        "RUN_TAU_EVAL": "false",
        "MODE": "build",
    }
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(ROOT / path)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_liar_raw_ebs_lr_matrix_dry_run_expands_four_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_v0_7_liar_raw_lora_ebs_lr_matrix.sh")

    for suffix in (
        "_lora_ebs16_lr1em5_ep8_eval100_pat8_liarw",
        "_lora_ebs16_lr2em5_ep8_eval100_pat8_liarw",
        "_lora_ebs32_lr1em5_ep8_eval100_pat8_liarw",
        "_lora_ebs32_lr2em5_ep8_eval100_pat8_liarw",
    ):
        assert suffix in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_LEARNING_RATE=2e-5" in output


def test_rawfc_selector_lr_matrix_dry_run_expands_eight_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_lora_selector_lr_matrix.sh")

    for suffix in (
        "_lora_ebs16_lr1em5_ep8_eval100_pat8_rawfc",
        "_lora_ebs16_lr5em6_ep8_eval100_pat8_rawfc",
    ):
        assert output.count(suffix) >= 4
    for selector in (
        "sentence_rule_step_adaptive5_10",
        "v0_7_budgeted_marginal_chain_adaptive3_10",
        "v0_7_budgeted_marginal_chain_adaptive5_10",
        "v0_7_budgeted_marginal_chain_adaptive5_12",
    ):
        assert selector in output
    assert "CASE_SUFFIX=__old_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_12" in output


def test_liar_raw_ministral3_transfer_matrix_matches_rawfc_migration_shape() -> None:
    output = _run_script("scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_transfer.sh")

    assert "MODELS=ministral3_8b" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10" in output
    assert "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw" in output
    assert "_lora_ebs32_lr2em5_ep12_eval100_pat8_liarw" in output
    assert "_lora_ebs16_lr1em5" not in output
    assert "_lora_ebs32_lr1em5" not in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=8" in output
    assert "configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "configs/deepspeed_zero2_bsz1_ga8.json" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_qec_v1_ministral3_prompt_matrix_dry_run_expands_only_new_qec_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_qec_v1_ministral3_prompt_matrix.sh")

    assert "liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_min" in output
    assert "liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_map" in output
    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_min" in output
    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_map" in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "TRACE_PROMPT_STYLE=qec_map" in output

    assert "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw" in output
    assert "_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc" in output
    assert "SFT_NUM_TRAIN_EPOCHS=10" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc" not in output


def test_aa_qec_stage1_ministral3_dry_run_expands_rawfc_view_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_aa_qec_stage1_ministral3.sh")

    assert "rawfc__ministral3_8b__aa_qec_o1_view_atom_order" in output
    assert "rawfc__ministral3_8b__aa_qec_o2_view_primary_secondary_order" in output
    assert "rawfc__ministral3_8b__aa_qec_o3_view_shuffled" in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=10" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "liar_raw__ministral3_8b__aa_qec" not in output
    assert "__aa_qec_f1" not in output


def test_aa_qec_stage2_liar_raw_atom_facts_abc_dry_run_expands_constrained_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_aa_qec_stage2_liar_raw_atom_facts_abc_ministral3.sh")

    assert "liar_raw__ministral3_8b__aa_qec_c1_atom_facts_abc_primary" in output
    assert "liar_raw__ministral3_8b__aa_qec_c2_atom_facts_abc_primary_secondary" in output
    assert "liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback" in output
    assert "liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary" in output
    assert "aa_qec_constrained_atom_facts_abc_primary_only_qd_prefer_selected_max10" in output
    assert "aa_qec_constrained_atom_facts_abc_primary_secondary_qd_prefer_selected_max10" in output
    assert "aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10" in output
    assert "aa_qec_constrained_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_selected_min5_10" in output
    assert "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "--expected-fingerprint d4cbf7c18126" in output
    assert "EXPECTED_CHUNK_MMR_FINGERPRINT=d4cbf7c18126" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "FORCE_AA_QEC_BUILD=true" in output
    assert "FORCE_STAGE=true" in output
    assert "DATASETS=liar_raw" in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "SFT_SAVE_STEPS=100" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "LIAR_CLASS_WEIGHTS=pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8" in output
    assert "rawfc__ministral3_8b" not in output
    assert "_rawfc" not in output


def test_aa_qec_stage2_liar_raw_atom_facts_abc_c3_c4_full_wrapper_targets_only_c3_c4() -> None:
    output = _run_script("scripts/sentence_trace_method/run_aa_qec_stage2_liar_raw_atom_facts_abc_c3_c4_full_ministral3.sh")

    assert "AA_QEC_STAGE2_CASES=C3,C4" in output
    assert "MODE=full" in output
    assert "RUN_TAU_EVAL=auto" in output
    assert "FORCE_AA_QEC_BUILD=false" in output
    assert "FORCE_STAGE=false" in output
    assert "liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback" in output
    assert "liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary" in output
    assert "liar_raw__ministral3_8b__aa_qec_c1_atom_facts_abc_primary" not in output
    assert "liar_raw__ministral3_8b__aa_qec_c2_atom_facts_abc_primary_secondary" not in output
    assert "-m sft.label_token_trainer" in output
    assert "-m sft.label_token_infer" in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_run_one_dry_run_uses_trace_prompt_style(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_one.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATASET": "liar_raw",
            "MODEL": "ministral3_8b",
            "CASE_SUFFIX": "__qec_dry_run",
            "TRACE_PROMPT_STYLE": "qec_min",
            "EVIDENCE_TEXT_MODE": "anchor_only",
        },
    )

    assert "--trace-prompt-style qec_min" in output
    assert "--evidence-text-mode anchor_only" in output


def test_rawfc_ministral3_lora_r32a64_d005_lr1e5_ep12_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r32a64_d005_lr1e5_ep12.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 32" in output
    assert "--alpha 64" in output
    assert "--dropout 0.05" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_rawfc_ministral3_qec_min_lora_r32a64_d005_lr1e5_ep12_wrapper_includes_test_eval() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_qec_min_lora_r32a64_d005_lr1e5_ep12_eval50.sh",
        {"MODE": "full"},
    )

    run_name = "rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
    assert run_name in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10__qec_min" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 32" in output
    assert "--alpha 64" in output
    assert "--dropout 0.05" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert f"outputs/sentence_trace_method/{run_name}/train" in output
    assert f"outputs/sentence_trace_method/{run_name}/train.resolved.yaml" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL=true" not in output
    assert "label_token_logit_adjust_tau" not in output


def test_rawfc_ministral3_lora_r16a32_d010_lr1e5_ep12_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.10" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_rawfc_ministral3_qec_min_lora_r16a32_d010_lr1e5_ep12_wrapper_includes_test_eval() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_qec_min_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {"MODE": "full"},
    )

    run_name = "rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
    assert run_name in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10__qec_min" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.10" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert f"outputs/sentence_trace_method/{run_name}/train" in output
    assert f"outputs/sentence_trace_method/{run_name}/train.resolved.yaml" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL=true" not in output
    assert "label_token_logit_adjust_tau" not in output


def test_rawfc_ministral3_atom_facts_lora_r16a32_d010_lr1e5_ep12_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_atom_facts_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh")

    assert "rawfc__ministral3_8b__v0_7_atom_facts_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_budgeted_marginal_adaptive5_10" in output
    assert "v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10" in output
    assert "--selector-name v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10" in output
    assert "--source-root outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_budgeted_marginal_adaptive5_10" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.10" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_rawfc_ministral3_atom_facts_abc_lora_r16a32_d010_lr1e5_ep12_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_atom_facts_abc_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh")

    assert "rawfc__ministral3_8b__v0_7_atom_facts_abc_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "--selector-name v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "--source-root outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.10" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_rawfc_ministral3_atom_facts_abc_qec_min_lora_r16a32_d010_lr1e5_ep12_wrapper_includes_test_eval() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_atom_facts_abc_adaptive5_10_qec_min_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {"MODE": "full"},
    )

    run_name = "rawfc__ministral3_8b__v0_7_atom_facts_abc_bm_adaptive5_10__qec_min_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
    assert run_name in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "CASE_SUFFIX=__v0_7_atom_facts_abc_bm_adaptive5_10__qec_min" in output
    assert "outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "--selector-name v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "--source-root outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.10" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert f"outputs/sentence_trace_method/{run_name}/train" in output
    assert f"outputs/sentence_trace_method/{run_name}/train.resolved.yaml" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL=true" not in output
    assert "label_token_logit_adjust_tau" not in output


def test_run_lora_matrix_threads_label_token_early_stopping_metric() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_lora_matrix.sh",
        {
            "DATASETS": "rawfc",
            "MODELS": "ministral3_8b",
            "CASE_SUFFIX": "__metric_override",
            "SFT_EARLY_STOPPING_METRIC": "macro_f1",
        },
    )

    assert "--early-stopping-metric macro_f1" in output


def test_rawfc_ministral3_atom_facts_abc_tight_qec_min_lora_wrapper_uses_macro_f1_best() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_atom_facts_abc_tight_adaptive5_10_qec_min_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {"MODE": "full"},
    )

    run_name = "rawfc__ministral3_8b__v0_7_atom_facts_abc_tight_bm_adaptive5_10__qec_min_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
    selector_name = "v0_7_atom_facts_abc_tight_budgeted_marginal_chain_adaptive5_10"
    source_root = "outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_tight_budgeted_marginal_adaptive5_10"

    assert run_name in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "CASE_SUFFIX=__v0_7_atom_facts_abc_tight_bm_adaptive5_10__qec_min" in output
    assert selector_name in output
    assert f"--selector-name {selector_name}" in output
    assert source_root in output
    assert f"--source-root {source_root}" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "SFT_EARLY_STOPPING_METRIC=macro_f1" in output
    assert "--early-stopping-metric macro_f1" in output
    assert "--learning-rate 1e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "label_token_logit_adjust_tau" not in output


def test_rawfc_ministral3_atom_facts_abc_tight_anchor_only_qec_min_lora_wrapper_threads_anchor_mode() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_atom_facts_abc_tight_anchor_only_adaptive5_10_qec_min_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {"MODE": "full"},
    )

    run_name = "rawfc__ministral3_8b__v0_7_atom_facts_abc_tight_anchor_only_bm_adaptive5_10__qec_min_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
    selector_name = "v0_7_atom_facts_abc_tight_budgeted_marginal_chain_adaptive5_10"
    source_root = "outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_tight_budgeted_marginal_adaptive5_10"

    assert run_name in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "EVIDENCE_TEXT_MODE=anchor_only" in output
    assert "CASE_SUFFIX=__v0_7_atom_facts_abc_tight_anchor_only_bm_adaptive5_10__qec_min" in output
    assert selector_name in output
    assert f"--selector-name {selector_name}" in output
    assert source_root in output
    assert f"--source-root {source_root}" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "SFT_EARLY_STOPPING_METRIC=macro_f1" in output
    assert "--early-stopping-metric macro_f1" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "label_token_logit_adjust_tau" not in output


def test_rawfc_ministral3_lora_r16a32_d005_lr5e6_ep12_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d005_lr5e6_ep12.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d005_ebs16_lr5em6_ep12_eval50_pat8_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "--learning-rate 5e-6" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 50" in output
    assert "--save-steps 50" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.05" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_rawfc_ministral3_lora_four_test_eval_dry_run_expands_plain_test_eval() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_lora_four_test_eval.sh")

    expected_runs = (
        "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc",
        "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc",
        "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc",
        "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d005_ebs16_lr5em6_ep12_eval50_pat8_rawfc",
    )

    assert output.count("-m sft.label_token_infer") == 4
    for run in expected_runs:
        assert f"outputs/sentence_trace_method/{run}/train" in output
        assert f"outputs/sentence_trace_method/{run}/train.resolved.yaml" in output
        assert f"outputs/sentence_trace_method/{run}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL" not in output
    assert "label_token_logit_adjust_tau" not in output


def test_liar_raw_ministral3_qec_map_test_eval_dry_run_targets_current_best() -> None:
    output = _run_script("scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_adaptive5_10_qec_map_test_eval.sh")

    run_name = "liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    assert output.count("-m sft.label_token_infer") == 1
    assert run_name in output
    assert f"outputs/sentence_trace_method/{run_name}/train" in output
    assert f"outputs/sentence_trace_method/{run_name}/train.resolved.yaml" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL" not in output
    assert "label_token_logit_adjust_tau" not in output


def test_liar_raw_ministral3_qec_min_test_eval_dry_run_targets_current_best() -> None:
    output = _run_script("scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_adaptive5_10_qec_min_test_eval.sh")

    run_name = "liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    assert output.count("-m sft.label_token_infer") == 1
    assert run_name in output
    assert f"outputs/sentence_trace_method/{run_name}/train" in output
    assert f"outputs/sentence_trace_method/{run_name}/train.resolved.yaml" in output
    assert f"outputs/sentence_trace_method/{run_name}/eval/test/best/label_token" in output
    assert "--split test" in output
    assert "--checkpoint best" in output
    assert "--logit-adjust off" in output
    assert "RUN_TAU_EVAL" not in output
    assert "label_token_logit_adjust_tau" not in output


def _assert_liar_raw_atom_facts_abc_qec_wrapper(
    output: str,
    *,
    style: str,
) -> None:
    run_name = f"liar_raw__ministral3_8b__v0_7_atom_facts_abc_bm_adaptive5_10__{style}_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    selector_name = "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10"
    source_root = "outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10"

    assert run_name in output
    assert f"TRACE_PROMPT_STYLE={style}" in output
    assert f"CASE_SUFFIX=__v0_7_atom_facts_abc_bm_adaptive5_10__{style}" in output
    assert "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw" in output
    assert selector_name in output
    assert f"--selector-name {selector_name}" in output
    assert source_root in output
    assert f"--source-root {source_root}" in output
    assert "--expected-fingerprint d4cbf7c18126" in output
    assert "EXPECTED_CHUNK_MMR_FINGERPRINT=d4cbf7c18126" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "--learning-rate 2e-5" in output
    assert "--num-train-epochs 12" in output
    assert "--eval-steps 100" in output
    assert "--save-steps 100" in output
    assert "--early-stopping-patience 8" in output
    assert "--gradient-accumulation-steps 4" in output
    assert "--deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--r 16" in output
    assert "--alpha 32" in output
    assert "--dropout 0.05" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "LIAR_CLASS_WEIGHTS=pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8" in output


def test_liar_raw_ministral3_atom_facts_abc_qec_min_lora_wrapper_targets_current_best_recipe() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_atom_facts_abc_adaptive5_10_qec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    )

    _assert_liar_raw_atom_facts_abc_qec_wrapper(output, style="qec_min")


def test_liar_raw_ministral3_atom_facts_abc_qec_map_lora_wrapper_targets_current_best_recipe() -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_atom_facts_abc_adaptive5_10_qec_map_lora_ebs16_lr2e5_ep12_eval100.sh"
    )

    _assert_liar_raw_atom_facts_abc_qec_wrapper(output, style="qec_map")


def test_run_one_dry_run_threads_source_root_expected_fingerprint_into_stage_and_build(tmp_path: Path) -> None:
    source_root = "outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10"
    output = _run_script(
        "scripts/sentence_trace_method/run_one.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATASET": "liar_raw",
            "MODEL": "ministral3_8b",
            "CASE_SUFFIX": "__abc_dry_run",
            "SOURCE_ROOT": source_root,
            "SELECTOR_NAME": "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10",
            "SELECTOR_GRAPH_VERSION": "evidence_chain_graph_v0_7",
            "SELECTOR_ADAPTIVE_POLICY": "budgeted_marginal_v0_7",
            "EXPECTED_CHUNK_MMR_FINGERPRINT": "d4cbf7c18126",
            "ALLOW_MULTI_SENTENCE_CANDIDATES": "true",
        },
    )

    assert f"--source-root {source_root}" in output
    assert "--expected-fingerprint d4cbf7c18126" in output
    assert "--allow-multi-sentence-candidates" in output
    assert "--expected-chunk-mmr-fingerprint d4cbf7c18126" in output


def test_rawfc_ministral3_fullft_aligned_wrapper_matches_best_lora_recipe() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12_fullft_aligned.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em5_ep12_eval100_pat8_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em5_ep12_eval100_pat8_rawfc" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "SFT_SAVE_STEPS=100" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "_lora_" not in output


def test_rawfc_ministral3_fullft_lr1e6_wrapper_keeps_aligned_recipe() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e6_ep12_fullft_aligned.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em6_ep12_eval100_pat8_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em6_ep12_eval100_pat8_rawfc" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=1e-6" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "SFT_SAVE_STEPS=100" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "_lora_" not in output


def test_rawfc_ministral3_fullft_lr1e6_ep3_eval50_pat3_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e6_ep3_eval50_pat3_fullft_aligned.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em6_ep3_eval50_pat3_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em6_ep3_eval50_pat3_rawfc" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=1e-6" in output
    assert "SFT_NUM_TRAIN_EPOCHS=3" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=3" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "_lora_" not in output


def test_rawfc_ministral3_fullft_lr5e7_ep5_eval50_pat4_wrapper() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr5e7_ep5_eval50_pat4_fullft_aligned.sh")

    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr5em7_ep5_eval50_pat4_rawfc" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10_fullft_ebs16_lr5em7_ep5_eval50_pat4_rawfc" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=5e-7" in output
    assert "SFT_NUM_TRAIN_EPOCHS=5" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=4" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "_lora_" not in output


def test_liar_raw_ministral3_fullft_aligned_wrapper_matches_best_lora_recipe() -> None:
    output = _run_script("scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_ep12_fullft_aligned.sh")

    assert "liar_raw__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr2em5_ep12_eval100_pat8_liarw" in output
    assert "v0_7_budgeted_marginal_chain_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_10_fullft_ebs16_lr2em5_ep12_eval100_pat8_liarw" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "SFT_SAVE_STEPS=100" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "LIAR_CLASS_WEIGHTS=pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8" in output
    assert "_lora_" not in output
