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


def test_run_one_dry_run_uses_trace_prompt_style(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_one.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATASET": "liar_raw",
            "MODEL": "ministral3_8b",
            "CASE_SUFFIX": "__qec_dry_run",
            "TRACE_PROMPT_STYLE": "qec_min",
        },
    )

    assert "--trace-prompt-style qec_min" in output


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
