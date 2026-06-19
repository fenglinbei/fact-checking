from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


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


def test_mrec_ministral3_main_dry_run_builds_mrec_sources_and_lora_cases(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_mrec_ministral3_main.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_MREC_BUILD": "true",
        },
    )

    assert "liar_raw__ministral3_8b__mrec_min" in output
    assert "rawfc__ministral3_8b__mrec_min_anchor_only" in output
    assert "TRACE_PROMPT_STYLE=mrec_min" in output
    assert "EVIDENCE_TEXT_MODE=anchor_only" in output
    assert "mrec_greedy_transition_v0_1" in output
    assert "minimal_resolving_chain_v0_1" in output
    assert "scripts/phase5_selectors/run/run_mrec_traces.sh" in output
    assert "outputs/selectors/mrec/liar_raw/mrec_greedy_transition_v0_1" in output
    assert "outputs/selectors/mrec/rawfc/mrec_greedy_transition_v0_1" in output
    assert "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "v0_7_atom_facts_abc_tight_budgeted_marginal_chain_adaptive5_10" in output
    assert "SFT_EARLY_STOPPING_METRIC=macro_f1" in output
    assert "--early-stopping-metric macro_f1" in output
    assert "check_mrec_diagnostics.py" in output
    assert "mrec_diagnostics_report.json" in output
    assert "--class-weight barely-true=1.5" in output
    assert "--class-weight true=1.8" in output


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


def test_aa_qec_stage2_liar_raw_atom_facts_abc_c3_c4_full_wrapper_targets_only_c3_c4(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_aa_qec_stage2_liar_raw_atom_facts_abc_c3_c4_full_ministral3.sh",
        {"OUTPUT_ROOT": str(tmp_path / "outputs")},
    )

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


def test_aa_qec_stage2_c3_c4_qec_map_full_wrapper_targets_only_c3_c4(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh",
        {"OUTPUT_ROOT": str(tmp_path / "outputs")},
    )

    assert "AA_QEC_STAGE2_CASES=C3,C4" in output
    assert "MODE=full" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "RUN_TAU_EVAL=auto" in output
    assert "FORCE_AA_QEC_BUILD=false" in output
    assert "FORCE_STAGE=false" in output
    assert "TRACE_PROMPT_STYLE=qec_map" in output
    assert "AA_QEC_STAGE2_CASE_SUFFIX_EXTRA=__qec_map" in output
    assert "liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback__qec_map" in output
    assert "liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary__qec_map" in output
    assert "liar_raw__ministral3_8b__aa_qec_c1_atom_facts_abc_primary" not in output
    assert "liar_raw__ministral3_8b__aa_qec_c2_atom_facts_abc_primary_secondary" not in output
    assert "-m sft.label_token_trainer" in output
    assert "-m sft.label_token_infer" in output
    assert "--split val" in output
    assert "--split test" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output
    assert "rawfc__ministral3_8b" not in output
    assert "_rawfc" not in output


def test_aa_qec_stage2_c3_c4_macro_f1_top3_checkpoint_eval_wrapper(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    c3_run = (
        "liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback"
        "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    )
    c4_run = (
        "liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary"
        "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    )

    def write_step(run_name: str, step: int, macro_f1: float) -> None:
        metrics_dir = output_root / run_name / "eval" / f"step-{step}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"macro_f1": macro_f1, "selection_score": macro_f1 + 0.1}),
            encoding="utf-8",
        )

    for step, macro_f1 in ((100, 0.1), (200, 0.4), (300, 0.3), (400, 0.2)):
        write_step(c3_run, step, macro_f1)
    for step, macro_f1 in ((110, 0.25), (220, 0.55), (330, 0.45), (440, 0.35)):
        write_step(c4_run, step, macro_f1)

    output = _run_script(
        "scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_macro_f1_top3_checkpoint_test_eval_ministral3.sh",
        {
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_SELECT_BIN": sys.executable,
        },
    )

    assert "METRIC=macro_f1" in output
    assert "TOP_K=3" in output
    assert "SPLITS=test" in output
    assert "LOGIT_ADJUST=off" in output
    assert c3_run in output
    assert c4_run in output
    assert "CHECKPOINTS=checkpoint-200,checkpoint-300,checkpoint-400" in output
    assert "CHECKPOINTS=checkpoint-220,checkpoint-330,checkpoint-440" in output
    assert "--split test" in output
    assert "--logit-adjust off" in output
    assert "-m sft.label_token_infer" in output
    assert "-m sft.label_token_trainer" not in output


def test_checkpoint_selection_gap_matrix_wrapper_expands_e0_to_e6_with_four_card_eval(tmp_path: Path) -> None:
    case_root = tmp_path / "outputs" / "liar_raw__ministral3_8b__mrec_min_lora"
    (case_root / "train" / "best").mkdir(parents=True)
    for step, macro_f1 in ((100, 0.610), (200, 0.625), (300, 0.630)):
        step_dir = case_root / "eval" / f"step-{step}"
        step_dir.mkdir(parents=True)
        (case_root / "train" / f"checkpoint-{step}").mkdir(parents=True)
        (step_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "macro_f1": macro_f1,
                    "macro_f1_se": 0.020,
                    "checkpoint_selection_score": macro_f1 + 0.1,
                }
            ),
            encoding="utf-8",
        )
    (case_root / "train.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")

    output = _run_script(
        "scripts/sentence_trace_method/run_checkpoint_selection_gap_matrix.sh",
        {
            "CASE_ROOT": str(case_root),
            "PYTHON_SELECT_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "EVAL_NPROC_PER_NODE": "4",
            "TAU_GRID": "0,1",
        },
    )

    for experiment in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
        assert f"[checkpoint-gap-matrix] EXPERIMENT={experiment}" in output
    assert "E0_current_macro_f1_tau1" in output
    assert "E1_val_macro_f1_tau1" in output
    assert "E2_val_macro_f1_tau0" in output
    assert "E3_val_macro_f1_val_selected_tau" in output
    assert "E4_one_se_tau1" in output
    assert "E5_one_se_tau0" in output
    assert "E6_one_se_val_selected_tau" in output
    assert "CHECKPOINT=checkpoint-300" in output
    assert "CHECKPOINT=checkpoint-100" in output
    assert "--num_processes 4" in output
    assert "LOGIT_ADJUST=on" in output
    assert "TAU=1.0" in output
    assert "TAU=0.0" in output
    assert "logit_adjust" in output
    assert "logit_adjust_tau" in output
    assert output.count("-m sft.label_token_multi_infer") == 2
    assert "--plan-json" in output
    assert "checkpoint_gap_E3_val_macro_f1_val_selected_tau_tau" in output
    assert "checkpoint_gap_E6_one_se_val_selected_tau_tau" in output
    assert "-m sft.label_token_infer" not in output
    assert "-m sft.label_token_trainer" not in output
    assert "label_token_logit_adjust" not in output
    assert "checkpoint-100/label_token" not in output
    assert "checkpoint-110/label_token" not in output


def test_checkpoint_selection_gap_matrix_wrapper_merges_same_selected_checkpoint(tmp_path: Path) -> None:
    case_root = tmp_path / "outputs" / "liar_raw__ministral3_8b__mrec_min_lora"
    (case_root / "train" / "best").mkdir(parents=True)
    for step, macro_f1, macro_f1_se in ((100, 0.610, 0.001), (200, 0.625, 0.001), (300, 0.630, 0.001)):
        step_dir = case_root / "eval" / f"step-{step}"
        step_dir.mkdir(parents=True)
        (case_root / "train" / f"checkpoint-{step}").mkdir(parents=True)
        (step_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "macro_f1": macro_f1,
                    "macro_f1_se": macro_f1_se,
                    "checkpoint_selection_score": macro_f1 + 0.1,
                }
            ),
            encoding="utf-8",
        )
    (case_root / "train.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")

    output = _run_script(
        "scripts/sentence_trace_method/run_checkpoint_selection_gap_matrix.sh",
        {
            "CASE_ROOT": str(case_root),
            "PYTHON_SELECT_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "EVAL_NPROC_PER_NODE": "4",
            "TAU_GRID": "0,1",
        },
    )

    assert output.count("-m sft.label_token_multi_infer") == 1
    assert "CHECKPOINT=checkpoint-300" in output
    assert "E1_val_macro_f1_tau1" in output
    assert "E6_one_se_val_selected_tau" in output


def test_aa_qec_stage2_c3_c4_macro_f1_top3_checkpoint_eval_wrapper_supports_distributed_dry_run(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    c3_run = (
        "liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback"
        "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    )
    c4_run = (
        "liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary"
        "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    )
    for run_name, step, macro_f1 in ((c3_run, 200, 0.4), (c4_run, 220, 0.5)):
        metrics_dir = output_root / run_name / "eval" / f"step-{step}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.json").write_text(json.dumps({"macro_f1": macro_f1}), encoding="utf-8")

    output = _run_script(
        "scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_macro_f1_top3_checkpoint_test_eval_ministral3.sh",
        {
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_SELECT_BIN": sys.executable,
            "ACCELERATE_BIN": "/opt/accelerate",
            "EVAL_NPROC_PER_NODE": "4",
            "TOP_K": "1",
        },
    )

    assert "EVAL_NPROC_PER_NODE=4" in output
    assert "+ /opt/accelerate launch" in output
    assert "--num_processes 4" in output
    assert "--num_machines 1" in output
    assert "--mixed_precision bf16" in output
    assert "-m sft.label_token_infer" in output
    assert "--split test" in output
    assert "--logit-adjust off" in output
    assert "+ /data/liaozijie/conda/accelerate-fc-gemma4/bin/python -m sft.label_token_infer" not in output
    assert "-m sft.label_token_trainer" not in output


def test_aa_qec_stage3_liar_raw_atom_facts_abc_dry_run_expands_full_top20_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_ministral3.sh")

    assert "liar_raw__ministral3_8b__aa_qec_f1_atom_facts_abc_primary_fallback_no_secondary" in output
    assert "liar_raw__ministral3_8b__aa_qec_f2_atom_facts_abc_primary_secondary_fallback" in output
    assert "liar_raw__ministral3_8b__aa_qec_f3_atom_facts_abc_primary_secondary_dynamic" in output
    assert "aa_qec_full_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_top20_min5_10" in output
    assert "aa_qec_full_atom_facts_abc_primary_secondary_fallback_qd_prefer_top20_min5_10" in output
    assert "aa_qec_full_atom_facts_abc_primary_secondary_dynamic_qd_prefer_top20" in output
    assert "CANDIDATE_SCOPE=top20" in output
    assert "SELECTOR_ADAPTIVE_POLICY=aa_qec_full_atom_facts_abc" in output
    assert "FORCE_AA_QEC_BUILD=true" in output
    assert "FORCE_STAGE=true" in output
    assert "MODE=build" in output
    assert "check_aa_qec_stage3_build_gate.py" in output
    assert "MIN_CHAIN_STEPS=0" in output
    assert "MAX_CHAIN_STEPS=0" in output
    assert "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10" in output
    assert "--expected-fingerprint d4cbf7c18126" in output
    assert "EXPECTED_CHUNK_MMR_FINGERPRINT=d4cbf7c18126" in output
    assert "--allow-multi-sentence-candidates" in output
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
    assert "aa_qec_constrained_atom_facts_abc_primary_only_qd_prefer_selected_max10" not in output
    assert "aa_qec_constrained_atom_facts_abc_primary_secondary_qd_prefer_selected_max10" not in output
    assert "aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10" not in output
    assert "aa_qec_constrained_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_selected_min5_10" not in output


def test_aa_qec_stage3_liar_raw_atom_facts_abc_f1_f3_full_wrapper_defaults_to_val_test(
    tmp_path: Path,
) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_f1_f3_full_ministral3.sh",
        {"OUTPUT_ROOT": str(tmp_path / "outputs")},
    )

    assert "AA_QEC_STAGE3_CASES=F1,F2,F3" in output
    assert "MODE=full" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "RUN_TAU_EVAL=auto" in output
    assert "RUN_STAGE3_BUILD_GATE=true" in output
    assert "check_aa_qec_stage3_build_gate.py" in output
    assert "FORCE_AA_QEC_BUILD=false" in output
    assert "FORCE_STAGE=false" in output
    assert "liar_raw__ministral3_8b__aa_qec_f1_atom_facts_abc_primary_fallback_no_secondary" in output
    assert "liar_raw__ministral3_8b__aa_qec_f2_atom_facts_abc_primary_secondary_fallback" in output
    assert "liar_raw__ministral3_8b__aa_qec_f3_atom_facts_abc_primary_secondary_dynamic" in output
    assert "CANDIDATE_SCOPE=top20" in output
    assert "SELECTOR_ADAPTIVE_POLICY=aa_qec_full_atom_facts_abc" in output
    assert "-m sft.label_token_trainer" in output
    assert "-m sft.label_token_infer" in output
    assert "--split val" in output
    assert "--split test" in output
    assert "TRACE_PROMPT_STYLE=qec_min" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=true" in output


def test_aa_qec_stage3_build_gate_uses_baseline_floors_build_rows_and_test_qd_warnings(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    graph_root = tmp_path / "graphs"
    source_selector = "source_selector"
    case_id = "F2"
    selector_name = "aa_qec_full_atom_facts_abc_primary_secondary_fallback_qd_prefer_top20_min5_10"
    baseline_selector = "aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10"
    case_root = output_root / "liar_raw__ministral3_8b__aa_qec_f2_atom_facts_abc_primary_secondary_fallback"

    metrics_by_split = {
        "train": (0.790, 0.960),
        "val": (0.840, 0.960),
        "test": (0.850, 0.940),
    }
    for split, (atom_coverage, qd_cue) in metrics_by_split.items():
        source_rows = [
            {"id": f"{split}-0", "selector_ordered_indices": [0]},
            {"id": f"{split}-1", "selector_ordered_indices": [0]},
        ]
        graph_rows = [
            {
                "id": f"{split}-0",
                "selector_ordered_indices": [1],
                "chain_diagnostics": {
                    "duplicate_evidence_rate": 0.0,
                    "atom_coverage_rate": atom_coverage,
                    "qd_cue_rate": qd_cue,
                },
            },
            {
                "id": f"{split}-1",
                "selector_ordered_indices": [1],
                "chain_diagnostics": {
                    "duplicate_evidence_rate": 0.0,
                    "atom_coverage_rate": atom_coverage,
                    "qd_cue_rate": qd_cue,
                },
            },
        ]
        baseline_rows = [
            {
                "id": f"{split}-0",
                "selector_ordered_indices": [0],
                "chain_diagnostics": {
                    "duplicate_evidence_rate": 0.0,
                    "atom_coverage_rate": atom_coverage,
                    "qd_cue_rate": 0.960,
                },
            },
            {
                "id": f"{split}-1",
                "selector_ordered_indices": [0],
                "chain_diagnostics": {
                    "duplicate_evidence_rate": 0.0,
                    "atom_coverage_rate": atom_coverage,
                    "qd_cue_rate": 0.960,
                },
            },
        ]
        build_rows = [
            {
                "id": f"{split}-0",
                "was_truncated": False,
                "evidence_text_truncated": False,
                "prompt_token_count": 512,
                "evidence_count": 5,
            },
            {
                "id": f"{split}-1",
                "was_truncated": False,
                "evidence_text_truncated": False,
                "prompt_token_count": 768,
                "evidence_count": 6,
            },
        ]

        _write_jsonl(
            output_root / "_sources" / "liar_raw" / source_selector / split / f"selection_trace_{split}.jsonl",
            source_rows,
        )
        _write_jsonl(
            output_root / "_sources" / "liar_raw" / selector_name / split / f"selection_trace_{split}.jsonl",
            graph_rows,
        )
        _write_jsonl(
            graph_root / f"{selector_name}_{split}" / f"selection_trace_{split}.jsonl",
            graph_rows,
        )
        _write_jsonl(
            graph_root / f"{baseline_selector}_{split}" / f"selection_trace_{split}.jsonl",
            baseline_rows,
        )
        _write_jsonl(case_root / "build" / f"build_{split}.jsonl", build_rows)

    report_path = tmp_path / "gate_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/check_aa_qec_stage3_build_gate.py"),
            "--output-root",
            str(output_root),
            "--graph-root",
            str(graph_root),
            "--source-selector-name",
            source_selector,
            "--cases",
            case_id,
            "--splits",
            "train,val,test",
            "--prompt-splits",
            "train,val,test",
            "--report-path",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[aa-qec-stage3-build-gate] PASSED" in result.stdout
    assert "[aa-qec-stage3-build-gate] WARNINGS" in result.stdout
    assert "F2/test: qd_cue_rate.mean=0.940000 < 0.950000" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["warnings"]
    assert report["cases"][case_id]["splits"]["train"]["atom_coverage_floor_source"] == baseline_selector
    assert report["cases"][case_id]["prompt_splits"]["train"]["source"] == "build_rows"
    assert report["cases"][case_id]["prompt_splits"]["test"]["truncation_rate"] == 0.0


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
