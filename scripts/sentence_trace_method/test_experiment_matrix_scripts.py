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


def test_atom_anchor_v0_1_full_wrapper_uses_existing_mrec_traces_for_ministral3_lora(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "liar_raw__ministral3_8b__atom_anchor_v0_1_mrec_min" in output
    assert "ATOM_ANCHOR_ROOT=outputs/selectors/atom_anchor/liar_raw_abc_v0_1" in output
    assert "TRACE_PROMPT_STYLE=mrec_min" in output
    assert "EVIDENCE_TEXT_MODE=full" in output
    assert "scripts/phase5_selectors/build/build_trace_verifier_data.py" in output
    assert "--train-trace outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec/selection_trace_train.jsonl" in output
    assert "--config scripts/sentence_trace_method/configs/liar_raw__ministral3_8b.yaml" in output
    assert "--top-k 10" in output
    assert "scripts/sentence_trace_method/prepare_lora_config.py" in output
    assert "--source-config" in output
    assert "sft.label_token_trainer" in output
    assert "sft.label_token_infer" in output
    assert "run_lora_label_token_logit_adjust_eval_only.sh" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "--early-stopping-metric macro_f1" in output


def test_atom_anchor_v0_2_learned_proxy_top5_full_wrapper_uses_existing_mrec_traces(
    tmp_path: Path,
) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_learned_marginal_proxy_top5_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_top5" in output
    assert "TRACE_ROOT=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy" in output
    assert "WEIGHT_FILE=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json" in output
    assert "TRACE_PROMPT_STYLE=mrec_min" in output
    assert "EVIDENCE_TEXT_MODE=full" in output
    assert "EXPECTED_SELECTOR_NAME=mrec_greedy_transition_v0_2_learned_marginal_proxy" in output
    assert "--expected-selector-name mrec_greedy_transition_v0_2_learned_marginal_proxy" in output
    assert "--train-trace outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/selection_trace_train.jsonl" in output
    assert "--top-k 5" in output
    assert "scripts/sentence_trace_method/prepare_lora_config.py" in output
    assert "sft.label_token_trainer" in output
    assert "sft.label_token_infer" in output
    assert "run_lora_label_token_logit_adjust_eval_only.sh" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output
    assert "--early-stopping-metric macro_f1" in output


def test_atom_anchor_v0_2_learned_proxy_budget1024_full_wrapper_builds_budget_traces(
    tmp_path: Path,
) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_learned_marginal_proxy_budget1024_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_budget1024" in output
    assert "TRACE_ROOT=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_budget1024" in output
    assert "WEIGHT_FILE=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json" in output
    assert "TOKEN_BUDGET=1024" in output
    assert "TRACE_TOP_K=100" in output
    assert "EXPECTED_SELECTOR_NAME=mrec_greedy_transition_v0_2_learned_marginal_proxy_budget1024" in output
    assert "scripts/phase5_selectors/build/build_mrec_traces.py" in output
    assert "--input outputs/selectors/atom_anchor/liar_raw_abc_v0_1/04_evidence_map/candidate_evidence_map_features_train.jsonl" in output
    assert "--selection-policy learned_marginal_proxy" in output
    assert "--weight-file outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json" in output
    assert "--token-budget 1024" in output
    assert "--max-steps 100" in output
    assert "--stop-threshold -1000000000" in output
    assert "--expected-selector-name mrec_greedy_transition_v0_2_learned_marginal_proxy_budget1024" in output
    assert "--top-k 100" in output
    assert "scripts/sentence_trace_method/prepare_lora_config.py" in output
    assert "sft.label_token_trainer" in output
    assert "sft.label_token_infer" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output


def test_atom_anchor_v0_2_fullpool_policy_wrapper_reads_yaml_config(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "MREC_POLICY_CONFIG=configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_policy.yaml" in output
    assert "PROMPT_EVIDENCE_POLICY=fixed_topk" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=5" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=5" in output
    assert "TRACE_CANDIDATE_TOP_N=0" in output
    assert "TRACE_MAX_STEPS=0" in output
    assert "scripts/phase5_selectors/build/build_mrec_traces.py" in output
    assert "--candidate-top-n 0" in output
    assert "--max-steps 0" in output
    assert "--config configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_policy.yaml" in output
    assert "--prompt-evidence-policy fixed_topk" in output
    assert "--prompt-evidence-min-count 5" in output
    assert "--prompt-evidence-max-count 5" in output
    assert "--swanlab-project fact-checking-sentence-trace-method-atom-anchor-v0-2" in output
    assert "-u SWANLAB_PROJECT" in output
    assert "TOKEN_BUDGET=1024" not in output
    assert "BUDGET_MAX_STEPS=100" not in output


def test_selector_mechanism_s0_s4_wrapper_dry_run_expands_cases(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_selector_mechanism_s0_s4_plain_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "SELECTOR_MECH_MODE": "full",
            "PREPARE_SELECTOR_MECH_TRACES": "true",
            "RUN_TAU_EVAL": "false",
        },
    )

    for selector in (
        "selector_mech_s0_no_evidence",
        "selector_mech_s1_claim_pool_random_top5",
        "selector_mech_s2_claim_pool_hybrid_top5",
        "selector_mech_s3_claim_pool_hybrid_mmr_top5",
        "selector_mech_s4_atom_union_source_score_top5",
    ):
        assert selector in output
        assert f"CASE_SUFFIX=__{selector}_plain" in output
    assert "run_liar_raw_selector_mechanism_s0_s4.sh" in output
    assert "ALLOW_EMPTY_EVIDENCE=true" in output
    assert "ALLOW_EMPTY_CANDIDATE_POOL=true" in output
    assert "TRACE_PROMPT_STYLE=plain" in output
    assert "SFT_LEARNING_RATE=2e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output


def test_lora_matrix_dry_run_passes_prompt_evidence_policy_args(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_lora_matrix.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATASETS": "liar_raw",
            "MODELS": "ministral3_8b",
            "CASE_SUFFIX": "__prompt_policy_probe",
            "SELECTOR_NAME": "selector_mech_s4_atom_union_source_score_ordered",
            "EXPECTED_SELECTOR_NAME": "selector_mech_s4_atom_union_source_score_ordered",
            "SOURCE_ROOT": "outputs/selectors/selector_mechanism_ablation_chunking/liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered",
            "TRACE_TOP_K": "20",
            "PROMPT_EVIDENCE_POLICY": "budget",
            "PROMPT_EVIDENCE_MIN_COUNT": "1",
            "PROMPT_EVIDENCE_MAX_COUNT": "20",
            "PROMPT_EVIDENCE_TOKEN_BUDGET": "541",
            "PROMPT_EVIDENCE_MAX_LENGTH_GUARD": "warn",
        },
    )

    assert "PROMPT_EVIDENCE_POLICY=budget" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=1" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=20" in output
    assert "PROMPT_EVIDENCE_TOKEN_BUDGET=541" in output
    assert "PROMPT_EVIDENCE_MAX_LENGTH_GUARD=warn" in output


def test_run_one_dry_run_passes_prompt_evidence_policy_args(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_one.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATASET": "liar_raw",
            "MODEL": "ministral3_8b",
            "CASE_SUFFIX": "__prompt_policy_probe",
            "SELECTOR_NAME": "selector_mech_s4_atom_union_source_score_ordered",
            "EXPECTED_SELECTOR_NAME": "selector_mech_s4_atom_union_source_score_ordered",
            "SOURCE_ROOT": "outputs/selectors/selector_mechanism_ablation_chunking/liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered",
            "TRACE_TOP_K": "20",
            "PROMPT_EVIDENCE_POLICY": "budget",
            "PROMPT_EVIDENCE_MIN_COUNT": "1",
            "PROMPT_EVIDENCE_MAX_COUNT": "20",
            "PROMPT_EVIDENCE_TOKEN_BUDGET": "541",
            "PROMPT_EVIDENCE_MAX_LENGTH_GUARD": "warn",
        },
    )

    assert "--top-k 20" in output
    assert "--prompt-evidence-policy budget" in output
    assert "--prompt-evidence-min-count 1" in output
    assert "--prompt-evidence-max-count 20" in output
    assert "--prompt-evidence-token-budget 541" in output
    assert "--prompt-evidence-max-length-guard warn" in output


def test_lora_matrix_skips_train_when_best_checkpoint_exists(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    lora_root = output_root / "liar_raw__ministral3_8b__resume_probe_lora"
    (lora_root / "train" / "best").mkdir(parents=True)
    (lora_root / "train.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/sentence_trace_method/run_lora_matrix.sh"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
            "PYTHON_BIN": sys.executable,
            "ACCELERATE_BIN": "/bin/false",
            "OUTPUT_ROOT": str(output_root),
            "DATASETS": "liar_raw",
            "MODELS": "ministral3_8b",
            "CASE_SUFFIX": "__resume_probe",
            "LORA_SUFFIX": "_lora",
            "MODE": "train",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "LoRA training artifacts already exist" in result.stdout
    assert "sft.label_token_trainer" not in result.stdout


def test_lora_matrix_skips_eval_when_nested_label_token_metrics_exist(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    lora_root = output_root / "liar_raw__ministral3_8b__eval_probe_lora"
    (lora_root / "eval" / "val" / "best" / "label_token").mkdir(parents=True)
    (lora_root / "eval" / "val" / "best" / "label_token" / "metrics.json").write_text(
        json.dumps({"macro_f1": 1.0}),
        encoding="utf-8",
    )
    (lora_root / "train.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/sentence_trace_method/run_lora_matrix.sh"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
            "PYTHON_BIN": "/bin/false",
            "OUTPUT_ROOT": str(output_root),
            "DATASETS": "liar_raw",
            "MODELS": "ministral3_8b",
            "CASE_SUFFIX": "__eval_probe",
            "LORA_SUFFIX": "_lora",
            "MODE": "eval",
            "EVAL_SPLITS": "val",
            "CHECKPOINTS": "best",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "LoRA eval already exists" in result.stdout


def test_atom_union_s4_chunking_ablation_wrapper_dry_run_expands_cases(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_union_s4_chunking_ablation.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "MODE": "full",
            "RUN_TAU_EVAL": "false",
        },
    )

    for case in ("abc", "sentence", "sentwin1", "semantic07", "report"):
        assert f"chunk_{case}_s4_union_top5_plain" in output
        assert f"chunk_{case}_s4_union_budget_promptmatched_plain" in output
    assert "selector_mech_s4_atom_union_source_score_ordered" in output
    assert "PROMPT_EVIDENCE_POLICY=fixed_topk" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=5" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=5" in output
    assert "PROMPT_EVIDENCE_POLICY=budget" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=1" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=20" in output
    assert "TRACE_PROMPT_STYLE=plain" in output
    assert "EVIDENCE_TEXT_MODE=full" in output
    assert "NPROC_PER_NODE=4" in output
    assert "NCCL_CUMEM_HOST_ENABLE=0" in output
    assert "OMP_NUM_THREADS=1" in output


def test_atom_union_s4_chunking_ablation_wrapper_rejects_too_many_processes_for_visible_devices(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "DRY_RUN": "true",
        "MODE": "train",
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "CUDA_VISIBLE_DEVICES": "0",
        "RUN_TAU_EVAL": "false",
    }
    result = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_union_s4_chunking_ablation.sh"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert (
        "Requested NPROC_PER_NODE=4 but CUDA_VISIBLE_DEVICES exposes only 1 device(s): 0"
        in result.stderr
    )
    assert "sft.label_token_trainer" not in result.stdout


def test_selector_mechanism_s5_s6_wrapper_dry_run_expands_chain_cases(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_selector_mechanism_s5_s6_plain_lora_ebs16_lr2e5_ep12_eval100.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "SELECTOR_MECH_MODE": "full",
            "FORCE_MREC_BUILD": "true",
            "RUN_TAU_EVAL": "false",
        },
    )

    assert "selector_mech_s5_map_quality_greedy" in output
    assert "selector_mech_s6_learned_marginal_proxy_trace_shuffle" in output
    assert "configs/experiment/mrec_v0.2/selector_mech_s5_map_quality_greedy.yaml" in output
    assert "configs/experiment/mrec_v0.2/selector_mech_s6_learned_marginal_proxy_trace_shuffle.yaml" in output
    assert "CASE_SUFFIX=__selector_mech_s5_map_quality_greedy" in output
    assert "CASE_SUFFIX=__selector_mech_s6_learned_marginal_proxy_trace_shuffle" in output
    assert "--selection-policy map_quality_greedy" in output
    assert "scripts/phase5_selectors/build/shuffle_mrec_trace_order.py" in output
    assert "--seed 0" in output
    assert "TRACE_PROMPT_STYLE=mrec_min" in output
    assert "PROMPT_EVIDENCE_POLICY=fixed_topk" in output
    assert "SFT_LEARNING_RATE=2e-05" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=100" in output


def test_rawfc_atom_anchor_v0_2_fullpool_policy_wrapper_reads_rawfc_yaml_config(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_fullpool_policy_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "FORCE_WEIGHT_TRAIN": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "MREC_POLICY_CONFIG=configs/experiment/mrec_v0.2/rawfc_learned_marginal_proxy_fullpool_minmax5_10.yaml" in output
    assert "SOURCE_FEATURE_ROOT=outputs/selectors/atom_anchor/rawfc_abc_v0_1/04_evidence_map" in output
    assert "TRACE_ROOT=outputs/selectors/atom_anchor/rawfc_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool" in output
    assert "WEIGHT_FILE=outputs/selectors/atom_anchor/rawfc_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json" in output
    assert "rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10" in output
    assert "PROMPT_EVIDENCE_POLICY=minmax" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=5" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=10" in output
    assert "QUALITY_AUDIT_MODE=source_only" in output
    assert "scripts/phase5_selectors/train/train_mrec_learned_marginal_proxy.py" in output
    assert "--train-input outputs/selectors/atom_anchor/rawfc_abc_v0_1/04_evidence_map/candidate_evidence_map_features_train.jsonl" in output
    assert "--val-input outputs/selectors/atom_anchor/rawfc_abc_v0_1/04_evidence_map/candidate_evidence_map_features_val.jsonl" in output
    assert "--input outputs/selectors/atom_anchor/rawfc_abc_v0_1/04_evidence_map/candidate_evidence_map_features_train.jsonl" in output
    assert "--source-selector-name v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10" in output
    assert "--dataset rawfc" in output
    assert "--label-schema rawfc3" in output
    assert "--train-raw data/raw/RAWFC/train.json" in output
    assert "--val-raw data/raw/RAWFC/val.json" in output
    assert "--test-raw data/raw/RAWFC/test.json" in output
    assert "--config configs/experiment/mrec_v0.2/rawfc_learned_marginal_proxy_fullpool_minmax5_10.yaml" in output
    assert "--class-weight false=1.0" in output
    assert "--class-weight half=1.0" in output
    assert "--class-weight true=1.0" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "pants-fire" not in output


def test_rawfc_atom_anchor_baseline20_fullpool_policy_wrapper_reads_baseline20_yaml_config(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_fullpool_policy_baseline20_lora_r16a32_d010_lr1e5_ep12_eval50.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "FORCE_WEIGHT_TRAIN": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "MREC_POLICY_CONFIG=configs/experiment/mrec_v0.2/rawfc_learned_marginal_proxy_fullpool_minmax5_10_baseline20.yaml" in output
    assert "SOURCE_FEATURE_ROOT=outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/04_evidence_map" in output
    assert "TRACE_ROOT=outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/05_mrec_v0_2_learned_marginal_proxy_fullpool" in output
    assert "WEIGHT_FILE=outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json" in output
    assert "rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_baseline20" in output
    assert "--train-input outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/04_evidence_map/candidate_evidence_map_features_train.jsonl" in output
    assert "--config configs/experiment/mrec_v0.2/rawfc_learned_marginal_proxy_fullpool_minmax5_10_baseline20.yaml" in output
    assert "--dataset rawfc" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_EVAL_STEPS=50" in output


def test_rawfc_llama31_atom_anchor_fullpool_minmax5_10_lora_matches_rawfc_recipe(
    tmp_path: Path,
) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_lora_ebs16_lr1e5_ep12_eval50.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "FORCE_WEIGHT_TRAIN": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "MREC_POLICY_CONFIG=configs/experiment/mrec_v0.2/rawfc_llama31_learned_marginal_proxy_fullpool_minmax5_10_lora.yaml" in output
    assert "BASE_CASE_NAME=rawfc__llama31_8b" in output
    assert "MODEL_PATH=/data/models/Meta-Llama-3.1-8B-Instruct" in output
    assert "rawfc__llama31_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_lora_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "PROMPT_EVIDENCE_POLICY=minmax" in output
    assert "PROMPT_EVIDENCE_MIN_COUNT=5" in output
    assert "PROMPT_EVIDENCE_MAX_COUNT=10" in output
    assert "--dataset rawfc" in output
    assert "--label-schema rawfc3" in output
    assert "--prompt-model-name-or-path /data/models/Meta-Llama-3.1-8B-Instruct" in output
    assert "--train-model-name-or-path /data/models/Meta-Llama-3.1-8B-Instruct" in output
    assert "--class-weight false=1.0" in output
    assert "--class-weight half=1.0" in output
    assert "--class-weight true=1.0" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "DEEPSPEED_CONFIG=configs/deepspeed/deepspeed_zero2_bsz1_ga4.json" in output
    assert "--deepspeed-config configs/deepspeed/deepspeed_zero2_bsz1_ga4.json" in output
    assert "LORA_R=16" in output
    assert "LORA_ALPHA=32" in output
    assert "LORA_DROPOUT=0.1" in output
    assert "REQUIRE_PROMPT_INPUT_IDS=false" in output
    assert "configs/deepspeed_zero2_bsz1_ga4.json" not in output
    assert "pants-fire" not in output


def test_rawfc_llama31_atom_anchor_fullpool_minmax5_10_fullft_matches_rawfc_recipe(
    tmp_path: Path,
) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_fullft_ebs16_lr1e5_ep12_eval50.sh",
        {
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FORCE_BUILD": "true",
            "FORCE_MREC_BUILD": "true",
            "FORCE_WEIGHT_TRAIN": "true",
            "MODE": "full",
            "RUN_TAU_EVAL": "auto",
        },
    )

    assert "MREC_POLICY_CONFIG=configs/experiment/mrec_v0.2/rawfc_llama31_learned_marginal_proxy_fullpool_minmax5_10_fullft.yaml" in output
    assert "FINETUNE_MODE=fullft" in output
    assert "BASE_CASE_NAME=rawfc__llama31_8b" in output
    assert "MODEL_PATH=/data/models/Meta-Llama-3.1-8B-Instruct" in output
    assert "CASE_NAME=rawfc__llama31_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_fullft_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "rawfc__llama31_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_fullft_ebs16_lr1em5_ep12_eval50_pat8_rawfc_lora" not in output
    assert "configs/deepspeed/deepspeed_zero3_bsz1_ga4_lowpeak.json" in output
    assert "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json" not in output
    assert "SFT_GRADIENT_ACCUMULATION_STEPS=4" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_NUM_TRAIN_EPOCHS=12" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "SFT_SAVE_STEPS=50" in output
    assert "SFT_EARLY_STOPPING_PATIENCE=8" in output
    assert "--config" in output
    assert "train.resolved.yaml" in output
    assert "prepare_lora_config.py" not in output
    assert "REQUIRE_PROMPT_INPUT_IDS=false" in output
    assert "pants-fire" not in output


def test_mrec_policy_config_does_not_export_swanlab_project_env() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sentence_trace_method/mrec_policy_config.py"),
            "--config",
            "configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_policy.yaml",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "export WRAPPER_SWANLAB_PROJECT=" in result.stdout
    assert "export SWANLAB_PROJECT=" not in result.stdout


def test_atom_anchor_v0_2_fullpool_policy_wrapper_rejects_too_many_processes_for_visible_devices(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "DRY_RUN": "true",
        "MODE": "train",
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "CUDA_VISIBLE_DEVICES": "0",
        "RUN_TAU_EVAL": "false",
    }
    result = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert (
        "Requested NPROC_PER_NODE=4 but CUDA_VISIBLE_DEVICES exposes only 1 device(s): 0"
        in result.stderr
    )
    assert "sft.label_token_trainer" not in result.stdout


def test_mrec_quick_ablation_build_wrappers_set_capacity_knobs(tmp_path: Path) -> None:
    cases = [
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t1p0_build.sh",
            "__mrec_t1p0",
            "mrec_greedy_transition_v0_1_t1p0",
            "TARGET_RESOLVED_RATE=1.0",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=false",
            "CANDIDATE_TOP_N=20",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t08_contrast_build.sh",
            "__mrec_t08_contrast",
            "mrec_greedy_transition_v0_1_t08_contrast",
            "TARGET_RESOLVED_RATE=0.80",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=20",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t1p0_contrast_build.sh",
            "__mrec_t1p0_contrast",
            "mrec_greedy_transition_v0_1_t1p0_contrast",
            "TARGET_RESOLVED_RATE=1.0",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=20",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t08_contrast_top50_build.sh",
            "__mrec_t08_contrast_top50",
            "mrec_greedy_transition_v0_1_t08_contrast_top50",
            "TARGET_RESOLVED_RATE=0.80",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=50",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t08_contrast_top100_build.sh",
            "__mrec_t08_contrast_top100",
            "mrec_greedy_transition_v0_1_t08_contrast_top100",
            "TARGET_RESOLVED_RATE=0.80",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=100",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t08_fill_min3_build.sh",
            "__mrec_t08_fill_min3",
            "mrec_greedy_transition_v0_1_t08_fill_min3",
            "TARGET_RESOLVED_RATE=0.80",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=20",
            "POST_TARGET_FILL_POLICY=contrast_then_support",
            "MIN_STEPS=3",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_t08_fill_min5_build.sh",
            "__mrec_t08_fill_min5",
            "mrec_greedy_transition_v0_1_t08_fill_min5",
            "TARGET_RESOLVED_RATE=0.80",
            "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true",
            "CANDIDATE_TOP_N=20",
            "POST_TARGET_FILL_POLICY=contrast_then_support",
            "MIN_STEPS=5",
        ),
    ]

    for script, case_suffix, selector_name, target_rate, contrast_policy, top_n, fill_policy, min_steps in cases:
        output = _run_script(script, {"OUTPUT_ROOT": str(tmp_path / Path(script).stem)})

        assert "DATASETS=liar_raw" in output
        assert "rawfc__ministral3_8b" not in output
        assert "MODE=build" in output
        assert "TRACE_PROMPT_STYLE=mrec_min" in output
        assert "RUN_MREC_DIAGNOSTICS=false" in output
        assert f"CASE_SUFFIX={case_suffix}" in output
        assert selector_name in output
        assert target_rate in output
        assert contrast_policy in output
        assert top_n in output
        assert fill_policy in output
        assert min_steps in output
        assert "-m sft.label_token_trainer" not in output


def test_mrec_final_full_wrappers_select_contrast_only_and_fill_min5(tmp_path: Path) -> None:
    cases = [
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_contrast_only_min0_full.sh",
            "__mrec_contrast_only_min0",
            "mrec_greedy_transition_v0_1_contrast_only_min0",
            "POST_TARGET_FILL_POLICY=contrast_only",
            "MIN_STEPS=0",
        ),
        (
            "scripts/sentence_trace_method/run_liar_raw_ministral3_mrec_contrast_then_support_min5_full.sh",
            "__mrec_contrast_then_support_min5",
            "mrec_greedy_transition_v0_1_contrast_then_support_min5",
            "POST_TARGET_FILL_POLICY=contrast_then_support",
            "MIN_STEPS=5",
        ),
    ]

    for script, case_suffix, selector_name, fill_policy, min_steps in cases:
        output = _run_script(
            script,
            {
                "MODE": "",
                "OUTPUT_ROOT": str(tmp_path / Path(script).stem),
            },
        )

        assert "RUN_LIAR_RAW=true" in output
        assert "RUN_RAWFC=false" in output
        assert "MODE=full" in output
        assert "TRACE_PROMPT_STYLE=mrec_min" in output
        assert "RUN_MREC_DIAGNOSTICS=false" in output
        assert "FORCE_MREC_BUILD=false" in output
        assert f"CASE_SUFFIX={case_suffix}" in output
        assert selector_name in output
        assert "TARGET_RESOLVED_RATE=0.80" in output
        assert "CONTINUE_AFTER_TARGET_FOR_CONTRAST=true" in output
        assert "CANDIDATE_TOP_N=20" in output
        assert fill_policy in output
        assert min_steps in output
        assert "liar_raw__ministral3_8b" in output
        assert "rawfc__ministral3_8b" not in output
        assert "-m sft.label_token_trainer" in output
        assert "-m sft.label_token_infer" in output


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


def test_liar_raw_ministral3_map_selector_s0_s2_plain_wrapper_dry_run(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_map_selector_s0_s2_plain_lora_ebs16_lr2e5_ep12_eval100.sh",
        {"OUTPUT_ROOT": str(tmp_path / "outputs")},
    )

    assert "DATASETS=liar_raw" in output
    assert "MODELS=ministral3_8b" in output
    assert "TRACE_PROMPT_STYLE=plain" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "RUN_TAU_EVAL=auto" in output
    assert "FORCE_STAGE=true" in output
    assert "map_selector_s0_retrieval_top5" in output
    assert "map_selector_s1_mmr_pool_top5" in output
    assert "map_selector_s2_map_quality_top5" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s0_retrieval_top5" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s1_mmr_pool_top5" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s2_map_quality_top5" in output
    assert "liar_raw__ministral3_8b__map_selector_s0_retrieval_top5_plain" in output
    assert "liar_raw__ministral3_8b__map_selector_s1_mmr_pool_top5_plain" in output
    assert "liar_raw__ministral3_8b__map_selector_s2_map_quality_top5_plain" in output
    assert "TRACE_PROMPT_STYLE=qec_min" not in output
    assert "TRACE_PROMPT_STYLE=qec_map" not in output
    assert "rawfc__ministral3_8b" not in output


def test_liar_raw_ministral3_map_selector_s3_s5_plain_wrapper_dry_run(tmp_path: Path) -> None:
    output = _run_script(
        "scripts/sentence_trace_method/run_liar_raw_ministral3_map_selector_s3_s5_plain_lora_ebs16_lr2e5_ep12_eval100.sh",
        {"OUTPUT_ROOT": str(tmp_path / "outputs")},
    )

    assert "DATASETS=liar_raw" in output
    assert "MODELS=ministral3_8b" in output
    assert "TRACE_PROMPT_STYLE=plain" in output
    assert "EVAL_SPLITS=val,test" in output
    assert "RUN_TAU_EVAL=auto" in output
    assert "FORCE_STAGE=true" in output
    assert "map_selector_s3_weighted_set_cover_top5" in output
    assert "map_selector_s4_minimal_evidence_group_top5" in output
    assert "map_selector_s5_fixed_budget_marginal_greedy_top5" in output
    assert output.count("SELECTOR_ADAPTIVE_POLICY=fixed_top5") >= 2
    assert "SELECTOR_ADAPTIVE_POLICY=fixed_budget_marginal_greedy" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s3_weighted_set_cover_top5" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s4_minimal_evidence_group_top5" in output
    assert "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s5_fixed_budget_marginal_greedy_top5" in output
    assert "liar_raw__ministral3_8b__map_selector_s3_weighted_set_cover_top5_plain" in output
    assert "liar_raw__ministral3_8b__map_selector_s4_minimal_evidence_group_top5_plain" in output
    assert "liar_raw__ministral3_8b__map_selector_s5_fixed_budget_marginal_greedy_top5_plain" in output
    assert "TRACE_PROMPT_STYLE=qec_min" not in output
    assert "TRACE_PROMPT_STYLE=qec_map" not in output
    assert "adaptive5_10" not in output
    assert "rawfc__ministral3_8b" not in output
