from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_evidence_map_atom_facts_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_map_selector_v0_7_atom_facts.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "outputs/selectors/evidence_map_selector/v0_7_atom_facts_${SPLIT}" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts}\"" in text
    assert "BUILD_VERIFIER_DATA=\"${BUILD_VERIFIER_DATA:-false}\"" in text
    assert "TEACHER_MODEL=\"${TEACHER_MODEL:-deepseek-v4-flash}\"" in text
    assert "CONCURRENCY=\"${CONCURRENCY:-128}\"" in text
    assert "THINKING_TYPE=\"${THINKING_TYPE:-disabled}\"" in text


def test_evidence_map_atom_facts_val_teacher_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_map_selector_v0_7_atom_facts_val_teacher.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "source \"${REPO_ROOT}/.env\"" in text
    assert "RUN_TEACHER=\"${RUN_TEACHER:-true}\"" in text
    assert "BUILD_VERIFIER_DATA=\"${BUILD_VERIFIER_DATA:-false}\"" in text
    assert "CONCURRENCY=\"${CONCURRENCY:-128}\"" in text
    assert "THINKING_TYPE=\"${THINKING_TYPE:-disabled}\"" in text
    assert "run_evidence_map_selector_v0_7_atom_facts.sh" in text


def test_evidence_chain_atom_facts_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7_atom_facts.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "v0_7_atom_facts_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl" in text
    assert "v0_7_atom_facts_budgeted_marginal_adaptive5_10_${SPLIT}" in text
    assert "MIN_TOP_K=\"${MIN_TOP_K:-5}\"" in text
    assert "MAX_TOP_K=\"${MAX_TOP_K:-10}\"" in text


def test_rawfc_atom_facts_all_splits_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_rawfc_v0_7_atom_facts_all_splits.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "TRAIN_RAW=\"${TRAIN_RAW:-data/raw/RAWFC/train.json}\"" in text
    assert "VAL_RAW=\"${VAL_RAW:-data/raw/RAWFC/val.json}\"" in text
    assert "TEST_RAW=\"${TEST_RAW:-data/raw/RAWFC/test.json}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0}\"" in text
    assert "EVIDENCE_MAP_ROOT=\"${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_7_atom_facts}\"" in text
    assert "GRAPH_ROOT=\"${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_budgeted_marginal_adaptive5_10}\"" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts}\"" in text
    assert "RUN_TEACHER=\"${RUN_TEACHER:-true}\"" in text
    assert "CONCURRENCY=\"${CONCURRENCY:-128}\"" in text
    assert "THINKING_TYPE=\"${THINKING_TYPE:-disabled}\"" in text
    assert "run_evidence_map_selector_v0_7_atom_facts.sh" in text
    assert "run_evidence_chain_graph_v0_7_atom_facts.sh" in text


def test_rawfc_atom_facts_abc_config_inherits_rawfc_v0_6c_and_only_switches_chunking() -> None:
    path = ROOT / "configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml"

    text = path.read_text(encoding="utf-8")

    assert "- v0_6c_rawfc3_rule_step_adaptive5_10" in text
    assert "name: v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking" in text
    assert "variant: v0_7_atom_facts_abc" in text
    assert "chunking_strategy: abc_claim_aware" in text
    assert "strategy: abc_claim_aware" in text
    assert "label_schema: rawfc3" not in text


def test_rawfc_atom_facts_abc_tight_config_inherits_rawfc_v0_6c_and_uses_tight_strategy() -> None:
    path = ROOT / "configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_tight_chunking.yaml"

    text = path.read_text(encoding="utf-8")

    assert "- v0_6c_rawfc3_rule_step_adaptive5_10" in text
    assert "name: v0_6c_rawfc3_rule_step_adaptive5_10_abc_tight_chunking" in text
    assert "variant: v0_7_atom_facts_abc_tight" in text
    assert "chunking_strategy: abc_claim_aware_rawfc_tight" in text
    assert "strategy: abc_claim_aware_rawfc_tight" in text
    assert "max_sent_per_chunk: 2" in text
    assert "lambda_std: 0.35" in text
    assert "w_sem: 0.65" in text
    assert "w_rel: 0.35" in text


def test_liar_raw_atom_facts_abc_config_inherits_liar_abc_chunking_lineage() -> None:
    path = ROOT / "configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml"

    text = path.read_text(encoding="utf-8")

    assert "- b3_abc_claim_aware" in text
    assert "name: v0_7_liar_raw_atom_facts_abc_chunking" in text
    assert "variant: v0_7_atom_facts_abc" in text
    assert "experiment_name: v0_7_liar_raw_atom_facts_abc" in text
    assert "label_schema: rawfc3" not in text


def test_evidence_map_atom_facts_abc_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_map_selector_v0_7_atom_facts_abc.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "outputs/selectors/evidence_map_selector/v0_7_atom_facts_abc_${SPLIT}" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}\"" in text
    assert "CHILD_SAMPLE_LIMIT=\"\"" in text
    assert "SAMPLE_LIMIT=\"${CHILD_SAMPLE_LIMIT}\"" in text
    assert "run_evidence_map_selector_v0_7_atom_facts.sh" in text


def test_evidence_chain_atom_facts_abc_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7_atom_facts_abc.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}\"" in text
    assert "outputs/selectors/evidence_map_selector/v0_7_atom_facts_abc_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl" in text
    assert "outputs/selectors/evidence_chain_graph/v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10_${SPLIT}" in text
    assert "print_chunk_mmr_fingerprint.py" in text
    assert "run_evidence_chain_graph_v0_7_atom_facts.sh" in text


def test_rawfc_atom_facts_abc_qd_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_rawfc_v0_7_atom_facts_abc_qd.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0_abc}\"" in text
    assert "QUESTION_API_KEY_ENV=\"${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}\"" in text
    assert "print_chunk_mmr_fingerprint.py" in text
    assert "outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl" in text
    assert "generate_question_decomp_cache.py" in text
    assert "build_question_decomp_retrieval.py" in text
    assert "build_question_decomp_union.py" in text


def test_rawfc_atom_facts_abc_all_splits_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_rawfc_v0_7_atom_facts_abc_all_splits.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0_abc}\"" in text
    assert "EVIDENCE_MAP_ROOT=\"${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_7_atom_facts_abc}\"" in text
    assert "GRAPH_ROOT=\"${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}\"" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}\"" in text
    assert "QUESTION_API_KEY_ENV=\"${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}\"" in text
    assert "CHILD_SAMPLE_LIMIT=\"\"" in text
    assert "SAMPLE_LIMIT=\"${CHILD_SAMPLE_LIMIT}\"" in text
    assert "run_rawfc_v0_7_atom_facts_abc_qd.sh" in text
    assert "run_evidence_map_selector_v0_7_atom_facts_abc.sh" in text
    assert "run_evidence_chain_graph_v0_7_atom_facts_abc.sh" in text


def test_rawfc_atom_facts_abc_tight_all_splits_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_rawfc_v0_7_atom_facts_abc_tight_all_splits.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_tight_chunking.yaml}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0_abc_tight}\"" in text
    assert "EVIDENCE_MAP_ROOT=\"${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_7_atom_facts_abc_tight}\"" in text
    assert "GRAPH_ROOT=\"${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_tight_budgeted_marginal_adaptive5_10}\"" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}\"" in text
    assert "SELECTOR_DIRECTNESS_WEIGHT=\"${SELECTOR_DIRECTNESS_WEIGHT:-0.30}\"" in text
    assert "SELECTOR_BACKGROUND_PENALTY=\"${SELECTOR_BACKGROUND_PENALTY:-0.30}\"" in text
    assert "OBJECTIVE_BACKGROUND_OR_IRRELEVANT=\"${OBJECTIVE_BACKGROUND_OR_IRRELEVANT:-0.24}\"" in text
    assert "OBJECTIVE_LENGTH=\"${OBJECTIVE_LENGTH:-0.08}\"" in text
    assert "DRY_RUN=\"${DRY_RUN:-false}\"" in text
    assert "run_rawfc_v0_7_atom_facts_abc_qd.sh" in text
    assert "run_evidence_map_selector_v0_7_atom_facts_abc.sh" in text
    assert "run_evidence_chain_graph_v0_7_atom_facts_abc.sh" in text


def test_atom_facts_evidence_map_wrapper_forwards_selector_weight_overrides() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_map_selector_v0_7_atom_facts.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SELECTOR_DIRECTNESS_WEIGHT" in text
    assert "--selector-directness-weight" in text
    assert "SELECTOR_BACKGROUND_PENALTY" in text
    assert "--selector-background-penalty" in text


def test_atom_facts_chain_graph_wrapper_forwards_objective_weight_overrides() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7_atom_facts.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "OBJECTIVE_BACKGROUND_OR_IRRELEVANT" in text
    assert "--objective-background-or-irrelevant" in text
    assert "OBJECTIVE_LENGTH" in text
    assert "--objective-length" in text


def test_liar_raw_atom_facts_abc_build_cache_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_v0_7_atom_facts_abc_build_cache.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "EXPERIMENT=\"${EXPERIMENT:-v0_7_liar_raw_atom_facts_abc_chunking}\"" in text
    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}\"" in text
    assert "print_chunk_mmr_fingerprint.py" in text
    assert "\"experiment=${EXPERIMENT}\"" in text
    assert "\"pipeline.mode=build\"" in text
    assert "\"+build.data.sample_limit=${SAMPLE_LIMIT}\"" in text
    assert "outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/train.pkl" in text


def test_liar_raw_atom_facts_abc_qd_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_v0_7_atom_facts_abc_qd.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}\"" in text
    assert "TRAIN_RAW=\"${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}\"" in text
    assert "VAL_RAW=\"${VAL_RAW:-data/raw/LIAR-RAW/val.json}\"" in text
    assert "TEST_RAW=\"${TEST_RAW:-data/raw/LIAR-RAW/test.json}\"" in text
    assert "RAW_DATASET=\"${RAW_DATASET:-liar_raw}\"" in text
    assert "LABEL_SCHEMA=\"${LABEL_SCHEMA:-liar6}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/liar_raw_deepseek_v0_abc}\"" in text
    assert "QUESTION_API_KEY_ENV=\"${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}\"" in text
    assert "run_liar_raw_v0_7_atom_facts_abc_build_cache.sh" in text
    assert "outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl" in text
    assert "--input-mode raw_split" in text
    assert "--dataset \"${RAW_DATASET}\"" in text
    assert "--label-schema \"${LABEL_SCHEMA}\"" in text
    assert "build_question_decomp_retrieval.py" in text
    assert "build_question_decomp_union.py" in text


def test_liar_raw_atom_facts_abc_stage_sources_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_v0_7_atom_facts_abc_stage_sources.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}\"" in text
    assert "SOURCE_ROOT=\"${SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}\"" in text
    assert "SELECTOR_NAME=\"${SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}\"" in text
    assert "print_chunk_mmr_fingerprint.py" in text
    assert "stage_sources.py" in text
    assert "--dataset liar_raw" in text
    assert "--expected-fingerprint \"${CHUNK_MMR_FINGERPRINT}\"" in text
    assert "--allow-multi-sentence-candidates" in text


def test_liar_raw_atom_facts_abc_all_splits_wrapper_defaults() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_v0_7_atom_facts_abc_all_splits.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "CONFIG=\"${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}\"" in text
    assert "TRAIN_RAW=\"${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}\"" in text
    assert "QUESTION_OUTPUT_ROOT=\"${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/liar_raw_deepseek_v0_abc}\"" in text
    assert "EVIDENCE_MAP_ROOT=\"${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/liar_raw_v0_7_atom_facts_abc}\"" in text
    assert "GRAPH_ROOT=\"${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}\"" in text
    assert "PROMPT_VERSION=\"${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}\"" in text
    assert "QUESTION_API_KEY_ENV=\"${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}\"" in text
    assert "RUN_STAGE_SOURCES=\"${RUN_STAGE_SOURCES:-true}\"" in text
    assert "RAW_DATASET=liar_raw" in text
    assert "LABEL_SCHEMA=liar6" in text
    assert "run_liar_raw_v0_7_atom_facts_abc_qd.sh" in text
    assert "run_evidence_map_selector_v0_7_atom_facts_abc.sh" in text
    assert "run_evidence_chain_graph_v0_7_atom_facts_abc.sh" in text
    assert "run_liar_raw_v0_7_atom_facts_abc_stage_sources.sh" in text


def test_liar_raw_map_selector_ablation_s0_s2_wrapper_defaults_and_dry_run() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_map_selector_ablation_s0_s2.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "SELECTORS=\"${SELECTORS:-map_selector_s0_retrieval_top5 map_selector_s1_mmr_pool_top5 map_selector_s2_map_quality_top5}\"" in text
    assert "outputs/selectors/evidence_map_selector/liar_raw_v0_7_atom_facts_abc_${split}/candidate_evidence_map_features_${split}.jsonl" in text
    assert "outputs/selectors/evidence_chain_graph/liar_raw_${selector}_${split}" in text
    assert "build_map_selector_ablation_traces.py" in text
    assert "--top-k \"${TOP_K}\"" in text
    assert "--chunk-mmr-fingerprint \"${CHUNK_MMR_FINGERPRINT}\"" in text

    result = subprocess.run(
        ["bash", str(path)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "true"},
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "map_selector_s0_retrieval_top5" in output
    assert "map_selector_s1_mmr_pool_top5" in output
    assert "map_selector_s2_map_quality_top5" in output
    assert "qec_min" not in output
    assert "qec_map" not in output


def test_liar_raw_map_selector_ablation_s3_s5_wrapper_defaults_and_dry_run() -> None:
    path = ROOT / "scripts/phase5_selectors/run/run_liar_raw_map_selector_ablation_s3_s5.sh"

    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
    text = path.read_text(encoding="utf-8")

    assert "SPLITS=\"${SPLITS:-train val test}\"" in text
    assert "SELECTORS=\"${SELECTORS:-map_selector_s3_weighted_set_cover_top5 map_selector_s4_minimal_evidence_group_top5 map_selector_s5_fixed_budget_marginal_greedy_top5}\"" in text
    assert "outputs/selectors/evidence_map_selector/liar_raw_v0_7_atom_facts_abc_${split}/candidate_evidence_map_features_${split}.jsonl" in text
    assert "outputs/selectors/evidence_chain_graph/liar_raw_${selector}_${split}" in text
    assert "build_map_selector_ablation_traces.py" in text

    result = subprocess.run(
        ["bash", str(path)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "true"},
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "map_selector_s3_weighted_set_cover_top5" in output
    assert "map_selector_s4_minimal_evidence_group_top5" in output
    assert "map_selector_s5_fixed_budget_marginal_greedy_top5" in output
    assert "adaptive5_10" not in output
    assert "qec_min" not in output
    assert "qec_map" not in output
