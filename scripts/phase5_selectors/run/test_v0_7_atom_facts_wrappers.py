from __future__ import annotations

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
