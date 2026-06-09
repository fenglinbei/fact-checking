#!/usr/bin/env bash

# Shared roots for the phase9 dense-only FullFT reruns.

PHASE9_FULLFT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE9_FULLFT_PROJECT_ROOT="$(cd "${PHASE9_FULLFT_SCRIPT_DIR}/../.." && pwd)"
cd "${PHASE9_FULLFT_PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin}"
export PATH="${CONDA_BIN}:${PATH}"
export PYTHONPATH="${PHASE9_FULLFT_PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PHASE_OUTPUT_ROOT="${PHASE_OUTPUT_ROOT:-outputs/selector_trace_verifier/phase9_dense_only_fullft}"
RAWFC_OUTPUT_ROOT="${RAWFC_OUTPUT_ROOT:-${PHASE_OUTPUT_ROOT}/rawfc_dense_v0_6c_eval25_backbone}"
LIAR_OUTPUT_ROOT="${LIAR_OUTPUT_ROOT:-${PHASE_OUTPUT_ROOT}/liar_raw_dense_v0_6c_backbone}"

PHASE_CONFIG_CACHE_ROOT="${PHASE_CONFIG_CACHE_ROOT:-outputs/cache/dense_only/phase9_dense_only_fullft}"
RAWFC_CONFIG_CACHE_ROOT="${RAWFC_CONFIG_CACHE_ROOT:-${PHASE_CONFIG_CACHE_ROOT}/rawfc_backbone_configs}"
LIAR_CONFIG_CACHE_ROOT="${LIAR_CONFIG_CACHE_ROOT:-${PHASE_CONFIG_CACHE_ROOT}/liar_raw_backbone_configs}"

PHASE_RUN_ROOT="${PHASE_RUN_ROOT:-outputs/runs/phase9_dense_only_fullft}"
RAWFC_RUN_ROOT="${RAWFC_RUN_ROOT:-${PHASE_RUN_ROOT}/rawfc_dense_v0_6c_eval25_backbone}"
LIAR_RUN_ROOT="${LIAR_RUN_ROOT:-${PHASE_RUN_ROOT}/liar_raw_dense_v0_6c_backbone}"

SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-true}"
