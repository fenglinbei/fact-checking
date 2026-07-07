#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROJECT_PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_PYTHONPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${PROJECT_PYTHONPATH}"
fi

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
export MODE="${MODE:-full}"
export DRY_RUN="${DRY_RUN:-false}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
export EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
export CHECKPOINTS="${CHECKPOINTS:-best}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export DATASETS="liar_raw"
export MODELS="ministral3_8b"
export TRACE_PROMPT_STYLE="plain"
export EVIDENCE_TEXT_MODE="full"
export SELECTOR_NAME="selector_mech_s4_atom_union_source_score_ordered"
export EXPECTED_SELECTOR_NAME="$SELECTOR_NAME"
export SELECTOR_GRAPH_VERSION="selector_mechanism_ablation_v0"
export SELECTOR_ADAPTIVE_POLICY="source_score_ordered"
export ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"
export ALLOW_EMPTY_CANDIDATE_POOL="false"
export ALLOW_EMPTY_EVIDENCE="false"
export FORCE_STAGE="${FORCE_STAGE:-true}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"

export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export FORCE_LORA_CONFIG="${FORCE_LORA_CONFIG:-true}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export SFT_EARLY_STOPPING_METRIC="${SFT_EARLY_STOPPING_METRIC:-macro_f1}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-chunking-ablation}"

CHUNKING_CASES="${CHUNKING_CASES:-abc sentence sentwin1 semantic07 report}"
POLICIES="${POLICIES:-top5 budget}"
SPLITS="${SPLITS:-train val test}"
SELECTOR_ROOT="${SELECTOR_ROOT:-outputs/selectors/atom_union_s4_chunking_ablation}"
SOURCE_BASE_ROOT="${SOURCE_BASE_ROOT:-outputs/selectors/selector_mechanism_ablation_chunking}"
EXISTING_ABC_ATOM_ROOT="${EXISTING_ABC_ATOM_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
REUSE_ABC_ATOM_ANCHOR="${REUSE_ABC_ATOM_ANCHOR:-true}"

FORCE_CACHE_BUILD="${FORCE_CACHE_BUILD:-false}"
FORCE_CLAIM_ATOMS="${FORCE_CLAIM_ATOMS:-false}"
FORCE_ATOM_RETRIEVAL="${FORCE_ATOM_RETRIEVAL:-false}"
FORCE_ATOM_UNION="${FORCE_ATOM_UNION:-false}"
FORCE_ORDERED_TRACE="${FORCE_ORDERED_TRACE:-false}"
FORCE_BUDGET_CALIBRATION="${FORCE_BUDGET_CALIBRATION:-false}"
FORCE_BUILD="${FORCE_BUILD:-auto}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"

EMBEDDER_MODEL="${EMBEDDER_MODEL:-/data/models/bge-base-en-v1.5}"
EMBEDDER_DEVICE="${EMBEDDER_DEVICE:-cuda}"
EMBEDDER_MAX_LENGTH="${EMBEDDER_MAX_LENGTH:-256}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-64}"
EMBEDDER_PRECISION="${EMBEDDER_PRECISION:-fp32}"
PER_ATOM_KEEP="${PER_ATOM_KEEP:-20}"
MERGED_POOL_SIZE="${MERGED_POOL_SIZE:-20}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
BASELINE_TOP_K="${BASELINE_TOP_K:-$SELECTOR_TOP_K}"

ATOM_BASE_URL="${ATOM_BASE_URL:-https://api.deepseek.com}"
ATOM_MODEL="${ATOM_MODEL:-deepseek-v4-flash}"
ATOM_API_KEY_ENV="${ATOM_API_KEY_ENV:-DEEPSEEK_API_KEY}"
ATOM_API_CONCURRENCY="${ATOM_API_CONCURRENCY:-128}"
ATOM_MAX_TOKENS="${ATOM_MAX_TOKENS:-2048}"
ATOM_THINKING_TYPE="${ATOM_THINKING_TYPE:-disabled}"
MOCK_ATOMS="${MOCK_ATOMS:-false}"

BUDGET_MIN_COUNT="${BUDGET_MIN_COUNT:-1}"
BUDGET_MAX_COUNT="${BUDGET_MAX_COUNT:-20}"
BUDGET_MIN="${BUDGET_MIN:-1}"
BUDGET_MAX="${BUDGET_MAX:-4096}"
BUDGET_LOCAL_WINDOW="${BUDGET_LOCAL_WINDOW:-32}"
BUDGET_DRY_RUN_DEFAULT="${BUDGET_DRY_RUN_DEFAULT:-541}"
REFERENCE_TOP5_BUILD_REPORT="${REFERENCE_TOP5_BUILD_REPORT:-${OUTPUT_ROOT}/liar_raw__ministral3_8b__chunk_abc_s4_union_top5_plain/build/build_report.json}"
FALLBACK_REFERENCE_TOP5_BUILD_REPORT="${FALLBACK_REFERENCE_TOP5_BUILD_REPORT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__selector_mech_s4_atom_union_source_score_top5_plain/build/build_report.json}"
SUMMARY_ANALYSIS_DIR="${SUMMARY_ANALYSIS_DIR:-outputs/analysis/chunking_ablation_atom_union_s4}"
SKIP_SUMMARY="${SKIP_SUMMARY:-false}"
RUN_TAU_EVAL="${RUN_TAU_EVAL:-false}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

visible_cuda_device_count() {
  local devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$devices" || "$devices" == "NoDevFiles" ]]; then
    printf '\n'
    return 0
  fi
  local raw_device device count=0
  IFS=',' read -r -a cuda_device_array <<< "$devices"
  for raw_device in "${cuda_device_array[@]}"; do
    device="${raw_device// /}"
    [[ -z "$device" ]] && continue
    count=$((count + 1))
  done
  printf '%s\n' "$count"
}

check_distributed_device_request() {
  case "$MODE" in
    train|eval|full) ;;
    *) return 0 ;;
  esac
  if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
    printf 'NPROC_PER_NODE must be a positive integer: %s\n' "$NPROC_PER_NODE" >&2
    exit 2
  fi

  local visible_count
  visible_count="$(visible_cuda_device_count)"
  if [[ -n "$visible_count" && "$visible_count" -gt 0 && "$NPROC_PER_NODE" -gt "$visible_count" ]]; then
    printf 'Requested NPROC_PER_NODE=%s but CUDA_VISIBLE_DEVICES exposes only %s device(s): %s\n' \
      "$NPROC_PER_NODE" "$visible_count" "${CUDA_VISIBLE_DEVICES:-}" >&2
    printf 'Set NPROC_PER_NODE<=%s, expand CUDA_VISIBLE_DEVICES, or request a matching GPU allocation before launching training.\n' \
      "$visible_count" >&2
    exit 2
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

case_items() {
  printf '%s\n' "${1//,/ }"
}

experiment_for_case() {
  case "$1" in
    abc) printf '%s\n' "v0.7/v0_7_liar_raw_atom_facts_abc_chunking" ;;
    sentence) printf '%s\n' "v0.7/v0_7_liar_raw_atom_facts_sentence_chunking" ;;
    sentwin1) printf '%s\n' "v0.7/v0_7_liar_raw_atom_facts_sentwin1_chunking" ;;
    semantic07) printf '%s\n' "v0.7/v0_7_liar_raw_atom_facts_semantic07_chunking" ;;
    report) printf '%s\n' "v0.7/v0_7_liar_raw_atom_facts_report_chunking" ;;
    *) printf 'Unsupported chunking case: %s\n' "$1" >&2; exit 2 ;;
  esac
}

config_for_case() {
  case "$1" in
    abc) printf '%s\n' "configs/experiment/v0.7/v0_7_liar_raw_atom_facts_abc_chunking.yaml" ;;
    sentence) printf '%s\n' "configs/experiment/v0.7/v0_7_liar_raw_atom_facts_sentence_chunking.yaml" ;;
    sentwin1) printf '%s\n' "configs/experiment/v0.7/v0_7_liar_raw_atom_facts_sentwin1_chunking.yaml" ;;
    semantic07) printf '%s\n' "configs/experiment/v0.7/v0_7_liar_raw_atom_facts_semantic07_chunking.yaml" ;;
    report) printf '%s\n' "configs/experiment/v0.7/v0_7_liar_raw_atom_facts_report_chunking.yaml" ;;
    *) printf 'Unsupported chunking case: %s\n' "$1" >&2; exit 2 ;;
  esac
}

raw_path_for_split() {
  case "$1" in
    train) printf '%s\n' "${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}" ;;
    val) printf '%s\n' "${VAL_RAW:-data/raw/LIAR-RAW/val.json}" ;;
    test) printf '%s\n' "${TEST_RAW:-data/raw/LIAR-RAW/test.json}" ;;
    *) printf 'Unsupported split: %s\n' "$1" >&2; exit 2 ;;
  esac
}

atom_root_for_case() {
  local case_name="$1"
  if [[ "$case_name" == "abc" && "$REUSE_ABC_ATOM_ANCHOR" == "true" ]]; then
    printf '%s\n' "$EXISTING_ABC_ATOM_ROOT"
  else
    printf '%s\n' "${SELECTOR_ROOT}/liar_raw_${case_name}"
  fi
}

claim_atom_root() {
  if [[ -n "${CLAIM_ATOM_ROOT:-}" ]]; then
    printf '%s\n' "$CLAIM_ATOM_ROOT"
    return 0
  fi
  local existing="${EXISTING_ABC_ATOM_ROOT}/01_claim_atoms"
  if [[ "$DRY_RUN" == "true" || -s "${existing}/claim_atoms_train.jsonl" ]]; then
    printf '%s\n' "$existing"
  else
    printf '%s\n' "${SELECTOR_ROOT}/liar_raw_claim_atoms"
  fi
}

fingerprint_for_case() {
  local case_name="$1"
  local config="$2"
  local env_name="CHUNK_MMR_FINGERPRINT_${case_name^^}"
  env_name="${env_name//[^A-Z0-9_]/_}"
  local override="${!env_name:-}"
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
    return 0
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    printf 'dry-run-%s\n' "$case_name"
    return 0
  fi
  local fp_args=(--config "$config")
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    fp_args+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  PYTHONPATH="${PROJECT_PYTHONPATH}:${PYTHONPATH:-}" "$PYTHON_BIN" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${fp_args[@]}"
}

all_cache_files_exist() {
  local fingerprint="$1"
  local split
  for split in ${SPLITS}; do
    [[ -f "outputs/cache/chunk_mmr/${fingerprint}/${split}.pkl" ]] || return 1
  done
  return 0
}

ensure_chunk_cache() {
  local case_name="$1"
  local experiment="$2"
  local fingerprint="$3"
  if all_cache_files_exist "$fingerprint" && [[ "$FORCE_CACHE_BUILD" != "true" ]]; then
    printf '[atom-union-s4-chunking] reuse chunk cache case=%s fingerprint=%s\n' "$case_name" "$fingerprint"
    return 0
  fi
  local pipeline_args=("experiment=${experiment}" "pipeline.mode=build" "pipeline.force.build=${FORCE_CACHE_BUILD}")
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    pipeline_args+=("+build.data.sample_limit=${SAMPLE_LIMIT}")
  fi
  run_cmd env PYTHONPATH="${PROJECT_PYTHONPATH}:${PYTHONPATH:-}" "$PYTHON_BIN" -m fact_checking.pipeline.run "${pipeline_args[@]}"
}

sample_args() {
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    printf '%s\n' "--sample-limit" "$SAMPLE_LIMIT"
  fi
}

ensure_atom_union_for_split() {
  local case_name="$1"
  local fingerprint="$2"
  local atom_root="$3"
  local split="$4"
  local claim_root="$5"
  local raw_path
  raw_path="$(raw_path_for_split "$split")"
  local chunk_cache="outputs/cache/chunk_mmr/${fingerprint}/${split}.pkl"
  local retrieval_root="${atom_root}/02_atom_retrieval"
  local union_root="${atom_root}/03_atom_union"
  local claim_atoms="${claim_root}/claim_atoms_${split}.jsonl"
  local retrieval_trace="${retrieval_root}/retrieval_trace_${split}.jsonl"
  local atom_union_pool="${union_root}/atom_union_candidate_pool_${split}.jsonl"
  local extra_sample_args=()
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    extra_sample_args=(--sample-limit "$SAMPLE_LIMIT")
  fi

  if [[ -s "$claim_atoms" && "$FORCE_CLAIM_ATOMS" != "true" ]]; then
    printf '[atom-union-s4-chunking] reuse claim atoms: %s\n' "$claim_atoms"
  else
    local mock_args=()
    if [[ "$MOCK_ATOMS" == "true" || "$MOCK_ATOMS" == "1" ]]; then
      mock_args=(--mock-atoms)
    fi
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/generate_claim_atom_cache.py \
      --input-mode raw_split \
      --dataset liar_raw \
      --label-schema liar6 \
      --raw-path "$raw_path" \
      --output-dir "$claim_root" \
      --split "$split" \
      --atom-cache-dir "${claim_root}/cache" \
      --atom-base-url "$ATOM_BASE_URL" \
      --atom-model "$ATOM_MODEL" \
      --atom-api-key-env "$ATOM_API_KEY_ENV" \
      --api-concurrency "$ATOM_API_CONCURRENCY" \
      --max-tokens "$ATOM_MAX_TOKENS" \
      --thinking-type "$ATOM_THINKING_TYPE" \
      --no-progress \
      "${mock_args[@]}" \
      "${extra_sample_args[@]}"
  fi

  if [[ -s "$retrieval_trace" && "$FORCE_ATOM_RETRIEVAL" != "true" ]]; then
    printf '[atom-union-s4-chunking] reuse atom retrieval: %s\n' "$retrieval_trace"
  else
    require_path "$claim_atoms" "${split} claim atoms"
    require_path "$chunk_cache" "${split} chunk cache"
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
      --claim-atoms-jsonl "$claim_atoms" \
      --chunk-cache-path "$chunk_cache" \
      --split "$split" \
      --output-dir "$retrieval_root" \
      --embedder-model "$EMBEDDER_MODEL" \
      --device "$EMBEDDER_DEVICE" \
      --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
      --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
      --precision "$EMBEDDER_PRECISION" \
      --per-atom-keep "$PER_ATOM_KEEP" \
      --merged-pool-size "$MERGED_POOL_SIZE" \
      --selector-top-k "$SELECTOR_TOP_K" \
      --baseline-top-k "$BASELINE_TOP_K" \
      --oracle-results "" \
      --no-progress \
      "${extra_sample_args[@]}"
  fi

  if [[ -s "$atom_union_pool" && "$FORCE_ATOM_UNION" != "true" ]]; then
    printf '[atom-union-s4-chunking] reuse atom union: %s\n' "$atom_union_pool"
  else
    require_path "${retrieval_root}/baseline_claim_mmr_selected_${split}.jsonl" "${split} baseline claim-MMR selected rows"
    require_path "${retrieval_root}/merged_candidate_pool_${split}.jsonl" "${split} atom merged candidate pool"
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_retrieval_union.py \
      --baseline-jsonl "${retrieval_root}/baseline_claim_mmr_selected_${split}.jsonl" \
      --atom-pool-jsonl "${retrieval_root}/merged_candidate_pool_${split}.jsonl" \
      --split "$split" \
      --output-dir "$union_root" \
      --selector-top-k "$SELECTOR_TOP_K" \
      --oracle-results "" \
      "${extra_sample_args[@]}"
  fi
}

ensure_ordered_trace_for_split() {
  local case_name="$1"
  local fingerprint="$2"
  local atom_root="$3"
  local split="$4"
  local output_dir="${SOURCE_BASE_ROOT}/liar_raw_${case_name}_${SELECTOR_NAME}_${split}"
  local trace_path="${output_dir}/selection_trace_${split}.jsonl"
  if [[ -s "$trace_path" && "$FORCE_ORDERED_TRACE" != "true" ]]; then
    printf '[atom-union-s4-chunking] reuse ordered trace: %s\n' "$trace_path"
    return 0
  fi
  require_path "outputs/cache/chunk_mmr/${fingerprint}/${split}.pkl" "${split} chunk cache"
  require_path "${atom_root}/03_atom_union/atom_union_candidate_pool_${split}.jsonl" "${split} atom union pool"
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_selector_mechanism_ablation_traces.py \
    --chunk-cache-path "outputs/cache/chunk_mmr/${fingerprint}/${split}.pkl" \
    --atom-union-jsonl "${atom_root}/03_atom_union/atom_union_candidate_pool_${split}.jsonl" \
    --output-dir "$output_dir" \
    --split "$split" \
    --sample-limit "$SAMPLE_LIMIT" \
    --selector-name "$SELECTOR_NAME" \
    --top-k "$SELECTOR_TOP_K" \
    --claim-pool-top-n "$MERGED_POOL_SIZE" \
    --random-seed 0 \
    --merge-mmr-lambda 0.70 \
    --chunk-mmr-fingerprint "$fingerprint"
}

target_prompt_mean() {
  local path="$REFERENCE_TOP5_BUILD_REPORT"
  if [[ ! -f "$path" && -f "$FALLBACK_REFERENCE_TOP5_BUILD_REPORT" ]]; then
    path="$FALLBACK_REFERENCE_TOP5_BUILD_REPORT"
  fi
  if [[ "$DRY_RUN" == "true" && ! -f "$path" ]]; then
    printf '%s\n' "$BUDGET_DRY_RUN_DEFAULT"
    return 0
  fi
  if [[ ! -f "$path" ]]; then
    printf 'Missing ABC top5 build report for budget calibration: %s\n' "$REFERENCE_TOP5_BUILD_REPORT" >&2
    exit 2
  fi
  "$PYTHON_BIN" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(float(data["splits"]["train"]["prompt_token_count"]["mean"]))' "$path"
}

calibrated_budget_for_case() {
  local case_name="$1"
  local config="$2"
  local fingerprint="$3"
  local calibration_file="${OUTPUT_ROOT}/_calibration/chunk_${case_name}_s4_union_budget_promptmatched.json"
  local selected_budget
  if [[ -f "$calibration_file" && "$FORCE_BUDGET_CALIBRATION" != "true" ]]; then
    selected_budget="$("$PYTHON_BIN" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(int(json.load(f)["selected_budget"]))' "$calibration_file")"
    printf '[atom-union-s4-chunking] reuse budget calibration case=%s budget=%s file=%s\n' \
      "$case_name" "$selected_budget" "$calibration_file" >&2
    printf '%s\n' "$selected_budget"
    return 0
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[atom-union-s4-chunking] dry-run budget calibration case=%s budget=%s\n' \
      "$case_name" "$BUDGET_DRY_RUN_DEFAULT" >&2
    printf '%s\n' "$BUDGET_DRY_RUN_DEFAULT"
    return 0
  fi
  local target
  target="$(target_prompt_mean)"
  printf '[atom-union-s4-chunking] calibrate budget case=%s target_prompt_mean=%s output=%s\n' \
    "$case_name" "$target" "$calibration_file" >&2
  run_cmd env PYTHONPATH="${PROJECT_PYTHONPATH}:${PYTHONPATH:-}" "$PYTHON_BIN" scripts/sentence_trace_method/calibrate_prompt_evidence_budget.py \
    --config scripts/sentence_trace_method/configs/liar_raw__ministral3_8b.yaml \
    --trace "${SOURCE_BASE_ROOT}/liar_raw_${case_name}_${SELECTOR_NAME}_train/selection_trace_train.jsonl" \
    --raw "${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}" \
    --dataset liar_raw \
    --label-schema liar6 \
    --target-prompt-mean "$target" \
    --output "$calibration_file" \
    --prompt-model-name-or-path /data/models/Ministral-3-8B-Instruct-2512 \
    --trace-prompt-style plain \
    --evidence-text-mode full \
    --expected-selector-name "$SELECTOR_NAME" \
    --expected-chunk-mmr-fingerprint "$fingerprint" \
    --min-count "$BUDGET_MIN_COUNT" \
    --max-count "$BUDGET_MAX_COUNT" \
    --budget-min "$BUDGET_MIN" \
    --budget-max "$BUDGET_MAX" \
    --local-window "$BUDGET_LOCAL_WINDOW" \
    --sample-limit "$SAMPLE_LIMIT" \
    --no-progress
  selected_budget="$("$PYTHON_BIN" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(int(json.load(f)["selected_budget"]))' "$calibration_file")"
  printf '[atom-union-s4-chunking] calibrated budget case=%s budget=%s file=%s\n' \
    "$case_name" "$selected_budget" "$calibration_file" >&2
  printf '%s\n' "$selected_budget"
}

run_policy_case() {
  local case_name="$1"
  local policy="$2"
  local fingerprint="$3"
  local config="$4"
  local budget=""
  export SOURCE_ROOT="${SOURCE_BASE_ROOT}/liar_raw_${case_name}_${SELECTOR_NAME}"
  export EXPECTED_CHUNK_MMR_FINGERPRINT="$fingerprint"
  export TRACE_TOP_K="$BUDGET_MAX_COUNT"
  export PROMPT_EVIDENCE_MAX_LENGTH_GUARD="${PROMPT_EVIDENCE_MAX_LENGTH_GUARD:-warn}"
  export RUN_TAU_EVAL="$RUN_TAU_EVAL"
  export FORCE_BUILD="$FORCE_BUILD"
  export FORCE_TRAIN="$FORCE_TRAIN"
  export FORCE_EVAL="$FORCE_EVAL"

  case "$policy" in
    top5|fixed_top5)
      export CASE_SUFFIX="__chunk_${case_name}_s4_union_top5_plain"
      export PROMPT_EVIDENCE_POLICY="fixed_topk"
      export PROMPT_EVIDENCE_MIN_COUNT="5"
      export PROMPT_EVIDENCE_MAX_COUNT="5"
      export PROMPT_EVIDENCE_TOKEN_BUDGET=""
      export TRACE_TOP_K="5"
      ;;
    budget|budget_promptmatched)
      budget="$(calibrated_budget_for_case "$case_name" "$config" "$fingerprint")"
      export CASE_SUFFIX="__chunk_${case_name}_s4_union_budget_promptmatched_plain"
      export PROMPT_EVIDENCE_POLICY="budget"
      export PROMPT_EVIDENCE_MIN_COUNT="$BUDGET_MIN_COUNT"
      export PROMPT_EVIDENCE_MAX_COUNT="$BUDGET_MAX_COUNT"
      export PROMPT_EVIDENCE_TOKEN_BUDGET="$budget"
      export TRACE_TOP_K="$BUDGET_MAX_COUNT"
      ;;
    *) printf 'Unsupported policy: %s\n' "$policy" >&2; exit 2 ;;
  esac

  printf '\n[atom-union-s4-chunking] CHUNKING_CASE=%s POLICY=%s CASE_SUFFIX=%s SOURCE_ROOT=%s EXPECTED_CHUNK_MMR_FINGERPRINT=%s PROMPT_EVIDENCE_POLICY=%s PROMPT_EVIDENCE_MIN_COUNT=%s PROMPT_EVIDENCE_MAX_COUNT=%s PROMPT_EVIDENCE_TOKEN_BUDGET=%s TRACE_PROMPT_STYLE=%s EVIDENCE_TEXT_MODE=%s TRACE_TOP_K=%s MODE=%s\n' \
    "$case_name" "$policy" "$CASE_SUFFIX" "$SOURCE_ROOT" "$EXPECTED_CHUNK_MMR_FINGERPRINT" "$PROMPT_EVIDENCE_POLICY" "$PROMPT_EVIDENCE_MIN_COUNT" "$PROMPT_EVIDENCE_MAX_COUNT" "${PROMPT_EVIDENCE_TOKEN_BUDGET:-}" "$TRACE_PROMPT_STYLE" "$EVIDENCE_TEXT_MODE" "$TRACE_TOP_K" "$MODE"

  bash "${SCRIPT_DIR}/run_lora_matrix.sh"
}

prepare_case() {
  local case_name="$1"
  local config experiment fingerprint atom_root claim_root split
  config="$(config_for_case "$case_name")"
  experiment="$(experiment_for_case "$case_name")"
  fingerprint="$(fingerprint_for_case "$case_name" "$config")"
  atom_root="$(atom_root_for_case "$case_name")"
  claim_root="$(claim_atom_root)"

  printf '\n[atom-union-s4-chunking] case=%s config=%s experiment=%s fingerprint=%s atom_root=%s claim_atom_root=%s\n' \
    "$case_name" "$config" "$experiment" "$fingerprint" "$atom_root" "$claim_root"

  if [[ "$MODE" == "build" || "$MODE" == "full" ]]; then
    ensure_chunk_cache "$case_name" "$experiment" "$fingerprint"
    for split in ${SPLITS}; do
      ensure_atom_union_for_split "$case_name" "$fingerprint" "$atom_root" "$split" "$claim_root"
      ensure_ordered_trace_for_split "$case_name" "$fingerprint" "$atom_root" "$split"
    done
  fi

  local policy
  for policy in $(case_items "$POLICIES"); do
    run_policy_case "$case_name" "$policy" "$fingerprint" "$config"
  done
}

check_distributed_device_request

printf '[atom-union-s4-chunking] MODE=%s CHUNKING_CASES=%s POLICIES=%s EVAL_SPLITS=%s CHECKPOINTS=%s OUTPUT_ROOT=%s DRY_RUN=%s PYTHON_BIN=%s CUDA_VISIBLE_DEVICES=%s NPROC_PER_NODE=%s NUM_MACHINES=%s MIXED_PRECISION=%s NCCL_CUMEM_HOST_ENABLE=%s OMP_NUM_THREADS=%s\n' \
  "$MODE" "$CHUNKING_CASES" "$POLICIES" "$EVAL_SPLITS" "$CHECKPOINTS" "$OUTPUT_ROOT" "$DRY_RUN" "$PYTHON_BIN" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$NPROC_PER_NODE" "$NUM_MACHINES" "$MIXED_PRECISION" "$NCCL_CUMEM_HOST_ENABLE" "$OMP_NUM_THREADS"

for chunking_case in $(case_items "$CHUNKING_CASES"); do
  prepare_case "$chunking_case"
done

if [[ "$SKIP_SUMMARY" != "true" && "$SKIP_SUMMARY" != "1" ]]; then
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/summarize_atom_union_s4_chunking_ablation.py \
    --output-root "$OUTPUT_ROOT" \
    --analysis-dir "$SUMMARY_ANALYSIS_DIR" \
    --chunking-cases "$CHUNKING_CASES" \
    --policies "$POLICIES" \
    --splits "$EVAL_SPLITS" \
    --checkpoint "${CHECKPOINTS%%,*}" \
    --lora-suffix "$LORA_SUFFIX"
fi
