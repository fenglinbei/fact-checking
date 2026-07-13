#!/usr/bin/env bash
# LIAR-RAW Atom-Union candidate-pool ablation.
#
# The four variants share claim atoms, chunks, selector weights, prompt policy,
# and the main-method verifier checkpoint. Only the candidate pool changes:
# baseline_only | atom_only | union_no_mmr | union_full.
#
# Typical usage:
#   DRY_RUN=true bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
#   SAMPLE_LIMIT=32 MOCK_EVIDENCE_MAPS=true bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
#   MODE=build bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
#   MODE=eval bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

LOAD_ENV="${LOAD_ENV:-true}"
if [[ "$LOAD_ENV" == "true" || "$LOAD_ENV" == "1" ]]; then
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
    set +a
  fi
fi

MODE="${MODE:-full}" # build|eval|summarize|full
POOL_MODES="${POOL_MODES:-baseline_only atom_only union_no_mmr union_full}"
SPLITS="${SPLITS:-train val test}"
EVAL_SPLITS="${EVAL_SPLITS:-val test}"
DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
FORCE_RETRIEVAL="${FORCE_RETRIEVAL:-false}"
FORCE_POOL_BUILD="${FORCE_POOL_BUILD:-false}"
FORCE_MAP_BUILD="${FORCE_MAP_BUILD:-false}"
FORCE_TRACE_BUILD="${FORCE_TRACE_BUILD:-false}"
FORCE_VERIFIER_BUILD="${FORCE_VERIFIER_BUILD:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
EMBEDDER_MODEL="${EMBEDDER_MODEL:-/data/models/bge-base-en-v1.5}"
EMBEDDER_DEVICE="${EMBEDDER_DEVICE:-cuda}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"
SOURCE_ATOM_ROOT="${SOURCE_ATOM_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
CLAIM_ATOM_ROOT="${CLAIM_ATOM_ROOT:-${SOURCE_ATOM_ROOT}/01_claim_atoms}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ -z "$RUN_SUFFIX" && "$SAMPLE_LIMIT" != "0" ]]; then
  RUN_SUFFIX="_sample${SAMPLE_LIMIT}"
  case "$MOCK_EVIDENCE_MAPS" in
    true|1|yes|on) RUN_SUFFIX="${RUN_SUFFIX}_mock" ;;
  esac
fi
ABLATION_ROOT="${ABLATION_ROOT:-outputs/selectors/atom_union_pool_ablation/liar_raw_abc_n20${RUN_SUFFIX}}"
RETRIEVAL_ROOT="${RETRIEVAL_ROOT:-${ABLATION_ROOT}/02_atom_retrieval}"
CASE_OUTPUT_ROOT="${CASE_OUTPUT_ROOT:-outputs/sentence_trace_method}"

POOL_SIZE="${POOL_SIZE:-20}"
PER_ATOM_KEEP="${PER_ATOM_KEEP:-20}"
BASELINE_TOP_K="${BASELINE_TOP_K:-20}"
ATOM_POOL_SIZE="${ATOM_POOL_SIZE:-20}"
UNION_MMR_LAMBDA="${UNION_MMR_LAMBDA:-0.70}"

MAIN_POLICY_CONFIG="${MAIN_POLICY_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_10.yaml}"
MAIN_WEIGHT_FILE="${MAIN_WEIGHT_FILE:-${SOURCE_ATOM_ROOT}/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json}"
MAIN_RUN_ROOT="${MAIN_RUN_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
SELECTOR_NAME="${SELECTOR_NAME:-mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"

RUN_TEACHER="${RUN_TEACHER:-true}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-128}"
TEACHER_REQUESTS_PER_MINUTE="${TEACHER_REQUESTS_PER_MINUTE:-2048}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
TEACHER_THINKING_TYPE="${TEACHER_THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-atom_evidence_map_v0_1}"

enabled() {
  case "${1:-false}" in true|1|yes|on) return 0 ;; *) return 1 ;; esac
}

run_cmd() {
  if enabled "$DRY_RUN"; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

progress_log() {
  if enabled "$SHOW_PROGRESS"; then
    printf '[atom-union-pool-ablation][%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
  fi
}

word_count() {
  local values="$1"
  local items=()
  # shellcheck disable=SC2206
  items=($values)
  printf '%d' "${#items[@]}"
}

BUILD_STAGE=0
BUILD_STAGE_TOTAL=0

run_build_stage() {
  local label="$1"
  shift
  BUILD_STAGE=$((BUILD_STAGE + 1))
  local started=$SECONDS
  progress_log "[stage ${BUILD_STAGE}/${BUILD_STAGE_TOTAL}] START ${label}"
  if enabled "$DRY_RUN"; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    progress_log "[stage ${BUILD_STAGE}/${BUILD_STAGE_TOTAL}] DRY-RUN ${label}"
  elif "$@"; then
    progress_log "[stage ${BUILD_STAGE}/${BUILD_STAGE_TOTAL}] DONE ${label} elapsed=$((SECONDS - started))s"
  else
    local status=$?
    progress_log "[stage ${BUILD_STAGE}/${BUILD_STAGE_TOTAL}] FAILED ${label} status=${status} elapsed=$((SECONDS - started))s"
    return "$status"
  fi
}

reuse_build_stage() {
  local label="$1"
  local artifact="$2"
  BUILD_STAGE=$((BUILD_STAGE + 1))
  progress_log "[stage ${BUILD_STAGE}/${BUILD_STAGE_TOTAL}] REUSE ${label} artifact=${artifact}"
}

require_path() {
  local path="$1"
  local label="$2"
  if enabled "$DRY_RUN"; then return 0; fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

sample_args=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  sample_args=(--sample-limit "$SAMPLE_LIMIT")
fi

should_build() { [[ "$MODE" == "build" || "$MODE" == "full" ]]; }
should_eval() { [[ "$MODE" == "eval" || "$MODE" == "full" ]]; }
should_summarize() { [[ "$MODE" == "summarize" || "$MODE" == "full" ]]; }

case "$MODE" in
  build|eval|summarize|full) ;;
  *) printf 'Unsupported MODE=%s. Use build, eval, summarize, or full.\n' "$MODE" >&2; exit 2 ;;
esac

printf '[atom-union-pool-ablation] MODE=%s POOL_MODES=[%s] SPLITS=[%s] POOL_SIZE=%s SAMPLE_LIMIT=%s DRY_RUN=%s\n' \
  "$MODE" "$POOL_MODES" "$SPLITS" "$POOL_SIZE" "$SAMPLE_LIMIT" "$DRY_RUN"
printf '[atom-union-pool-ablation] selector_weight=%s\n' "$MAIN_WEIGHT_FILE"
printf '[atom-union-pool-ablation] verifier_checkpoint=%s/train/best\n' "$MAIN_RUN_ROOT"

if should_build; then
  split_count="$(word_count "$SPLITS")"
  pool_mode_count="$(word_count "$POOL_MODES")"
  teacher_stage_count=0
  if enabled "$RUN_TEACHER"; then teacher_stage_count=1; fi
  BUILD_STAGE_TOTAL=$((split_count + pool_mode_count * (split_count * (4 + teacher_stage_count) + 1)))
  progress_log "build plan: stages=${BUILD_STAGE_TOTAL} variants=${pool_mode_count} splits=${split_count} teacher=${RUN_TEACHER}"
  require_path "$MAIN_WEIGHT_FILE" "main selector weight file"
  for split in $SPLITS; do
    claim_atoms="${CLAIM_ATOM_ROOT}/claim_atoms_${split}.jsonl"
    chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"
    retrieval_manifest="${RETRIEVAL_ROOT}/retrieval_manifest_${split}.json"
    if [[ -s "$retrieval_manifest" && "$FORCE_RETRIEVAL" != "true" ]]; then
      reuse_build_stage "retrieval split=${split}" "$retrieval_manifest"
    else
      require_path "$claim_atoms" "${split} claim atoms"
      require_path "$chunk_cache" "${split} chunk cache"
      run_build_stage "retrieval split=${split}" \
        "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
        --claim-atoms-jsonl "$claim_atoms" \
        --chunk-cache-path "$chunk_cache" \
        --output-dir "$RETRIEVAL_ROOT" \
        --split "$split" \
        --embedder-model "$EMBEDDER_MODEL" \
        --device "$EMBEDDER_DEVICE" \
        --precision fp32 \
        --per-atom-keep "$PER_ATOM_KEEP" \
        --merged-pool-size "$ATOM_POOL_SIZE" \
        --selector-top-k 5 \
        --baseline-top-k "$BASELINE_TOP_K" \
        --oracle-results "" \
        "${sample_args[@]}"
    fi
  done

  for pool_mode in $POOL_MODES; do
    variant_root="${ABLATION_ROOT}/${pool_mode}"
    union_root="${variant_root}/03_atom_union"
    map_root="${variant_root}/04_evidence_map"
    trace_root="${variant_root}/05_mrec_v0_2_learned_marginal_proxy_fullpool_minmax5_10"
    case_root="${CASE_OUTPUT_ROOT}/liar_raw__ministral3_8b__atom_union_pool_ablation_${pool_mode}_reuse_main_ckpt${RUN_SUFFIX}"

    printf '\n[atom-union-pool-ablation] build variant=%s\n' "$pool_mode"
    for split in $SPLITS; do
      chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"
      union_pool="${union_root}/atom_union_candidate_pool_${split}.jsonl"
      map_pool="${map_root}/evidence_map_candidate_pool_${split}.jsonl"
      annotations="${map_root}/deepseek_evidence_map_annotations_${split}.jsonl"
      features="${map_root}/candidate_evidence_map_features_${split}.jsonl"
      trace="${trace_root}/selection_trace_${split}.jsonl"

      if [[ ! -s "$union_pool" || "$FORCE_POOL_BUILD" == "true" ]]; then
        require_path "${RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" "${split} baseline top-${BASELINE_TOP_K} pool"
        require_path "${RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" "${split} atom RRF pool"
        union_args=(
          --baseline-jsonl "${RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl"
          --atom-pool-jsonl "${RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl"
          --output-dir "$union_root"
          --split "$split"
          --pool-mode "$pool_mode"
          --final-pool-size "$POOL_SIZE"
          --union-mmr-lambda "$UNION_MMR_LAMBDA"
          --oracle-results ""
        )
        if [[ "$pool_mode" == "union_full" ]]; then
          union_args+=(--chunk-cache-path "$chunk_cache")
        fi
        run_build_stage "candidate-pool variant=${pool_mode} split=${split}" \
          "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_retrieval_union.py \
          "${union_args[@]}" "${sample_args[@]}"
      else
        reuse_build_stage "candidate-pool variant=${pool_mode} split=${split}" "$union_pool"
      fi

      if [[ ! -s "$map_pool" || "$FORCE_MAP_BUILD" == "true" ]]; then
        require_path "$union_pool" "${split} ${pool_mode} candidate pool"
        run_build_stage "map-pool variant=${pool_mode} split=${split}" \
          "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
          --input-candidate-file "$union_pool" \
          --output-dir "$map_root" \
          --split "$split" \
          --candidate-source atom_union \
          --candidate-top-n "$POOL_SIZE" \
          "${sample_args[@]}"
      else
        reuse_build_stage "map-pool variant=${pool_mode} split=${split}" "$map_pool"
      fi

      if enabled "$RUN_TEACHER"; then
        teacher_args=()
        if enabled "$MOCK_EVIDENCE_MAPS"; then teacher_args+=(--mock-maps); fi
        require_path "$map_pool" "${split} ${pool_mode} evidence-map pool"
        run_build_stage "teacher-api variant=${pool_mode} split=${split}" \
          "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
          --candidate-pool "$map_pool" \
          --output-dir "$map_root" \
          --split "$split" \
          --prompt-version "$PROMPT_VERSION" \
          --base-url "$TEACHER_BASE_URL" \
          --model "$TEACHER_MODEL" \
          --api-key-env "$TEACHER_API_KEY_ENV" \
          --concurrency "$TEACHER_CONCURRENCY" \
          --requests-per-minute "$TEACHER_REQUESTS_PER_MINUTE" \
          --max-tokens "$TEACHER_MAX_TOKENS" \
          --top-p "$TEACHER_TOP_P" \
          --thinking-type "$TEACHER_THINKING_TYPE" \
          --resume \
          "${teacher_args[@]}" "${sample_args[@]}"
      fi

      if [[ ! -s "$features" || "$FORCE_MAP_BUILD" == "true" ]]; then
        require_path "$map_pool" "${split} ${pool_mode} evidence-map pool"
        require_path "$annotations" "${split} ${pool_mode} evidence-map annotations"
        run_build_stage "map-postprocess variant=${pool_mode} split=${split}" \
          "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
          --candidate-pool "$map_pool" \
          --annotations "$annotations" \
          --output-dir "$map_root" \
          --split "$split" \
          "${sample_args[@]}"
      else
        reuse_build_stage "map-postprocess variant=${pool_mode} split=${split}" "$features"
      fi

      if [[ ! -s "$trace" || "$FORCE_TRACE_BUILD" == "true" ]]; then
        require_path "$features" "${split} ${pool_mode} evidence-map features"
        run_build_stage "mrec-trace variant=${pool_mode} split=${split}" \
          "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
          --input "$features" \
          --output-dir "$trace_root" \
          --split "$split" \
          --candidate-top-n 0 \
          --max-steps 0 \
          --min-steps 0 \
          --target-resolved-rate 1.0 \
          --post-target-fill-policy contrast_only \
          --selector-name "$SELECTOR_NAME" \
          --selection-policy learned_marginal_proxy \
          --weight-file "$MAIN_WEIGHT_FILE" \
          --stop-threshold -1000000000 \
          --source-selector-name "$SOURCE_SELECTOR_NAME" \
          "${sample_args[@]}"
      else
        reuse_build_stage "mrec-trace variant=${pool_mode} split=${split}" "$trace"
      fi
    done

    if [[ ! -s "${case_root}/train.resolved.yaml" || "$FORCE_VERIFIER_BUILD" == "true" ]]; then
      run_build_stage "verifier-data variant=${pool_mode}" \
        "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
        --config "$MAIN_POLICY_CONFIG" \
        --train-trace "${trace_root}/selection_trace_train.jsonl" \
        --val-trace "${trace_root}/selection_trace_val.jsonl" \
        --test-trace "${trace_root}/selection_trace_test.jsonl" \
        --train-raw data/raw/LIAR-RAW/train.json \
        --val-raw data/raw/LIAR-RAW/val.json \
        --test-raw data/raw/LIAR-RAW/test.json \
        --dataset liar_raw \
        --label-schema liar6 \
        --output-dir "$case_root" \
        --selection-mode trace \
        --trace-prompt-style mrec_min \
        --evidence-text-mode full \
        --expected-selector-name "$SELECTOR_NAME" \
        --top-k 10 \
        --prompt-evidence-policy minmax \
        --prompt-evidence-min-count 5 \
        --prompt-evidence-max-count 10 \
        --prompt-evidence-max-length-guard warn \
        --expected-chunk-mmr-fingerprint "" \
        --prompt-model-name-or-path "$MODEL_PATH" \
        --train-model-name-or-path "$MODEL_PATH"
    else
      reuse_build_stage "verifier-data variant=${pool_mode}" "${case_root}/train.resolved.yaml"
    fi
  done
fi

if should_eval; then
  require_path "${MAIN_RUN_ROOT}/train/best" "main verifier best checkpoint"
  for pool_mode in $POOL_MODES; do
    case_root="${CASE_OUTPUT_ROOT}/liar_raw__ministral3_8b__atom_union_pool_ablation_${pool_mode}_reuse_main_ckpt${RUN_SUFFIX}"
    require_path "${case_root}/train.resolved.yaml" "${pool_mode} verifier-data config"
    for split in $EVAL_SPLITS; do
      for eval_mode in label_token label_token_logit_adjust_tau0p75; do
        output_dir="${case_root}/eval/${split}/best/${eval_mode}"
        metrics="${output_dir}/metrics.json"
        if [[ -s "$metrics" && "$FORCE_EVAL" != "true" ]]; then
          printf '[atom-union-pool-ablation] reuse eval: %s\n' "$metrics"
          continue
        fi
        eval_args=(
          --run-dir "${MAIN_RUN_ROOT}/train"
          --checkpoint best
          --split "$split"
          --config "${case_root}/train.resolved.yaml"
          --output-dir "$output_dir"
        )
        if [[ "$eval_mode" == "label_token_logit_adjust_tau0p75" ]]; then
          eval_args+=(--logit-adjust on --logit-adjust-tau 0.75)
        else
          eval_args+=(--logit-adjust off)
        fi
        run_cmd "$PYTHON_BIN" -m sft.label_token_infer "${eval_args[@]}"
      done
    done
  done
fi

if should_summarize; then
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/summarize_atom_union_pool_ablation.py \
    --ablation-root "$ABLATION_ROOT" \
    --case-output-root "$CASE_OUTPUT_ROOT" \
    --case-suffix "$RUN_SUFFIX" \
    --pool-modes $POOL_MODES
fi

printf '\n[atom-union-pool-ablation] done.\n'
