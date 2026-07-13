#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

REFERENCE_CONTRACT="${REFERENCE_CONTRACT:-configs/validation/baces_native_label_token_reference_v0_1.json}"
SOURCE_FACTORIAL_MANIFEST="${SOURCE_FACTORIAL_MANIFEST:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/08_baces_factorial_v0_1/val/manifest.json}"
SOURCE_CONTROLLER="${SOURCE_CONTROLLER:-ordinal_replay_minmax5_10}"
SELECTORS="${SELECTORS:-baces_exact,learned_marginal}"
MIN_K="${MIN_K:-1}"
MAX_K="${MAX_K:-10}"
SPLIT="${SPLIT:-val}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
if [[ -n "$SAMPLE_LIMIT" ]] && ! [[ "$SAMPLE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SAMPLE_LIMIT must be empty or a positive integer: %s\n' "$SAMPLE_LIMIT" >&2
  exit 2
fi
SAMPLE_NAMESPACE=""
if [[ -n "$SAMPLE_LIMIT" ]]; then
  SAMPLE_NAMESPACE="__sample${SAMPLE_LIMIT}"
fi

if [[ ! -f "$REFERENCE_CONTRACT" ]]; then
  printf 'Reference contract does not exist: %s\n' "$REFERENCE_CONTRACT" >&2
  exit 2
fi

# Keep model identity, native-gate artifacts, and inference configuration on the
# same frozen contract used by the deduplicated factorial runner.
contract_python_bin="$(jq -er '.native_command[0]' "$REFERENCE_CONTRACT")"
contract_run_dir="$(jq -er '.checkpoint.run_dir' "$REFERENCE_CONTRACT")"
contract_checkpoint="$(jq -er '.checkpoint.checkpoint' "$REFERENCE_CONTRACT")"
contract_adapter_sha256="$(jq -er '.checkpoint.adapter_sha256' "$REFERENCE_CONTRACT")"
contract_config="$(jq -er '.artifacts.inference_config.path' "$REFERENCE_CONTRACT")"
contract_gate_predictions="$(jq -er '.artifacts.predictions.path' "$REFERENCE_CONTRACT")"
contract_gate_metrics="$(jq -er '.artifacts.metrics.path' "$REFERENCE_CONTRACT")"
contract_gate_build="$(jq -er '.artifacts.build.path' "$REFERENCE_CONTRACT")"

PYTHON_BIN="${PYTHON_BIN:-$contract_python_bin}"
BUILD_CONFIG="${BUILD_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_10.yaml}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
TRACE_ORDER_FIELD="${TRACE_ORDER_FIELD:-selector_available_ordered_indices}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/09_baces_capacity_prefix_v0_1/${SPLIT}${SAMPLE_NAMESPACE}}"
DIAGNOSTIC_BUILD_ROOT="${DIAGNOSTIC_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_capacity_prefix_diagnostic_v0_1__${SPLIT}${SAMPLE_NAMESPACE}}"
MAX_PLAN_ROOT="${MAX_PLAN_ROOT:-$ARTIFACT_ROOT/maximal_plans}"
PLAN_ROOT="${PLAN_ROOT:-$ARTIFACT_ROOT/prefix_plans}"
PLAN_MANIFEST="${PLAN_MANIFEST:-$PLAN_ROOT/manifest.json}"
POLICY_ROOT="${POLICY_ROOT:-$ARTIFACT_ROOT/policies}"
SOURCE_POLICY_PATH="${SOURCE_POLICY_PATH:-$POLICY_ROOT/${SOURCE_CONTROLLER}.jsonl}"
SOURCE_POLICY_SUMMARY="${SOURCE_POLICY_PATH%.jsonl}.summary.json"
FORMAL_BUILD_ROOT="${FORMAL_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_capacity_prefix_v0_1__${SPLIT}${SAMPLE_NAMESPACE}}"
MATRIX_ROOT="${MATRIX_ROOT:-$ARTIFACT_ROOT/matrix}"
MATRIX_MANIFEST="${MATRIX_MANIFEST:-$MATRIX_ROOT/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation_artifacts/baces_capacity_prefix_v0_1/${SPLIT}${SAMPLE_NAMESPACE}/atom_anchor_v0_2_fullpool_minmax5_10_best_${contract_adapter_sha256:0:8}_noadjust}"
RUN_CONTRACT="${RUN_CONTRACT:-$ARTIFACT_ROOT/prefix_run_contract.json}"

RUN_DIR="${RUN_DIR:-$contract_run_dir}"
CHECKPOINT="${CHECKPOINT:-$contract_checkpoint}"
EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256:-$contract_adapter_sha256}"
CONFIG="${CONFIG:-$contract_config}"
GATE_CELL="${GATE_CELL:-baces_exact__prefix_k05}"
GATE_PREDICTIONS="${GATE_PREDICTIONS:-$contract_gate_predictions}"
GATE_METRICS="${GATE_METRICS:-$contract_gate_metrics}"
GATE_BUILD="${GATE_BUILD:-$contract_gate_build}"

PHASES="${PHASES:-diagnostic_build,project,plans,policy,formal_build,manifest,prepare,infer,fanout,stats}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
FORCE_DIAGNOSTIC_BUILD="${FORCE_DIAGNOSTIC_BUILD:-false}"
FORCE_PROJECT="${FORCE_PROJECT:-false}"
FORCE_PLANS="${FORCE_PLANS:-false}"
FORCE_POLICY="${FORCE_POLICY:-false}"
FORCE_FORMAL_BUILD="${FORCE_FORMAL_BUILD:-false}"
FORCE_MANIFEST="${FORCE_MANIFEST:-false}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"
FORCE_STATS="${FORCE_STATS:-false}"
FORCE_ALL="${FORCE_ALL:-false}"
UNSAFE_SKIP_EQUIVALENCE_GATE="${UNSAFE_SKIP_EQUIVALENCE_GATE:-false}"
INCLUDE_FIXED_POLICIES="${INCLUDE_FIXED_POLICIES:-true}"
INCLUDE_SOURCE_POLICY="${INCLUDE_SOURCE_POLICY:-true}"
CAPACITY_POLICIES="${CAPACITY_POLICIES:-}"
TOKEN_PENALTY_PER_1K="${TOKEN_PENALTY_PER_1K:-0.0}"
TIE_ATOL="${TIE_ATOL:-1e-12}"
DRY_RUN="${DRY_RUN:-false}"
CONTRACT_PROMOTE_PENDING=false
RUN_CONTRACT_PENDING="${RUN_CONTRACT}.pending"

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  python_dir="$(dirname "$PYTHON_BIN")"
  if [[ -x "${python_dir}/accelerate" ]]; then
    ACCELERATE_BIN="${python_dir}/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

phase_enabled() {
  local needle="$1"
  [[ ",${PHASES}," == *",${needle},"* ]]
}

validate_boolean() {
  local name="$1"
  local value="$2"
  if [[ "$value" != "true" && "$value" != "false" ]]; then
    printf '%s must be true or false: %s\n' "$name" "$value" >&2
    exit 2
  fi
}

resolve_source_trace() {
  local selector="$1"
  local relative_trace
  local manifest_dir
  if [[ ! -f "$SOURCE_FACTORIAL_MANIFEST" ]]; then
    printf 'Source factorial manifest does not exist: %s\n' "$SOURCE_FACTORIAL_MANIFEST" >&2
    return 2
  fi
  relative_trace="$(jq -er \
    --arg selector "$selector" \
    --arg controller "$SOURCE_CONTROLLER" \
    '[.cells[] | select(.ready == true and .selector_level == $selector and .controller_level == $controller) | .trace_file]
     | if length == 1 then .[0] else error("expected exactly one ready source trace") end' \
    "$SOURCE_FACTORIAL_MANIFEST")"
  if [[ "$relative_trace" = /* ]]; then
    printf '%s\n' "$relative_trace"
  else
    manifest_dir="$(dirname "$SOURCE_FACTORIAL_MANIFEST")"
    printf '%s/%s\n' "$manifest_dir" "$relative_trace"
  fi
}

sha256_file() {
  local path="$1"
  local digest
  local ignored
  if [[ ! -f "$path" ]]; then
    printf 'Run-contract input does not exist: %s\n' "$path" >&2
    return 2
  fi
  read -r digest ignored < <(sha256sum "$path")
  printf '%s\n' "$digest"
}

cpu_rebuild_phases_enabled() {
  local phase
  for phase in diagnostic_build project plans policy formal_build manifest prepare; do
    if ! phase_enabled "$phase"; then
      return 1
    fi
  done
  return 0
}

stage_outputs_exist() {
  local root
  for root in "$ARTIFACT_ROOT" "$DIAGNOSTIC_BUILD_ROOT" "$FORMAL_BUILD_ROOT" "$OUTPUT_DIR"; do
    if [[ -d "$root" ]] && [[ -n "$(find "$root" -mindepth 1 -print -quit)" ]]; then
      return 0
    fi
  done
  return 1
}

write_contract_payload() {
  local target="$1"
  local payload="$2"
  local temporary="${target}.tmp.$$"
  mkdir -p "$(dirname "$target")"
  printf '%s\n' "$payload" > "$temporary"
  mv "$temporary" "$target"
}

validate_or_stage_run_contract() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[capacity-prefix] run_contract=%s (dry-run; not materialized)\n' "$RUN_CONTRACT"
    return 0
  fi

  local normalized_selectors
  local selectors_json
  local source_traces_json='[]'
  local implementations_json='[]'
  local selector
  local source_trace
  local source_trace_sha
  local implementation
  local implementation_sha
  local gate_contract_json
  local contract_core
  local contract_fingerprint
  local ignored
  local contract_payload
  local existing_fingerprint
  local contract_mismatch=false

  normalized_selectors="$(IFS=,; printf '%s' "${selector_values[*]}")"
  selectors_json="$(jq -cn --arg value "$normalized_selectors" '$value | split(",") | sort')"
  for selector in "${selector_values[@]}"; do
    source_trace="$(resolve_source_trace "$selector")"
    source_trace_sha="$(sha256_file "$source_trace")"
    source_traces_json="$(
      jq -c \
        --arg selector "$selector" \
        --arg path "$(realpath "$source_trace")" \
        --arg sha256 "$source_trace_sha" \
        '. + [{selector_level:$selector,path:$path,sha256:$sha256}] | sort_by(.selector_level)' \
        <<< "$source_traces_json"
    )"
  done
  for implementation in \
    scripts/phase5_selectors/build/build_trace_verifier_data.py \
    scripts/phase5_selectors/build/project_capacity_prefix_plan.py \
    scripts/phase5_selectors/build/expand_capacity_prefix_plans.py \
    scripts/phase5_selectors/build/materialize_capacity_policy_from_traces.py \
    scripts/phase5_selectors/build/materialize_capacity_prefix_matrix.py \
    src/sft/label_token_matrix_infer.py \
    src/sft/capacity_prefix_analysis.py \
    scripts/phase5_selectors/eval/run_baces_capacity_prefix_matrix.sh; do
    implementation_sha="$(sha256_file "$implementation")"
    implementations_json="$(
      jq -c \
        --arg path "$(realpath "$implementation")" \
        --arg sha256 "$implementation_sha" \
        '. + [{path:$path,sha256:$sha256}]' \
        <<< "$implementations_json"
    )"
  done
  if [[ "$UNSAFE_SKIP_EQUIVALENCE_GATE" == "true" ]]; then
    gate_contract_json="$(
      jq -cn \
        --arg cell "$GATE_CELL" \
        '{mode:"diagnostic_unsafe_skip",cell:$cell}'
    )"
  else
    gate_contract_json="$(
      jq -cn \
        --arg cell "$GATE_CELL" \
        --arg predictions "$(realpath "$GATE_PREDICTIONS")" \
        --arg predictions_sha256 "$(sha256_file "$GATE_PREDICTIONS")" \
        --arg metrics "$(realpath "$GATE_METRICS")" \
        --arg metrics_sha256 "$(sha256_file "$GATE_METRICS")" \
        --arg build "$(realpath "$GATE_BUILD")" \
        --arg build_sha256 "$(sha256_file "$GATE_BUILD")" \
        --arg expected_adapter_sha256 "$EXPECTED_ADAPTER_SHA256" \
        '{
          mode:"frozen_native_equivalence_gate",
          cell:$cell,
          predictions:{path:$predictions,sha256:$predictions_sha256},
          metrics:{path:$metrics,sha256:$metrics_sha256},
          build:{path:$build,sha256:$build_sha256},
          expected_adapter_sha256:$expected_adapter_sha256
        }'
    )"
  fi

  contract_core="$(
    jq -cn \
      --arg split "$SPLIT" \
      --arg sample_limit "$SAMPLE_LIMIT" \
      --argjson selectors "$selectors_json" \
      --arg min_k "$MIN_K" \
      --arg max_k "$MAX_K" \
      --arg source_controller "$SOURCE_CONTROLLER" \
      --arg trace_order_field "$TRACE_ORDER_FIELD" \
      --arg trace_prompt_style "$TRACE_PROMPT_STYLE" \
      --arg expected_chunk_mmr_fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT" \
      --arg reference_contract "$(realpath "$REFERENCE_CONTRACT")" \
      --arg reference_contract_sha256 "$(sha256_file "$REFERENCE_CONTRACT")" \
      --arg build_config "$(realpath "$BUILD_CONFIG")" \
      --arg build_config_sha256 "$(sha256_file "$BUILD_CONFIG")" \
      --arg source_factorial_manifest "$(realpath "$SOURCE_FACTORIAL_MANIFEST")" \
      --arg source_factorial_manifest_sha256 "$(sha256_file "$SOURCE_FACTORIAL_MANIFEST")" \
      --argjson source_traces "$source_traces_json" \
      --arg python_bin "$(realpath "$PYTHON_BIN")" \
      --arg python_bin_sha256 "$(sha256_file "$PYTHON_BIN")" \
      --arg run_dir "$(realpath -m "$RUN_DIR")" \
      --arg checkpoint "$CHECKPOINT" \
      --arg expected_adapter_sha256 "$EXPECTED_ADAPTER_SHA256" \
      --arg inference_config "$(realpath "$CONFIG")" \
      --arg inference_config_sha256 "$(sha256_file "$CONFIG")" \
      --argjson gate_contract "$gate_contract_json" \
      --arg artifact_root "$(realpath -m "$ARTIFACT_ROOT")" \
      --arg diagnostic_build_root "$(realpath -m "$DIAGNOSTIC_BUILD_ROOT")" \
      --arg maximal_plan_root "$(realpath -m "$MAX_PLAN_ROOT")" \
      --arg plan_root "$(realpath -m "$PLAN_ROOT")" \
      --arg source_policy_path "$(realpath -m "$SOURCE_POLICY_PATH")" \
      --arg formal_build_root "$(realpath -m "$FORMAL_BUILD_ROOT")" \
      --arg matrix_root "$(realpath -m "$MATRIX_ROOT")" \
      --arg output_dir "$(realpath -m "$OUTPUT_DIR")" \
      --argjson implementations "$implementations_json" \
      '{
        split:$split,
        sample_limit:(if $sample_limit == "" then null else ($sample_limit|tonumber) end),
        selector_levels:$selectors,
        capacities:{min:($min_k|tonumber),max:($max_k|tonumber)},
        source_controller:$source_controller,
        trace_order_field:$trace_order_field,
        trace_prompt_style:$trace_prompt_style,
        expected_chunk_mmr_fingerprint:$expected_chunk_mmr_fingerprint,
        inputs:{
          reference_contract:{path:$reference_contract,sha256:$reference_contract_sha256},
          build_config:{path:$build_config,sha256:$build_config_sha256},
          source_factorial_manifest:{
            path:$source_factorial_manifest,
            sha256:$source_factorial_manifest_sha256
          },
          source_traces:$source_traces
        },
        verifier_runtime:{
          python:{path:$python_bin,sha256:$python_bin_sha256},
          run_dir:$run_dir,
          checkpoint:$checkpoint,
          expected_adapter_sha256:$expected_adapter_sha256,
          config:{path:$inference_config,sha256:$inference_config_sha256}
        },
        equivalence_gate:$gate_contract,
        roots:{
          artifact:$artifact_root,
          diagnostic_build:$diagnostic_build_root,
          maximal_plan:$maximal_plan_root,
          plan:$plan_root,
          source_policy:$source_policy_path,
          formal_build:$formal_build_root,
          matrix:$matrix_root,
          inference:$output_dir
        },
        implementations:$implementations
      }'
  )"
  read -r contract_fingerprint ignored < <(printf '%s' "$contract_core" | sha256sum)
  contract_payload="$(
    jq -cn \
      --arg fingerprint "$contract_fingerprint" \
      --argjson contract "$contract_core" \
      '{
        schema_version:"capacity_prefix_run_contract_v0_1",
        status:"frozen",
        contract_fingerprint:$fingerprint,
        contract:$contract
      }'
  )"

  if [[ -f "$RUN_CONTRACT" ]]; then
    if ! existing_fingerprint="$(
      jq -er \
        'select(.schema_version == "capacity_prefix_run_contract_v0_1" and .status == "frozen") | .contract_fingerprint' \
        "$RUN_CONTRACT" 2>/dev/null
    )"; then
      printf 'Existing run contract is invalid: %s\n' "$RUN_CONTRACT" >&2
      return 2
    fi
    if [[ "$existing_fingerprint" != "$contract_fingerprint" ]]; then
      contract_mismatch=true
    fi
  elif stage_outputs_exist; then
    contract_mismatch=true
  else
    write_contract_payload "$RUN_CONTRACT" "$contract_payload"
    printf '[capacity-prefix] froze run contract: %s\n' "$RUN_CONTRACT"
    return 0
  fi

  if [[ "$contract_mismatch" == "true" ]]; then
    if [[ "$FORCE_ALL" != "true" ]] || ! cpu_rebuild_phases_enabled; then
      printf 'Capacity-prefix run contract mismatch (or legacy outputs without a contract): %s\n' \
        "$RUN_CONTRACT" >&2
      printf 'Use new artifact roots, or rerun all CPU phases through prepare with FORCE_ALL=true.\n' >&2
      return 2
    fi
    write_contract_payload "$RUN_CONTRACT_PENDING" "$contract_payload"
    CONTRACT_PROMOTE_PENDING=true
    printf '[capacity-prefix] staged replacement run contract: %s\n' "$RUN_CONTRACT_PENDING"
  else
    printf '[capacity-prefix] verified run contract: %s\n' "$RUN_CONTRACT"
  fi
}

for boolean_name in \
  FORCE_DIAGNOSTIC_BUILD FORCE_PROJECT FORCE_PLANS FORCE_POLICY FORCE_FORMAL_BUILD \
  FORCE_MANIFEST FORCE_PREPARE FORCE_INFER FORCE_FANOUT FORCE_STATS \
  FORCE_ALL UNSAFE_SKIP_EQUIVALENCE_GATE INCLUDE_FIXED_POLICIES \
  INCLUDE_SOURCE_POLICY DRY_RUN; do
  validate_boolean "$boolean_name" "${!boolean_name}"
done
if ! [[ "$MIN_K" =~ ^[1-9][0-9]*$ && "$MAX_K" =~ ^[1-9][0-9]*$ ]] || (( MIN_K > MAX_K )); then
  printf 'Require positive integer capacities with MIN_K <= MAX_K; got %s..%s\n' "$MIN_K" "$MAX_K" >&2
  exit 2
fi
if [[ "$FORCE_ALL" == "true" ]]; then
  FORCE_DIAGNOSTIC_BUILD=true
  FORCE_PROJECT=true
  FORCE_PLANS=true
  FORCE_POLICY=true
  FORCE_FORMAL_BUILD=true
  FORCE_MANIFEST=true
  FORCE_PREPARE=true
  FORCE_INFER=true
  FORCE_FANOUT=true
  FORCE_STATS=true
fi
# Rebuilding any upstream artifact invalidates every enabled downstream artifact.
if [[ "$FORCE_DIAGNOSTIC_BUILD" == "true" ]]; then
  FORCE_PROJECT=true
fi
if [[ "$FORCE_PROJECT" == "true" ]]; then
  FORCE_PLANS=true
fi
if [[ "$FORCE_PLANS" == "true" ]]; then
  FORCE_FORMAL_BUILD=true
fi
if [[ "$FORCE_FORMAL_BUILD" == "true" ]]; then
  FORCE_MANIFEST=true
fi
if [[ "$FORCE_MANIFEST" == "true" ]]; then
  FORCE_PREPARE=true
fi
if [[ "$FORCE_PREPARE" == "true" ]]; then
  FORCE_INFER=true
fi
if [[ "$FORCE_INFER" == "true" ]]; then
  FORCE_FANOUT=true
fi
if [[ "$FORCE_FANOUT" == "true" || "$FORCE_POLICY" == "true" ]]; then
  FORCE_STATS=true
fi
if ! [[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EVAL_NPROC_PER_NODE must be a positive integer: %s\n' "$EVAL_NPROC_PER_NODE" >&2
  exit 2
fi
if [[ "$SPLIT" != "val" ]]; then
  printf 'The v0.1 capacity-prefix wrapper is validation-only; got SPLIT=%s\n' "$SPLIT" >&2
  exit 2
fi
if [[ "$MATRIX_MANIFEST" != "$MATRIX_ROOT/manifest.json" ]]; then
  printf 'MATRIX_MANIFEST must equal MATRIX_ROOT/manifest.json: %s != %s/manifest.json\n' \
    "$MATRIX_MANIFEST" "$MATRIX_ROOT" >&2
  exit 2
fi
if [[ "$PLAN_MANIFEST" != "$PLAN_ROOT/manifest.json" ]]; then
  printf 'PLAN_MANIFEST must equal PLAN_ROOT/manifest.json: %s != %s/manifest.json\n' \
    "$PLAN_MANIFEST" "$PLAN_ROOT" >&2
  exit 2
fi

IFS=',' read -r -a raw_selectors <<< "$SELECTORS"
selector_values=()
for raw_selector in "${raw_selectors[@]}"; do
  selector="$(printf '%s' "$raw_selector" | tr -d '[:space:]')"
  if [[ -z "$selector" || ! "$selector" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'Invalid selector in SELECTORS=%s: %s\n' "$SELECTORS" "$raw_selector" >&2
    exit 2
  fi
  selector_values+=("$selector")
done
if [[ "${#selector_values[@]}" -eq 0 ]]; then
  printf 'SELECTORS must contain at least one selector\n' >&2
  exit 2
fi
if phase_enabled fanout && [[ "$UNSAFE_SKIP_EQUIVALENCE_GATE" != "true" ]]; then
  if [[ -n "$SAMPLE_LIMIT" ]]; then
    printf 'SAMPLE_LIMIT cannot use the full native equivalence reference; either omit fanout or set UNSAFE_SKIP_EQUIVALENCE_GATE=true for a diagnostic-only smoke run\n' >&2
    exit 2
  fi
  if [[ ! "$GATE_CELL" =~ ^(.+)__prefix_k([0-9]+)$ ]]; then
    printf 'GATE_CELL must have form SELECTOR__prefix_kNN: %s\n' "$GATE_CELL" >&2
    exit 2
  fi
  gate_selector="${BASH_REMATCH[1]}"
  gate_k="$((10#${BASH_REMATCH[2]}))"
  gate_selector_found=false
  for selector in "${selector_values[@]}"; do
    if [[ "$selector" == "$gate_selector" ]]; then
      gate_selector_found=true
      break
    fi
  done
  if [[ "$gate_selector_found" != "true" || "$gate_k" -lt "$MIN_K" || "$gate_k" -gt "$MAX_K" ]]; then
    printf 'GATE_CELL is outside the configured selector/capacity grid: %s\n' "$GATE_CELL" >&2
    exit 2
  fi
fi

validate_or_stage_run_contract

printf '[capacity-prefix] source_manifest=%s source_controller=%s selectors=%s K=%s..%s\n' \
  "$SOURCE_FACTORIAL_MANIFEST" "$SOURCE_CONTROLLER" "$SELECTORS" "$MIN_K" "$MAX_K"
printf '[capacity-prefix] build_config=%s reference_contract=%s\n' \
  "$BUILD_CONFIG" "$REFERENCE_CONTRACT"
printf '[capacity-prefix] plan=%s matrix=%s output=%s phases=%s sample_limit=%s\n' \
  "$PLAN_MANIFEST" "$MATRIX_MANIFEST" "$OUTPUT_DIR" "$PHASES" "${SAMPLE_LIMIT:-all}"

if phase_enabled diagnostic_build; then
  for selector in "${selector_values[@]}"; do
    source_trace="$(resolve_source_trace "$selector")"
    diagnostic_dir="$DIAGNOSTIC_BUILD_ROOT/${selector}__prefix_k$(printf '%02d' "$MAX_K")"
    diagnostic_build="$diagnostic_dir/build/build_${SPLIT}.jsonl"
    if [[ -f "$diagnostic_build" && "$FORCE_DIAGNOSTIC_BUILD" != "true" ]]; then
      printf '[capacity-prefix] skip diagnostic_build (exists): %s\n' "$diagnostic_build"
      continue
    fi
    diagnostic_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py
      --config "$BUILD_CONFIG"
      --val-trace "$source_trace"
      --output-dir "$diagnostic_dir"
      --selection-mode trace
      --trace-order-field "$TRACE_ORDER_FIELD"
      --trace-prompt-style "$TRACE_PROMPT_STYLE"
      --top-k "$MAX_K"
      --prompt-evidence-policy fixed_topk
      --prompt-evidence-min-count "$MAX_K"
      --prompt-evidence-max-count "$MAX_K"
      --prompt-evidence-max-length-guard warn
      --expected-chunk-mmr-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT"
      --val-only
      --no-progress)
    if [[ -n "$SAMPLE_LIMIT" ]]; then
      diagnostic_cmd+=(--sample-limit "$SAMPLE_LIMIT")
    fi
    run_cmd "${diagnostic_cmd[@]}"
  done
fi

if phase_enabled project; then
  for selector in "${selector_values[@]}"; do
    source_trace="$(resolve_source_trace "$selector")"
    diagnostic_dir="$DIAGNOSTIC_BUILD_ROOT/${selector}__prefix_k$(printf '%02d' "$MAX_K")"
    output_plan="$MAX_PLAN_ROOT/$selector/selection_plan_${SPLIT}.jsonl"
    if [[ -f "$output_plan" && "$FORCE_PROJECT" != "true" ]]; then
      printf '[capacity-prefix] skip project (exists): %s\n' "$output_plan"
      continue
    fi
    project_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/project_capacity_prefix_plan.py
      --source-trace "$source_trace"
      --source-build "$diagnostic_dir/build/build_${SPLIT}.jsonl"
      --output-plan "$output_plan"
      --requested-prefix-k "$MAX_K"
      --trace-order-field "$TRACE_ORDER_FIELD")
    if [[ -n "$SAMPLE_LIMIT" ]]; then
      project_cmd+=(--sample-limit "$SAMPLE_LIMIT")
    fi
    if [[ "$FORCE_PROJECT" == "true" ]]; then
      project_cmd+=(--overwrite)
    fi
    run_cmd "${project_cmd[@]}"
  done
fi

if phase_enabled plans; then
  if [[ -f "$PLAN_MANIFEST" && "$FORCE_PLANS" != "true" ]]; then
    printf '[capacity-prefix] skip plans (exists): %s\n' "$PLAN_MANIFEST"
  else
    plans_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/expand_capacity_prefix_plans.py
      --output-dir "$PLAN_ROOT"
      --split "$SPLIT"
      --min-k "$MIN_K"
      --max-k "$MAX_K"
      --source-controller "$SOURCE_CONTROLLER")
    for selector in "${selector_values[@]}"; do
      plans_cmd+=(--selector-plan "$selector=$MAX_PLAN_ROOT/$selector/selection_plan_${SPLIT}.jsonl")
    done
    if [[ "$FORCE_PLANS" == "true" ]]; then
      plans_cmd+=(--overwrite)
    fi
    run_cmd "${plans_cmd[@]}"
  fi
fi

if phase_enabled policy; then
  if [[ -f "$SOURCE_POLICY_PATH" && -f "$SOURCE_POLICY_SUMMARY" && "$FORCE_POLICY" != "true" ]]; then
    printf '[capacity-prefix] skip policy (exists): %s\n' "$SOURCE_POLICY_PATH"
  elif [[ "$FORCE_POLICY" != "true" && ( -e "$SOURCE_POLICY_PATH" || -e "$SOURCE_POLICY_SUMMARY" ) ]]; then
    printf 'Policy artifact is partial; set FORCE_POLICY=true to rebuild both files: %s / %s\n' \
      "$SOURCE_POLICY_PATH" "$SOURCE_POLICY_SUMMARY" >&2
    exit 2
  else
    policy_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/materialize_capacity_policy_from_traces.py
      --policy-id "$SOURCE_CONTROLLER"
      --output-policy "$SOURCE_POLICY_PATH"
      --expected-controller "$SOURCE_CONTROLLER"
      --source-factorial-manifest "$SOURCE_FACTORIAL_MANIFEST"
      --split "$SPLIT"
      --trace-order-field "$TRACE_ORDER_FIELD"
      --min-k "$MIN_K"
      --max-k "$MAX_K")
    for selector in "${selector_values[@]}"; do
      source_trace="$(resolve_source_trace "$selector")"
      policy_cmd+=(--selector-trace "$selector=$source_trace")
    done
    if [[ -n "$SAMPLE_LIMIT" ]]; then
      policy_cmd+=(--sample-limit "$SAMPLE_LIMIT")
    fi
    if [[ "$FORCE_POLICY" == "true" ]]; then
      policy_cmd+=(--overwrite)
    fi
    run_cmd "${policy_cmd[@]}"
  fi
fi

if phase_enabled formal_build; then
  for selector in "${selector_values[@]}"; do
    source_trace="$(resolve_source_trace "$selector")"
    for (( k=MIN_K; k<=MAX_K; k++ )); do
      capacity_level="prefix_k$(printf '%02d' "$k")"
      cell_id="${selector}__${capacity_level}"
      formal_dir="$FORMAL_BUILD_ROOT/$cell_id"
      formal_build="$formal_dir/build/build_${SPLIT}.jsonl"
      selection_plan="$PLAN_ROOT/$cell_id/selection_plan_${SPLIT}.jsonl"
      if [[ -f "$formal_build" && "$FORCE_FORMAL_BUILD" != "true" ]]; then
        printf '[capacity-prefix] skip formal_build (exists): %s\n' "$formal_build"
        continue
      fi
      formal_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py
        --config "$BUILD_CONFIG"
        --val-trace "$source_trace"
        --val-selection-plan "$selection_plan"
        --output-dir "$formal_dir"
        --selection-mode trace
        --trace-order-field "$TRACE_ORDER_FIELD"
        --trace-prompt-style "$TRACE_PROMPT_STYLE"
        --prompt-evidence-policy selected_set
        --prompt-evidence-min-count 0
        --prompt-evidence-max-count "$MAX_K"
        --prompt-evidence-max-length-guard error
        --expected-chunk-mmr-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT"
        --forbid-prompt-truncation
        --val-only
        --no-progress)
      if [[ -n "$SAMPLE_LIMIT" ]]; then
        formal_cmd+=(--sample-limit "$SAMPLE_LIMIT")
      fi
      run_cmd "${formal_cmd[@]}"
    done
  done
fi

if phase_enabled manifest; then
  if [[ -f "$MATRIX_MANIFEST" && "$FORCE_MANIFEST" != "true" ]]; then
    printf '[capacity-prefix] skip manifest (exists): %s\n' "$MATRIX_MANIFEST"
  else
    manifest_cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/materialize_capacity_prefix_matrix.py
      --plan-manifest "$PLAN_MANIFEST"
      --build-root "$FORMAL_BUILD_ROOT"
      --output-dir "$MATRIX_ROOT"
      --split "$SPLIT")
    if [[ "$FORCE_MANIFEST" == "true" ]]; then
      manifest_cmd+=(--overwrite)
    fi
    run_cmd "${manifest_cmd[@]}"
  fi
fi

if phase_enabled prepare; then
  prepare_cmd=("$PYTHON_BIN" -m sft.label_token_matrix_infer prepare
    --matrix-manifest "$MATRIX_MANIFEST"
    --build-root "$FORMAL_BUILD_ROOT"
    --output-dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --label-prefix 'Label:')
  if [[ "$FORCE_PREPARE" == "true" ]]; then
    prepare_cmd+=(--force-prepare)
  fi
  run_cmd "${prepare_cmd[@]}"
  if [[ "$CONTRACT_PROMOTE_PENDING" == "true" ]]; then
    mv "$RUN_CONTRACT_PENDING" "$RUN_CONTRACT"
    CONTRACT_PROMOTE_PENDING=false
    printf '[capacity-prefix] promoted replacement run contract: %s\n' "$RUN_CONTRACT"
  fi
fi

if phase_enabled infer; then
  infer_args=(-m sft.label_token_matrix_infer infer
    --output-dir "$OUTPUT_DIR"
    --run-dir "$RUN_DIR"
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --split "$SPLIT"
    --expected-world-size "$EVAL_NPROC_PER_NODE"
    --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256")
  if [[ "$FORCE_INFER" == "true" ]]; then
    infer_args+=(--force-infer)
  fi
  if [[ "$EVAL_NPROC_PER_NODE" -gt 1 ]]; then
    infer_cmd=("$ACCELERATE_BIN" launch
      --multi_gpu
      --num_processes "$EVAL_NPROC_PER_NODE"
      --num_machines "$NUM_MACHINES"
      --mixed_precision "$MIXED_PRECISION"
      "${infer_args[@]}")
  else
    infer_cmd=("$PYTHON_BIN" "${infer_args[@]}")
  fi
  run_cmd "${infer_cmd[@]}"
fi

if phase_enabled fanout; then
  fanout_cmd=("$PYTHON_BIN" -m sft.label_token_matrix_infer fanout
    --output-dir "$OUTPUT_DIR")
  if [[ "$UNSAFE_SKIP_EQUIVALENCE_GATE" == "true" ]]; then
    fanout_cmd+=(--unsafe-skip-equivalence-gate)
  else
    fanout_cmd+=(
      --equivalence-gate-cell "$GATE_CELL"
      --equivalence-gate-predictions "$GATE_PREDICTIONS"
      --equivalence-gate-metrics "$GATE_METRICS"
      --equivalence-gate-build "$GATE_BUILD"
      --equivalence-gate-expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256"
      --equivalence-gate-reference-contract "$REFERENCE_CONTRACT")
  fi
  if [[ "$FORCE_FANOUT" == "true" ]]; then
    fanout_cmd+=(--force-fanout)
  fi
  run_cmd "${fanout_cmd[@]}"
fi

if phase_enabled stats; then
  stats_cmd=("$PYTHON_BIN" -m sft.capacity_prefix_analysis
    --matrix-manifest "$OUTPUT_DIR/materialized/matrix_manifest.json"
    --output-dir "$OUTPUT_DIR/capacity_analysis"
    --token-penalty-per-1k "$TOKEN_PENALTY_PER_1K"
    --tie-atol "$TIE_ATOL")
  if [[ "$INCLUDE_FIXED_POLICIES" == "true" ]]; then
    stats_cmd+=(--include-fixed-policies)
  fi
  if [[ "$INCLUDE_SOURCE_POLICY" == "true" ]]; then
    stats_cmd+=(--policy "$SOURCE_CONTROLLER=$SOURCE_POLICY_PATH")
  fi
  if [[ -n "$CAPACITY_POLICIES" ]]; then
    IFS=',' read -r -a policy_values <<< "$CAPACITY_POLICIES"
    for policy in "${policy_values[@]}"; do
      if [[ -z "$policy" || "$policy" != *=* ]]; then
        printf 'CAPACITY_POLICIES entries must be POLICY_ID=JSONL_PATH: %s\n' "$policy" >&2
        exit 2
      fi
      stats_cmd+=(--policy "$policy")
    done
  fi
  if [[ "$FORCE_STATS" == "true" ]]; then
    stats_cmd+=(--force)
  fi
  run_cmd "${stats_cmd[@]}"
fi
