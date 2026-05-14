#!/usr/bin/env bash
# ==============================================================================
# Sensitivity-gated MMR — 阶段 A 离线网格搜索 (无需重训)
#
# 复用 fixed_07 的 SFT checkpoint，对 λ 网格各跑一次 val build + infer，
# 再调用 scripts/rl_mmr/search_sensitivity_thresholds.py 扫
# (θ_s, θ_r, λ_low, gating_mode, ε) 网格。
#
# 用法:
#   bash scripts/rl_mmr/run_stage_a_search.sh \
#        <FIXED_TRAIN_DIR>                  # e.g. outputs/runs/b3_mmr_topk_sweep_1024/.../train
#        [LAMBDA_GRID="0.2 0.3 0.4 0.7"]
#        [SPLIT=val]
#        [BASE_EXPERIMENT=mmr_sensitivity_gated]
#        [SEARCH_OUT=outputs/rl_mmr/sensitivity_search]
#        [VLLM_PORT=8000]
#
# Notes:
#   - FIXED_TRAIN_DIR 必须直接指向训练目录（其下应有 best/ 子目录）。
#   - λ 网格至少包含: lambda_base(默认 0.7) 与 search.lambda_low 中所有候选值。
#   - vLLM 在首个 λ 启动后保持运行，最后由本脚本停掉。
#   - 重复执行幂等：build / infer 阶段命中缓存即跳过。
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

FIXED_TRAIN_DIR="${1:?需要传入 fixed_07 训练目录（例如 outputs/runs/<exp>/<id>/train）}"
LAMBDA_GRID="${2:-0.2 0.3 0.4 0.7}"
SPLIT="${3:-val}"
BASE_EXPERIMENT="${4:-mmr_sensitivity_gated}"
SEARCH_OUT="${5:-outputs/rl_mmr/sensitivity_search}"
VLLM_PORT="${6:-8000}"

if [ ! -d "${FIXED_TRAIN_DIR}/best" ]; then
    echo "[stage-A][fatal] best checkpoint not found at ${FIXED_TRAIN_DIR}/best" >&2
    exit 1
fi
FIXED_TRAIN_DIR="$(cd "${FIXED_TRAIN_DIR}" && pwd)"
echo "[stage-A] fixed train dir : ${FIXED_TRAIN_DIR}"
echo "[stage-A] lambda grid     : ${LAMBDA_GRID}"
echo "[stage-A] split           : ${SPLIT}"
echo "[stage-A] base experiment : ${BASE_EXPERIMENT}"
echo "[stage-A] search out      : ${SEARCH_OUT}"
echo "[stage-A] vllm port       : ${VLLM_PORT}"

SCRATCH_ROOT="${SEARCH_OUT}/_scratch"
mkdir -p "${SCRATCH_ROOT}" "${SEARCH_OUT}"

VLLM_PID_FILE="${SCRATCH_ROOT}/vllm.pid.json"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

cleanup_vllm() {
    if [ -f "${VLLM_PID_FILE}" ]; then
        local pid
        pid=$(python -c "import json,sys; print(json.load(open('${VLLM_PID_FILE}')).get('pid',''))" 2>/dev/null || true)
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            echo "[stage-A] stopping vLLM (pid=${pid})"
            kill -TERM "${pid}" 2>/dev/null || true
            for _ in $(seq 1 30); do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        rm -f "${VLLM_PID_FILE}"
    fi
}
trap cleanup_vllm EXIT

# Collect (lambda, build_jsonl, predictions_jsonl) entries.
RUNS_FRAGMENT="${SCRATCH_ROOT}/runs_fragment.yaml"
: > "${RUNS_FRAGMENT}"
echo "runs:" >> "${RUNS_FRAGMENT}"

CHUNK_MMR_PATH=""
FIRST_LAMBDA=1

for LAM in ${LAMBDA_GRID}; do
    SCRATCH="${SCRATCH_ROOT}/lambda_${LAM}"
    BUILD_LOG="${SCRATCH}/build.log"
    INFER_LOG="${SCRATCH}/infer.log"
    mkdir -p "${SCRATCH}"
    echo ""
    echo "[stage-A] ========== λ=${LAM} =========="
    echo "[stage-A]   scratch run_dir : ${SCRATCH}"

    # ---- Build ----
    echo "[stage-A]   build → ${BUILD_LOG}"
    python -m fact_checking.pipeline.run \
        "experiment=${BASE_EXPERIMENT}" \
        pipeline.mode=build \
        "pipeline.run_dir=${SCRATCH}" \
        "build.retrieval.mmr_lambda=${LAM}" \
        "build.retrieval.learned_lambda.enabled=false" \
        2>&1 | tee "${BUILD_LOG}"

    BUILD_VAL=$(python -c "
import json, sys
m = json.load(open('${SCRATCH}/manifest.json'))
print(m['phases']['build']['outputs']['${SPLIT}'])
")
    if [ -z "${CHUNK_MMR_PATH}" ]; then
        CHUNK_DIR=$(grep -oE 'Chunk-MMR cache dir: [^ ]+' "${BUILD_LOG}" | tail -1 | awk '{print $4}')
        if [ -z "${CHUNK_DIR}" ]; then
            echo "[stage-A][fatal] could not parse 'Chunk-MMR cache dir' from ${BUILD_LOG}" >&2
            exit 2
        fi
        CHUNK_MMR_PATH="${CHUNK_DIR}/${SPLIT}.pkl"
        echo "[stage-A]   chunk_mmr_path : ${CHUNK_MMR_PATH}"
        if [ ! -f "${CHUNK_MMR_PATH}" ]; then
            echo "[stage-A][fatal] chunk_mmr pickle not found: ${CHUNK_MMR_PATH}" >&2
            exit 3
        fi
    fi
    echo "[stage-A]   build_val      : ${BUILD_VAL}"

    # ---- Infer ----
    # First λ starts and keeps vLLM alive; subsequent λ reuse the live server.
    if [ "${FIRST_LAMBDA}" -eq 1 ]; then
        SERVER_OVERRIDES=(
            "infer.server.manage=true"
            "infer.server.stop_after_infer=false"
            "infer.server.pid_file=${VLLM_PID_FILE}"
            "infer.port=${VLLM_PORT}"
        )
    else
        SERVER_OVERRIDES=(
            "infer.server.manage=false"
            "infer.base_url=${VLLM_BASE_URL}"
        )
    fi

    echo "[stage-A]   infer → ${INFER_LOG}"
    python -m fact_checking.pipeline.run \
        "experiment=${BASE_EXPERIMENT}" \
        pipeline.mode=infer \
        "pipeline.run_dir=${SCRATCH}" \
        "train.run_dir=${FIXED_TRAIN_DIR}" \
        "build.retrieval.mmr_lambda=${LAM}" \
        "build.retrieval.learned_lambda.enabled=false" \
        "infer.split=${SPLIT}" \
        "${SERVER_OVERRIDES[@]}" \
        2>&1 | tee "${INFER_LOG}"

    INFER_OUT=$(python -c "
import json
m = json.load(open('${SCRATCH}/manifest.json'))
print(m['phases']['infer']['output_dir'])
")
    PRED_FILE="${INFER_OUT}/api/${SPLIT}_predictions.jsonl"
    if [ ! -f "${PRED_FILE}" ]; then
        echo "[stage-A][fatal] predictions not found: ${PRED_FILE}" >&2
        exit 4
    fi
    echo "[stage-A]   predictions    : ${PRED_FILE}"

    cat >> "${RUNS_FRAGMENT}" <<EOF
  - lambda: ${LAM}
    build_jsonl: ${BUILD_VAL}
    predictions_jsonl: ${PRED_FILE}
EOF

    FIRST_LAMBDA=0
done

# ---- Build search config ----
SEARCH_CFG="${SEARCH_OUT}/search_config.yaml"

# Pull retrieval coefficients & top_k from the BASE_EXPERIMENT's resolved config.
read -r ALPHA_DENSE ALPHA_LEX ALPHA_BM25 TOP_K <<< "$(python - <<PY
from hydra import compose, initialize_config_dir
with initialize_config_dir(version_base=None, config_dir='${ROOT}/configs', job_name='stage_a'):
    cfg = compose(config_name='pipeline/default', overrides=['experiment=${BASE_EXPERIMENT}'])
r = cfg.build.retrieval
print(float(r.alpha_dense), float(r.alpha_lexical), float(r.alpha_bm25), int(r.top_k))
PY
)"
echo "[stage-A] retrieval recipe : alpha_dense=${ALPHA_DENSE} alpha_lex=${ALPHA_LEX} alpha_bm25=${ALPHA_BM25} top_k=${TOP_K}"

cat > "${SEARCH_CFG}" <<EOF
# Auto-generated by run_stage_a_search.sh on $(date -Iseconds)
chunk_mmr_path: ${CHUNK_MMR_PATH}
output_dir: ${SEARCH_OUT}

build_cfg:
  alpha_dense: ${ALPHA_DENSE}
  alpha_lexical: ${ALPHA_LEX}
  alpha_bm25: ${ALPHA_BM25}
  top_k: ${TOP_K}

fixed_baseline_lambda: 0.7

metric:
  w_acc: 1.0
  w_macro_f1: 0.0

search:
  theta_s: [0.2, 0.4, 0.6, 0.8]
  theta_r: [0.3, 0.4, 0.5, 0.6]
  lambda_low: [0.2, 0.3, 0.4]
  lambda_base: 0.7
  lambda_probe: 1.0
  gating_modes: [basic, conservative]
  epsilons: [0.02, 0.05, 0.10]
  pool_redundancy_topn: 32
  min_n_candidates_for_gate: 2
  relevance_floor_mode: mean_delta
  p_floor: 0.5

top_k_summary: 20

EOF
cat "${RUNS_FRAGMENT}" >> "${SEARCH_CFG}"

echo ""
echo "[stage-A] search config: ${SEARCH_CFG}"
echo "[stage-A] launching threshold search …"

python scripts/rl_mmr/search_sensitivity_thresholds.py --config "${SEARCH_CFG}"

echo ""
echo "[stage-A] done."
echo "[stage-A] CSV : ${SEARCH_OUT}/dev_grid.csv"
echo "[stage-A] JSON: ${SEARCH_OUT}/dev_grid.json"
