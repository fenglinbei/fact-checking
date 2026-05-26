#!/usr/bin/env bash
# Serve the oracle-direct sentence verifier through vLLM's OpenAI-compatible API.
#
# Default use:
#   bash scripts/verifier/run_oracle_direct_vllm_server.sh
#
# Common overrides:
#   GPU_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 PORT=8000 \
#     bash scripts/verifier/run_oracle_direct_vllm_server.sh
#
# Background mode:
#   BACKGROUND=true WAIT_READY=true bash scripts/verifier/run_oracle_direct_vllm_server.sh
#   ACTION=status bash scripts/verifier/run_oracle_direct_vllm_server.sh
#   ACTION=stop bash scripts/verifier/run_oracle_direct_vllm_server.sh
#
# The verifier-proxy label builder expects:
#   VERIFIER_BASE_URL=http://127.0.0.1:8000/v1
#   VERIFIER_MODEL=fact-checking-sft
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
VERIFIER_CHECKPOINT="${VERIFIER_CHECKPOINT:-best}"
BASE_MODEL="${BASE_MODEL:-/data/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-fact-checking-sft}"

ACTION="${ACTION:-serve}"  # serve|check|cmd|status|stop
SERVE_MODE="${SERVE_MODE:-merged}"  # merged|lora
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

GPU_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
MERGE_DTYPE="${MERGE_DTYPE:-${DTYPE}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1025}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MERGE_LORA_CACHE_DIR="${MERGE_LORA_CACHE_DIR:-outputs/cache/merged_lora}"
FORCE_MERGE_REBUILD="${FORCE_MERGE_REBUILD:-false}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
CHECK_VLLM_VERSION="${CHECK_VLLM_VERSION:-true}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.8.5.post}"

BACKGROUND="${BACKGROUND:-false}"
WAIT_READY="${WAIT_READY:-false}"
LOG_DIR="${LOG_DIR:-outputs/logs/vllm}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/oracle_direct_vllm_server_${PORT}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/oracle_direct_vllm_server_${PORT}.pid.json}"

bool_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

fatal() {
  echo "[oracle-direct-vllm][fatal] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[oracle-direct-vllm] missing ${label}: ${path}" >&2
    echo "[oracle-direct-vllm] sync the verifier checkpoint before serving; no fallback verifier is used." >&2
    exit 1
  fi
}

infer_tp_from_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo "1"
    return
  fi
  python - "$devices" <<'PY'
import sys
devices = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(max(len(devices), 1))
PY
}

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf "%q " "${arg}"
  done
  printf "\n"
}

pid_from_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    return 1
  fi
  python - "$file" <<'PY'
import json
import sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    print(payload.get("pid", ""))
except Exception:
    print("")
PY
}

status_server() {
  echo "[oracle-direct-vllm] pid file : ${PID_FILE}"
  local pid=""
  pid="$(pid_from_file "${PID_FILE}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[oracle-direct-vllm] pid      : ${pid} (running)"
  elif [[ -n "${pid}" ]]; then
    echo "[oracle-direct-vllm] pid      : ${pid} (not running)"
  else
    echo "[oracle-direct-vllm] pid      : none"
  fi
  python - "${BASE_URL}" <<'PY' || true
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = [item.get("id") for item in payload.get("data", [])]
    print(f"[oracle-direct-vllm] /models  : reachable {model_ids}")
except Exception as exc:
    print(f"[oracle-direct-vllm] /models  : not reachable ({type(exc).__name__}: {exc})")
PY
}

stop_server() {
  local pid=""
  pid="$(pid_from_file "${PID_FILE}" || true)"
  if [[ -z "${pid}" ]]; then
    echo "[oracle-direct-vllm] no pid found in ${PID_FILE}"
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[oracle-direct-vllm] pid ${pid} is not running"
    rm -f "${PID_FILE}"
    return 0
  fi
  local cmdline=""
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" != *"vllm.entrypoints.openai.api_server"* ]]; then
    fatal "refusing to stop pid=${pid}; it does not look like a vLLM OpenAI server"
  fi
  echo "[oracle-direct-vllm] stopping pid=${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}

check_vllm_version() {
  if ! bool_true "${CHECK_VLLM_VERSION}"; then
    return 0
  fi
  python - "${EXPECTED_VLLM_VERSION}" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
try:
    version = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError as exc:
    raise SystemExit("[oracle-direct-vllm][fatal] vLLM is not installed in this environment.") from exc
print(f"[oracle-direct-vllm] vllm version : {version}", file=sys.stderr)
if version != expected:
    print(
        f"[oracle-direct-vllm][warn] expected vLLM {expected}; serving may still work, "
        "but this wrapper was written for that remote version.",
        file=sys.stderr,
    )
PY
}

wait_for_ready() {
  python - "${BASE_URL}" "${WAIT_SECONDS}" <<'PY'
import json
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
deadline = time.time() + int(float(sys.argv[2]))
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [item.get("id") for item in payload.get("data", [])]
        print(f"[oracle-direct-vllm] server ready: {model_ids}")
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)
raise TimeoutError(f"Timed out waiting for {base_url}/models: {last_error}")
PY
}

case "${ACTION}" in
  serve|check|cmd|status|stop) ;;
  *) fatal "ACTION must be one of serve|check|cmd|status|stop, got: ${ACTION}" ;;
esac

if [[ "${ACTION}" == "status" ]]; then
  status_server
  exit 0
fi

if [[ "${ACTION}" == "stop" ]]; then
  stop_server
  exit 0
fi

if [[ "${VERIFIER_CHECKPOINT}" == "final" ]]; then
  fatal "VERIFIER_CHECKPOINT=final is not allowed here; use best or checkpoint-600."
fi

CHECKPOINT_DIR="${DIRECT_VERIFIER_RUN_DIR}/${VERIFIER_CHECKPOINT}"
require_file "${CHECKPOINT_DIR}/adapter_config.json" "verifier adapter config"
require_file "${CHECKPOINT_DIR}/adapter_model.safetensors" "verifier adapter weights"
require_file "${CHECKPOINT_DIR}/tokenizer_config.json" "verifier tokenizer config"
require_file "${DIRECT_VERIFIER_RUN_DIR}/label_token_ce_meta.json" "label-token metadata"

if [[ "${BASE_MODEL}" = /* && ! -d "${BASE_MODEL}" ]]; then
  fatal "base model path not found: ${BASE_MODEL}"
fi

if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
  TENSOR_PARALLEL_SIZE="$(infer_tp_from_devices "${GPU_DEVICES}")"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

if [[ "${ACTION}" == "check" ]]; then
  check_vllm_version
  echo "[oracle-direct-vllm] verifier run : ${DIRECT_VERIFIER_RUN_DIR}"
  echo "[oracle-direct-vllm] checkpoint   : ${VERIFIER_CHECKPOINT}"
  echo "[oracle-direct-vllm] checkpoint files ok"
  echo "[oracle-direct-vllm] base model   : ${BASE_MODEL}"
  echo "[oracle-direct-vllm] serve mode   : ${SERVE_MODE}"
  echo "[oracle-direct-vllm] base url     : ${BASE_URL}"
  echo "[oracle-direct-vllm] served model : ${SERVED_MODEL_NAME}"
  echo "[oracle-direct-vllm] cuda devices : ${CUDA_VISIBLE_DEVICES:-}"
  echo "[oracle-direct-vllm] tp size      : ${TENSOR_PARALLEL_SIZE}"
  exit 0
fi

MODEL_PATH="${BASE_MODEL}"
TOKENIZER_PATH="${BASE_MODEL}"
LORA_ARGS=()

case "${SERVE_MODE}" in
  merged)
    check_vllm_version
    MERGE_ARGS=()
    if bool_true "${FORCE_MERGE_REBUILD}"; then
      MERGE_ARGS=(--force-rebuild)
    fi
    echo "[oracle-direct-vllm] preparing merged LoRA cache..." >&2
    MODEL_PATH="$(
      PYTHONPATH=src python scripts/verifier/prepare_lora_vllm_model.py \
        --base-model "${BASE_MODEL}" \
        --adapter-dir "${CHECKPOINT_DIR}" \
        --tokenizer-dir "${CHECKPOINT_DIR}" \
        --dtype "${MERGE_DTYPE}" \
        --merge-cache-dir "${MERGE_LORA_CACHE_DIR}" \
        "${MERGE_ARGS[@]}"
    )"
    TOKENIZER_PATH="${MODEL_PATH}"
    ;;
  lora)
    check_vllm_version
    LORA_ARGS=(
      --enable-lora
      --max-lora-rank "${MAX_LORA_RANK}"
      --lora-modules "${SERVED_MODEL_NAME}=${CHECKPOINT_DIR}"
    )
    ;;
  *)
    fatal "SERVE_MODE must be merged or lora, got: ${SERVE_MODE}"
    ;;
esac

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --tokenizer "${TOKENIZER_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
)
CMD+=("${LORA_ARGS[@]}")

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  CMD+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if bool_true "${ENABLE_PREFIX_CACHING}"; then
  CMD+=(--enable-prefix-caching)
fi
if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_VLLM_ARGS})
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[oracle-direct-vllm] verifier run : ${DIRECT_VERIFIER_RUN_DIR}"
echo "[oracle-direct-vllm] checkpoint   : ${VERIFIER_CHECKPOINT}"
echo "[oracle-direct-vllm] serve mode   : ${SERVE_MODE}"
echo "[oracle-direct-vllm] model        : ${MODEL_PATH}"
echo "[oracle-direct-vllm] tokenizer    : ${TOKENIZER_PATH}"
echo "[oracle-direct-vllm] base url     : ${BASE_URL}"
echo "[oracle-direct-vllm] served model : ${SERVED_MODEL_NAME}"
echo "[oracle-direct-vllm] cuda devices : ${CUDA_VISIBLE_DEVICES:-}"
echo "[oracle-direct-vllm] tp size      : ${TENSOR_PARALLEL_SIZE}"
echo "[oracle-direct-vllm] max len      : ${MAX_MODEL_LEN}"
echo "[oracle-direct-vllm] command:"
quote_cmd "${CMD[@]}"

if [[ "${ACTION}" == "cmd" ]]; then
  exit 0
fi

mkdir -p "${LOG_DIR}"

if bool_true "${BACKGROUND}"; then
  existing_pid="$(pid_from_file "${PID_FILE}" || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    fatal "pid file already points to a running process: ${PID_FILE} pid=${existing_pid}"
  fi
  {
    echo "$ $(quote_cmd "${CMD[@]}")"
    echo
  } > "${LOG_FILE}"
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  pid="$!"
  python - "${PID_FILE}" "${pid}" "${BASE_URL}" "${LOG_FILE}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "pid": int(sys.argv[2]),
    "base_url": sys.argv[3],
    "log_file": sys.argv[4],
    "served_model_name": sys.argv[5],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "[oracle-direct-vllm] started in background pid=${pid}"
  echo "[oracle-direct-vllm] log: ${LOG_FILE}"
  if bool_true "${WAIT_READY}"; then
    if ! wait_for_ready; then
      echo "[oracle-direct-vllm] startup log tail:" >&2
      tail -n 80 "${LOG_FILE}" >&2 || true
      exit 1
    fi
  fi
else
  exec "${CMD[@]}"
fi
