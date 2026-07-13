#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
DATA_ROOT="${DATA_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_verifier_training_v0_1}"
ORDINAL_CELL="baces_exact__ordinal_replay_minmax5_10"
MATCHED_CELL="baces_exact__matched_token_cap"
ORDINAL_SOURCE_ROOT="${ORDINAL_SOURCE_ROOT:-${DATA_ROOT}/${ORDINAL_CELL}}"
MATCHED_SOURCE_ROOT="${MATCHED_SOURCE_ROOT:-${DATA_ROOT}/${MATCHED_CELL}}"
LORA_SUFFIX="${LORA_SUFFIX:-__lora_ebs16_lr2em5_ep12_eval100_pat12_liarw}"
ORDINAL_LORA_ROOT="${ORDINAL_LORA_ROOT:-${ORDINAL_SOURCE_ROOT}${LORA_SUFFIX}}"
MATCHED_LORA_ROOT="${MATCHED_LORA_ROOT:-${MATCHED_SOURCE_ROOT}${LORA_SUFFIX}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga8.json}"
GRACEFUL_STOP_AT="${GRACEFUL_STOP_AT:-$(date +%F) 09:45:00}"
HARD_STOP_AT="${HARD_STOP_AT:-$(date +%F) 10:00:00}"
ORDINAL_GPUS="${ORDINAL_GPUS:-0,1}"
MATCHED_GPUS="${MATCHED_GPUS:-2,3}"
ORDINAL_PORT="${ORDINAL_PORT:-29521}"
MATCHED_PORT="${MATCHED_PORT:-29522}"
QUEUE_ID="${QUEUE_ID:-baces_dual_lora_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs/sentence_trace_method/queues}"
QUEUE_DIR="${QUEUE_ROOT}/${QUEUE_ID}"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_ROOT}/baces_dual_lora.lock"

mkdir -p "$QUEUE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another BACES dual-LoRA queue is already running. Lock: $LOCK_FILE" >&2
  exit 2
fi
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

graceful_epoch="$(date -d "$GRACEFUL_STOP_AT" +%s)"
hard_epoch="$(date -d "$HARD_STOP_AT" +%s)"
ordinal_pgid=""
matched_pgid=""
started_pgid=""
stop_signaled=false

timestamp() {
  date '+%F %T %Z %z'
}

status() {
  local message="$1"
  printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$STATUS_FILE"
}

process_group_alive() {
  local pgid="$1"
  [[ -n "$pgid" ]] && kill -0 "-${pgid}" 2>/dev/null
}

descendant_pids() {
  local root_pid="$1"
  local frontier="$root_pid"
  local all_pids=""
  local next_frontier children pid
  while [[ -n "$frontier" ]]; do
    next_frontier=""
    for pid in $frontier; do
      children="$(ps --ppid "$pid" -o pid= 2>/dev/null || true)"
      if [[ -n "$children" ]]; then
        all_pids="${all_pids} ${children}"
        next_frontier="${next_frontier} ${children}"
      fi
    done
    frontier="$next_frontier"
  done
  echo "$all_pids"
}

trainer_pids_for_tree() {
  local root_pid="$1"
  local pid ppid cmd parent_cmd
  for pid in $(descendant_pids "$root_pid"); do
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *"sft.label_token_trainer"* ]] || continue
    ppid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ' || true)"
    parent_cmd="$(ps -p "$ppid" -o args= 2>/dev/null || true)"
    [[ "$parent_cmd" == *"accelerate launch"* ]] && echo "$pid"
  done
}

trainer_pids_for_config() {
  local config_path="$1"
  ps -eo pid=,args= | awk -v config="$config_path" '
    index($0, "sft.label_token_trainer") && index($0, config) { print $1 }
  '
}

force_kill_task() {
  local name="$1"
  local pgid="$2"
  local config_path="$3"
  local pids=""
  if [[ -n "$pgid" ]]; then
    pids="$(descendant_pids "$pgid")"
  fi
  pids="${pids} $(trainer_pids_for_config "$config_path")"
  if [[ -n "${pids// /}" ]]; then
    status "hard-killing ${name} worker tree: ${pids}"
    kill -KILL $pids 2>/dev/null || true
  fi
  process_group_alive "$pgid" && kill -KILL "-${pgid}" 2>/dev/null || true
}

signal_trainers() {
  local name="$1"
  local pgid="$2"
  local trainer_pids
  if ! process_group_alive "$pgid"; then
    return 0
  fi
  trainer_pids="$(trainer_pids_for_tree "$pgid")"
  if [[ -n "$trainer_pids" ]]; then
    status "sending SIGTERM to ${name} trainer ranks; pgid=${pgid}; pids=$(echo "$trainer_pids" | tr '\n' ' ')"
    kill -TERM $trainer_pids 2>/dev/null || true
  else
    status "${name} has no trainer rank yet; sending SIGTERM to pgid=${pgid}"
    kill -TERM "-${pgid}" 2>/dev/null || true
  fi
}

stop_process_groups() {
  local reason="$1"
  if [[ "$stop_signaled" != "true" ]]; then
    stop_signaled=true
    status "graceful stop requested: ${reason}"
    signal_trainers "ordinal" "$ordinal_pgid"
    signal_trainers "matched-token" "$matched_pgid"
  fi
}

handle_stop_signal() {
  stop_process_groups "queue received stop signal"
}

trap handle_stop_signal TERM INT

validate_source_root() {
  local name="$1"
  local root="$2"
  "$PYTHON_BIN" - "$name" "$root" <<'PY'
import json
import sys
from pathlib import Path

import yaml

name, raw_root = sys.argv[1:]
root = Path(raw_root)
config_path = root / "train.resolved.yaml"
if not config_path.is_file():
    raise SystemExit(f"{name}: missing source config: {config_path}")
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {"train": 10065, "val": 1274, "test": 1251}
seen_paths = []
for split, expected_rows in expected.items():
    key = f"{split}_candidates"
    path = Path(str(cfg["data"][key]))
    if not path.is_file():
        raise SystemExit(f"{name}: missing {key}: {path}")
    seen_paths.append(path.resolve())
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_ids = row.get("prompt_input_ids")
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise SystemExit(f"{name}:{split}:{line_number}: empty prompt_input_ids")
            if row.get("was_truncated") is not False:
                raise SystemExit(f"{name}:{split}:{line_number}: was_truncated is not false")
            if row.get("evidence_text_truncated") is not False:
                raise SystemExit(
                    f"{name}:{split}:{line_number}: evidence_text_truncated is not false"
                )
            rows += 1
    if rows != expected_rows:
        raise SystemExit(
            f"{name}:{split}: row_count={rows}, expected={expected_rows}"
        )
if len(set(seen_paths)) != 3:
    raise SystemExit(f"{name}: train/val/test candidate paths are not distinct")
if bool(cfg.get("sft_train", {}).get("lora", {}).get("enabled")):
    raise SystemExit(f"{name}: source build config unexpectedly enables LoRA")
print(f"{name}: source data gate PASS")
PY
}

prepare_lora_root() {
  local name="$1"
  local source_root="$2"
  local lora_root="$3"
  local experiment_name
  experiment_name="$(basename "$lora_root")"
  "$PYTHON_BIN" scripts/sentence_trace_method/prepare_lora_config.py \
    --source-config "${source_root}/train.resolved.yaml" \
    --output-root "$lora_root" \
    --experiment-name "$experiment_name" \
    --swanlab-project fact-checking-baces-verifier-lora \
    --r 16 \
    --alpha 32 \
    --dropout 0.1 \
    --bias none \
    --deepspeed-config "$DEEPSPEED_CONFIG" \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --num-train-epochs 12 \
    --eval-steps 100 \
    --save-steps 100 \
    --early-stopping-patience 12 \
    --early-stopping-metric macro_f1

  "$PYTHON_BIN" - "$name" "$lora_root" "$DEEPSPEED_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

name, raw_root, expected_ds = sys.argv[1:]
root = Path(raw_root)
cfg = yaml.safe_load((root / "train.resolved.yaml").read_text(encoding="utf-8"))
train = cfg.get("sft_train", {})
assert train.get("lora", {}).get("enabled") is True, f"{name}: LoRA is disabled"
assert train.get("gradient_accumulation_steps") == 8, f"{name}: GA is not 8"
assert float(train.get("learning_rate")) == 2e-5, f"{name}: LR mismatch"
assert float(train.get("num_train_epochs")) == 12.0, f"{name}: epoch mismatch"
assert train.get("eval_steps") == 100 and train.get("save_steps") == 100
assert train.get("early_stopping_patience") == 12
assert cfg.get("train", {}).get("deepspeed_config") == expected_ds
for split in ("train", "val", "test"):
    path = Path(str(cfg["data"][f"{split}_candidates"]))
    assert path.resolve() == (root / "build" / f"build_{split}.jsonl").resolve()
print(f"{name}: LoRA config gate PASS")
PY
}

start_training() {
  local name="$1"
  local gpus="$2"
  local port="$3"
  local lora_root="$4"
  local runtime_root="outputs/cache/runtime/baces_dual/${QUEUE_ID}/${name}"
  local log_path="${QUEUE_DIR}/${name}.train.log"
  started_pgid=""
  mkdir -p "$runtime_root/xdg" "$runtime_root/triton" "$runtime_root/torchinductor"
  if [[ -f "${lora_root}/train/training_complete.json" ]] && \
     grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "${lora_root}/train/training_complete.json"; then
    status "${name} training is already complete; not launching"
    return 0
  fi
  status "starting ${name}; gpus=${gpus}; port=${port}; config=${lora_root}/train.resolved.yaml"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    XDG_CACHE_HOME="$runtime_root/xdg" \
    TRITON_CACHE_DIR="$runtime_root/triton" \
    TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor" \
    "$ACCELERATE_BIN" launch \
      --num_processes 2 \
      --num_machines 1 \
      --main_process_port "$port" \
      --mixed_precision bf16 \
      --use_deepspeed \
      --deepspeed_config_file "$DEEPSPEED_CONFIG" \
      -m sft.label_token_trainer \
      --config "${lora_root}/train.resolved.yaml" \
      >"$log_path" 2>&1 &
  local pgid="$!"
  status "${name} launched; pgid=${pgid}; log=${log_path}"
  started_pgid="$pgid"
}

validate_terminal_state() {
  local name="$1"
  local lora_root="$2"
  "$PYTHON_BIN" - "$name" "$lora_root" <<'PY'
import json
import sys
from pathlib import Path

name, raw_root = sys.argv[1:]
train = Path(raw_root) / "train"
complete = train / "training_complete.json"
latest = train / "latest_state" / "trainer_state.json"
if complete.is_file() and json.loads(complete.read_text(encoding="utf-8")).get("completed") is True:
    print(f"{name}: completed")
elif latest.is_file():
    state = json.loads(latest.read_text(encoding="utf-8"))
    if state.get("completed") is True:
        raise SystemExit(f"{name}: latest_state is unexpectedly marked completed")
    print(f"{name}: resumable at global_step={state.get('global_step')}")
else:
    raise SystemExit(f"{name}: neither training_complete nor resumable latest_state exists")
PY
}

status "queue_id=${QUEUE_ID}"
status "graceful_stop_at=${GRACEFUL_STOP_AT}; hard_stop_at=${HARD_STOP_AT}"
status "ordinal_source=${ORDINAL_SOURCE_ROOT}; matched_source=${MATCHED_SOURCE_ROOT}"

now_epoch="$(date +%s)"
if (( graceful_epoch <= now_epoch || hard_epoch <= graceful_epoch )); then
  status "invalid stop window; now=${now_epoch}, graceful=${graceful_epoch}, hard=${hard_epoch}"
  exit 2
fi

validate_source_root "ordinal" "$ORDINAL_SOURCE_ROOT"
validate_source_root "matched-token" "$MATCHED_SOURCE_ROOT"
prepare_lora_root "ordinal" "$ORDINAL_SOURCE_ROOT" "$ORDINAL_LORA_ROOT"
prepare_lora_root "matched-token" "$MATCHED_SOURCE_ROOT" "$MATCHED_LORA_ROOT"

start_training "ordinal" "$ORDINAL_GPUS" "$ORDINAL_PORT" "$ORDINAL_LORA_ROOT"
ordinal_pgid="$started_pgid"
start_training "matched-token" "$MATCHED_GPUS" "$MATCHED_PORT" "$MATCHED_LORA_ROOT"
matched_pgid="$started_pgid"
printf '%s\n' "$ordinal_pgid" >"${QUEUE_DIR}/ordinal.pgid"
printf '%s\n' "$matched_pgid" >"${QUEUE_DIR}/matched_token.pgid"

while process_group_alive "$ordinal_pgid" || process_group_alive "$matched_pgid"; do
  now_epoch="$(date +%s)"
  if (( now_epoch >= graceful_epoch )); then
    stop_process_groups "deadline reached at ${GRACEFUL_STOP_AT}"
  fi
  if [[ "$stop_signaled" == "true" ]]; then
    if process_group_alive "$ordinal_pgid" && [[ -z "$(trainer_pids_for_tree "$ordinal_pgid")" ]]; then
      kill -TERM "-${ordinal_pgid}" 2>/dev/null || true
    fi
    if process_group_alive "$matched_pgid" && [[ -z "$(trainer_pids_for_tree "$matched_pgid")" ]]; then
      kill -TERM "-${matched_pgid}" 2>/dev/null || true
    fi
  fi
  if (( now_epoch >= hard_epoch )); then
    status "hard stop reached; sending SIGKILL to any remaining process groups"
    force_kill_task \
      "ordinal" \
      "$ordinal_pgid" \
      "${ORDINAL_LORA_ROOT}/train.resolved.yaml"
    force_kill_task \
      "matched-token" \
      "$matched_pgid" \
      "${MATCHED_LORA_ROOT}/train.resolved.yaml"
    break
  fi
  sleep 20
done

[[ -z "$ordinal_pgid" ]] || wait "$ordinal_pgid" 2>/dev/null || true
[[ -z "$matched_pgid" ]] || wait "$matched_pgid" 2>/dev/null || true
validate_terminal_state "ordinal" "$ORDINAL_LORA_ROOT"
validate_terminal_state "matched-token" "$MATCHED_LORA_ROOT"
status "queue stopped cleanly; both runs are complete or resumable"
