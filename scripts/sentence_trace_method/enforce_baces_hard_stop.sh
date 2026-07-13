#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
HARD_STOP_AT="${HARD_STOP_AT:-$(date +%F) 10:00:00}"
DATA_ROOT="${DATA_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_verifier_training_v0_1}"
LORA_SUFFIX="${LORA_SUFFIX:-__lora_ebs16_lr2em5_ep12_eval100_pat12_liarw}"
ORDINAL_CONFIG="${DATA_ROOT}/baces_exact__ordinal_replay_minmax5_10${LORA_SUFFIX}/train.resolved.yaml"
MATCHED_CONFIG="${DATA_ROOT}/baces_exact__matched_token_cap${LORA_SUFFIX}/train.resolved.yaml"
LOG_PATH="${LOG_PATH:-outputs/sentence_trace_method/queues/baces_dual_20260714/hard_stop.log}"

mkdir -p "$(dirname "$LOG_PATH")"
hard_epoch="$(date -d "$HARD_STOP_AT" +%s)"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z %z')" "$1" | tee -a "$LOG_PATH"
}

log "hard-stop watchdog armed for ${HARD_STOP_AT}"
while (( $(date +%s) < hard_epoch )); do
  remaining=$(( hard_epoch - $(date +%s) ))
  sleep_seconds=30
  if (( remaining < sleep_seconds )); then
    sleep_seconds="$remaining"
  fi
  (( sleep_seconds > 0 )) && sleep "$sleep_seconds"
done

"$PYTHON_BIN" - "$ORDINAL_CONFIG" "$MATCHED_CONFIG" <<'PY' | tee -a "$LOG_PATH"
import os
import signal
import sys
from pathlib import Path

patterns = tuple(sys.argv[1:])
victims = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    pid = int(proc.name)
    if pid in {os.getpid(), os.getppid()}:
        continue
    try:
        raw = (proc / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    if not any(pattern in command for pattern in patterns):
        continue
    if "sft.label_token_trainer" not in command and "accelerate launch" not in command:
        continue
    victims.append((pid, command))

for pid, command in victims:
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"hard-killed pid={pid}: {command}")
    except ProcessLookupError:
        pass
print(f"hard-stop matched process count={len(victims)}")
PY
log "hard-stop watchdog finished"

