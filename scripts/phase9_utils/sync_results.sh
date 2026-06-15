#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE="${REMOTE:-fenglin@yd.frp-ski.com:/home/fenglin/project/fact-checking/}"
SSH_PORT="${SSH_PORT:-16880}"
MODE="${MODE:-core}"
DRY_RUN_ARGS=()
SHOW_RULES=false
ALLOW_RESUME_STATE=false
RESUME_PATH=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/phase9_utils/sync_results.sh [--mode core|audit|resume-state] [--dry-run] [--show-rules]

Modes:
  core          Sync docs plus lightweight metrics/config/status artifacts.
  audit         core plus prediction JSONL and confusion-matrix PNG files.
  resume-state  Blocked by default; requires --allow-resume-state and --path outputs/.../latest_state.

Environment:
  REMOTE        Destination project root. Default: fenglin@yd.frp-ski.com:/home/fenglin/project/fact-checking/
  SSH_PORT      SSH port. Default: 16880
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#--mode=}"
            shift
            ;;
        --dry-run)
            DRY_RUN_ARGS+=(--dry-run)
            shift
            ;;
        --show-rules)
            SHOW_RULES=true
            shift
            ;;
        --remote)
            REMOTE="${2:?--remote requires a value}"
            shift 2
            ;;
        --remote=*)
            REMOTE="${1#--remote=}"
            shift
            ;;
        --ssh-port)
            SSH_PORT="${2:?--ssh-port requires a value}"
            shift 2
            ;;
        --ssh-port=*)
            SSH_PORT="${1#--ssh-port=}"
            shift
            ;;
        --allow-resume-state)
            ALLOW_RESUME_STATE=true
            shift
            ;;
        --path)
            RESUME_PATH="${2:?--path requires a value}"
            shift 2
            ;;
        --path=*)
            RESUME_PATH="${1#--path=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    core|audit|resume-state) ;;
    *)
        echo "Invalid mode: ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac

FILTERS=()

add_common_excludes() {
    FILTERS+=(
        "--exclude=outputs/cache/***"
        "--exclude=outputs/**/_cache_build/***"
        "--exclude=outputs/**/latest_state/***"
        "--exclude=outputs/**/checkpoint-*/***"
        "--exclude=outputs/**/_raw_sources/***"
        "--exclude=data/raw/***"
        "--exclude=data/processed/coverage/***"
        "--exclude=logs/***"
        "--exclude=wandb/***"
        "--exclude=multirun/***"
        "--exclude=swanlog/***"
    )
}

add_core_includes() {
    FILTERS+=(
        "--include=*/"
        "--include=docs/**/*.md"
        "--include=docs/**/*.csv"
        "--include=docs/**/*.jsonl"
        "--include=outputs/**/metrics.json"
        "--include=outputs/**/*metrics*.json"
        "--include=outputs/**/*summary*.json"
        "--include=outputs/**/*comparison*.json"
        "--include=outputs/**/manifest.json"
        "--include=outputs/**/*.resolved.yaml"
        "--include=outputs/**/training_complete.json"
        "--include=outputs/**/confusion_matrix.json"
    )
}

add_audit_includes() {
    FILTERS+=(
        "--include=outputs/**/*predictions*.jsonl"
        "--include=outputs/**/confusion_matrix.png"
    )
}

add_late_excludes() {
    FILTERS+=(
        "--exclude=*.safetensors"
        "--exclude=*.pt"
        "--exclude=*.pth"
        "--exclude=*.bin"
        "--exclude=*.ckpt"
        "--exclude=tokenizer.json"
        "--exclude=spm.model"
        "--exclude=sentencepiece.bpe.model"
        "--exclude=*.log"
        "--exclude=*.out"
        "--exclude=.env"
        "--exclude=*"
    )
}

if [[ "$MODE" == "resume-state" ]]; then
    if [[ "$ALLOW_RESUME_STATE" != true || -z "$RESUME_PATH" ]]; then
        echo "resume-state mode is blocked by default; pass --allow-resume-state and --path outputs/.../latest_state." >&2
        exit 2
    fi
    if [[ "$RESUME_PATH" = /* || "$RESUME_PATH" != outputs/* || "$RESUME_PATH" != *latest_state* ]]; then
        echo "resume-state path must be a relative outputs/.../latest_state path: ${RESUME_PATH}" >&2
        exit 2
    fi
    FILTERS+=(
        "--include=*/"
        "--include=${RESUME_PATH}/***"
        "--exclude=*"
    )
else
    add_common_excludes
    add_core_includes
    if [[ "$MODE" == "audit" ]]; then
        add_audit_includes
    fi
    add_late_excludes
fi

if [[ -z "$REMOTE" ]]; then
    echo "REMOTE is empty; set REMOTE to the destination project root." >&2
    exit 2
fi

if [[ "$SHOW_RULES" == true ]]; then
    echo "MODE=${MODE}"
    echo "REMOTE=${REMOTE}"
    echo "SSH_PORT=${SSH_PORT}"
    for rule in "${FILTERS[@]}"; do
        echo "$rule"
    done
    exit 0
fi

cd "$REPO_ROOT"

echo "Syncing result artifacts..."
echo "mode=${MODE}"
echo "remote=${REMOTE}"
rsync -avzP "${DRY_RUN_ARGS[@]}" -e "ssh -p ${SSH_PORT}" "${FILTERS[@]}" ./ "$REMOTE"
echo "Sync complete."
