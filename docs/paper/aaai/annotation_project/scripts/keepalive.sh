#!/bin/bash
# Keep Label Studio and its reverse SSH tunnel alive.

set -u

PROJECT_DIR="/data/liaozijie/fact-checking/docs/paper/aaai/annotation_project"
LS_BIN="/data/liaozijie/conda/fc-annotation/bin/label-studio"
LS_DATA_DIR="$PROJECT_DIR/label_studio_data"
LS_PORT=8090
LS_LOG="$LS_DATA_DIR/service.log"
CSRF_TRUSTED_ORIGINS="https://fc.fenglin.pro"

TUNNEL_REMOTE_PORT=18080
TUNNEL_LOCAL_PORT=8090
SSH_HOST="dig"

CHECK_INTERVAL=30
TUNNEL_CHECK_INTERVAL=120

PID_FILE="/tmp/ls_keepalive.pid"
mkdir -p "$LS_DATA_DIR"

if [ -f "$LS_DATA_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$LS_DATA_DIR/.env"
    set +a
fi

echo $$ > "$PID_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_label_studio() {
    log "启动 Label Studio，数据目录: $LS_DATA_DIR"
    cd "$PROJECT_DIR" || return 1
    CSRF_TRUSTED_ORIGINS="$CSRF_TRUSTED_ORIGINS" \
    nohup "$LS_BIN" start --data-dir "$LS_DATA_DIR" \
        --host 127.0.0.1 --port "$LS_PORT" --enable-legacy-api-token \
        >> "$LS_LOG" 2>&1 &
    LS_PID=$!
    echo "$LS_PID" > /tmp/ls_labelstudio.pid
    for _ in $(seq 1 60); do
        if curl -sI --max-time 3 "http://127.0.0.1:$LS_PORT/" >/dev/null 2>&1; then
            log "Label Studio 端口 $LS_PORT 已就绪 (PID=$LS_PID)"
            return 0
        fi
        sleep 1
    done
    log "警告: Label Studio 60 秒内未就绪"
}

start_tunnel() {
    log "启动 autossh 隧道（本地 $TUNNEL_LOCAL_PORT -> 公网 $TUNNEL_REMOTE_PORT）"
    AUTOSSH_POLL=30 AUTOSSH_GATETIME=0 \
    nohup /usr/bin/autossh -M 0 -N \
        -R "${TUNNEL_REMOTE_PORT}:127.0.0.1:${TUNNEL_LOCAL_PORT}" "$SSH_HOST" \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
        >> "$LS_DATA_DIR/tunnel.log" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > /tmp/ls_tunnel.pid
    sleep 3
}

check_label_studio() {
    if [ -f /tmp/ls_labelstudio.pid ]; then
        local pid
        pid=$(cat /tmp/ls_labelstudio.pid 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    curl -sI --max-time 3 "http://127.0.0.1:$LS_PORT/" >/dev/null 2>&1
}

check_tunnel_process() {
    if [ -f /tmp/ls_tunnel.pid ]; then
        local pid
        pid=$(cat /tmp/ls_tunnel.pid 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

check_tunnel_connectivity() {
    local result
    result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" \
        "curl -sI --max-time 5 http://127.0.0.1:$TUNNEL_REMOTE_PORT/ 2>&1 | head -4" 2>/dev/null)
    echo "$result" | grep -q "WSGIServer"
}

log "===== 保活脚本启动 (PID=$$) ====="
last_tunnel_check=0

while true; do
    now=$(date +%s)
    if ! check_label_studio; then
        log "Label Studio 未运行，重启"
        start_label_studio
        sleep 10
        continue
    fi
    if ! check_tunnel_process; then
        log "autossh 不存在，重启隧道"
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" \
            "for pid in \$(lsof -ti :$TUNNEL_REMOTE_PORT 2>/dev/null); do kill -9 \$pid 2>/dev/null; done" \
            2>/dev/null
        start_tunnel
        sleep 5
        continue
    fi
    if [ $((now - last_tunnel_check)) -ge $TUNNEL_CHECK_INTERVAL ]; then
        if ! check_tunnel_connectivity; then
            log "隧道连通性失败，重建"
            kill "$(cat /tmp/ls_tunnel.pid 2>/dev/null)" 2>/dev/null
            sleep 2
            ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" \
                "for pid in \$(lsof -ti :$TUNNEL_REMOTE_PORT 2>/dev/null); do kill -9 \$pid 2>/dev/null; done" \
                2>/dev/null
            start_tunnel
            sleep 5
        else
            log "隧道连通性正常"
        fi
        last_tunnel_check=$now
    fi
    sleep "$CHECK_INTERVAL"
done
