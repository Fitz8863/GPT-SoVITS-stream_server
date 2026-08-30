#!/bin/bash
# 一键停止 TTS 全套服务(9880 / 9872 / 9873 / 保活心跳)
# 注意: 不要用 pkill -f api_v2 之类的方式(会误杀包含关键字的 shell 自身)
BASE="$(cd "$(dirname "$0")" && pwd)"
port_up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
pid_of()  { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1; }

echo "== 停止 TTS 服务 =="
date +%s > "$BASE/.manually_stopped"   # 抑制看门狗自动拉起

stop_one() {  # stop_one 端口 名称
    if port_up "$1"; then
        local pid; pid=$(pid_of "$1")
        if [ -n "$pid" ]; then
            kill "$pid" && echo "[停止] $2 ($1, pid $pid)"
        fi
    else
        echo "[跳过] $2 未在运行"
    fi
}

stop_one 9873 "音色管理后台"
stop_one 9872 "流式测试页"
stop_one 9880 "TTS API"

# 等待进程退出(最多 15s)
for _ in $(seq 1 15); do
    port_up 9880 || port_up 9872 || port_up 9873 || break
    sleep 1
done

# 残留强杀
for p in 9873 9872 9880; do
    if port_up "$p"; then
        local_pid=$(pid_of "$p")
        [ -n "$local_pid" ] && kill -9 "$local_pid" && echo "[强杀] 端口 $p (pid $local_pid)"
    fi
done

# 停止保活心跳
if [ -f "$BASE/keepalive.pid" ]; then
    kp=$(cat "$BASE/keepalive.pid" 2>/dev/null)
    if [ -n "$kp" ] && kill -0 "$kp" 2>/dev/null; then
        kill "$kp" && echo "[停止] 保活心跳 (pid $kp)"
    fi
    rm -f "$BASE/keepalive.pid"
fi
# 兜底: 按脚本路径清理残留心跳进程
pgrep -f "tts-server/keepalive\.sh" | while read p; do kill "$p" 2>/dev/null && echo "[清理] 残留心跳 pid $p"; done

echo "== 完成 =="
port_up 9880 || port_up 9872 || port_up 9873 || echo "所有 TTS 服务已停止"
