#!/bin/bash
# 保活守护: 三件事
#   1) 心跳: 每 INTERVAL 秒发一个微小合成请求(保持 GPU 高性能 + 验证链路健康)
#   2) 看门狗: 任一服务端口失联时自动调 start.sh 拉起(手动 stop 后有抑制标记, 不会自作主张)
#   3) 日志轮转: api_v2.log 超 10MB 自动切到 .1
# 由 start.sh 拉起, stop.sh 停止; 日志: keepalive.log(超 1MB 自动截断)
BASE="$(cd "$(dirname "$0")" && pwd)"
INTERVAL="${KEEPALIVE_INTERVAL:-600}"    # 秒, 默认 10 分钟
API="http://127.0.0.1:9880/tts"
REF="$BASE/voices/demo_female_zh.wav"

echo "$(date '+%F %T') keepalive started (interval ${INTERVAL}s, pid $$)" >> "$BASE/keepalive.log"

while true; do
    sleep "$INTERVAL"

    # --- 看门狗: 服务失联自动拉起(手动停止模式下跳过) ---
    HEAL=0
    for p in 9880 9872 9873; do
        ss -ltn 2>/dev/null | grep -q ":$p " || HEAL=1
    done
    if [ "$HEAL" = "1" ]; then
        if [ -f "$BASE/.manually_stopped" ]; then
            echo "$(date '+%F %T') watchdog: 服务未运行(手动停止模式, 跳过拉起)" >> "$BASE/keepalive.log"
        else
            echo "$(date '+%F %T') watchdog: 检测到服务失联, 自动拉起..." >> "$BASE/keepalive.log"
            bash "$BASE/start.sh" > /dev/null 2>&1
            echo "$(date '+%F %T') watchdog: 拉起完成" >> "$BASE/keepalive.log"
        fi
    fi

    # --- 日志轮转: api_v2.log 超 10MB 复制为 .1 并清空 ---
    if [ -f "$BASE/api_v2.log" ] && [ "$(stat -c%s "$BASE/api_v2.log" 2>/dev/null || echo 0)" -gt 10485760 ]; then
        cp "$BASE/api_v2.log" "$BASE/api_v2.log.1" && truncate -s 0 "$BASE/api_v2.log"
        echo "$(date '+%F %T') rotate: api_v2.log -> api_v2.log.1" >> "$BASE/keepalive.log"
    fi

    # --- 心跳 ---
    ss -ltn 2>/dev/null | grep -q ":9880 " || continue
    t=$(curl -s -o /dev/null -w "%{time_total}" -m 120 -X POST "$API" \
        -H "Content-Type: application/json" \
        -d '{"text":"keepalive","text_lang":"en","ref_audio_path":"'"$REF"'","prompt_text":"希望你以后能够做的比我还好呦。","prompt_lang":"zh","media_type":"raw","streaming_mode":3}')
    if [ -f "$BASE/keepalive.log" ] && [ "$(stat -c%s "$BASE/keepalive.log" 2>/dev/null || echo 0)" -gt 1048576 ]; then
        tail -n 500 "$BASE/keepalive.log" > "$BASE/keepalive.log.tmp" && mv "$BASE/keepalive.log.tmp" "$BASE/keepalive.log"
    fi
    echo "$(date '+%F %T') beat ok, took ${t}s" >> "$BASE/keepalive.log"
done
