#!/bin/bash
# 一键启动 TTS 全套服务: 合成API(9880) + 流式测试页(9872) + 音色管理后台(9873)
# 幂等: 已在运行的服务自动跳过
BASE="$(cd "$(dirname "$0")" && pwd)"
# Python 解释器解析: TTS_PYTHON 环境变量 > conda gpt-sovits 环境 > 系统 python3
if [ -n "$TTS_PYTHON" ]; then
    PY="$TTS_PYTHON"
elif [ -x "$(conda info --base 2>/dev/null)/envs/gpt-sovits/bin/python" ]; then
    PY="$(conda info --base 2>/dev/null)/envs/gpt-sovits/bin/python"
elif [ -x "$HOME/anaconda3/envs/gpt-sovits/bin/python" ]; then
    PY="$HOME/anaconda3/envs/gpt-sovits/bin/python"
else
    PY="python3"
fi

port_up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
pid_of()  { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1; }
wait_port() {  # wait_port 端口 超时次数
    for _ in $(seq 1 "$2"); do
        port_up "$1" && return 0
        echo -n "."; sleep 2
    done
    return 1
}

echo "== TTS 服务启动 =="
rm -f "$BASE/.manually_stopped"   # 手动启动后解除看门狗抑制
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 降低热切换模型时的显存碎片

# 1) 核心合成 API
if port_up 9880; then
    echo "[跳过] TTS API 已在运行 (pid $(pid_of 9880))"
else
    echo -n "[启动] TTS API (9880, 加载模型约 20~40s)"
    (cd "$BASE/GPT-SoVITS" && nohup "$PY" api_v2.py -a 0.0.0.0 -p 9880 \
        -c GPT_SoVITS/configs/tts_infer_v2proplus.yaml > "$BASE/api_v2.log" 2>&1 &)
    if wait_port 9880 40; then echo " ✓ (pid $(pid_of 9880))"
    else echo " ✗ 启动失败, 查看日志: $BASE/api_v2.log"; exit 1; fi
fi

# 1.5) 预热: 消除各语言路径首次请求的初始化开销(冷启动首请求可达 10s+)
if port_up 9880; then
    REF="$BASE/voices/demo_female_zh.wav"
    PTEXT="希望你以后能够做的比我还好呦。"
    echo -n "[预热] 语言路径"
    for lt in "zh:你好。" "en:Hello, this is a warmup request." "ja:こんにちは。"; do
        L="${lt%%:*}"; T="${lt#*:}"
        code=$(curl -s -o /dev/null -w "%{http_code}" -m 180 -X POST http://127.0.0.1:9880/tts \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$T\",\"text_lang\":\"$L\",\"ref_audio_path\":\"$REF\",\"prompt_text\":\"$PTEXT\",\"prompt_lang\":\"zh\",\"media_type\":\"raw\",\"streaming_mode\":3}")
        echo -n " $L=$code"
    done
    echo " ✓ (此后首请求即为热态)"
fi

# 2) 流式测试页
if port_up 9872; then
    echo "[跳过] 流式测试页已在运行"
else
    echo -n "[启动] 流式测试页 (9872)"
    nohup "$PY" "$BASE/webui_stream.py" > "$BASE/webui.log" 2>&1 &
    if wait_port 9872 20; then echo " ✓"
    else echo " ✗ 查看日志: $BASE/webui.log"; fi
fi

# 3) 音色管理后台
if port_up 9873; then
    echo "[跳过] 音色管理后台已在运行"
else
    echo -n "[启动] 音色管理后台 (9873)"
    nohup "$PY" "$BASE/voice_admin.py" > "$BASE/voice_admin.log" 2>&1 &
    if wait_port 9873 20; then echo " ✓"
    else echo " ✗ 查看日志: $BASE/voice_admin.log"; fi
fi

# 4) 保活心跳(每 10 分钟一次微小合成, 保持 GPU 高性能, 闲置后首包依然最低延迟)
if [ -f "$BASE/keepalive.pid" ] && kill -0 "$(cat "$BASE/keepalive.pid" 2>/dev/null)" 2>/dev/null; then
    echo "[跳过] 保活心跳已在运行 (pid $(cat "$BASE/keepalive.pid"))"
else
    rm -f "$BASE/keepalive.pid"
    nohup bash "$BASE/keepalive.sh" > /dev/null 2>&1 &
    echo $! > "$BASE/keepalive.pid"
    echo "[启动] 保活心跳 (pid $(cat "$BASE/keepalive.pid"), 每 10 分钟一次微小合成)"
fi

echo
echo "== 状态 =="
port_up 9880 && echo "  TTS API        http://100.95.19.17:9880     ✓" || echo "  TTS API        ✗"
port_up 9872 && echo "  流式测试页     http://100.95.19.17:9872     ✓" || echo "  流式测试页     ✗"
port_up 9873 && echo "  音色管理后台   http://100.95.19.17:9873/ui  ✓" || echo "  音色管理后台   ✗"
