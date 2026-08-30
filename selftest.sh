#!/bin/bash
# 一键自检: 回归测试 TTS 全部端点。用法: bash selftest.sh (需服务已启动)
# 会合成少量语音(约 30s), 全部通过输出 "N 通过 / 0 失败" 且退出码 0
BASE="$(cd "$(dirname "$0")" && pwd)"
PASS=0; FAIL=0
H9873="http://127.0.0.1:9873"
H9880="http://127.0.0.1:9880"
REF="$BASE/voices/demo_female_zh.wav"

ok()  { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "== TTS 服务自检 =="

for p in 9880 9872 9873; do
    ss -ltn 2>/dev/null | grep -q ":$p " && ok "端口 $p 监听" || bad "端口 $p 未监听(先跑 start.sh)"
done

# 1. 直连流式合成(9880)
code=$(curl -s -o /tmp/selftest_1.wav -w "%{http_code}:%{size_download}" -m 120 -X POST $H9880/tts \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"自检。\",\"text_lang\":\"zh\",\"ref_audio_path\":\"$REF\",\"prompt_text\":\"希望你以后能够做的比我还好呦。\",\"prompt_lang\":\"zh\",\"media_type\":\"wav\",\"streaming_mode\":3}")
[[ "$code" == 200:* && "${code#*:}" -gt 1000 ]] && ok "9880 直连流式合成" || bad "9880 直连流式合成 ($code)"

# 2. 按名调用(9873)
code=$(curl -s -o /tmp/selftest_2.wav -w "%{http_code}:%{size_download}" -m 120 -X POST $H9873/tts \
    -H "Content-Type: application/json" -d '{"voice":"demo_female_zh","text":"按名调用自检。","streaming_mode":3}')
[[ "$code" == 200:* && "${code#*:}" -gt 1000 ]] && ok "9873 按名调用" || bad "9873 按名调用 ($code)"

# 3. 注册表
n=$(curl -s -m 10 $H9873/voices | python3 -c "import json,sys; print(len(json.load(sys.stdin)['voices']))" 2>/dev/null)
[ -n "$n" ] && [ "$n" -ge 1 ] 2>/dev/null && ok "注册表可读($n 个音色)" || bad "注册表不可读"

# 4. PATCH 编辑(改备注后还原)
curl -s -m 10 $H9873/voices | python3 -c "import json,sys; print(json.load(sys.stdin)['voices']['demo_female_zh']['note'])" > /tmp/st_old.txt 2>/dev/null
r1=$(curl -s -X PATCH $H9873/voices/demo_female_zh -H "Content-Type: application/json" -d '{"note":"selftest"}' -m 10 | grep -c "已更新")
python3 -c "import json; print(json.dumps({'note': open('/tmp/st_old.txt').read().strip()}))" > /tmp/st_restore.json
r2=$(curl -s -X PATCH $H9873/voices/demo_female_zh -H "Content-Type: application/json" -d @/tmp/st_restore.json -m 10 | grep -c "已更新")
[ "$r1" = "1" ] && [ "$r2" = "1" ] && ok "PATCH 编辑(备注已还原)" || bad "PATCH 编辑"

# 5. ASR 自动转写
r=$(curl -s -X POST $H9873/asr -H "Content-Type: application/json" -d "{\"file_path\":\"$REF\"}" -m 180 | grep -c '"prompt_lang"')
[ "$r" = "1" ] && ok "ASR 自动转写" || bad "ASR 自动转写"

# 6. OpenAI 兼容端点(mp3)
code=$(curl -s -o /tmp/selftest_6.mp3 -w "%{http_code}:%{size_download}" -m 120 -X POST $H9873/v1/audio/speech \
    -H "Content-Type: application/json" -d '{"input":"openai 自检。","response_format":"mp3"}')
[[ "$code" == 200:* && "${code#*:}" -gt 1000 ]] && ok "OpenAI 兼容端点(mp3)" || bad "OpenAI 兼容端点 ($code)"

# 7. 备份 zip 完整性
code=$(curl -s -o /tmp/selftest_bk.zip -w "%{http_code}" -m 60 $H9873/voices/backup)
unzip -tq /tmp/selftest_bk.zip > /dev/null 2>&1 && [ "$code" = "200" ] && ok "备份 zip 完整" || bad "备份 zip ($code)"

# 8. 页面
c1=$(curl -sL -o /dev/null -w "%{http_code}" -m 15 $H9873/ui)
c2=$(curl -sL -o /dev/null -w "%{http_code}" -m 15 http://127.0.0.1:9872/)
[ "$c1" = "200" ] && ok "管理后台页面" || bad "管理后台页面 ($c1)"
[ "$c2" = "200" ] && ok "流式测试页" || bad "流式测试页 ($c2)"

rm -f /tmp/selftest_*.wav /tmp/selftest_6.mp3 /tmp/selftest_bk.zip /tmp/st_old.txt /tmp/st_restore.json
echo
echo "== 自检结果: $PASS 通过 / $FAIL 失败 =="
[ $FAIL -eq 0 ] || exit 1
