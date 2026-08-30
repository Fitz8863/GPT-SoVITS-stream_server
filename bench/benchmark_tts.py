#!/usr/bin/env python3
"""GPT-SoVITS api_v2 流式基准测试: 首包延迟(TTFB) / 总耗时 / RTF / 分块情况。

用法示例:
  python benchmark_tts.py --ref /path/to/ref.wav --ref-text "参考音频的转写文本" --ref-lang zh
  python benchmark_tts.py --ref ... --modes 0 2 3 --min-chunk 16 8
"""
import argparse
import json
import os
import sys
import time

import requests

SR = 32000  # GPT-SoVITS 输出采样率 32k
BYTES_PER_SAMPLE = 2  # int16
WAV_HEADER = 44

CASES = [
    ("zh_short", "你好，欢迎使用语音合成服务。", "zh"),
    ("zh_med", "今天下午三点半我们开产品评审会，请提前准备演示材料。会议大概一个小时，结束后我找你对一下下周的安排。", "zh"),
    ("en_short", "Hello, this is a streaming test of the speech synthesis service.", "en"),
    ("ja_short", "こんにちは、これは音声合成のストリーミングテストです。", "ja"),
]


def run_once(url, payload, save_path=None):
    t0 = time.perf_counter()
    first_byte = None
    first_audio = None
    header_buf = b""
    total = 0
    chunks = 0
    last_time = t0
    gaps = []  # 块间隔

    r = requests.post(f"{url}/tts", json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}

    for chunk in r.iter_content(chunk_size=4096):
        now = time.perf_counter()
        if chunks > 0:
            gaps.append(now - last_time)
        last_time = now
        chunks += 1
        if first_byte is None:
            first_byte = now - t0
        if first_audio is None:
            need = WAV_HEADER - len(header_buf)
            header_buf += chunk[:need]
            rest = chunk[need:]
            if len(header_buf) >= WAV_HEADER and rest:
                first_audio = now - t0
        elif first_audio is None and len(header_buf) >= WAV_HEADER:
            first_audio = now - t0
        total += len(chunk)
        if save_path:
            with open(save_path, "ab") as f:
                f.write(chunk)

    end = time.perf_counter() - t0
    audio_sec = max(total - WAV_HEADER, 0) / (SR * BYTES_PER_SAMPLE)
    gen_time = last_time - t0  # 最后一个字节到达 = 生成完成时刻
    return {
        "ttfb_s": round(first_byte, 3) if first_byte is not None else None,
        "first_audio_s": round(first_audio, 3) if first_audio is not None else None,
        "total_s": round(end, 3),
        "gen_complete_s": round(gen_time, 3),
        "audio_s": round(audio_sec, 3),
        "rtf": round(gen_time / audio_sec, 4) if audio_sec > 0 else None,
        "chunks": chunks,
        "bytes": total,
        "max_gap_s": round(max(gaps), 3) if gaps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9880")
    ap.add_argument("--ref", required=True, help="参考音频路径(服务端本地路径)")
    ap.add_argument("--ref-text", default="", help="参考音频转写文本")
    ap.add_argument("--ref-lang", default="zh")
    ap.add_argument("--modes", type=int, nargs="+", default=[2, 3], help="streaming_mode 列表, 0=非流式")
    ap.add_argument("--min-chunk", type=int, nargs="+", default=[16], help="min_chunk_length 列表")
    ap.add_argument("--cases", nargs="*", default=[c[0] for c in CASES])
    ap.add_argument("--out-dir", default="bench_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    case_map = {c[0]: c for c in CASES}

    print(f"参考音频: {args.ref} (lang={args.ref_lang})")
    print(f"{'case':<10} {'mode':<5} {'minchk':<7} {'TTFB(s)':<9} {'首音频(s)':<10} {'生成完(s)':<10} {'音频(s)':<9} {'RTF':<8} {'块数':<5} {'最大间隔':<8}")
    print("-" * 92)

    results = []
    for mode in args.modes:
        for mc in args.min_chunk:
            if mode in (0, 1) and mc != args.min_chunk[0]:
                continue  # min_chunk 只对流式 2/3 有意义, 每种模式只跑一次
            for case in args.cases:
                name, text, lang = case_map[case]
                payload = {
                    "text": text,
                    "text_lang": lang,
                    "ref_audio_path": args.ref,
                    "prompt_text": args.ref_text,
                    "prompt_lang": args.ref_lang,
                    "media_type": "wav",
                    "streaming_mode": mode,
                    "min_chunk_length": mc,
                    "text_split_method": "cut0",  # 不切分, 单句直出, 流式行为最可控
                }
                save = os.path.join(args.out_dir, f"{name}_m{mode}_mc{mc}.wav")
                if os.path.exists(save):
                    os.remove(save)
                try:
                    res = run_once(args.url, payload, save_path=save)
                except requests.RequestException as e:
                    res = {"error": str(e)[:200]}
                res.update({"case": name, "mode": mode, "min_chunk": mc})
                results.append(res)
                if "error" in res:
                    print(f"{name:<10} {mode:<5} {mc:<7} ERROR: {res['error'][:80]}")
                else:
                    print(f"{name:<10} {mode:<5} {mc:<7} {res['ttfb_s']:<9} {res['first_audio_s']:<10} "
                          f"{res['gen_complete_s']:<10} {res['audio_s']:<9} {res['rtf']:<8} {res['chunks']:<5} {res['max_gap_s']:<8}")

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out_dir}/results.json (音频文件同目录, 可回听对比)")


if __name__ == "__main__":
    main()
