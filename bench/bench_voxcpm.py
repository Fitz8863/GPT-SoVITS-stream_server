#!/usr/bin/env python3
"""VoxCPM 流式基准(进程内): 首包延迟 / RTF, 与 GPT-SoVITS 基准同口径对比。

用法:
  python bench_voxcpm.py --model /path/to/VoxCPM1.5 [--ref wav] [--ref-text "..."]
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf

CASES = [
    ("zh_short", "你好，欢迎使用语音合成服务。"),
    ("zh_med", "今天下午三点半我们开产品评审会，请提前准备演示材料。会议大概一个小时，结束后我找你对一下下周的安排。"),
    ("en_short", "Hello, this is a streaming test of the speech synthesis service."),
    ("ja_short", "こんにちは、これは音声合成のストリーミングテストです。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="/home/hwj/AI/tts-server/tts-server/voices/demo_female_zh.wav")
    ap.add_argument("--ref-text", default="希望你以后能够做的比我还好呦。")
    ap.add_argument("--out-dir", default="bench_out_voxcpm")
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--cases", nargs="*", default=[c[0] for c in CASES])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    case_map = dict(CASES)

    from voxcpm import VoxCPM
    print(f"加载模型: {args.model}")
    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False)
    print(f"模型加载耗时 {time.perf_counter()-t0:.1f}s, 采样率 {model.tts_model.sample_rate}")

    sr = model.tts_model.sample_rate

    # 预热
    print("预热中...")
    for _ in model.generate_streaming(text="预热。", prompt_wav_path=args.ref, prompt_text=args.ref_text,
                                      inference_timesteps=args.timesteps):
        pass

    print(f"{'case':<10} {'TTFB(s)':<9} {'首音频(s)':<10} {'生成完(s)':<10} {'音频(s)':<9} {'RTF':<8} {'块数':<5}")
    print("-" * 70)
    for case in args.cases:
        text = case_map[case]
        chunks = []
        t0 = time.perf_counter()
        first_chunk = None
        for chunk in model.generate_streaming(
            text=text,
            prompt_wav_path=args.ref,
            prompt_text=args.ref_text,
            inference_timesteps=args.timesteps,
        ):
            now = time.perf_counter() - t0
            if first_chunk is None:
                first_chunk = now
            chunks.append(chunk)
        gen_complete = time.perf_counter() - t0
        wav = np.concatenate(chunks)
        audio_s = len(wav) / sr
        rtf = gen_complete / audio_s
        save = os.path.join(args.out_dir, f"{case}.wav")
        sf.write(save, wav, sr)
        print(f"{case:<10} {first_chunk:<9.3f} {'-':<10} {gen_complete:<10.3f} {audio_s:<9.3f} {rtf:<8.4f} {len(chunks):<5}")

    print(f"\n音频已保存到 {args.out_dir}/ 可回听对比")


if __name__ == "__main__":
    main()
