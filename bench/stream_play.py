#!/usr/bin/env python3
"""GPT-SoVITS 流式播放客户端: 发出请求即开始播放, 边收边播。

依赖: pip install sounddevice requests
用法:
  python stream_play.py --text "你好，这是流式播放测试。" --text-lang zh \
      --ref /path/to/ref.wav --ref-text "参考音频转写" --ref-lang zh \
      [--streaming-mode 3] [--save out.wav] [--speed 1.0]

无声卡环境(如 WSL2)会自动降级为仅保存模式。
"""
import argparse
import contextlib
import time
import wave

import numpy as np
import requests
import sounddevice as sd

WAV_HEADER = 44


def parse_wav_header(h: bytes):
    assert h[0:4] == b"RIFF" and h[8:12] == b"WAVE", f"不是 WAV 流: {h[:12]!r}"
    channels = int.from_bytes(h[22:24], "little")
    sr = int.from_bytes(h[24:28], "little")
    bits = int.from_bytes(h[34:36], "little")
    return sr, channels, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9880")
    ap.add_argument("--text", required=True)
    ap.add_argument("--text-lang", default="zh")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--ref-lang", default="zh")
    ap.add_argument("--streaming-mode", type=int, default=3, help="2=质量优先, 3=首包最快; 注意必须传整数")
    ap.add_argument("--min-chunk", type=int, default=16)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save", default="", help="可选, 保存为 wav 文件")
    args = ap.parse_args()

    payload = {
        "text": args.text,
        "text_lang": args.text_lang,
        "ref_audio_path": args.ref,
        "prompt_text": args.ref_text,
        "prompt_lang": args.ref_lang,
        "media_type": "wav",
        "streaming_mode": args.streaming_mode,
        "min_chunk_length": args.min_chunk,
        "speed_factor": args.speed,
    }

    t0 = time.perf_counter()
    r = requests.post(f"{args.url}/tts", json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        print("错误:", r.text)
        return

    # WSL2/无声卡环境下 PortAudio 查不到输出设备, 自动降级为仅保存
    try:
        sd.query_devices(kind="output")["name"]
        has_audio_device = True
    except Exception:
        has_audio_device = False
        print("未检测到音频输出设备, 降级为仅保存模式")

    it = r.iter_content(chunk_size=4096)
    buf = b""
    header_done = False
    stream = None
    wf = None
    n_received = 0
    sr, ch, bits = 32000, 1, 16

    with contextlib.ExitStack() as stack:
        try:
            for chunk in it:
                if not header_done:
                    # 第一阶段: 先凑齐 44 字节 WAV 头, 剩余部分即是音频
                    buf += chunk
                    if len(buf) < WAV_HEADER:
                        continue
                    sr, ch, bits = parse_wav_header(buf[:WAV_HEADER])
                    print(f"采样率 {sr} Hz / {ch}ch / {bits}bit, 首包延迟 {time.perf_counter()-t0:.3f}s")
                    if has_audio_device:
                        stream = stack.enter_context(
                            sd.OutputStream(samplerate=sr, channels=ch, dtype="int16")
                        )
                        stream.start()
                    if args.save or not has_audio_device:
                        wf = stack.enter_context(wave.open(args.save or "stream_out.wav", "wb"))
                        wf.setnchannels(ch)
                        wf.setsampwidth(bits // 8)
                        wf.setframerate(sr)
                    header_done = True
                    audio_bytes = buf[WAV_HEADER:]
                else:
                    audio_bytes = chunk

                if audio_bytes:
                    n_received += len(audio_bytes)
                    if stream is not None:
                        audio = np.frombuffer(audio_bytes, dtype=np.int16)
                        if audio.size:
                            stream.write(audio.reshape(-1, 1) if stream.channels > 1 else audio)
                    if wf is not None:
                        wf.writeframes(audio_bytes)
        finally:
            if stream is not None:
                stream.stop()
            print(f"完成: 收到 {n_received / (sr * bits // 8):.2f}s 音频, 总耗时 {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
