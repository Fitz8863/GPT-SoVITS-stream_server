#!/usr/bin/env python3
"""GPT-SoVITS 流式测试 WebUI — 输入文字, 边合成边播放(真正流式, 请求发出即出声)。

通过 HTTP 流式对接本机 api_v2 服务(9880), 不重复加载模型, 不占额外显存。
启动: conda activate gpt-sovits && python webui_stream.py
访问: http://localhost:9872  (Windows 浏览器直接开, WSL2 自动转发)
"""
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests

API = "http://127.0.0.1:9880"
DEFAULT_REF = str(Path(__file__).resolve().parent / "voices" / "demo_female_zh.wav")
DEFAULT_REF_TEXT = "希望你以后能够做的比我还好呦。"

LANG_CHOICES = [
    ("中文(中英混合)", "zh"),
    ("日语(日英混合)", "ja"),
    ("英语", "en"),
    ("韩语(韩英混合)", "ko"),
    ("粤语(粤英混合)", "yue"),
    ("多语种自动切分", "auto"),
    ("多语种自动(中文按粤语)", "auto_yue"),
    ("全部按中文识别", "all_zh"),
    ("全部按日文识别", "all_ja"),
    ("全部按粤语识别", "all_yue"),
    ("全部按韩文识别", "all_ko"),
]


def tts_stream(text, text_lang, ref_audio, prompt_text, prompt_lang, speed, mode):
    """生成器: 从 api_v2 流式取 WAV(44字节头+裸PCM), 逐块转 float32 交给 Gradio 播放。"""
    if not text.strip():
        raise gr.Error("请输入要合成的文本")
    if not ref_audio:
        raise gr.Error("请提供参考音频(上传或填写服务器本地路径)")

    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "media_type": "wav",
        "streaming_mode": int(mode),  # 必须 2/3 整数; true 会退化成假流式
        "min_chunk_length": 16,
        "speed_factor": float(speed),
    }
    t0 = time.perf_counter()
    r = requests.post(f"{API}/tts", json=payload, stream=True, timeout=300)
    if r.status_code != 200:
        raise gr.Error(f"API 返回 {r.status_code}: {r.text[:300]}")

    buf = b""
    header_done = False
    sr = 32000
    first_pkg = None
    n_audio = 0
    pending = bytearray()  # 未达到 yields 阈值的音频累积
    MIN_YIELD_BYTES = 32000  # ~0.5s @32k/16bit, 避免过碎的前端解码调度

    for chunk in r.iter_content(chunk_size=65536):
        buf += chunk
        if not header_done:
            if len(buf) < 44:
                continue
            if buf[0:4] != b"RIFF" or buf[36:40] != b"data":
                raise gr.Error("返回流不是标准 WAV(服务端可能报错, 查看服务日志)")
            sr = int.from_bytes(buf[24:28], "little")
            header_done = True
            first_pkg = time.perf_counter() - t0
            buf = buf[44:]
        pending += buf
        buf = b""
        # int16 对齐: 奇数字节留到下一轮
        n = len(pending) - (len(pending) % 2)
        if n < MIN_YIELD_BYTES:
            continue
        arr = np.frombuffer(pending[:n], dtype=np.int16)
        del pending[:n]
        n_audio += n
        # 直接喂 int16! gradio 对 float 输入会做逐 chunk 峰值归一化,
        # 导致停顿/低电平段被放大成噪音
        yield (sr, arr), ""

    if pending:
        n = len(pending) - (len(pending) % 2)
        if n > 0:
            arr = np.frombuffer(pending[:n], dtype=np.int16)
            n_audio += n
            yield (sr, arr), ""

    stats = (
        f"首包延迟 **{first_pkg*1000:.0f} ms** | 音频时长 **{n_audio/(sr*2):.2f} s** | "
        f"总耗时 **{time.perf_counter()-t0:.2f} s** | 采样率 {sr} Hz"
    )
    yield None, stats


def build():
    with gr.Blocks(title="GPT-SoVITS 流式测试") as demo:
        gr.Markdown(
            "## 🎧 GPT-SoVITS 流式 TTS 测试(v2ProPlus)\n"
            "输入文字点【开始合成】,声音会**立即**开始播放(边合成边播)。"
            "换音色 = 换参考音频 + 对应转写文本。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="要合成的文本",
                    value="你好,这是流式合成测试,点击开始后声音应该马上响起来。",
                    lines=4,
                )
                mode = gr.Radio(
                    choices=[("3 = 极速首包(推荐)", 3), ("2 = 质量优先", 2)],
                    value=3,
                    type="value",
                    label="流式模式",
                )
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速")
                btn = gr.Button("🔊 开始合成", variant="primary")
            with gr.Column(scale=2):
                lang_in = gr.Dropdown(choices=LANG_CHOICES, value="zh", label="文本语言")
                ref = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="参考音频(上传 3~10s 干净人声;不上传则用默认示例)",
                    value=DEFAULT_REF,
                )
                ref_path = gr.Textbox(
                    label="或参考音频的服务器本地路径", value=DEFAULT_REF, lines=1
                )
                prompt_text = gr.Textbox(
                    label="参考音频转写文本", value=DEFAULT_REF_TEXT, lines=2
                )
                prompt_lang = gr.Dropdown(choices=LANG_CHOICES, value="zh", label="参考音频语言")

        audio_out = gr.Audio(
            label="流式播放", streaming=True, autoplay=True, show_download_button=True
        )
        stats = gr.Markdown("")

        # 上传文件优先于路径
        def use_upload(up, path):
            return up if up else path

        ref.upload(fn=use_upload, inputs=[ref, ref_path], outputs=[ref_path])

        btn.click(
            fn=tts_stream,
            inputs=[text_in, lang_in, ref_path, prompt_text, prompt_lang, speed, mode],
            outputs=[audio_out, stats],
        )

    return demo


if __name__ == "__main__":
    demo = build()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=9872, inbrowser=False, show_error=True)
