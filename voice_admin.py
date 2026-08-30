#!/usr/bin/env python3
"""音色管理后台 + 按名调用代理(端口 9873)

- /ui      管理界面:音色注册/试听/删除、默认参数设置
- /tts     按名调用代理:{"voice":"音色ID","text":"..."} → 解析注册表 → 转发 api_v2
           兼容透传:带 ref_audio_path 的完整请求体原样转发
- /voices  注册表 API:GET 列表 / POST 注册(服务端本地路径)/ DELETE /voices/{id}

依赖本机 api_v2(9880)运行。启动: bash /home/hwj/AI/tts-server/start_voice_admin.sh
"""
import json
import re
import shutil
import subprocess
import threading
import time
import zipfile
from datetime import date
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

VOICES_DIR = Path("/home/hwj/AI/tts-server/voices")
BASE_DIR = Path(__file__).resolve().parent
GPTSOVITS_DIR = BASE_DIR / "GPT-SoVITS"
REG_PATH = VOICES_DIR / "registry.json"
API = "http://127.0.0.1:9880"
_lang_lock = threading.Lock()


def _resolve_model_path(p):
    """模型权重相对路径按 api_v2 工作目录(GPT-SoVITS/)解析, 返回绝对路径字符串。"""
    p = (p or "").strip()
    if p and not Path(p).is_absolute():
        p = str((GPTSOVITS_DIR / p).resolve())
    return p


# ---------------- 注册表 ----------------

def load_reg() -> dict:
    with _lang_lock:
        try:
            reg = json.loads(REG_PATH.read_text(encoding="utf-8"))
        except Exception:
            reg = {}
    reg.setdefault("voices", {})
    reg.setdefault("models", {})
    st = reg.setdefault("settings", {})
    st.setdefault("default_voice", "")
    st.setdefault("default_streaming_mode", 3)
    st.setdefault("default_speed", 1.0)
    st.setdefault("default_text_lang", "zh")
    st.setdefault("active_model", "base")
    return reg


def save_reg(reg: dict):
    with _lang_lock:
        tmp = REG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(REG_PATH)


# ---------------- FastAPI: 代理 + 注册表 API ----------------

app = FastAPI(title="TTS Voice Admin")


@app.get("/voices")
def voices_list():
    reg = load_reg()
    return {"settings": reg["settings"], "voices": reg["voices"]}


@app.post("/voices")
async def voices_add(body: dict):
    """从服务端本地路径注册音色。body: voice_id, file_path, prompt_text, prompt_lang, note"""
    ok, msg = _register(body.get("voice_id", ""), body.get("file_path", ""),
                        body.get("prompt_text", ""), body.get("prompt_lang", "zh"),
                        body.get("note", ""), copy_file=False)
    return JSONResponse(status_code=200 if ok else 400, content={"message": msg})


@app.delete("/voices/{voice_id}")
def voices_del(voice_id: str):
    reg = load_reg()
    if voice_id not in reg["voices"]:
        return JSONResponse(status_code=404, content={"message": f"voice '{voice_id}' not found"})
    del reg["voices"][voice_id]
    if reg["settings"].get("default_voice") == voice_id:
        reg["settings"]["default_voice"] = ""
    save_reg(reg)
    return {"message": f"deleted '{voice_id}' (音频文件保留在磁盘)"}


@app.post("/asr")
async def asr_endpoint(body: dict):
    """自动转写服务端本地音频: {"file_path": "..."} → {"text": "...", "prompt_lang": "zh"}"""
    fp = body.get("file_path", "")
    if not fp or not Path(fp).exists():
        return JSONResponse(status_code=400, content={"message": f"音频不存在: {fp}"})
    try:
        text, lang = await run_in_threadpool(transcribe_file, fp)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"ASR failed: {e}"})
    if not text:
        return JSONResponse(status_code=400, content={"message": "未识别到语音内容"})
    return {"text": text, "prompt_lang": lang}


# ---------------- 微调模型管理(注册/启用/切回底模) ----------------

# 官方 v2ProPlus 底模对(与 tts_infer_v2proplus.yaml 一致, 相对 api_v2 工作目录)
BASE_MODEL = {
    "gpt_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "sovits_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    "note": "官方 v2ProPlus 底模(零样本音色库基于此)",
}


def list_models():
    reg = load_reg()
    items = [{"id": "base", **BASE_MODEL, "active": reg["settings"]["active_model"] == "base"}]
    for mid, m in sorted(reg["models"].items()):
        items.append({"id": mid, **m, "active": reg["settings"]["active_model"] == mid})
    return items


def _register_model(mid, gpt_path, sovits_path, note):
    mid = re.sub(r'[\\/:*?"<>|\s]+', "_", (mid or "").strip())
    if not mid or mid == "base":
        return False, "模型ID 无效(不能为 base 或空)"
    gpt_path = _resolve_model_path(gpt_path)
    sovits_path = _resolve_model_path(sovits_path)
    if not gpt_path or not Path(gpt_path).exists():
        return False, f"GPT 权重不存在: {gpt_path}"
    if not sovits_path or not Path(sovits_path).exists():
        return False, f"SoVITS 权重不存在: {sovits_path}"
    reg = load_reg()
    if mid in reg["models"]:
        return False, f"模型 '{mid}' 已注册"
    reg["models"][mid] = {"gpt_path": gpt_path, "sovits_path": sovits_path,
                          "note": note or "", "created_at": str(date.today())}
    save_reg(reg)
    return True, f"模型 '{mid}' 已注册"


def activate_model(mid):
    """调 api_v2 官方热切换端点启用模型对, 成功后记录到注册表(重启自动恢复)。"""
    reg = load_reg()
    if mid == "base":
        m = BASE_MODEL
    else:
        m = reg["models"].get(mid)
        if not m:
            return False, f"模型 '{mid}' 未注册"
        m = dict(m)
        m["gpt_path"] = _resolve_model_path(m.get("gpt_path"))
        m["sovits_path"] = _resolve_model_path(m.get("sovits_path"))
        if not (Path(m["gpt_path"]).exists() and Path(m["sovits_path"]).exists()):
            return False, "权重文件不存在,请检查路径(可能被移动或删除)"
    r1 = requests.get(f"{API}/set_gpt_weights",
                      params={"weights_path": m["gpt_path"]}, timeout=180)
    r2 = requests.get(f"{API}/set_sovits_weights",
                      params={"weights_path": m["sovits_path"]}, timeout=180)
    if r1.status_code != 200 or r2.status_code != 200:
        return False, (f"切换失败: GPT({r1.status_code}) SoVITS({r2.status_code}) "
                       f"{r1.text[:120]} {r2.text[:120]}")
    reg["settings"]["active_model"] = mid
    save_reg(reg)
    # 切换后自动预热: 用微小请求预编译新权重路径, 避免切换后首个真实请求变慢
    try:
        warm_ref = VOICES_DIR / "demo_female_zh.wav"
        if warm_ref.exists():
            requests.post(f"{API}/tts", timeout=180, json={
                "text": "预热。", "text_lang": "zh",
                "ref_audio_path": str(warm_ref),
                "prompt_text": "希望你以后能够做的比我还好呦。",
                "prompt_lang": "zh", "media_type": "raw", "streaming_mode": 3})
    except Exception:
        pass
    return True, f"已启用模型 '{mid}'(已完成预热,重启服务也会自动恢复)"


@app.get("/models")
def models_list():
    return {"active_model": load_reg()["settings"]["active_model"], "models": list_models()}


@app.post("/models")
async def models_add(body: dict):
    ok, msg = _register_model(body.get("id", ""), body.get("gpt_path", ""),
                              body.get("sovits_path", ""), body.get("note", ""))
    return JSONResponse(status_code=200 if ok else 400, content={"message": msg})


@app.delete("/models/{mid}")
def models_del(mid: str):
    reg = load_reg()
    if mid not in reg["models"]:
        return JSONResponse(status_code=404, content={"message": f"模型 '{mid}' 未注册"})
    del reg["models"][mid]
    if reg["settings"].get("active_model") == mid:
        reg["settings"]["active_model"] = "base"
    save_reg(reg)
    return {"message": f"已删除注册 '{mid}'(权重文件保留)"}


@app.post("/models/{mid}/activate")
async def models_activate(mid: str):
    ok, msg = await run_in_threadpool(activate_model, mid)
    return JSONResponse(status_code=200 if ok else 400, content={"message": msg})


# ---------------- OpenAI 兼容端点(/v1/audio/speech) ----------------

@app.get("/v1/models")
async def openai_models():
    return {"object": "list",
            "data": [{"id": "gpt-sovits-v2proplus", "object": "model", "owned_by": "self"}]}


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    """OpenAI TTS 协议: input=文本, voice=音色ID(未知则用默认音色), speed, response_format(mp3/wav/flac/opus/aac/pcm)"""
    body = await request.json()
    text = (body.get("input") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"message": "input is required"})
    fmt = (body.get("response_format") or "mp3").lower()
    if fmt not in ("mp3", "wav", "flac", "opus", "aac", "pcm"):
        fmt = "mp3"
    reg = load_reg()
    st = reg["settings"]
    v = reg["voices"].get(body.get("voice") or "") or reg["voices"].get(st.get("default_voice") or "")
    if not v:
        return JSONResponse(status_code=400, content={"message": "no voice registered"})
    payload = {
        "text": text,
        "text_lang": body.get("text_lang") or st.get("default_text_lang", "zh"),
        "ref_audio_path": v["file"],
        "prompt_text": v.get("prompt_text", ""),
        "prompt_lang": v.get("prompt_lang", "zh"),
        "streaming_mode": 0, "speed_factor": float(body.get("speed", 1.0)),
        "media_type": "wav", "text_split_method": "cut5",
    }

    def _gen_wav():
        r = requests.post(f"{API}/tts", json=payload, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(r.text[:300])
        return r.content

    try:
        wav = await run_in_threadpool(_gen_wav)
    except RuntimeError as e:
        return JSONResponse(status_code=502, content={"message": str(e)})
    if len(wav) <= 44:
        return JSONResponse(status_code=502, content={"message": "empty audio from engine"})

    def _convert(target):
        src = f"/tmp/oai_{int(time.time()*1000)}.wav"
        dst = f"{src.rsplit('.',1)[0]}.{target}"
        Path(src).write_bytes(wav)
        cmd = ["ffmpeg", "-y", "-i", src]
        cmd += ["-c:a", "flac"] if target == "flac" else \
               ["-c:a", "libopus", "-b:a", "64k"] if target == "opus" else \
               ["-c:a", "aac", "-b:a", "96k"] if target == "aac" else \
               ["-c:a", "libmp3lame", "-b:a", "96k"]
        subprocess.run(cmd + [dst], check=True, capture_output=True)
        data = Path(dst).read_bytes()
        Path(src).unlink(missing_ok=True); Path(dst).unlink(missing_ok=True)
        return data

    if fmt == "wav":
        return Response(wav, media_type="audio/wav")
    if fmt == "pcm":
        return Response(wav[44:], media_type="audio/pcm")
    try:
        data = await run_in_threadpool(_convert, fmt)
    except Exception as e:
        return JSONResponse(status_code=502, content={"message": f"transcode failed: {e}"})
    return Response(data, media_type=f"audio/{fmt}")


# ---------------- 音色库备份 / 恢复 ----------------

def _make_backup() -> str:
    reg = load_reg()
    bdir = Path(VOICES_DIR) / "backups"
    bdir.mkdir(exist_ok=True)
    zpath = str(bdir / f"voices_backup_{date.today().strftime('%Y%m%d')}_{int(time.time())%100000}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for vid, v in reg["voices"].items():
            f = v.get("file", "")
            if f and Path(f).exists():
                z.write(f, arcname=f"{vid}.wav")
        z.writestr("registry.json", json.dumps(reg, ensure_ascii=False, indent=2))
    # 备份保留策略: 只留最近 10 份
    zlist = sorted(bdir.glob("voices_backup_*.zip"))
    for old in zlist[:-10]:
        old.unlink(missing_ok=True)
    return zpath


def _restore_zip(zpath, overwrite=False):
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    backup_reg = {"voices": {}}
    if "registry.json" in names:
        backup_reg = json.loads(z.read("registry.json"))
    reg = load_reg()
    restored, skipped = [], []
    for n in names:
        if not n.lower().endswith(".wav"):
            continue
        vid = Path(n).stem
        if vid in reg["voices"] and not overwrite:
            skipped.append(vid)
            continue
        dest = VOICES_DIR / f"{vid}.wav"
        with z.open(n) as fsrc, open(dest, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        entry = backup_reg.get("voices", {}).get(vid, {})
        reg["voices"][vid] = {
            "file": str(dest), "prompt_text": entry.get("prompt_text", ""),
            "prompt_lang": entry.get("prompt_lang", "zh"),
            "note": (entry.get("note", "") + " (恢复自备份)").strip(),
            "duration": entry.get("duration"),
            "created_at": entry.get("created_at", str(date.today())),
        }
        restored.append(vid)
    save_reg(reg)
    return restored, skipped


@app.patch("/voices/{voice_id}")
async def voices_edit_api(voice_id: str, body: dict):
    """修改音色: body 可含 voice_id(改名)/prompt_text/prompt_lang/note"""
    ok, msg, final_id = _edit_voice(voice_id, body.get("voice_id"), body.get("prompt_text"),
                                    body.get("prompt_lang"), body.get("note"))
    if not ok:
        return JSONResponse(status_code=400, content={"message": msg})
    return {"message": msg, "voice_id": final_id}


@app.post("/voices/{voice_id}/audio")
async def voices_audio_replace_api(voice_id: str, file: UploadFile, re_asr: bool = True):
    """替换音色的参考音频(multipart 上传), re_asr=true 自动重新识别转写"""
    tmp = f"/tmp/repl_{int(time.time()*1000)}.wav"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        ok, msg, text, lang = _replace_audio(voice_id, tmp, re_asr)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if not ok:
        return JSONResponse(status_code=400, content={"message": msg})
    return {"message": msg, "text": text, "prompt_lang": lang}


@app.get("/voices/backup")
def voices_backup_dl():
    zpath = _make_backup()
    return FileResponse(zpath, filename=Path(zpath).name, media_type="application/zip")


@app.post("/voices/restore")
async def voices_restore_api(file: UploadFile, overwrite: bool = False):
    tmp = f"/tmp/restore_{int(time.time())}.zip"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        restored, skipped = _restore_zip(tmp, overwrite)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"恢复失败: {e}"})
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {"restored": restored, "skipped": skipped}


@app.post("/tts")
async def tts_proxy(request: Request):
    body = await request.json()
    reg = load_reg()
    st = reg["settings"]

    if body.get("voice"):
        v = reg["voices"].get(body["voice"])
        if not v:
            return JSONResponse(status_code=400, content={
                "message": f"voice '{body['voice']}' not registered. GET /voices 查看可用音色"})
        if not body.get("text"):
            return JSONResponse(status_code=400, content={"message": "text is required"})
        payload = {
            "text": body["text"],
            "text_lang": body.get("text_lang") or st.get("default_text_lang", "zh"),
            "ref_audio_path": v["file"],
            "prompt_text": v.get("prompt_text", ""),
            "prompt_lang": v.get("prompt_lang", "zh"),
            "media_type": body.get("media_type", "wav"),
            "streaming_mode": int(body.get("streaming_mode", st.get("default_streaming_mode", 3))),
            "speed_factor": float(body.get("speed", st.get("default_speed", 1.0))),
            "min_chunk_length": int(body.get("min_chunk_length", 16)),
        }
        if body.get("text_split_method"):
            payload["text_split_method"] = body["text_split_method"]
        if body.get("seed", -1) != -1:
            payload["seed"] = int(body["seed"])
        for k in ("top_k", "top_p", "temperature", "repetition_penalty", "fragment_interval"):
            if body.get(k) is not None:
                payload[k] = body[k]
    else:
        payload = body  # 透传完整 api_v2 请求体

    def gen(r):
        for chunk in r.iter_content(chunk_size=65536):
            yield chunk

    r = await run_in_threadpool(
        lambda: requests.post(f"{API}/tts", json=payload, stream=True, timeout=300)
    )
    if r.status_code != 200:
        return JSONResponse(status_code=r.status_code, content={"message": r.text[:500]})
    return StreamingResponse(gen(r), media_type=r.headers.get("content-type", "audio/wav"))


# ---------------- ASR 自动转写(SenseVoiceSmall, 多语种 zh/en/ja/ko/yue) ----------------

_asr_model = None
_asr_lock = threading.RLock()


def get_asr_model():
    """懒加载 SenseVoiceSmall(首次调用自动从 ModelScope 下载, 之后常驻)。"""
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            import torch
            from funasr import AutoModel
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            _asr_model = AutoModel(model="iic/SenseVoiceSmall", device=dev,
                                   disable_update=True, disable_pbar=True)
        return _asr_model


def transcribe_file(path):
    """返回 (转写文本, 检测到的语言代码)。非 wav 格式先经 ffmpeg 转码。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频不存在: {path}")
    tmp = None
    try:
        if path.suffix.lower() != ".wav":
            tmp = f"/tmp/asr_{int(time.time()*1000)}.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", tmp],
                           check=True, capture_output=True)
            src = tmp
        else:
            src = str(path)
        model = get_asr_model()
        with _asr_lock:
            res = model.generate(input=src, cache={}, language="auto",
                                 use_itn=True, batch_size_s=60)
    finally:
        if tmp and Path(tmp).exists():
            Path(tmp).unlink()
    raw = res[0]["text"]
    mlang = re.search(r"<\|(zh|en|ja|ko|yue)\|>", raw)
    lang = mlang.group(1) if mlang else "zh"
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    text = rich_transcription_postprocess(raw).strip()
    # SenseVoice 会把情绪/事件标签转成 emoji(如 😊), 清掉避免污染转写文本
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\uFE0F\u200d]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, lang


def ui_autotranscribe(upload, srv_path):
    path = upload or srv_path
    if not path:
        raise gr.Error("请先上传音频或填写服务器本地路径")
    try:
        text, lang = transcribe_file(path)
    except Exception as e:
        raise gr.Error(f"识别失败: {e}")
    if not text:
        raise gr.Error("未识别到语音内容(音频可能是静音或纯噪声)")
    upd = gr.update(value=lang)
    return text, upd, f"🎙️ 已自动识别(prompt_lang=**{lang}**): {text}\n\n请人工核对转写后点击注册。"


# ---------------- 音色注册核心逻辑 ----------------

def _register(voice_id, audio_path, prompt_text, prompt_lang, note, copy_file=True):
    voice_id = re.sub(r'[\\/:*?"<>|\s]+', "_", (voice_id or "").strip())
    if not voice_id:
        return False, "音色ID不能为空(只允许字母数字下划线中文等)"
    if not audio_path or not Path(audio_path).exists():
        return False, f"音频文件不存在: {audio_path}"
    try:
        info = sf.info(str(audio_path))
        dur = info.duration
    except Exception as e:
        return False, f"无法读取音频: {e}"
    if dur < 3.05 or dur > 9.9:
        return False, (f"❌ 语音时长 {dur:.1f} 秒,要求 3~10 秒(引擎硬性限制,超限会被拒绝合成),"
                       f"请裁剪后重新上传")

    dest = VOICES_DIR / f"{voice_id}.wav"
    if copy_file:
        if Path(audio_path).resolve() != dest.resolve():
            shutil.copy(str(audio_path), str(dest))
        sf_to_check = dest
    else:
        sf_to_check = Path(audio_path)
        dest = Path(audio_path).resolve()

    reg = load_reg()
    reg["voices"][voice_id] = {
        "file": str(dest),
        "prompt_text": (prompt_text or "").strip(),
        "prompt_lang": prompt_lang or "zh",
        "note": note or "",
        "duration": round(dur, 2),
        "created_at": str(date.today()),
    }
    save_reg(reg)
    return True, (f"音色 '{voice_id}' 已注册({dur:.1f}s, prompt_lang={prompt_lang})"
                  + ("" if (prompt_text or "").strip() else "\n⚠️ 未填转写文本,建议补充以提高相似度"))


# ---------------- 流式试音(选音色 → 输文本 → 立即播放) ----------------

def tts_stream_play(voice, text, text_lang, mode, speed, seed, repetition_penalty,
                    top_k, top_p, temperature, text_split_method):
    reg = load_reg()
    st = reg["settings"]
    voice = voice or st.get("default_voice", "")
    v = reg["voices"].get(voice)
    if not v:
        raise gr.Error(f"音色 '{voice}' 未注册,请先在「注册音色」页添加并刷新")
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本")
    payload = {
        "text": text,
        "text_lang": text_lang or st.get("default_text_lang", "zh"),
        "ref_audio_path": v["file"],
        "prompt_text": v.get("prompt_text", ""),
        "prompt_lang": v.get("prompt_lang", "zh"),
        "media_type": "wav",
        "streaming_mode": int(mode),
        "speed_factor": float(speed),
        "min_chunk_length": 16,
        "repetition_penalty": float(repetition_penalty),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "text_split_method": text_split_method or "cut5",
    }
    if seed is not None and int(seed) != -1:
        payload["seed"] = int(seed)
    t0 = time.perf_counter()
    r = requests.post(f"{API}/tts", json=payload, stream=True, timeout=300)
    if r.status_code != 200:
        raise gr.Error(f"API {r.status_code}: {r.text[:200]}")

    buf = b""
    header_done = False
    sr = 32000
    first = None
    n_audio = 0
    pending = bytearray()
    MIN_YIELD = 32000  # ~0.5s @32k/16bit

    for chunk in r.iter_content(chunk_size=65536):
        buf += chunk
        if not header_done:
            if len(buf) < 44:
                continue
            sr = int.from_bytes(buf[24:28], "little")
            header_done = True
            first = time.perf_counter() - t0
            buf = buf[44:]
        pending += buf
        buf = b""
        n = len(pending) - (len(pending) % 2)
        if n < MIN_YIELD:
            continue
        arr = np.frombuffer(pending[:n], dtype=np.int16)  # int16 直通, 防 gradio 逐块归一化噪音
        del pending[:n]
        n_audio += n
        yield (sr, arr), ""

    if pending:
        n = len(pending) - (len(pending) % 2)
        if n > 0:
            n_audio += n
            yield (sr, np.frombuffer(pending[:n], dtype=np.int16)), ""

    if n_audio == 0:
        raise gr.Error("服务端没有返回任何音频。常见原因:参考音频时长超出 3~10s 硬性限制、"
                       "文件损坏或转写不匹配,详情查看 api_v2.log")

    stats = (f"音色 **{voice}** | 首包 **{first*1000:.0f} ms** | "
             f"时长 **{n_audio/(sr*2):.2f} s** | 总耗时 **{time.perf_counter()-t0:.2f} s** | {sr} Hz")
    yield None, stats


LANG_FULL = [
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


CALL_DOC = """## 📡 服务总览

| 服务 | 地址 | 用途 |
|---|---|---|
| **按名调用 API(设备接入首选)** | `http://100.95.19.17:9873/tts` | 按音色 ID 合成,支持流式 |
| **OpenAI 兼容端点** | `http://100.95.19.17:9873/v1/audio/speech` | 现成 TTS 客户端即插即用 |
| 直连 api_v2(官方) | `http://100.95.19.17:9880/tts` | 每次带参考音频路径的完整接口 |
| 流式测试页 | `http://100.95.19.17:9872` | 网页试听 |
| 完整接口文档(参数/示例/FAQ) | `/home/hwj/AI/tts-server/TTS接口文档.md` | 强烈建议阅读 |

服务启停:`bash /home/hwj/AI/tts-server/start.sh`(自动预热)/ `stop.sh`;内置 GPU 保活,闲置后首包无惩罚。接口**无鉴权**,请勿暴露公网。

---

## 1️⃣ 按名调用(其他设备推荐)

```bash
curl -X POST http://100.95.19.17:9873/tts \\
  -H "Content-Type: application/json" \\
  -d '{"voice":"demo_female_zh","text":"你好","streaming_mode":3}' \\
  -o out.wav
```

**可用参数**(全部可选,缺省用「⚙️ 默认参数」页保存的值):

| 参数 | 默认 | 说明 |
|---|---|---|
| `voice` | 默认音色 | 注册表中的音色 ID(本页「音色列表」可查) |
| `text` | —(必填) | 要合成的文本 |
| `text_lang` | zh | 11 种语言模式:zh/ja/en/ko/yue/auto/auto_yue/all_zh/all_ja/all_yue/all_ko |
| `streaming_mode` | 3 | **整数!** 3=极速首包(推荐) 2=质量优先 0=非流式;`true` 是假流式勿用 |
| `speed` | 1.0 | 语速 0.5~2.0 |
| `media_type` | wav | wav / raw(嵌入式推荐) / ogg / aac |
| `top_k` / `top_p` / `temperature` | 15 / 1.0 / 1.0 | 采样:越小越稳,越大越有情感 |
| `repetition_penalty` | 1.35 | **出现复读/卡字就调大**(最高 3.0) |
| `seed` | -1 | 固定数值可复现同一结果 |
| `text_split_method` | cut5 | 长文切句:cut0 不切 / cut5 按标点 / cut3 按句号… |
| `fragment_interval` | 0.3 | 句间静音秒数 |

**响应格式(流式)**:先收到 44 字节 WAV 头,之后每个 HTTP chunk 都是裸 PCM
(**32000Hz / 16bit / 单声道 / 小端**)。客户端读满 44 字节即可起播,后续块直接喂播放器。
带 `ref_audio_path` 的完整请求体会原样透传到 9880(兼容老代码)。

**错误处理**:所有错误统一 `HTTP 400 + {"message":"..."}`;成功才是 200 音频流。

---

## 2️⃣ OpenAI 兼容端点(现成客户端即插即用)

```bash
curl -X POST http://100.95.19.17:9873/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"model":"tts-1","input":"要合成的话","voice":"anke","response_format":"mp3","speed":1.0}' \\
  -o out.mp3
```

- `voice` 填音色 ID;**填未知值(如客户端默认 alloy)自动回退默认音色**,零配置可用
- `response_format`:mp3(默认)/ wav / flac / opus / aac / pcm,非流式整段返回
- Python SDK 接法:`base_url="http://100.95.19.17:9873/v1"`,model 任意填

---

## 3️⃣ 音色管理 API

```bash
curl http://100.95.19.17:9873/voices                              # 列出全部 + 默认参数
curl -X POST http://100.95.19.17:9873/voices -H "Content-Type: application/json" \\
  -d '{"voice_id":"xm","file_path":"/home/hwj/AI/tts-server/voices/xm.wav",
       "prompt_text":"转写","prompt_lang":"zh"}'                    # 注册(音频须在服务端)
curl -X PATCH http://100.95.19.17:9873/voices/xm -H "Content-Type: application/json" \\
  -d '{"voice_id":"xm2","note":"新备注"}'                           # 修改:改ID/转写/语言/备注
curl -X DELETE http://100.95.19.17:9873/voices/xm2                 # 删除(音频保留)
curl -X POST http://100.95.19.17:9873/asr \\
  -d '{"file_path":"/home/hwj/AI/tts-server/voices/x.wav"}'        # 自动转写+检测语言
curl http://100.95.19.17:9873/voices/backup -o backup.zip          # 备份(全部音色+注册表)
curl -X POST "http://100.95.19.17:9873/voices/restore?overwrite=true" -F "file=@backup.zip"
```

参考音频要求:**3~10 秒**(引擎硬限,上传时自动校验拦截)、干净人声无背景乐;
不知道台词用「注册音色」页的【🎙️ 自动识别转写】按钮。

---

## 4️⃣ 直连 9880(官方完整接口)

参数与按名调用一致但必填 `ref_audio_path`+`prompt_text`+`prompt_lang`,
另有 `batch_size`/`split_bucket` 等训练级参数(一般不动)。
其他端点:`/control?command=restart`(重启)、`/set_gpt_weights`、`/set_sovits_weights`(微调模型热切换)。

---

## ❓ 常见问题

| 现象 | 原因与解决 |
|---|---|
| 播放全是噪音 | 采样率处理错误:必须按 32000Hz/16bit/单声道 播放;浏览器 AudioContext 显式 `sampleRate: 32000` |
| 合成到一半复读/卡字 | 调大 `repetition_penalty`(1.6~1.8 起试) |
| 报"语音要求 3~10 秒" | 参考音频时长超引擎硬限,裁剪后重新上传 |
| 首包延迟大 | 正常情况不会发生(启动预热+GPU 保活);若服务重启过,等 start.sh 预热完成 |
| 想要更高音色相似度 | 用 1 分钟数据微调(官方 webui.py),导出后热挂载,详见部署文档 |"""


def _voice_choices_with_default():
    reg = load_reg()
    ids = sorted(reg["voices"].keys())
    dv = reg["settings"].get("default_voice") or (ids[0] if ids else None)
    return gr.update(choices=ids, value=dv)


# ---------------- 管理界面(gradio, 挂载在 /ui) ----------------

def ui_check_upload(fpath):
    """上传即校验时长: 不在 3~10s 内直接报错并清空, 不允许进入注册流程。"""
    if not fpath:
        return "", None
    try:
        dur = sf.info(str(fpath)).duration
    except Exception as e:
        raise gr.Error(f"无法读取音频文件: {e}")
    if dur < 3 or dur > 10:
        raise gr.Error(f"❌ 语音时长 {dur:.1f} 秒,要求 3~10 秒(引擎硬性限制),请裁剪后重新上传")
    return f"✓ 音频时长 {dur:.1f}s,符合 3~10s 要求", None


def ui_register(voice_id, upload, srv_path, prompt_text, prompt_lang, note):
    path = upload or srv_path
    ok, msg = _register(voice_id, path, prompt_text, prompt_lang, note, copy_file=bool(upload))
    upd = _voice_choices_with_default() if ok else gr.update()
    return msg, upd


def ui_list():
    reg = load_reg()
    rows = [[vid, v.get("duration", "?"), v.get("prompt_lang", ""), v.get("prompt_text", ""),
             v.get("file", ""), v.get("note", "")] for vid, v in sorted(reg["voices"].items())]
    return rows


def ui_pick(voice_id):
    reg = load_reg()
    v = reg["voices"].get(voice_id)
    return v["file"] if v else None


def ui_pick_full(voice_id):
    reg = load_reg()
    v = reg["voices"].get(voice_id)
    if not v:
        return None, "", gr.update(), "", ""
    return (v.get("file"), v.get("prompt_text", ""),
            gr.update(value=v.get("prompt_lang", "zh")),
            voice_id, v.get("note", ""))


def _edit_voice(voice_id, new_id, prompt_text, prompt_lang, note):
    """修改音色: 改ID(重命名)/转写/语言/备注。返回 (ok, msg, 最终ID)"""
    reg = load_reg()
    if voice_id not in reg["voices"]:
        return False, f"音色 '{voice_id}' 不存在", voice_id
    v = reg["voices"][voice_id]
    final_id = voice_id
    target = (new_id or "").strip()
    if target and target != voice_id:
        target = re.sub(r'[\\/:*?"<>|\s]+', "_", target)
        if target in reg["voices"]:
            return False, f"音色ID '{target}' 已存在,请换一个", voice_id
        old_file = Path(v["file"])
        # 仅当文件按标准命名放在音色库时才同步重命名文件
        if old_file.parent.resolve() == VOICES_DIR.resolve() and old_file.name == f"{voice_id}.wav":
            new_file = VOICES_DIR / f"{target}.wav"
            old_file.rename(new_file)
            v["file"] = str(new_file)
        reg["voices"][target] = v
        del reg["voices"][voice_id]
        if reg["settings"].get("default_voice") == voice_id:
            reg["settings"]["default_voice"] = target
        final_id = target
    v = reg["voices"][final_id]
    if prompt_text is not None:
        v["prompt_text"] = prompt_text.strip()
    if prompt_lang:
        v["prompt_lang"] = prompt_lang
    if note is not None:
        v["note"] = note
    save_reg(reg)
    return True, f"音色 '{final_id}' 已更新", final_id


def ui_save_edit(voice_id, new_id, prompt_text, prompt_lang, note):
    ok, msg, final_id = _edit_voice(voice_id, new_id, prompt_text, prompt_lang, note)
    if not ok:
        raise gr.Error(msg)
    ids = sorted(load_reg()["voices"].keys())
    return (msg, ui_list(),
            gr.update(choices=ids, value=final_id),
            _voice_choices_with_default())


def _replace_audio(voice_id, src_path, re_asr=True):
    """替换音色的参考音频(标准化到音色库, 可选 ASR 重新转写)。返回 (ok, msg, text, lang)"""
    reg = load_reg()
    if voice_id not in reg["voices"]:
        return False, f"音色 '{voice_id}' 不存在", None, None
    try:
        dur = sf.info(str(src_path)).duration
    except Exception as e:
        return False, f"无法读取音频: {e}", None, None
    if dur < 3.05 or dur > 9.9:
        return False, f"❌ 语音时长 {dur:.1f} 秒,要求 3~10 秒(引擎硬性限制)", None, None
    dest = VOICES_DIR / f"{voice_id}.wav"
    shutil.copy(str(src_path), str(dest))
    v = reg["voices"][voice_id]
    v["file"] = str(dest)
    text = lang = None
    if re_asr:
        try:
            text, lang = transcribe_file(str(dest))
            v["prompt_text"], v["prompt_lang"] = text, lang
        except Exception as e:
            text = f"(ASR 识别失败,转写未更新: {e})"
    v["duration"] = round(dur, 2)
    save_reg(reg)
    msg = f"音色 '{voice_id}' 参考音频已替换({dur:.1f}s)" + (f",转写已更新: {text}" if text else "")
    return True, msg, text, lang


def ui_replace_audio(voice_id, upload, re_asr):
    if not voice_id:
        raise gr.Error("请先在上方选择音色")
    if not upload:
        raise gr.Error("请先上传新的参考音频")
    ok, msg, text, lang = _replace_audio(voice_id, upload, bool(re_asr))
    if not ok:
        raise gr.Error(msg)
    upd = gr.update(value=lang) if lang else gr.update()
    return msg, str(VOICES_DIR / f"{voice_id}.wav"), text or "", upd, ui_list()


def ui_delete(voice_id):
    if not voice_id:
        return "请先在上方选择要删除的音色", ui_list()
    reg = load_reg()
    if voice_id in reg["voices"]:
        del reg["voices"][voice_id]
        if reg["settings"].get("default_voice") == voice_id:
            reg["settings"]["default_voice"] = ""
        save_reg(reg)
    return f"已删除 '{voice_id}'(文件保留)", ui_list()


def ui_save_settings(default_voice, default_streaming_mode, default_speed, default_text_lang):
    reg = load_reg()
    reg["settings"].update({
        "default_voice": default_voice or "",
        "default_streaming_mode": int(default_streaming_mode),
        "default_speed": float(default_speed),
        "default_text_lang": default_text_lang,
    })
    save_reg(reg)
    return f"已保存默认参数: {json.dumps(reg['settings'], ensure_ascii=False)}"


def ui_backup():
    zpath = _make_backup()
    size = Path(zpath).stat().st_size / 1024
    return zpath, f"✅ 备份完成: {Path(zpath).name}({size:.0f} KB,{len(load_reg()['voices'])} 个音色)"


def ui_restore(fpath, overwrite):
    if not fpath:
        raise gr.Error("请先上传备份包")
    restored, skipped = _restore_zip(fpath, bool(overwrite))
    return (f"✅ 恢复完成: 新增/覆盖 {len(restored)} 个音色 {restored};"
            f"跳过同名 {len(skipped)} 个 {skipped}"), ui_list(), _voice_choices_with_default()


def ui_active_model():
    mid = load_reg()["settings"].get("active_model", "base")
    tag = "(官方底模, 音色库克隆模式)" if mid == "base" else "(微调模型, 专属音色模式)"
    return f"**当前启用模型: `{mid}`** {tag}"


def ui_m_list():
    return [[m["id"], m.get("gpt_path", ""), m.get("sovits_path", ""),
             m.get("note", ""), "✅ 当前启用" if m["active"] else ""] for m in list_models()]


def ui_m_register(mid, gpt_path, sovits_path, note):
    ok, msg = _register_model(mid, gpt_path, sovits_path, note)
    if not ok:
        raise gr.Error(msg)
    ids = [m["id"] for m in list_models() if m["id"] != "base"]
    return msg, ui_m_list(), gr.update(choices=ids, value=mid)


def ui_m_activate(mid):
    if not mid:
        raise gr.Error("请先选择要启用的模型")
    ok, msg = activate_model(mid)
    if not ok:
        raise gr.Error(msg)
    return msg + "(请搭配该音色对应的参考音频使用)", ui_m_list(), ui_active_model()


def ui_m_del(mid):
    reg = load_reg()
    if not mid or mid not in reg["models"]:
        raise gr.Error("请先选择要删除的模型注册(不能删 base)")
    del reg["models"][mid]
    if reg["settings"].get("active_model") == mid:
        reg["settings"]["active_model"] = "base"
    save_reg(reg)
    return f"已删除注册 '{mid}'(权重文件保留)", ui_m_list()


def build_ui():
    st0 = load_reg()["settings"]
    with gr.Blocks(title="TTS 音色管理后台") as demo:
        gr.Markdown("## 🗂️ GPT-SoVITS 音色管理后台\n注册后的音色,任何设备可通过 "
                    "`POST :9873/tts {\"voice\":\"音色ID\",\"text\":\"...\"}` 直接调用。")
        with gr.Tab("🔊 流式试音"):
            with gr.Row():
                with gr.Column():
                    t_voice = gr.Dropdown(choices=[], value=None,
                                          label="选择音色", interactive=True)
                    t_model = gr.Markdown(ui_active_model())
                    t_text = gr.Textbox(
                        label="要合成的文本",
                        value="你好,这是管理后台的流式试音,点击开始后声音马上播放。",
                        lines=3)
                    with gr.Row():
                        t_lang = gr.Dropdown(choices=LANG_FULL,
                                             value=st0.get("default_text_lang", "zh"),
                                             label="文本语言")
                        t_mode = gr.Radio(choices=[("3 = 极速首包", 3), ("2 = 质量优先", 2)],
                                          value=st0.get("default_streaming_mode", 3), type="value",
                                          label="流式模式")
                    t_speed = gr.Slider(0.5, 2.0, value=st0.get("default_speed", 1.0),
                                        step=0.05, label="语速")
                    with gr.Accordion("⚙️ 进阶参数(采样/复现/长文本)", open=False):
                        with gr.Row():
                            a_seed = gr.Number(value=-1, precision=0,
                                               label="随机种子(-1=随机,固定值可复现)")
                            a_rep = gr.Slider(1.0, 3.0, value=1.35, step=0.05,
                                              label="重复惩罚(复读/卡字时调大)")
                        with gr.Row():
                            a_topk = gr.Slider(1, 100, value=15, step=1,
                                               label="Top-K(越小越稳,朗读建议 10~15)")
                            a_topp = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P")
                            a_temp = gr.Slider(0.1, 2.0, value=1.0, step=0.05,
                                               label="温度(高=更有情感但易不稳)")
                        a_cut = gr.Dropdown(
                            choices=[("按标点切(推荐长文)", "cut5"), ("不切,整句直出", "cut0"),
                                     ("按中文句号切", "cut3"), ("按英文句号切", "cut4"),
                                     ("凑四句一切", "cut1"), ("凑50字一切", "cut2")],
                            value="cut5", label="长文本切句方式")
                    t_btn = gr.Button("🔊 开始合成", variant="primary")
                with gr.Column():
                    t_audio = gr.Audio(label="流式播放(边合成边播)", streaming=True,
                                       autoplay=True, show_download_button=True)
                    t_stats = gr.Markdown("")
            t_btn.click(tts_stream_play,
                        [t_voice, t_text, t_lang, t_mode, t_speed,
                         a_seed, a_rep, a_topk, a_topp, a_temp, a_cut],
                        [t_audio, t_stats])

        with gr.Tab("➕ 注册音色"):
            with gr.Row():
                with gr.Column():
                    up = gr.Audio(sources=["upload"], type="filepath",
                                  label="上传参考音频(要求 3~10 秒,超限会被拒绝)")
                    up_status = gr.Markdown("")
                    srv = gr.Textbox(label="或填写服务器本地音频路径(与上传二选一)", lines=1)
                with gr.Column():
                    vid = gr.Textbox(label="音色ID(调用时用的名字,如 xiaoming)", lines=1)
                    ptext = gr.Textbox(label="参考音频逐字转写(强烈建议填写)", lines=2)
                    plang = gr.Dropdown(choices=LANG_FULL, value="zh", label="参考音频语言")
                    asr_btn = gr.Button("🎙️ 自动识别转写(从上方音频识别台词和语言)")
                    note = gr.Textbox(label="备注(可选)", lines=1)
            reg_btn = gr.Button("💾 注册音色", variant="primary")
            reg_out = gr.Markdown("")
            reg_btn.click(ui_register, [vid, up, srv, ptext, plang, note], [reg_out, t_voice])
            asr_btn.click(ui_autotranscribe, [up, srv], [ptext, plang, reg_out])
            up.upload(ui_check_upload, [up], [up_status, up])

        with gr.Tab("📋 音色列表 / 删除 / 试听"):
            lst = gr.Dataframe(
                headers=["音色ID", "时长s", "语言", "转写文本", "文件路径", "备注"],
                interactive=False, wrap=True)
            refresh_btn = gr.Button("🔄 刷新列表")
            with gr.Row():
                pick = gr.Dropdown(choices=[], label="选择音色", interactive=True)
                play = gr.Audio(label="试听参考音频", interactive=False)
            with gr.Row():
                e_audio = gr.Audio(sources=["upload"], type="filepath",
                                   label="替换选中音色的参考音频(可选,要求 3~10s;留空=不替换)")
                with gr.Column():
                    e_asr = gr.Checkbox(value=True, label="替换后自动重新识别转写")
                    repl_btn = gr.Button("🔄 替换参考音频", variant="secondary")
            with gr.Row():
                e_prompt = gr.Textbox(label="转写文本(可修改)", lines=2)
                with gr.Column():
                    e_lang = gr.Dropdown(choices=LANG_FULL, label="参考音频语言(可修改)")
                    e_newid = gr.Textbox(label="新音色ID(留空=不改名)", lines=1)
                    e_note = gr.Textbox(label="备注(可修改)", lines=1)
            with gr.Row():
                edit_btn = gr.Button("💾 保存修改", variant="primary")
                del_btn = gr.Button("🗑️ 删除选中音色", variant="stop")
            with gr.Row():
                edit_out = gr.Markdown("")
                del_out = gr.Markdown("")
            refresh_btn.click(lambda: (ui_list(), gr.update(choices=[r[0] for r in ui_list()])),
                              None, [lst, pick])
            pick.change(ui_pick_full, [pick], [play, e_prompt, e_lang, e_newid, e_note])
            repl_btn.click(ui_replace_audio, [pick, e_audio, e_asr],
                           [edit_out, play, e_prompt, e_lang, lst])
            edit_btn.click(ui_save_edit, [pick, e_newid, e_prompt, e_lang, e_note],
                           [edit_out, lst, pick, t_voice])
            del_btn.click(ui_delete, [pick], [del_out, lst])

        with gr.Tab("⚙️ 默认参数(按名调用时生效)"):
            st = load_reg()["settings"]
            _ids = sorted(load_reg()["voices"].keys())
            _dv = st.get("default_voice")
            d_voice = gr.Dropdown(choices=_ids,
                                  value=_dv if _dv in _ids else None,
                                  label="默认音色(voice 缺省时使用)")
            d_mode = gr.Radio(choices=[("3 = 极速首包(推荐)", 3), ("2 = 质量优先", 2)],
                              value=st.get("default_streaming_mode", 3), type="value",
                              label="默认流式模式")
            d_speed = gr.Slider(0.5, 2.0, value=st.get("default_speed", 1.0), step=0.05, label="默认语速")
            d_lang = gr.Dropdown(choices=LANG_FULL,
                                 value=st.get("default_text_lang", "zh"), label="默认文本语言")
            save_btn = gr.Button("💾 保存默认参数", variant="primary")
            st_out = gr.Markdown("")
            save_btn.click(ui_save_settings, [d_voice, d_mode, d_speed, d_lang], [st_out])

        with gr.Tab("🧠 微调模型"):
            gr.Markdown(
                "### 🧠 微调(专属音色)模型管理\n"
                "官方 webui 训练产物是**一对权重**(`GPT_SoVITS/logs/<实验名>/` 下的 `GPT-*.pth` 与 `SoVITS-*.pth`,"
                "建议先把要用的迭代拷到固定目录再注册)。\n"
                "注册后可随时一键启用/切回底模,重启服务自动恢复;**同一时间只有一个模型生效**,"
                "微调模型请搭配对应说话人的参考音频;切换请在无合成任务时进行。微调需基于 v2ProPlus 底模训练。")
            with gr.Row():
                with gr.Column():
                    m_id = gr.Textbox(label="模型ID(如 anke_ft)", lines=1)
                    m_gpt = gr.Textbox(label="GPT 权重路径(服务端, *.pth)", lines=1)
                    m_sovits = gr.Textbox(label="SoVITS 权重路径(服务端, *.pth)", lines=1)
                    m_note = gr.Textbox(label="备注", lines=1)
                with gr.Column():
                    m_tbl = gr.Dataframe(headers=["模型ID", "GPT权重", "SoVITS权重", "备注", "状态"],
                                         interactive=False, wrap=True)
                    m_pick = gr.Dropdown(choices=[], label="选择要启用的模型(含 base 底模)")
            with gr.Row():
                m_reg_btn = gr.Button("📥 注册模型")
                m_act_btn = gr.Button("🚀 启用选中模型", variant="primary")
                m_del_btn = gr.Button("🗑️ 删除选中注册")
            m_out = gr.Markdown("")
            m_reg_btn.click(ui_m_register, [m_id, m_gpt, m_sovits, m_note], [m_out, m_tbl, m_pick])
            m_act_btn.click(ui_m_activate, [m_pick], [m_out, m_tbl, t_model])
            m_del_btn.click(ui_m_del, [m_pick], [m_out, m_tbl])

        with gr.Tab("💾 备份 / 恢复"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⬇️ 备份\n打包全部音色音频 + 注册表,下载保存")
                    bk_btn = gr.Button("📦 生成备份包", variant="primary")
                    bk_file = gr.File(label="备份包(右键/点击下载)")
                    bk_out = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### ⬆️ 恢复\n上传备份包合并音色(默认跳过同名,勾选后覆盖)")
                    rs_file = gr.File(label="上传备份包", file_types=[".zip"])
                    rs_overwrite = gr.Checkbox(value=False, label="覆盖同名音色")
                    rs_btn = gr.Button("♻️ 恢复备份", variant="primary")
                    rs_out = gr.Markdown("")
            bk_btn.click(ui_backup, None, [bk_file, bk_out])
            rs_btn.click(ui_restore, [rs_file, rs_overwrite], [rs_out, lst, t_voice])

        with gr.Tab("📖 调用说明"):
            gr.Markdown(CALL_DOC)
        demo.load(lambda: (ui_list(), _voice_choices_with_default(), _voice_choices_with_default(),
                           ui_m_list(), gr.update(choices=[m["id"] for m in list_models()]),
                           ui_active_model()),
                  None, [lst, pick, t_voice, m_tbl, m_pick, t_model])
    return demo


demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn

    def _preload_asr():
        try:
            get_asr_model()
            print("[ASR] SenseVoiceSmall 已加载就绪", flush=True)
        except Exception as e:
            print(f"[ASR] 预加载失败(注册时点击会重试): {e}", flush=True)

    def _gpu_keeper():
        """持续微小 GPU 负载(约 5% 占空比), 防止降频到 P8 导致闲置后首包延迟增大。
        每 0.4s 做 ~20ms 的 fp16 矩阵乘, 平均功耗增加约 3~6W。"""
        try:
            import torch
            if not torch.cuda.is_available():
                print("[Keeper] 无 CUDA, GPU 保活线程不启动", flush=True)
                return
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            print("[Keeper] GPU 保活线程启动(5% 占空比)", flush=True)
            while True:
                for _ in range(400):
                    _ = torch.mm(a, b)
                torch.cuda.synchronize()
                time.sleep(0.4)
        except Exception as e:
            print(f"[Keeper] GPU 保活线程退出: {e}", flush=True)

    def _apply_active_model():
        """启动后自动恢复上次启用的微调模型(9880 端口就绪后执行)。"""
        try:
            mid = load_reg()["settings"].get("active_model", "base")
            if mid and mid != "base":
                time.sleep(3)
                ok, msg = activate_model(mid)
                print(f"[Model] 启动恢复微调模型 '{mid}': {msg}", flush=True)
        except Exception as e:
            print(f"[Model] 启动恢复微调模型失败: {e}", flush=True)

    threading.Thread(target=_preload_asr, daemon=True).start()
    threading.Thread(target=_gpu_keeper, daemon=True).start()
    threading.Thread(target=_apply_active_model, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=9873)
