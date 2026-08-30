# GPT-SoVITS 流式服务端(stream_server)

基于 **GPT-SoVITS v2ProPlus** 的自托管流式语音合成服务:真流式(边合成边播放)、零样本音色克隆、
带完整管理后台,面向个人/局域网(Tailscale)多设备使用。

> 实测环境:RTX 4060 Laptop 8GB / WSL2 — 真流式首包 **~0.15s**,RTF **0.13~0.21**(中英日),
> 模型常驻显存 3.6GB。含 GPU 保活,闲置任意时长首包无降频惩罚。

## ✨ 功能

- **真流式合成**:官方 `streaming_mode=2/3` 子句级流式(HTTP chunked,44 字节 WAV 头 + 裸 PCM)
- **音色注册中心**:上传 3~10s 参考音频即注册音色,设备按名调用 `{"voice":"id","text":"..."}`
- **ASR 自动转写**:注册时一键识别台词和语言(SenseVoiceSmall,zh/en/ja/ko/yue)
- **OpenAI TTS 兼容端点**:`/v1/audio/speech`,现成客户端(mp3/wav/flac/opus/aac/pcm)零代码接入
- **管理后台**:流式试音(含进阶采样参数)、音色增删改查/试听/替换音频、默认参数、备份/恢复
- **11 种语言模式**:中(中英混)/日(日英混)/英/韩/粤/多语种自动/强制单语种
- **自愈运维**:一键启停、三语预热、10 分钟心跳 + 看门狗自动拉起、GPU keeper 防降频、日志轮转
- **一键自检**:`selftest.sh` 回归测试全部端点

## 🧱 架构

```
官方 GPT-SoVITS api_v2 (9880)   ← 流式合成引擎(官方源码,零侵入)
    ↑ 纯转发,逐块透传
voice_admin (9873)              ← 自研:按名调用代理 + 管理后台(/ui)+ OpenAI 端点 + ASR
webui_stream (9872)             ← 自研:流式测试网页
```

## 📦 部署指南(Linux/WSL2 + NVIDIA GPU)

### 1. 克隆本仓库与官方 GPT-SoVITS

```bash
git clone git@github.com:Fitz8863/GPT-SoVITS-stream_server.git
cd GPT-SoVITS-stream_server
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git
```

### 2. 创建 conda 环境并安装依赖

```bash
conda create -n gpt-sovits python=3.10 -y
conda activate gpt-sovits
cd GPT-SoVITS
# 官方安装脚本: 装 PyTorch CUDA + 全部依赖 + 预训练模型(支持国内 ModelScope 源)
bash install.sh --device CU126 --source ModelScope
```

### 3. 关键版本钉扎(必做!)

gradio 4.44 与新版 fastapi 不兼容(管理后台会 500):

```bash
pip install "fastapi==0.115.6" "starlette==0.41.3" sounddevice
```

### 4. 修改启动脚本里的 Python 路径

`start.sh` / `stop.sh` 中 `PY=` 改为你的环境解释器路径(默认
`/home/hwj/anaconda3/envs/gpt-sovits/bin/python`,按需修改);`voice_admin.py` 顶部
`VOICES_DIR` 确认指向本仓库 `voices/` 目录。

### 5. 启动与自检

```bash
bash start.sh      # 拉起 9880(API)+ 9872(测试页)+ 9873(管理后台),自动三语预热
bash selftest.sh   # 12 项端点回归自检
bash stop.sh       # 一键停止
```

- 管理后台:`http://<主机IP>:9873/ui`
- 在「注册音色」上传 3~10s 干净人声(超限自动拦截),点【🎙️ 自动识别转写】补台词,即可按名调用

## ⚠️ 已知的关键坑(踩过的)

| 坑 | 说明 |
|---|---|
| `streaming_mode` 必须传**整数 2/3** | 传 `true` 因 Python `True==1` 落入官方旧版"整句分段返回"(假流式) |
| `min_chunk_length=16` 即最优 | 调小到 8 实测 RTF 从 0.15 恶化到 0.3+ |
| gradio 4.44 × fastapi≥0.120 不兼容 | 必须钉 `fastapi==0.115.6 + starlette==0.41.3`,否则 /ui 全 500 |
| gradio 对 float32 音频分块做**逐块峰值归一化** | 网页播放必须喂 int16,否则低电平段被放大成噪音 |
| 参考音频硬限 **3~10 秒** | 引擎源码强制,超限中途抛异常;本服务注册/上传时已拦截 |
| 英文合成需要 nltk_data,日语需要 open_jtalk 词典 | 官方 install.sh 会自动装 |
| BERT 特征仅中文使用 | en/ja/ko 无需额外模型 |

## 📁 目录结构

```
├── start.sh / stop.sh          # 一键启停(幂等, 自动预热)
├── selftest.sh                 # 12 项端点回归自检
├── voice_admin.py              # 管理后台 + 按名调用代理 + OpenAI 端点 + ASR + GPU keeper
├── webui_stream.py             # 流式测试网页
├── keepalive.sh                # 心跳 + 看门狗自动拉起 + 日志轮转
├── bench/                      # 基准与客户端示例(benchmark_tts / stream_play / bench_voxcpm)
├── voices/                     # 音色库(registry.json + 音频, 不入库; 见 registry.example.json)
├── TTS接口文档.md               # 全参数/流式格式/多语言客户端示例/FAQ
└── TTS部署说明.md               # 部署细节与基准数据
```

## 📄 许可

本仓库自研部分 MIT。GPT-SoVITS 为 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
(MIT);其预训练模型使用条款见官方仓库。请勿将克隆的他人声音用于违规用途。
