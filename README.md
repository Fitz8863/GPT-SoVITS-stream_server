# GPT-SoVITS 流式服务端(stream_server)

基于 **GPT-SoVITS v2ProPlus** 的自托管流式语音合成服务:真流式(边合成边播放)、双模式音色体系
(零样本克隆 / 微调专属音色)、全功能管理后台,面向个人/局域网(Tailscale)多设备使用。

> 实测环境:RTX 4060 Laptop 8GB / WSL2 — 真流式首包 **~0.15s**,RTF **0.13~0.21**(中英日),
> 服务常驻显存 3.6GB。内置 GPU 保活,闲置任意时长首包无降频惩罚。

## ✨ 功能特性

- **真流式合成**:官方 `streaming_mode=2/3` 子句级流式(HTTP chunked,44 字节 WAV 头 + 裸 PCM),
  首包 ~0.15s,生成速度约为播放速度 5~8 倍,永不断流
- **克隆模式(零样本)**:上传 3~10s 参考音频即注册音色,设备按名调用 `{"voice":"id","text":"..."}`
- **专属模式(微调)**:上传自己微调的模型对 + 捆绑参考音频,注册为"专属音色包",固定音色长期使用
- **双模式自动路由**:调用克隆音色自动用 base 底模,调用专属音色自动热切换其微调模型,设备端无感
- **ASR 自动转写**:上传参考音频即自动识别台词和语言(SenseVoiceSmall,zh/en/ja/ko/yue)
- **OpenAI TTS 兼容端点**:`/v1/audio/speech`,现成客户端(mp3/wav/flac/opus/aac/pcm)零代码接入
- **管理后台**:双模式独立工作区(试音/注册/管理),流式试音含进阶采样参数,备份/恢复
- **11 种语言模式**:中(中英混)/日(日英混)/英/韩/粤/多语种自动/强制单语种
- **自愈运维**:一键启停(幂等)、三语预热、10 分钟心跳 + 看门狗自动拉起、GPU keeper 防降频、日志轮转
- **一键自检**:`selftest.sh` 13 项端点回归测试

## 🧱 架构

```
官方 GPT-SoVITS api_v2 (9880)   ← 流式合成引擎(官方源码,零侵入,可直接 git pull 升级)
    ↑ 纯转发,逐块透传
voice_admin (9873)              ← 自研:按名调用代理 + 管理后台(/ui)+ OpenAI 端点 + ASR + GPU keeper
webui_stream (9872)             ← 自研:克隆模式流式测试网页
```

## 🎭 两种使用模式(自动路由,设备无感)

| 模式 | 调用方式 | 引擎行为 |
|---|---|---|
| **克隆模式** | `{"voice":"音色ID","text":"..."}` | 自动确保 base 底模 + 音色库参考音频 |
| **专属模式** | `{"model":"微调模型ID","text":"..."}` | 自动热切换微调模型 + 用注册时捆绑的参考音频 |

切换含权重加载与自动预热(数秒);频繁交替会反复热切换,建议分批使用。

## 📦 部署指南(Linux/WSL2 + NVIDIA GPU)

### 0. 环境要求

- Linux 或 WSL2;NVIDIA GPU(≥6GB 显存,实测 4060 Laptop 8GB)+ CUDA 12.x 驱动
- conda(miniconda/anaconda 均可)、git;系统含 ffmpeg(官方 install.sh 也会安装)
- 国内网络建议配置 pip/conda 镜像(可选,详见 TTS部署说明.md)

### 1. 克隆仓库(本服务 + 官方引擎)

```bash
git clone https://github.com/Fitz8863/GPT-SoVITS-stream_server.git
cd GPT-SoVITS-stream_server
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git
```

### 2. 创建 conda 环境并安装依赖与预训练模型

```bash
conda create -n gpt-sovits python=3.10 -y
conda activate gpt-sovits
cd GPT-SoVITS
# 官方安装脚本: PyTorch CUDA + 全部依赖 + 预训练底模(国内可选 ModelScope 源)
bash install.sh --device CU126 --source ModelScope
```

### 3. 版本钉扎(必做!)

gradio 4.44 与新版 fastapi 不兼容(管理后台会 500):

```bash
pip install "fastapi==0.115.6" "starlette==0.41.3" sounddevice
```

### 4. 配置启动脚本(可选)

`start.sh` 按以下顺序自动解析 Python 解释器,一般无需修改:

1. 环境变量 `TTS_PYTHON`(显式指定)
2. conda 的 `gpt-sovits` 环境
3. 系统 `python3`

其他可配置项:`KEEPALIVE_INTERVAL`(心跳周期,秒,默认 600)、
`PYTORCH_CUDA_ALLOC_CONF`(已默认 `expandable_segments:True`)。

### 5. 启动与自检

```bash
bash start.sh      # 拉起 9880(API)+ 9872(测试页)+ 9873(管理后台),自动三语预热
bash selftest.sh   # 13 项端点回归自检
bash stop.sh       # 一键停止
```

| 服务 | 端口 | 用途 |
|---|---|---|
| TTS API(api_v2) | 9880 | 官方完整接口 |
| 管理后台 | 9873 | `/ui` 双模式工作区 + 按名调用 API + OpenAI 端点 |
| 流式测试页 | 9872 | 克隆模式网页试音 |

## 🔊 快速使用

### 1. 注册音色(管理后台)

浏览器打开 `http://<服务IP>:9873/ui`:

- **克隆音色**:「克隆模式 → 添加克隆音色」上传 3~10s 干净人声(超限自动拦截),
  转写自动 ASR 识别(可人工核对),填音色 ID 点注册
- **专属音色**:先在官方 webui 用你的数据集微调(底模选 v2ProPlus),然后
  「专属模式 → 添加专属音色包」上传 `GPT-*.ckpt` + `SoVITS-*.pth` + 一条该说话人参考音频,
  转写自动识别;点启用即热切换

### 2. API 调用

```bash
# 克隆模式
curl -X POST http://<服务IP>:9873/tts -H "Content-Type: application/json" \
  -d '{"voice":"音色ID","text":"你好","streaming_mode":3}' -o out.wav

# 专属模式(自动切换到微调模型)
curl -X POST http://<服务IP>:9873/tts -H "Content-Type: application/json" \
  -d '{"model":"anke_ft","text":"你好","streaming_mode":3}' -o out.wav

# OpenAI 兼容(现成客户端零代码接入)
curl -X POST http://<服务IP>:9873/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好","voice":"音色或模型ID","response_format":"mp3"}' -o out.mp3
```

流式响应格式:**44 字节 WAV 头 + 裸 PCM**(32000Hz/16bit/单声道),收到头即可起播。
完整参数表、Python/JS/嵌入式示例见 `TTS接口文档.md`。

## ⚙️ 配置说明

| 配置 | 位置 | 说明 |
|---|---|---|
| `TTS_PYTHON` | 环境变量 | 指定 Python 解释器(默认自动探测 conda 环境) |
| `KEEPALIVE_INTERVAL` | 环境变量 | 心跳周期秒数(默认 600) |
| `voices/registry.json` | 音色注册表 | 音色/模型包/`settings.default_voice`(设备省略 voice 时使用) |
| 端口 | start.sh 内 | 9880 / 9872 / 9873 |

## 📁 目录结构

```
├── start.sh / stop.sh          # 一键启停(幂等, 自动预热)
├── selftest.sh                 # 13 项端点回归自检
├── voice_admin.py              # 管理后台 + 按名调用代理 + OpenAI 端点 + ASR + 微调模型 + GPU keeper
├── webui_stream.py             # 克隆模式流式测试网页
├── keepalive.sh                # 心跳 + 看门狗自动拉起 + 日志轮转
├── bench/                      # 基准脚本与客户端示例(benchmark_tts / stream_play)
├── voices/                     # 音色库(registry.json + 音频, 不入库; 见 registry.example.json)
├── fine_tuned_models/          # 上传的微调模型专属音色包(权重+参考音频, 不入库)
├── GPT-SoVITS/                 # 官方引擎(自行克隆, 含底模; 零侵入可随时 git pull)
├── TTS接口文档.md               # 全参数/流式格式/多语言客户端示例/FAQ
└── TTS部署说明.md               # 本机部署实录与基准数据
```

## ⚠️ 已知的关键坑(踩过的)

| 坑 | 说明 |
|---|---|
| `streaming_mode` 必须传**整数 2/3** | 传 `true` 因 Python `True==1` 落入官方旧版"整句分段返回"(假流式) |
| `min_chunk_length=16` 即最优 | 调小到 8 实测 RTF 从 0.15 恶化到 0.3+ |
| gradio 4.44 × fastapi≥0.120 不兼容 | 必须钉 `fastapi==0.115.6 + starlette==0.41.3`,否则 /ui 全 500 |
| gradio 对 float32 音频分块做**逐块峰值归一化** | 网页播放必须喂 int16,否则低电平段被放大成噪音 |
| 参考音频硬限 **3~10 秒** | 引擎源码强制,超限中途抛异常;本服务注册/上传时已拦截 |
| 热切换需显存余量 | ASR 固定跑 CPU + expandable_segments 解决(本仓库已内置) |
| 英文合成需要 nltk_data,日语需要 open_jtalk 词典 | 官方 install.sh 会自动装 |
| BERT 特征仅中文使用 | en/ja/ko 无需额外模型 |
| 微调模型需基于 v2ProPlus 底模训练 | v3/v4 模型不支持流式,不兼容本服务 |

## 🔧 故障排查速查

| 现象 | 处理 |
|---|---|
| 播放全是噪音 | 客户端采样率处理错误:必须 32000Hz/16bit/单声道;浏览器 AudioContext 显式 `sampleRate: 32000` |
| 合成到一半复读/卡字 | 调大 `repetition_penalty`(1.6~1.8 起试) |
| 启用模型报 CUDA OOM | 显存不足:确认 ASR 在 CPU(本仓库已内置),关闭其他占卡程序 |
| 切换模型报"文件损坏" | 权重文件传输损坏:重新传输并用 md5 校验(注册时会自动校验拦截) |
| 管理后台 500 | 检查 fastapi/starlette 版本钉扎(见部署步骤 3) |
| 首包延迟大 | 正常不会发生(预热+保活);若服务重启过,等 start.sh 预热完成 |

## 📄 许可

本仓库自研部分 MIT。GPT-SoVITS 为 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
(MIT);其预训练模型使用条款见官方仓库。演示参考音频来自 CosyVoice 公开示例。
**请勿将克隆的他人声音用于违规用途。**
