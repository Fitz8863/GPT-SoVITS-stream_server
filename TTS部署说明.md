# 流式+克隆 TTS 服务部署说明(RTX 4060 Laptop 8GB)

> 部署日期:2026-08-29 | 环境:WSL2 (Ubuntu) + CUDA 12.6 + conda | 用途:个人/内部,中英混合,兼顾日语
>
> **其他设备/程序如何调用本服务 → 见配套的《[TTS接口文档](/home/hwj/AI/tts-server/TTS接口文档.md)》**(参数表、流式格式、curl/Python/JS/嵌入式示例、错误码、FAQ)
> 音色库目录:`/home/hwj/AI/tts-server/voices/`(参考音频统一放这里,内置示例 `demo_female_zh.wav`)

## 结论速览

| 指标 | GPT-SoVITS v2ProPlus ✅ 主力 | VoxCPM 1.5(备选) |
|---|---|---|
| 真流式首包 TTFB | **0.15~0.22s**(mode 3) | **0.10s** |
| RTF | **0.13~0.21**(中英日全部达标) | **0.60 ❌ 超标**(标准后端) |
| 零样本克隆 | ✅ 5 秒参考音频 | ✅ 参考音频+转写 |
| 语言 | zh(中英混)/ ja(日英混)/ en / ko / yue / auto | 30 语种+9 中文方言 |
| 许可 | 代码 MIT | 代码+权重 Apache-2.0 |

**定案:GPT-SoVITS v2ProPlus 作为常驻服务。** VoxCPM 首包更快但 RTF 0.6 超过 0.5
红线(其官方 0.15@4090 需要大卡;你的 4060 Laptop 上标准后端实测 0.6,如以后要用,
可尝试 nano-vllm-voxcpm 加速后端)。

**推荐参数:`streaming_mode=3`(整数!)+ `min_chunk_length=16`(默认即最优)。**
`min_chunk_length=8` 实测 RTF 反而恶化(0.28~0.42),不要调小。

## ⚠️ 最重要的一个坑

`api_v2.py` 的 `streaming_mode` 传 **布尔 true 不是真流式**!Python 里 `True==1`,
会落入旧版"整句分段返回"模式(每句合成完才返回,issue #1198/#1802 的差体验即此)。
真流式必须传**整数 2 或 3**:

- `streaming_mode: 3` — 首包最快(定长切块),实测首包 ~0.15s
- `streaming_mode: 2` — 质量优先(在"静音 token"处自然停顿切块),长文本首包 0.45~0.7s
- `streaming_mode: 1` / `true` — 旧版分段返回,**不是**子句级流式
- `streaming_mode: 0` / `false` — 非流式

## conda 环境

```bash
conda activate gpt-sovits    # GPT-SoVITS 主环境 (python 3.10, torch 2.7.0+cu126)
conda activate voxcpm        # VoxCPM 对比环境
```

> 配置了 tuna conda 镜像(~/.condarc);原 anaconda 安装器自带的
> `/home/hwj/anaconda3/.condarc`(指向被墙的 repo.anaconda.com)已移除(备份为 .condarc.bak.*),
> 这正是以前 conda 命令一直失败的原因。pip 走阿里云镜像(系统 pip.conf),共享缓存位于
> /home/hwj/AI/tts-server/pipcache(3.7G,可删,删后重装依赖需重新下载)。

## 启动 / 停止服务

```bash
bash /home/hwj/AI/tts-server/start.sh   # 一键启动全部 3 个服务(幂等,已在运行的自动跳过)
bash /home/hwj/AI/tts-server/stop.sh    # 一键停止全部
tail -f /home/hwj/AI/tts-server/api_v2.log   # 看 API 日志
```

| 启动后的服务 | 端口 | 说明 |
|---|---|---|
| TTS 合成 API | 9880 | v2ProPlus fp16,启动时自动加载模型(约 20~40s) |
| 流式测试网页 | 9872 | 快速试玩 |
| 音色管理后台 | 9873 | 音色注册/试音/默认参数 |

均绑定 0.0.0.0(Tailscale 内网可访问)。**不要用 `pkill -f api_v2` 停服务**
(会误杀包含关键字的 shell 自身),用 stop.sh 或按端口 kill。

浏览器打开(本机):**http://localhost:9872**
其他设备(Tailscale 内网):**http://100.95.19.17:9872**(本机 tailscale 节点 IP;API 同理 http://100.95.19.17:9880,两个服务均已绑 0.0.0.0)功能:输入文字点【开始合成】→ 声音立即边合成边播放;
可上传参考音频换音色、切语言(官方全部 11 种语言模式)、调速、切流式模式;
合成完显示首包延迟/时长统计。它只是 HTTP 调 9880 的 api_v2,不占额外显存。
(GPT-SoVITS 官方 inference_webui 播放**不是流式**且会重复加载模型,未采用。)

> ⚠️ WebUI 踩坑记录:gradio 对 **float32** numpy 分块会做**逐 chunk 峰值归一化**
> (`processing_utils.convert_to_16_bit_wav` 里 `data / data.abs().max()`),
> 停顿/低电平片段被放大成满幅 → 满耳杂音。必须 yield **int16** 数组(gradio 原样直通),
> 且用 ≥64KB 粒度读流、攒 ~0.5s 再 yield,避免前端解码调度过碎产生咔啦声。

## API 调用(零样本换音色)

```bash
curl -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" -d '{
  "text": "要合成的文本",
  "text_lang": "zh",                     // zh=中英混合 | ja=日英混合 | en | ko | yue | auto
  "ref_audio_path": "/path/to/ref.wav",  // 参考音频(服务端本地路径),建议 3~10s 干净人声
  "prompt_text": "参考音频的转写文本",      // 强烈建议提供,克隆相似度更高
  "prompt_lang": "zh",
  "media_type": "wav",
  "streaming_mode": 3,                   // 整数!
  "min_chunk_length": 16,
  "speed_factor": 1.0
}'
```

流式响应格式:**44 字节 WAV 头 + 后续每块裸 PCM**(32kHz/16bit/mono),客户端收到头即可起播。
也可用 `media_type: "raw"`(无头纯 PCM)或 `aac`/`ogg`。

其他端点:`/control?command=restart`(重启,慎用)、`/set_gpt_weights`、
`/set_sovits_weights`(热切换微调模型)。

## 客户端示例

`/home/hwj/AI/tts-server/bench/stream_play.py` — 发出请求即播放、边收边播(无声卡环境自动降级保存):

```bash
/home/hwj/anaconda3/envs/gpt-sovits/bin/python /home/hwj/AI/tts-server/bench/stream_play.py \
    --text "你好" --ref /path/to/ref.wav \
    --ref-text "参考转写" --text-lang zh --streaming-mode 3 --save out.wav
```

注意:WSL2 内无声卡(PortAudio 无输出设备),降级为保存文件;若在 Windows 侧或设备侧
(ESP32 等)消费流,直接按上面的"WAV 头 + 裸 PCM"格式处理即可。HTTP 服务绑定 127.0.0.1,
跨机访问已生效:三个服务均绑 0.0.0.0,Tailscale 设备直接访问 100.95.19.17。

## 实测基准(本机 RTX 4060 Laptop 8GB,fp16,conda 环境)

参考音频:`/home/hwj/AI/tts-server/tts-server/voices/demo_female_zh.wav`(3.48s,中文女声)
复现:`cd /home/hwj/AI/tts-server/bench && /home/hwj/anaconda3/envs/gpt-sovits/bin/python benchmark_tts.py --ref <ref.wav> --ref-text "..." --modes 2 3`

| case | 模式 | 首包 TTFB | 生成完 | 音频时长 | RTF |
|---|---|---|---|---|---|
| zh_short | 3 流式 | **0.147s** | 0.514s | 3.21s | 0.160 |
| zh_med(9s) | 3 流式 | 0.181s | 1.795s | 9.33s | 0.193 |
| en_short | 3 流式 | 0.218s | 0.850s | 4.12s | 0.206 |
| ja_short | 3 流式 | 0.214s | 0.839s | 4.00s | 0.210 |
| zh_short | 2 流式 | 0.698s | 0.700s | 2.86s | 0.245 |
| zh_med(9s) | 2 流式 | 0.452s | 1.407s | 9.30s | **0.151** |
| en_short | 2 流式 | 0.592s | 0.595s | 4.30s | 0.138 |
| ja_short | 2 流式 | 0.549s | 0.551s | 4.30s | 0.128 |
| zh_short | 0 非流式 | 1.39s | — | 3.34s | 0.418 |

- 首包 **~0.15s**:发出请求后 150ms 即可开始播放,满足"立马出声"
- RTF 0.13~0.21:远优于 RTF<0.5;边播边生成不会断流(生成速度≈播放速度 5~8 倍)
- 服务冷启动后首次请求需 10s 级初始化——start.sh 已内置 zh/en/ja 三语自动预热,启动完成即为热态
- 音频文件在 `/home/hwj/AI/tts-server/bench/bench_out/`,VoxCPM 的在 `bench_out_voxcpm/`,可回听对比

## VoxCPM 对比数据(同卡,标准 PyTorch 后端,inference_timesteps=10)

复现:`conda activate voxcpm && python /home/hwj/AI/tts-server/bench/bench_voxcpm.py --model /home/hwj/AI/tts-server/models/VoxCPM1.5`

| case | 首包 TTFB | 生成完 | RTF |
|---|---|---|---|
| zh_short | 0.102s | 2.107s | **0.599** |
| zh_med | 0.112s | 6.867s | 0.596 |
| en_short | 0.099s | 3.152s | 0.616 |
| ja_short | 0.103s | 2.783s | 0.600 |

首包 100ms 确实惊艳,但 RTF 0.6 在你的卡上超 RTF<0.5 要求 → 作为备选保留。

## 目录结构

```
/home/hwj/AI/tts-server/GPT-SoVITS/        # 主服务(含 pretrained_models 4.6G、G2PW、sv 等)
/home/hwj/AI/tts-server/start.sh
/home/hwj/AI/tts-server/stop.sh
/home/hwj/AI/tts-server/api_v2.log         # 服务日志
/home/hwj/AI/tts-server/bench/             # benchmark_tts.py / stream_play.py / bench_voxcpm.py / 结果
/home/hwj/AI/tts-server/VoxCPM/            # 对比方案源码
/home/hwj/AI/tts-server/models/VoxCPM1.5/  # VoxCPM 权重 (~2G)
/home/hwj/AI/tts-server/pipcache/          # pip 共享缓存 (3.7G, 可删)
/home/hwj/AI/tts-server/tts-server/voices/demo_female_zh.wav  # 测试参考音频(3.48s 中文女声)
/home/hwj/AI/tts-server/TTS部署说明.md      # 本文档
```

## 已知事项 / 后续

- 换音色 = 换 `ref_audio_path` + `prompt_text`;常用音色想要更高相似度,可用 WebUI
  (`conda activate gpt-sovits && cd /home/hwj/AI/tts-server/GPT-SoVITS && python webui.py`)
  做 1 分钟数据微调,再经 `/set_gpt_weights` + `/set_sovits_weights` 热加载
- 日语推理无需额外模型(open_jtalk 词典已装进 conda 环境;BERT 特征仅中文使用)
- 英文合成依赖 nltk_data(已装进 conda 环境);若重建环境需重新放置
- start.sh 已内置三语预热 + 保活心跳(10 分钟)+ **GPU keeper 线程**(voice_admin 内 5% 占空比微小矩阵乘,持续把 GPU 钉在 P0 满频,闲置后首包无降频惩罚,实测闲置 7 分钟 TTFB 仍为毫秒级;代价 GPU 常驻约 +8W)
