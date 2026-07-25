# TGAI-GUI-NLP —— 从训练到部署，一条龙全包

<p align="center">
  <strong>PyTorch 原生大语言模型训练框架 — 桌面 GUI · 云端 CLI · WebUI · QQ 机器人 · TGAI GO 引擎 · 一键导出</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.3.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-red" alt="pytorch">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/arch-MoE%20%2B%20SwiGLU%20%2B%20RoPE-purple" alt="arch">
</p>

---

## 🤔 这 B 玩意儿是啥？

**TGAI** 是一个 **从零手搓的大语言模型**，基于 PyTorch 原生实现，不依赖 HuggingFace Transformers。

注意，不是那种 `from transformers import AutoModel` 一键调包的二道贩子——**这玩意儿连 Transformer Block 都是自己一行行写的**。

你给它喂对话数据，它就学着回你话。训完的模型可以：
- 在桌面 GUI 里聊（PyQt6）
- 在手机浏览器上聊（Flask WebUI）
- 挂到 QQ 上当赛博陪聊（NapCat 协议）
- 转换为 .TG 格式，在 ESP32 单片机上跑（TGAI GO 引擎）
- 导出 ONNX 塞进手机 APP 离线跑（TG CHAT）

一句话：**数据进去，模型出来，部署一条龙。**

---

## ✨ 吹牛逼环节（但确实是真的）

- 🖥️ **双界面训练** — 桌面 GUI 点点点就能训，命令行 `tgai-cli` 一行命令也能训。**不用写代码**。
- ☁️ **云端一键训练** — 自带 YAML 配置文件，扔到 AutoDL / 矩池云上直接跑，**GPU 训练教程写进 README 了你还要怎样**。
- 🧠 **MoE 混合专家架构** — 4 个专家、每次激活 2 个，SwiGLU 前馈、RoPE 旋转位置编码、Flash Attention、KV Cache 加速，该有的都有。
- 🎛️ **GUI 参数面板** — 滑动条调参，实时 loss 曲线，训练过程中随时改超参数，不用重开。
- 📱 **手机端部署** — 训完导出 ONNX 模型，扔到 TG CHAT APP 里就能离线推理。支持 INT8 量化加速。
- 🌐 **WebUI 远程操控** — Flask + Socket.IO，手机/平板打开浏览器就能训练和聊天。
- 🤖 **QQ 机器人** — 接 NapCat，自动回复群聊和私聊，还可以语音。
- 🌐 **REST API 服务** — 内建 API 服务器，支持 `/api/chat`、`/api/generate`，可接入任何前端。
- 🔧 **LoRA 微调** — 低秩适配器训练，极低显存下微调已有模型。
- 🎛️ **词频偏置** — 热加载 `word_bias_config.json`，控制特定词的生成概率。
- 📊 **数据工具链** — 训练数据统计分析、清洗去重、JSONL 合并，反正就是伺候你那个屎山数据集。
- ⚡ **AMP 混合精度** — 自动混合精度训练，显卡不冒烟的同时快一截。
- 💾 **智能缓存** — 数据集 mmap 缓存，几 GB 的 JSONL 也不用把内存撑爆。
- 🔄 **断点续训** — 训练崩了？重开就接着跑，checkpoint 存得比你还勤快。
- 🚀 **TGAI GO 引擎联动** — 一键将 PyTorch 模型转换为 .TG 格式，部署到 ESP32 等嵌入式设备。支持 FP32/FP16/Q8_0/Q4_0 多种量化。
- 🔄 **GGUF 模型兼容** — 支持 Llama/Mistral/Qwen2/Phi-3 等社区模型 GGUF → .TG 转换。
- 🏗️ **模型嫁接** — V1 模型权重可膨胀到更大 V2 架构，无需从头训练。

---

## 📦 快速开始（3 秒装不完，你忍忍）

### 环境要求

| 项 | 最低配置 | 推荐配置 |
|----|----------|----------|
| Python | 3.9+ | 3.11+ |
| 内存 | 8 GB | 16+ GB |
| 显卡（可选） | GTX 1060 6GB | RTX 3060 12GB+ |
| 操作系统 | Windows / Linux | Ubuntu 22.04 |

### 安装

```bash
git clone https://github.com/JXW666NB/TGAI-GUI-NLP.git
cd TGAI-GUI-NLP

# CPU 训练
pip install -r requirements_cpu.txt

# 或者 GPU 训练（CUDA 12.4）
pip install -r requirements_gpu.txt
```

### 快速体验

```bash
# 启动桌面 GUI（Windows/Linux 都能跑）
python gui.py

# 或者启动 WebUI（手机也能用）
python webui.py
```

> ⚠️ 如果你用 GPU 训练，`requirements_gpu.txt` 里的 PyTorch 是 CUDA 12.4 版。显卡不是这版本的话自己去 [pytorch.org](https://pytorch.org) 换。

---

## 🖥️ GUI 使用教程（桌面版，适合新手）

启动 `gui.py` 后，你会看到一个 4 标签页的窗口。下面按使用顺序说。

### 1️⃣ 训练（Training）标签页

这是核心功能。界面分三块：

**左侧：超参数配置**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `d_model` | 模型隐藏层维度。越大越聪明，也越吃显存 | 256-896 |
| `n_layers` | Transformer 层数 | 8-20 |
| `n_heads` | 注意力头数 | 8-16 |
| `d_ff` | 前馈网络维度 | d_model × 4 |
| `n_experts` | MoE 专家数量 | 4 |
| `n_activated` | 每次激活的专家数 | 2 |
| `batch_size` | 每步训练的样本数 | 8-32 |
| `learning_rate` | 学习率 | 1e-4 |
| `epochs` | 训练轮数 | 10-50 |
| `grad_accum` | 梯度累积步数（等效批大小 = batch_size × grad_accum） | 4-8 |
| `dropout` | 丢弃率，防过拟合 | 0.1 |
| `warmup_steps` | 学习率预热步数 | 500-2000 |
| `seq_len` | 序列最大长度 | 256-512 |

**中间：训练控制**

- 点击「训练数据」旁边的「浏览」选择你的 `.jsonl` 文件
- 点击「输出目录」选择 checkpoint 存放位置
- 点击「🚀 开始训练」就开跑了

**右侧：实时监控**

- 训练过程中 loss 曲线实时更新
- 控制台日志输出每个 step 的 loss、学习率、perplexity

**训练数据格式（.jsonl）：**

```jsonl
{"text": "用户：你好\nTGAI：你好！有什么可以帮你的？"}
{"text": "用户：今天天气怎么样\nTGAI：抱歉我是AI，不知道实时天气哦。不过你可以看看窗外！"}
```

一行一个对话，`\n` 分隔用户和助手。

### 2️⃣ 聊天（Chat）标签页

训练完后，加载模型开始唠嗑。

1. 点击「加载模型」选择一个 `.pt` checkpoint 文件
2. 在输入框打字，回车发送
3. 右侧可调节温度、top-k、top-p、重复惩罚等生成参数

| 参数 | 说明 | 范围 |
|------|------|------|
| 温度 | 越高越浪，越低越保守 | 0.1-2.0 |
| Top-K | 每步只从 K 个最高概率词里抽 | 1-100 |
| Top-P | 核采样，累积概率截断 | 0-1.0 |
| 重复惩罚 | 大于 1.0 惩罚复读机行为 | 1.0-2.0 |

### 3️⃣ 分词器（Tokenizer）标签页

- 输入任意文本，看 tokenize 结果
- 浏览完整词表

### 4️⃣ 数据（Data）标签页

- 查看/编辑训练数据
- 统计问答长度、检查格式问题

---

## ☁️ CLI 云端服务器训练教程

如果你有一台云 GPU（AutoDL、矩池云、恒源云等），用命令行训练更方便。

### 第一步：连上服务器

```bash
ssh -p 端口号 root@你的服务器IP
```

### 第二步：装环境

```bash
# 克隆项目
git clone https://github.com/JXW666NB/TGAI-GUI-NLP.git
cd TGAI-GUI-NLP

# 装依赖（GPU 版依赖自带 CUDA 12.4 的 PyTorch）
pip install -r requirements_gpu.txt

# 如果你的 CUDA 版本不是 12.4，手动装 PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 第三步：准备数据

把你的 `.jsonl` 训练数据上传到服务器：

```bash
# 本地执行
scp your_data.jsonl root@服务器IP:/root/TGAI-GUI-NLP/data/

# 检查数据质量
python tgai_cli.py data stats --input data/your_data.jsonl
```

输出示例：
```
总样本数: 5230
最短 Q: 2 字 | 最长 Q: 87 字 | 平均: 15.3 字
最短 A: 5 字 | 最长 A: 243 字 | 平均: 48.7 字
[警告] 3 条样本答案长度 < 10 字，可能质量较低
```

### 第四步：生成训练配置文件

```bash
python tgai_cli.py config generate
```

这会生成一个 `train_config.yaml`，打开改改：

```yaml
# train_config.yaml
data_path: data/your_data.jsonl
output_dir: checkpoints/

# 模型参数
vocab_size: 16384
d_model: 512
n_layers: 12
n_heads: 8
d_ff: 2048
max_seq_len: 256

# MoE
n_experts: 4
n_activated: 2

# 训练参数
batch_size: 16
epochs: 30
learning_rate: 0.0001
grad_accum: 4
warmup_steps: 2000
dropout: 0.1
```

### 第五步：开训！

```bash
# 使用 YAML 配置文件训练
python tgai_cli.py train --config train_config.yaml

# 或者不用配置文件，直接命令行传参
python tgai_cli.py train \
    --data data/your_data.jsonl \
    --epochs 30 \
    --batch 16 \
    --d_model 512 \
    --n_layers 12 \
    --grad_accum 4
```

### 第六步：后台运行（防止 SSH 断开导致训练中断）

```bash
# 用 nohup 后台跑
nohup python tgai_cli.py train --config train_config.yaml > train.log 2>&1 &

# 查看进度
tail -f train.log

# 查看训练状态
python tgai_cli.py status
```

输出示例：
```
系统状态:
  CPU: Intel Xeon 8核 | 使用率: 34%
  内存: 32.0 GB | 已用: 18.2 GB (56.9%)
  GPU: NVIDIA RTX 4090 | 显存: 24.0 GB | 已用: 14.8 GB
  Python: 3.11.9

Checkpoints:
  milestone1000.pt     ( 839 MB)  2025-07-03 14:22
  milestone2000.pt     ( 839 MB)  2025-07-03 15:18
  last_step.pt         ( 839 MB)  2025-07-03 15:33

训练进程:
  PID 12345  运行中  |  日志: train.log (2.3 MB)
```

### 第七步：下载训练好的模型

```bash
# 查看有哪些 checkpoint
ls -lh checkpoints/

# 在本地电脑执行
scp root@服务器IP:/root/TGAI-GUI-NLP/checkpoints/milestone*.pt ./
```

### 常用 CLI 命令速查

```bash
# 查看帮助
python tgai_cli.py --help

# 训练
python tgai_cli.py train --config train_config.yaml

# 聊天
python tgai_cli.py chat --checkpoint checkpoints/milestone4000.pt

# 数据统计
python tgai_cli.py data stats --input data/train.jsonl

# 数据清洗
python tgai_cli.py data clean --input data/train.jsonl --output data/cleaned.jsonl

# 合并多个数据文件
python tgai_cli.py data merge --input data/a.jsonl data/b.jsonl --output data/merged.jsonl

# 生成默认配置
python tgai_cli.py config generate

# 系统状态
python tgai_cli.py status
```

---

## 🌐 WebUI 远程操控

如果你不想装 PyQt6（比如云服务器没桌面），可以用 WebUI：

```bash
python webui.py
```

默认监听 `0.0.0.0:5000`，手机/平板打开浏览器访问 `http://服务器IP:5000` 就能操作。功能跟桌面 GUI 差不多，暗色主题，移动端自适应。

---

## 🤖 QQ 机器人部署

### 准备工作

1. 装 [NapCat](https://github.com/NapNeko/NapCatQQ)（QQ 机器人框架）
2. 把 `qq_bot_config.json` 里的 WebSocket URL 和 token 填对：

```json
{
    "ws_url": "ws://127.0.0.1:6099/api/Debug/ws?token=你的token",
    "http_url": "http://127.0.0.1:6099",
    "bot_qq": "你的机器人QQ号",
    "temperature": 0.8
}
```

### 启动

```bash
python tgai_qq_bot.py --checkpoint checkpoints/milestone4000.pt
```

群聊里 @机器人 或以 "TGAI" 开头就会回复。私聊自动回复。

---

## 📱 导出模型到手机（ONNX + TG CHAT）

训完模型之后，可以导出为 ONNX 格式，扔到 TG CHAT APP 里离线推理。

### GUI 导出（最简单）

1. 打开 `gui.py`，切到「导出」标签页
2. 选择一个 `.pt` checkpoint 文件
3. 勾选「INT8 量化」（推荐，模型减半速度翻倍）
4. 点击「导出并打包」→ 自动生成 `.TG` 文件
5. 把 `.TG` 传到手机，在 TG CHAT 里一键导入

### 命令行导出

```bash
# 步骤 1: 导出 ONNX 模型
python scripts/export/export_onnx.py \
    --checkpoint checkpoints/milestone4000.pt \
    --out_dir exported/ \
    --int8

# 步骤 2: 打包为 .TG
python scripts/export/pack_tg.py \
    --model exported/tgai.onnx \
    --tokenizer exported/tokenizer.json \
    --out TGAI-4000.tg
```

### 导出 YUAZ（朋友模型）

支持标准 Llama 架构的模型（YUAZ 等），在 GUI 里模型类型选「YUAZ (Llama)」即可。

### 参数说明

| 参数 | 说明 |
|------|------|
| `--int8` | 启用 INT8 动态量化，模型大小减半，推理加速约 2x |
| `--opset` | ONNX opset 版本，默认 18 |
| `--max_seq_len` | 示例序列长度（不影响运行时），默认 512 |

### 相关脚本

| 脚本 | 作用 |
|------|------|
| `scripts/export/export_onnx.py` | PyTorch checkpoint → ONNX 模型 |
| `scripts/export/pack_tg.py` | ONNX + tokenizer → .TG 打包文件 |
| `scripts/export/export_tokenizer_mobile.py` | 单独导出手机端 tokenizer |
| `scripts/export/export_for_mobile.py` | 完整移动端导出（含自定义量化格式） |
| `scripts/export/export_executorch.py` | ExecuTorch 格式导出（Android/iOS） |
| `scripts/export/export_pytorch_mobile.py` | PyTorch Mobile 格式导出 |

> ⚠️ 导出需要 8GB+ 内存。如果你的 checkpoint 包含优化器状态（训练存档），内存需求更大。建议用只含模型权重的 checkpoint 来导出。如果内存不够，找台内存大的电脑跑。

---

## 🚀 TGAI GO 引擎 & PT→TG 转换（重点）

TGAI GO 是 TGAI 的嵌入式 C 推理引擎，能在 ESP32 等单片机上跑你的模型。**TGAI-NLP 与 TGAI GO 完整联动：**

### 什么是 .TG 格式？

`.TG` 是 TGAI GO 的自有模型格式，包含模型权重 + tokenizer + 元数据，单文件分发。支持多种量化方式。

### GUI 一键转换（推荐）

打开 `gui.py`，切换到「导出」标签页，有两张卡片：

**卡片 1：TGAI GO 引擎转换（PT/GGUF → .TG）**

| 输入 | 说明 |
|------|------|
| 源文件 | `.pt`（PyTorch checkpoint）或 `.gguf`（社区模型） |
| Tokenizer | `tokenizer.json` |
| 输出 | `.TG` 文件 |
| 量化 | FP32 / FP16 / Q8_0 / Q4_0 / INT8 |
| 架构 | tgai_moe / llama / mistral / qwen2 / gemma / phi3 |

点击「开始转换」即可。支持 GGUF v3 格式的 Llama/Mistral/Qwen2/Phi-3/Gemma 等社区模型。

**卡片 2：传统 ONNX 导出 + 打包**

| 输入 | 说明 |
|------|------|
| Checkpoint | `.pt` 文件 |
| 输出 | ONNX + tokenizer → `.TG`（ZIP 格式） |
| 量化 | 可选 INT8 |

### 命令行转换

```bash
# PyTorch → .TG
cd TGAI\ GO/tools
python tgai_convert.py --checkpoint model.pt --output model.TG \
    --tokenizer tokenizer.json --dtype q4_0 --arch tgai_moe

# GGUF → .TG（社区模型兼容）
python gguf_to_tg.py --input llama.gguf --output llama.TG --dtype fp16
```

### 云端转换

```bash
# 量化转换（调用 TGAI GO 工具）
python scripts/export/cloud_quantize.py \
    --input model.pt --output model.TG --dtype q4_0

# 批量转换
bash scripts/export/convert_cloud.sh

# 模型嫁接（V1 → V2 膨胀）
python scripts/export/graft_v2.py \
    --input tgai_v1.pt --output tgai_v2.pt --preset lite1
```

### 部署到 ESP32

转换完成后，用 TGAI GO 的上传工具烧录到 ESP32：

```
TGAI GO/tools/upload_model.bat → 选择 .TG → 上传到 ESP32 LittleFS
```

> 详细文档见 [TGAI GO 仓库](https://github.com/JXW666NB/TGAI-GO)

---

## 🧠 模型架构

```
TGAILanguageModel (V4 Decoder-Only MoE Transformer)

Embedding
  ↓
[TransformerBlock × n_layers]
  ├── RMSNorm
  ├── FlashSelfAttention (RoPE + KV Cache)
  ├── RMSNorm
  └── MoELayer
      ├── Router (Top-K Gating)
      └── Experts × n_experts (SwiGLUFFN)
  ↓
RMSNorm
  ↓
LM Head (weight tied with embedding)
```

| 组件 | 技术 |
|------|------|
| 位置编码 | RoPE（旋转位置编码） |
| 归一化 | RMSNorm（Root Mean Square） |
| 注意力 | Flash Attention（`scaled_dot_product_attention`） |
| 前馈网络 | SwiGLU（SwiGLUFFN） |
| 混合专家 | MoE（Top-K 路由 + 负载均衡 loss） |
| KV 缓存 | 预分配缓冲区，动态扩展 |
| 推理加速 | `torch.compile` + KV Cache 流式输出 |

---

## 🏗️ 项目结构

```
tgai_nlp/
├── gui.py                  # 🖥️ PyQt6 桌面 GUI（训练/聊天/分词器/数据编辑器）
├── webui.py                # 🌐 Flask + Socket.IO WebUI（手机也能用）
├── tgai_cli.py             # ☁️ 命令行工具（云端训练/数据管理/配置生成）
├── train.py                # 🔥 训练引擎（AMP 混合精度、梯度累积、断点续训）
├── train_lora.py           # 🔧 LoRA 低秩适配器微调
├── model.py                # 🧠 V4 MoE Transformer 模型定义
├── tokenizer.py            # 🔤 BPE 分词器（中文优化 CJK 预分词）
├── inference.py            # 💬 推理引擎（流式输出/KV Cache/采样策略）
├── tgai_qq_bot.py          # 🤖 QQ 机器人（NapCat 协议）
├── word_bias_config.json   # 🎛️ 词频偏置配置（热加载）
├── data/
│   ├── custom_train.jsonl  # 示例训练数据
│   └── train_all.jsonl     # 完整训练数据
├── requirements.txt        # 基础依赖
├── requirements_cpu.txt    # CPU 训练依赖
├── requirements_gpu.txt    # GPU 训练依赖（CUDA 12.4）
├── requirements_api.txt    # API 服务器依赖
├── requirements_lora.txt   # LoRA 训练依赖
├── scripts/
│   ├── api_server.py       # 🌐 REST API 服务器
│   ├── chat_cli.py         # 💬 命令行聊天
│   ├── benchmarks/         # 📊 基准测试
│   ├── deploy/             # 🚀 部署脚本（AutoDL, Cloudflare Tunnel）
│   ├── export/             # 📦 模型导出（ONNX, ExecuTorch, .TG）
│   └── tools/              # 🔧 辅助工具（GUI 打包器）
├── .gitignore
├── LICENSE
└── README.md               # 你他妈正在看的这玩意儿
```

---

## 🔧 参数估算

训练前不确定该用什么参数？以下是经验公式：

### 显存估算（FP16 训练）

```
显存 ≈ (参数量 × 2 字节) + (激活值 × 2 字节) × 2

参数量 ≈ vocab_size × d_model + n_layers × (4 × d_model² + 3 × d_model × d_ff × n_experts)
```

| 配置 | 参数量 | 显存占用 | 推荐 GPU |
|------|--------|----------|----------|
| `d_model=256, n_layers=8` | ~50M | ~2 GB | GTX 1060 |
| `d_model=384, n_layers=10` | ~120M | ~4 GB | RTX 2060 |
| `d_model=512, n_layers=12` | ~250M | ~8 GB | RTX 3060 |
| `d_model=768, n_layers=16` | ~600M | ~16 GB | RTX 3090 |
| `d_model=896, n_layers=20` | ~1000M | ~24 GB | RTX 4090 |

> ⚠️ 这是估计值，实际会因 `seq_len`、`batch_size`、`grad_accum` 而变化。拿不准？`tgai_cli.py train` 启动时会自动估算并提醒你。

---

## 🎮 快捷操作

| 操作 | GUI | CLI |
|------|-----|-----|
| 开始训练 | 点击「🚀 开始训练」 | `tgai_cli.py train --config xxx.yaml` |
| 断点续训 | 同上，自动检测 `last_step.pt` | `tgai_cli.py train --resume` |
| 加载模型聊天 | Chat 标签页 → 加载模型 | `tgai_cli.py chat --checkpoint xxx.pt` |
| 数据统计 | Data 标签页 | `tgai_cli.py data stats` |
| 后台训练 | 不支持 | `nohup tgai_cli.py train ... &` |

---

## 🤝 贡献与反馈

觉得这项目烂得清奇，或者想把你的屎山合并进来？欢迎提 Issue、PR，或者进 QQ 群 **1082708943** 开喷。如果你被这项目逗笑了，请点个 Star ⭐️，作者会感动到多吃一碗泡面。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。随便改、随便卖、随便塞进毕设里——**但得把原作者名字留着**，否则半夜会有 AI 爬你窗户。

---

## 🔗 相关项目

- 🚀 **TGAI GO**（嵌入式 C 推理引擎）：[github.com/JXW666NB/TGAI-GO](https://github.com/JXW666NB/TGAI-GO)
- 📱 **TG CHAT**（手机 APP）：[github.com/JXW666NB/TGAI_CHAT](https://github.com/JXW666NB/TGAI_CHAT)
- 🛠️ **TG-HELPER**（桌面 AI 助手）：[github.com/JXW666NB/TG-HELPER](https://github.com/JXW666NB/TG-HELPER)
- 🎮 **TGAI 模型导出**（ONNX / 手机端）：本项目的 `scripts/` 子目录

---

**踏马的终于肝完力**
—— JXW, 2025 年某个通宵的凌晨
