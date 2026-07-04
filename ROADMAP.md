# TGAI 技术路线图 & 升级规划

> 最后更新：2025-07-03
> 
> 本文档记录 TGAI 项目当前状态、技术债务、未来升级路线，以及各阶段讨论结论。
> 目标：让普通人或中小企业可以迅速开发和训练自己的 AI 模型（文本 → 多模态 → 图像生成 → 音频生成）。

---

## 0. 当前状态：TGAI v0.2.0

### 模型规格

| 项目 | 当前值 |
|------|--------|
| 架构 | Decoder-Only MoE Transformer |
| 参数量 | ~500M（`d_model=896, n_layers=20, n_experts=4, n_activated=2`） |
| 组件 | RoPE + RMSNorm + Flash Attention + SwiGLU + KV Cache |
| 词表 | 32768 tokens（BPE 中文优化） |
| 训练框架 | PyTorch 原生，AMP 混合精度，梯度累积，断点续训 |
| 推理 | KV Cache 流式输出，支持 `torch.compile` 加速 |

### 界面 & 部署

| 界面 | 技术 | 状态 |
|------|------|------|
| 桌面 GUI | PyQt6 | ✅ 训练/聊天/分词器/数据编辑/导出 |
| WebUI | Flask + Socket.IO | ✅ 手机/平板可用 |
| CLI | `tgai_cli.py` | ✅ 云端训练/数据管理 |
| QQ 机器人 | NapCat OneBot 协议 | ✅ |
| 手机 APP | TG CHAT (Flutter + ONNX Runtime) | ✅ 离线推理 |

### 手机端推理

| 项目 | 技术 | 性能 |
|------|------|------|
| 推理引擎 | ONNX Runtime Mobile | — |
| 执行提供器 | AUTO / NNAPI / ACL+CPU / CPU_ONLY | 可选 |
| 芯片检测 | 自动识别骁龙/天玑/麒麟/Exynos | 推荐最优后端 |
| 加速 | INT8 动态量化 | 模型减半，速度翻倍 |
| 解码策略 | 滑动窗口（prefill 64 + decode 16） | 平衡速度与质量 |
| 实测速度 | ~5s/token (FP16 CPU) | 待 INT8 模型验证 |

---

## 1. 技术债务 & 当前瓶颈

### 1.1 MoE Python 循环（最高优先级）

**位置**：`model.py` → `MoELayer.forward()`

```python
# 当前：纯 Python 串行循环
for e_idx in range(self.n_experts):  # ← 无法编译加速
    expert_out = self.experts[e_idx](x_flat)
```

**问题**：4 个 expert 还行，扩展到 32+ 个 expert 会严重拖慢训练/推理速度。

**方案**：改用 grouped GEMM（`torch.vmap` 或 `torch.ops.aten.grouped_mm`）。

### 1.2 分布式训练缺失

| 能力 | 状态 | 优先级 |
|------|------|--------|
| 单卡训练 | ✅ | — |
| 多卡 DDP | ❌ | 中 |
| DeepSpeed ZeRO-3 | ❌ | 高 |
| 张量并行 (TP) | ❌ | 高（10B+ 必需） |
| 专家并行 (EP) | ❌ | 高（MoE 天然优势） |
| 流水线并行 (PP) | ❌ | 中 |

**影响**：当前单卡最多跑 ~3B 参数模型（H100 80GB），无法支持 10B+ 训练。

### 1.3 数据加载

- `TextDataset` 是否流式读取？如果是全量加载到内存，1TB 数据会炸。
- 首次 tokenize 大数据集非常慢（需要多进程 tokenize 加速）。

### 1.4 手机端速度

- FP16 模型 ~5s/token，INT8 量化后预计 ~2-3s/token。
- NNAPI 对 MoE 架构支持有限，实际加速效果待验证。
- 模型文件 ~1.5GB（FP16），手机存储压力较大。

---

## 2. 规模扩展路线（1B → 10B → 20B → 40B）

### 2.1 各阶段需求

| 规模 | 参数量 | GPU 需求 | 关键技术 | 成本/周（云） |
|------|--------|---------|---------|-------------|
| 当前 | ~0.5B | 1×4090 | 无 | ¥0 |
| 中型 | 1-3B | 1×A100/H100 | — | ¥500 |
| 大型 | 7-10B | 4×A100 | ZeRO-3 + EP | ¥3000 |
| 超大 | 20B | 8×A100 | ZeRO-3 + EP + TP | ¥6000 |
| 超大 | 40B | 16×A100 | ZeRO-3 + EP + TP + PP | ¥12000 |

### 2.2 改造步骤

```
现有代码
    │
    ├─ ① DeepSpeed ZeRO-3 集成（1-2 天）
    │    → 权重/优化器/梯度分片到多卡
    │    → 4 卡等效显存 = 单卡 × 4
    │
    ├─ ② MoE Grouped GEMM 改造（2-3 天）
    │    → 消除 Python 循环，所有 expert 并行计算
    │
    ├─ ③ 专家并行 EP（1-2 天）
    │    → 不同 expert 放不同 GPU
    │    → router all-to-all 通信
    │
    ├─ ④ 张量并行 TP（2-3 天）
    │    → 单层 Linear 切分到多卡
    │    → 配合 EP 覆盖 10B+
    │
    └─ ⑤ FP8/INT4 训练（3-4 天）
         → 进一步压显存，40B 用更少卡跑
```

### 2.3 当前理论极限（不改代码）

- 单卡 80GB H100：最大 ~3B 参数
- 训练数据：流式读取无上限
- 训练速度：~5000 token/s（H100 FP16）

---

## 3. 多模态路线图

### 3.1 统一架构愿景

```
              TGAI MoE Transformer（统一主干）
             ┌──────────┼──────────┐
       文本输入        图像输入      音频输入
       (Token)     (CLIP编码)   (EnCodec Token)
             │           │           │
       文本输出        图像输出      音频输出
       (Token)     (DiT解码)   (EnCodec解码)
```

**核心理念**：一个 Transformer 主干，三种模态的头尾，共享底层注意力机制和 MoE 专家。

### 3.2 多模态理解（图文理解）—— 短期

```
图片 → CLIP/SigLIP 视觉编码器（冻结）→ 投影层（可训练）→ 拼到 token 序列 → 喂 TGAI
```

- 改动量：最小（加视觉编码器 + 投影层，Transformer 不动）
- 成本：单卡能训，类似 LLaVA 方案
- 数据：图文对（图 + 描述）
- 目标：看图说话、图片理解、OCR

### 3.3 图像生成（DiT 扩散模型）—— 中期

```
文字 → CLIP/T5 文本编码器（预训练）
噪声 + 时间步 → DiT 去噪网络（基于 TGAI 架构改造）→ 潜变量
                                                        ↓
                                                  VAE 解码器（预训练）
                                                        ↓
                                                     图片
```

**为什么复用 TGAI**：
- DiT（Diffusion Transformer）本质就是 Transformer + 时间步编码
- TGAI 的 TransformerBlock 可以直接复用，额外加：
  - 时间步嵌入（`TimeEmbedding`）
  - 噪声预测 head
  - 扩散 loss（`mse_loss`）

**阶段计划**：

| 阶段 | 模型 | 参数 | 输出 | 训练数据 | 硬件 |
|------|------|------|------|---------|------|
| P0: Micro DiT | TGAI-DiT-Small | 100M | 64×64 | 1 万张领域图 | 1×3090 |
| P1: Small DiT | TGAI-DiT-Mid | 500M | 256×256 | 10 万张图 | 1×4090 |
| P2: Mid DiT | TGAI-DiT-Large | 1B | 512×512 | 100 万张图 | 4×A100 |

**LoRA 微调优先**：先用 `diffusers` + Kohya LoRA 打通"用户扔 20 张图出同风格"流程，再训练自己的 DiT。

### 3.4 音频/音乐生成 —— 中长期

```
音频 → EnCodec/DAC 压缩为离散 token → TGAI Transformer（next-token 预测）
         ↑ 文本条件（T5/CLAP embedding）       ↓
                                              EnCodec 解码器
                                                   ↓
                                                 音频
```

**为什么可行**：
- Meta MusicGen 已经验证了这条路
- MusicGen Small 只有 300M 参数，单卡可训
- TGAI 的 Transformer 代码几乎不用改，输入从文本 token 换成音频 token 即可

**关键技术**：
- EnCodec（Meta）：音频 → 离散 token（50Hz, 2048 码本）
- CLAP：文本-音频对齐 embedding
- TGAI Transformer：next-token 预测（跟文本一模一样）

---

## 4. 关键决策记录

| # | 决策 | 日期 | 理由 |
|----|------|------|------|
| 1 | 手机推理选 ONNX Runtime 而非 ExecuTorch | 2025-07-02 | ExecuTorch 兼容性差，MoE 算子支持不全 |
| 2 | 执行提供器选 AUTO/NNAPI/ACL/CPU_ONLY 四模式 | 2025-07-03 | NNAPI 自动调度厂商 NPU，无需捆绑 SDK |
| 3 | INT8 动态量化优先于 INT4/Q4_K_S | 2025-07-04 | ONNX Runtime Mobile 不支持 INT4，INT8 改动最小 |
| 4 | 不捆绑厂商 NPU SDK | 2025-07-04 | QNN/Neuron/CANN 各需 +150-200MB，NPU 对 MoE 支持差 |
| 5 | 手机端用滑动窗口而非全上下文 | 2025-07-02 | 64+16 token 窗口解码，O(1) 推理复杂度 |
| 6 | 不追大模型，追专用模型 | 2025-07-05 | 1B 优质领域数据 > 70B 垃圾数据 |
| 7 | 多模态从 LoRA 图像微调起步 | 2025-07-05 | 成本最低，用户需求最明确 |
| 8 | 音频/音乐从零研发（不依赖 API） | 2025-07-05 | 市场未定，但有技术路径（MusicGen 方案） |
| 9 | 图像生成先 LoRA 后自研 DiT | 2025-07-05 | 先用成熟方案跑通流程，再逐步替换为自己组件 |

---

## 5. 当前待解决问题

### P0（必须解决）

- [x] ~~手机端推理速度优化（FP16）~~ → 已通过滑动窗口 + ACL 加速到 ~5s/token
- [ ] **手机端 INT8 模型验证** — 导出 INT8 ONNX 并实测速度
- [ ] **NNAPI 实际效果验证** — 在骁龙/天玑真机上测试加速效果
- [ ] **0.5B 模型训练验证** — 确保 GUI 训练 + CLI 训练流程完整可用

### P1（近期规划）

- [ ] DeepSpeed ZeRO-3 集成（支持多卡训练）
- [ ] MoE grouped GEMM 改造
- [ ] 数据加载流式优化（支持 1TB+ 数据）
- [ ] 多模态理解（CLIP + 投影层）
- [ ] LoRA 图像微调 GUI 集成

### P2（中期规划）

- [ ] 专家并行 EP
- [ ] 张量并行 TP
- [ ] 自研 DiT 扩散模型（基于 TGAI 架构）
- [ ] 音频 tokenizer（EnCodec）+ Transformer 音乐生成

### P3（长期规划）

- [ ] 3D 并行训练（DP + TP + PP）
- [ ] FP8/INT4 训练支持
- [ ] 40B+ 工业级训练
- [ ] 统一多模态前端（文/图/音输入输出）

---

## 6. 架构对比参考

| 框架 | TGAI | LLaMA-Factory | HuggingFace Trainer | Megatron-LM |
|------|------|--------------|-------------------|------------|
| GUI 训练 | ✅ PyQt6 | ✅ Gradio | ❌ | ❌ |
| MoE 原生支持 | ✅ | ❌ | ⚠️ 弱 | ✅ |
| 多模态 | 🔜 规划中 | ✅ | ✅ | ✅ |
| 手机端导出 | ✅ ONNX+TG | ❌ | ❌ | ❌ |
| 分布式训练 | 🔜 规划中 | ✅ | ✅ | ✅ |
| 上手难度 | 低 | 低 | 中 | 地狱级 |
| 定位 | 个人/中小企业全流程 | 学术微调 | 通用框架 | 工业超大模型 |

---

## 7. 参考项目

| 项目 | 用途 | 链接 |
|------|------|------|
| TG CHAT | 手机端推理 APP | [GitHub](https://github.com/JXW666NB/TGAI_CHAT) |
| TG-HELPER | 桌面 AI 助手 | [GitHub](https://github.com/JXW666NB/TG-HELPER) |
| LLaVA | 多模态理解参考 | [GitHub](https://github.com/haotian-liu/LLaVA) |
| MusicGen | 音乐生成参考 | [GitHub](https://github.com/facebookresearch/audiocraft) |
| Diffusers | 扩散模型库 | [GitHub](https://github.com/huggingface/diffusers) |
| DeepSpeed | 分布式训练 | [GitHub](https://github.com/microsoft/DeepSpeed) |

---

**踏马的终于肝完力**

—— JXW, 2025 年某个通宵的凌晨
