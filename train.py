"""
TGAI V4 训练脚本 — MoE Transformer + Flash Attention
=====================================================
新特性:
  - AMP 混合精度 (torch.cuda.amp)
  - 梯度累积 (grad_accum_steps)
  - 数据预缓存 (tokenize后存 .pt，秒加载)
  - 断点续训 (优化器/调度器/step 全量恢复)
  - MoE 负载均衡 loss
  - 余弦衰减 + warmup 学习率
  - 看门狗自动保存 (last_step.pt)
  - 定期评估 (perplexity)

用法:
    python train.py                          # 默认配置
    python train.py --epochs 100 --batch 32  # 自定义参数
    python train.py --resume checkpoints/last_step.pt  # 断点续训
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import ChineseTokenizer, PAD_ID, BOS_ID, EOS_ID, UNK_ID, SPECIAL_TOKENS
from model import TGAILanguageModel, TGAIConfig, create_model


# ═══════════════════════════════════════════════════════════
# 数据集 — 预缓存
# ═══════════════════════════════════════════════════════════
class TextDataset(Dataset):
    """将QA文本编码为固定长度序列，支持预缓存到磁盘（mmap 模式节省内存）。"""

    def __init__(
        self,
        texts: List[str],
        tokenizer: ChineseTokenizer,
        seq_len: int = 256,
        cache_path: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self._use_mmap = False

        if cache_path:
            # 检查缓存（.bin 格式）
            meta_path = cache_path.replace('.pt', '_meta.json')
            mmap_inputs = cache_path.replace('.pt', '_inputs.bin')
            mmap_targets = cache_path.replace('.pt', '_targets.bin')

            if os.path.exists(meta_path) and os.path.exists(mmap_inputs):
                import json as _json
                with open(meta_path) as f:
                    meta = _json.load(f)
                self._count = meta['count']
                self.seq_len = meta['seq_len']
                self._inputs_path = mmap_inputs
                self._targets_path = mmap_targets
                self._load_mmap()
                print(f"  [数据集] 从缓存加载 ({self._count} 样本)")
                return

            # 检查旧的 .pt 缓存
            if os.path.exists(cache_path):
                data = torch.load(cache_path, map_location='cpu', weights_only=True)
                self.inputs = data['inputs']
                self.targets = data['targets']
                self._use_mmap = False
                print(f"  [数据集] 从缓存加载: {cache_path} ({len(self.inputs)} 样本)")
                return

        # mmap 模式：边编码边写磁盘，避免内存爆满
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            self._encode_mmap(texts, tokenizer, cache_path)
        else:
            self._encode_inmemory(texts, tokenizer)

    def _encode_mmap(self, texts, tokenizer, cache_path):
        """逐批编码写入二进制文件，训练时用 mmap 读取"""
        print(f"  [数据集] 编码 {len(texts)} 条文本...")

        tmp_dir = os.path.dirname(cache_path)
        self._inputs_path = cache_path.replace('.pt', '_inputs.bin')
        self._targets_path = cache_path.replace('.pt', '_targets.bin')
        meta_path = cache_path.replace('.pt', '_meta.json')

        count = 0
        skipped = 0
        batch_size = 10000
        batch_inputs = []
        batch_targets = []

        # 用普通文件写入，不预分配
        f_in = open(self._inputs_path, 'wb')
        f_tg = open(self._targets_path, 'wb')

        for text in texts:
            ids = tokenizer.encode(text, add_special=False)
            if len(ids) < 2:
                skipped += 1
                continue

            max_content = self.seq_len - 2
            if len(ids) > max_content:
                ids = ids[:max_content]

            input_ids = [BOS_ID] + ids
            target_ids = ids + [EOS_ID]

            pad_len = self.seq_len - len(input_ids)
            input_ids += [PAD_ID] * pad_len
            target_ids += [PAD_ID] * pad_len

            batch_inputs.append(input_ids)
            batch_targets.append(target_ids)
            count += 1

            if len(batch_inputs) >= batch_size:
                arr_i = np.array(batch_inputs, dtype=np.int64)
                arr_t = np.array(batch_targets, dtype=np.int64)
                arr_i.tofile(f_in)
                arr_t.tofile(f_tg)
                batch_inputs.clear()
                batch_targets.clear()
                if count % 100000 == 0:
                    print(f"    已编码 {count} 条...")

        # 最后一批
        if batch_inputs:
            arr_i = np.array(batch_inputs, dtype=np.int64)
            arr_t = np.array(batch_targets, dtype=np.int64)
            arr_i.tofile(f_in)
            arr_t.tofile(f_tg)

        f_in.close()
        f_tg.close()

        if skipped:
            print(f"  [数据集] 跳过 {skipped} 条过短文本")

        # 保存元信息
        import json as _json
        with open(meta_path, 'w') as f:
            _json.dump({'count': count, 'seq_len': self.seq_len}, f)

        self._use_mmap = True
        self._count = count
        self._load_mmap()

        print(f"  [数据集] {self._count} 样本, seq_len={self.seq_len}, 已缓存")

    def _load_mmap(self):
        """从二进制文件 mmap 加载"""
        self._inputs_mmap = np.memmap(self._inputs_path, dtype=np.int64, mode='r',
                                       shape=(self._count, self.seq_len))
        self._targets_mmap = np.memmap(self._targets_path, dtype=np.int64, mode='r',
                                        shape=(self._count, self.seq_len))
        self._use_mmap = True

    def _encode_inmemory(self, texts, tokenizer):
        """原始内存模式（小数据集用）"""
        print(f"  [数据集] 编码 {len(texts)} 条文本...")
        all_inputs, all_targets = [], []
        skipped = 0

        for text in texts:
            ids = tokenizer.encode(text, add_special=False)
            if len(ids) < 2:
                skipped += 1
                continue

            max_content = self.seq_len - 2
            if len(ids) > max_content:
                ids = ids[:max_content]

            input_ids = [BOS_ID] + ids
            target_ids = ids + [EOS_ID]

            pad_len = self.seq_len - len(input_ids)
            input_ids += [PAD_ID] * pad_len
            target_ids += [PAD_ID] * pad_len

            all_inputs.append(input_ids)
            all_targets.append(target_ids)

        if skipped:
            print(f"  [数据集] 跳过 {skipped} 条过短文本")

        self.inputs = torch.tensor(all_inputs, dtype=torch.long)
        self.targets = torch.tensor(all_targets, dtype=torch.long)
        self._use_mmap = False
        print(f"  [数据集] {len(self.inputs)} 样本, seq_len={self.seq_len}")

    def __len__(self):
        if self._use_mmap:
            return self._count
        return len(self.inputs)

    def __getitem__(self, idx):
        if self._use_mmap:
            return torch.from_numpy(self._inputs_mmap[idx].copy()), torch.from_numpy(self._targets_mmap[idx].copy())
        return self.inputs[idx], self.targets[idx]


# ═══════════════════════════════════════════════════════════
# 学习率调度器
# ═══════════════════════════════════════════════════════════
class CosineWarmupScheduler:
    """线性 warmup + 余弦衰减"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * min(progress, 1.0))
            )
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.base_lrs[i] * scale

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> dict:
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state: dict):
        self.current_step = state['current_step']
        self.base_lrs = state['base_lrs']


# ═══════════════════════════════════════════════════════════
# 训练配置
# ═══════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    # 数据
    data_path: str = "data/train_all.jsonl"
    seq_len: int = 256
    cache_dir: str = "checkpoints/cache"

    # 分词器
    vocab_size: int = 16384
    tokenizer_path: str = "checkpoints/tokenizer.json"

    # 模型
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    n_experts: int = 4
    n_activated: int = 2

    # 训练
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4  # 降低学习率，更稳定
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # 保存
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    save_every_steps: int = 500
    eval_every_steps: int = 100

    # 硬件
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def use_amp(self) -> bool:
        return self.device.startswith("cuda")


# ═══════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: TGAILanguageModel, dataloader: DataLoader, max_batches: int = 20) -> float:
    """计算验证集困惑度 (perplexity)"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, _ = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=PAD_ID,
            reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (targets != PAD_ID).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ═══════════════════════════════════════════════════════════
# 保存/加载 checkpoint
# ═══════════════════════════════════════════════════════════
def save_model_only(path: str, model: TGAILanguageModel, tokenizer: ChineseTokenizer,
                     epoch: int, global_step: int, best_ppl: float):
    """仅保存模型权重（推理用，~3.3G）"""
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'n_layers': model.config.n_layers,
            'n_heads': model.config.n_heads,
            'd_ff': model.config.d_ff,
            'max_seq_len': model.config.max_seq_len,
            'dropout': model.config.dropout,
            'n_experts': model.config.n_experts,
            'n_activated': model.config.n_activated,
        },
    }
    torch.save(state, path)


def save_checkpoint(
    path: str,
    model: TGAILanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    tokenizer: ChineseTokenizer,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    loss: float,
    best_ppl: float,
):
    """完整checkpoint（含优化器状态，~10G，用于断点续训）"""
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'n_layers': model.config.n_layers,
            'n_heads': model.config.n_heads,
            'd_ff': model.config.d_ff,
            'max_seq_len': model.config.max_seq_len,
            'dropout': model.config.dropout,
            'n_experts': model.config.n_experts,
            'n_activated': model.config.n_activated,
        },
    }
    torch.save(state, path)


def load_checkpoint(path: str, device: str) -> dict:
    """加载checkpoint"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")
    return torch.load(path, map_location=device, weights_only=False)


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════
def load_texts(data_path: str) -> List[str]:
    """从jsonl/json文件加载文本"""
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                texts = [json.loads(line)['text'] for line in f if line.strip()]
            else:
                data = json.load(f)
                texts = data if isinstance(data, list) else data.get('texts', [])
        print(f"  从 {data_path} 加载 {len(texts)} 条文本")
        return texts

    # 回退
    alt_paths = ['data/train_qa.jsonl', 'data/custom_train.jsonl']
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"  自动检测到 {alt}")
            with open(alt, 'r', encoding='utf-8') as f:
                texts = [json.loads(line)['text'] for line in f if line.strip()]
            print(f"  加载 {len(texts)} 条文本")
            return texts

    print("  [警告] 无训练数据，使用内置示例")
    return _get_demo_texts()


def _get_demo_texts() -> List[str]:
    return [
        "苹果是食物", "苹果富有营养价值", "苹果手机是苹果公司的产品",
        "苹果手机很好用", "我是TGAI，你好", "我叫TGAI",
        "我的名字是TGAI", "灵感菇很灵", "灵感菇瓦达",
        "唱歌很好听", "唱歌让我快乐", "抱歉我无法理解你说的话",
        "数据库无记录", "好的", "你叫什么名字",
        "今天天气真好", "什么是人工智能？",
        "苹果是什么颜色的？苹果是红色的",
        "用户：苹果是什么\nTGAI：苹果是一种常见的水果，富含维生素和膳食纤维，非常健康。",
        "用户：你叫什么名字\nTGAI：我叫TGAI，很高兴认识你！",
        "用户：灵感菇好吃吗\nTGAI：灵感菇味道很不错，是很有营养的食材。",
        "用户：唱歌好听吗\nTGAI：唱歌能让人心情愉悦，每个人唱出来的声音都是独特的。",
        "用户：苹果手机怎么样\nTGAI：苹果手机以流畅的系统和优秀的生态系统著称，很受欢迎。",
        "用户：你好\nTGAI：你好！有什么可以帮你的吗？",
        "用户：你能做什么\nTGAI：我可以和你聊天、回答问题、提供建议。",
        "用户：介绍你自己\nTGAI：我是TGAI，一个基于Transformer的小型语言模型。",
    ]


# ═══════════════════════════════════════════════════════════
# 主训练函数
# ═══════════════════════════════════════════════════════════
def train(config: TrainConfig):
    device = torch.device(config.device)
    use_gpu = device.type == 'cuda'

    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap_major = torch.cuda.get_device_capability(0)[0]
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

        # TF32 加速 (Ampere+ 支持，吞吐提升 ~2x)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        print(f"  优化: TF32=ON, cudnn.benchmark=ON")
    else:
        # CPU 优化: 限制线程数避免争抢，启用高效内核
        import multiprocessing
        n_cpu = min(multiprocessing.cpu_count(), 8)
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(n_cpu)
        print(f"设备: CPU (线程: {n_cpu})")

    print(f"配置: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"MoE: {config.n_experts} experts, top-{config.n_activated}")
    print(f"Batch: {config.batch_size}, GradAccum: {config.grad_accum_steps}x, SeqLen: {config.seq_len}")

    # 单专家模式：跳过 MoE 负载均衡计算
    use_moe = config.n_experts > 1 and config.n_activated < config.n_experts
    if not use_moe:
        print("  (单专家模式，跳过 MoE 负载均衡)")

    # ── 1. 数据 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("1/5 加载数据")
    texts = load_texts(config.data_path)

    # ── 2. 分词器 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("2/5 分词器")
    if os.path.exists(config.tokenizer_path):
        tokenizer = ChineseTokenizer.load(config.tokenizer_path)
        print(f"  已加载: {tokenizer.vocab_size_actual} tokens")
    else:
        print("  训练新分词器...")
        tokenizer = ChineseTokenizer(vocab_size=config.vocab_size)
        tokenizer.train(texts)
        os.makedirs(os.path.dirname(config.tokenizer_path) or '.', exist_ok=True)
        tokenizer.save(config.tokenizer_path)
        print(f"  已保存: {config.tokenizer_path} ({tokenizer.vocab_size_actual} tokens)")

    # ── 3. 数据集 (预缓存) ───────────────────────────
    print("\n" + "=" * 60)
    print("3/5 构建数据集")

    random.seed(42)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.9 * len(texts))
    train_texts = [texts[i] for i in indices[:split]]
    val_texts = [texts[i] for i in indices[split:]]

    os.makedirs(config.cache_dir, exist_ok=True)

    # 数据集指纹：防止不同数据集共用缓存
    data_hash = hashlib.md5(config.data_path.encode()).hexdigest()[:8]

    train_dataset = TextDataset(
        train_texts, tokenizer, config.seq_len,
        cache_path=os.path.join(config.cache_dir, f"train_s{config.seq_len}_{data_hash}.pt"),
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, config.seq_len,
        cache_path=os.path.join(config.cache_dir, f"val_s{config.seq_len}_{data_hash}.pt"),
    )

    train_loader = DataLoader(
        train_dataset, config.batch_size, shuffle=True, drop_last=True,
        num_workers=2 if not use_gpu else 0,
        pin_memory=use_gpu,
        prefetch_factor=2 if not use_gpu else None,
    )
    val_loader = DataLoader(
        val_dataset, config.batch_size, shuffle=False, drop_last=True,
        num_workers=0, pin_memory=use_gpu,
    )
    print(f"  训练: {len(train_dataset)} 样本, {len(train_loader)} batches/epoch")
    print(f"  验证: {len(val_dataset)} 样本")

    # ── 4. 模型 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("4/5 创建模型")

    model = create_model(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.seq_len,
        dropout=config.dropout,
        n_experts=config.n_experts,
        n_activated=config.n_activated,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  参数: {n_params:,} (~{n_params/1e6:.1f}M)")

    # ── 5. 优化器 & 调度器 ───────────────────────────
    # 自动选择: 显存 ≥ 60GB → AdamW (更优), < 60GB → SGD+CPU-offload (省内存)
    big_gpu = use_gpu and gpu_mem >= 60
    if big_gpu:
        print(f"  优化器: AdamW (显存充足, {gpu_mem:.0f}GB)")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
    else:
        print(f"  优化器: SGD+CPU-offload (显存紧张, {gpu_mem:.0f}GB)")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True,
            foreach=False,
        )
        _opt_step = optimizer.step
        def _cpu_offload_step(closure=None):
            for state in optimizer.state.values():
                if 'momentum_buffer' in state:
                    buf = state['momentum_buffer']
                    if buf.device.type == 'cpu':
                        state['momentum_buffer'] = buf.to(device, non_blocking=True)
            ret = _opt_step(closure)
            for state in optimizer.state.values():
                if 'momentum_buffer' in state:
                    buf = state['momentum_buffer']
                    state['momentum_buffer'] = buf.detach().cpu()
            torch.cuda.empty_cache()
            return ret
        optimizer.step = _cpu_offload_step
    total_steps = (len(train_loader) // config.grad_accum_steps) * config.epochs
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
    )

    # AMP
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # 断点续训
    start_epoch = 1
    global_step = 0
    best_ppl = float('inf')

    # 优先使用命令行指定的 --resume 路径
    custom_resume = getattr(config, '_resume_path', '')
    if custom_resume and os.path.exists(custom_resume):
        resume_path = custom_resume
        print(f"  指定续训: {resume_path}")
    else:
        resume_path = os.path.join(config.checkpoint_dir, 'last_step.pt')
        if not os.path.exists(resume_path):
            resume_path = None
        else:
            print(f"  检测到断点: {resume_path}")

    if resume_path:
        try:
            ckpt = load_checkpoint(resume_path, str(device))
            state_dict = ckpt['model_state_dict']

            # 删除 RoPE cos/sin: checkpoint 的 max_seq_len 可能与当前训练不同
            # (如嫁接时 8192, 训练时 512)。RoPE 是确定性 buffer，模型初始化时已正确计算。
            rope_keys = [k for k in state_dict if '.rope.cos' in k or '.rope.sin' in k]
            for k in rope_keys:
                del state_dict[k]
            if rope_keys:
                print(f"  已跳过 {len(rope_keys)} 个 RoPE 参数 (尺寸不匹配, 使用当前模型)")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            # 过滤 RoPE 缺失 (已故意删除，用当前模型尺寸)
            missing = [k for k in missing if '.rope.cos' not in k and '.rope.sin' not in k]
            if missing:
                print(f"  缺失 key ({len(missing)}): {missing[:3]}...")
            if unexpected:
                print(f"  多余 key ({len(unexpected)}): {unexpected[:3]}...")

            # 续训 checkpoint 才有 optimizer/scheduler；嫁接初始权重没有
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt['epoch']
                global_step = ckpt['global_step']
                best_ppl = ckpt.get('best_ppl', float('inf'))
                print(f"  → 从 epoch {start_epoch}, step {global_step} 继续")
            else:
                print(f"  → 加载初始权重成功 (从头训练)")
        except Exception as e:
            print(f"  [警告] 续训失败: {e}，从头开始")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # torch.compile 加速（在续训加载权重之后编译）
    compile_model = getattr(config, 'compile_model', False)
    if compile_model and use_gpu and cap_major >= 7:
        try:
            print("  编译模型 (torch.compile)...", end=" ", flush=True)
            t_compile = time.time()
            model = torch.compile(model, mode="reduce-overhead")
            print(f"完成 ({time.time()-t_compile:.1f}s)")
        except Exception as e:
            print(f"跳过 ({e})")

    # ── 6. 训练循环 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("5/5 开始训练")
    print(f"  Epochs: {config.epochs} | Warmup: {config.warmup_steps} steps | Grad Accum: {config.grad_accum_steps}x")
    print(f"  Total optimizer steps: ~{total_steps}")
    print("=" * 60)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_moe_loss = 0.0
        accum_loss = 0.0
        accum_steps = 0
        t0 = time.time()

        import tqdm as _tqdm
        n_opt_steps = len(train_loader) // config.grad_accum_steps
        resume_step = global_step  # 续训起始步
        skip_batches = global_step * config.grad_accum_steps if epoch == start_epoch else 0
        if skip_batches > 0:
            print(f"  跳过 {skip_batches} 个已完成的 batch (从 step {global_step} 继续)...")
        pbar = _tqdm.tqdm(range(len(train_loader)), desc=f"Epoch {epoch}", leave=True)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx < skip_batches:
                pbar.update(1)
                continue
            inputs = inputs.to(device, non_blocking=use_gpu)
            targets = targets.to(device, non_blocking=use_gpu)

            # ── 前向传播 (AMP + 梯度检查点) ──
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(inputs, use_checkpoint=True)
                    ce_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=PAD_ID,
                    )
                    moe_loss = _collect_moe_loss(model) if use_moe else torch.tensor(0.0, device=device)
                    loss = (ce_loss + moe_loss) / config.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits, _ = model(inputs, use_checkpoint=True)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=PAD_ID,
                )
                moe_loss = _collect_moe_loss(model) if use_moe else torch.tensor(0.0, device=device)
                loss = (ce_loss + moe_loss) / config.grad_accum_steps
                loss.backward()

            accum_loss += ce_loss.item()
            epoch_moe_loss += moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss
            accum_steps += 1

            # ── 梯度累积更新 ──
            if accum_steps == config.grad_accum_steps or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_loss += accum_loss
                avg = epoch_loss / max(1, global_step - resume_step) / config.grad_accum_steps
                accum_loss = 0.0
                accum_steps = 0
                pbar.set_postfix(step=global_step, loss=f"{avg:.4f}")
                pbar.update(config.grad_accum_steps)

                # ── 看门狗保存 (每 N 步更新 last_step.pt，每 1000 步留里程碑) ──
                if global_step % config.save_every_steps == 0:
                    save_checkpoint(
                        os.path.join(config.checkpoint_dir, 'last_step.pt'),
                        model, optimizer, scheduler, tokenizer, config,
                        epoch, global_step,
                        epoch_loss / max(1, global_step - resume_step),
                        best_ppl,
                    )
                    msg = f"  → 步级保存: step{global_step} (last_step.pt)"
                    # 每 1000 步覆盖一个轻量里程碑（仅权重，下载测试用）
                    if global_step % 1000 == 0:
                        save_model_only(
                            os.path.join(config.checkpoint_dir, 'milestone.pt'),
                            model, tokenizer, epoch, global_step, best_ppl,
                        )
                        msg += f", 里程碑 milestone.pt (step{global_step}, ~3.3G)"
                    print(msg)

                # ── 步级评估 ──
                if global_step % config.eval_every_steps == 0:
                    val_ppl = evaluate(model, val_loader, max_batches=10)
                    lr = scheduler.get_lr()
                    moe_str = f"MoE: {moe_loss.item():.6f} | " if use_moe else ""
                    print(
                        f"  Step {global_step:6d} | CE: {ce_loss.item():.4f} | "
                        f"{moe_str}"
                        f"PPL: {val_ppl:.2f} | LR: {lr:.2e}"
                    )
                    # 专家使用率诊断 (选前/中/后 3 层)
                    if use_moe:
                        n_layers = len(model.blocks)
                        sample_layers = [0, n_layers // 2, n_layers - 1]
                        for li in sample_layers:
                            usage = model.blocks[li].moe.expert_usage
                            if usage is not None:
                                usage_list = [f"{u*100:.1f}%" for u in usage.tolist()]
                                print(f"  L{li:02d} 专家分配: {' | '.join(usage_list)}")

        # ── Epoch 结束 ──
        pbar.close()
        n_updates = (len(train_loader) + config.grad_accum_steps - 1) // config.grad_accum_steps
        avg_ce = epoch_loss / max(n_updates, 1)
        avg_moe = epoch_moe_loss / max(n_updates * config.grad_accum_steps, 1)
        elapsed = time.time() - t0

        # 验证
        val_ppl = evaluate(model, val_loader, max_batches=20)
        lr = scheduler.get_lr()

        # 显存报告
        mem_str = ""
        if use_gpu:
            mem_str = f" | GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB"
            torch.cuda.empty_cache()

        print(f"\n=== Epoch {epoch:3d}/{config.epochs} | CE: {avg_ce:.4f} | MoE: {avg_moe:.4f} | "
              f"PPL: {val_ppl:.2f} | LR: {lr:.2e} | {elapsed:.0f}s{mem_str} ===\n")

        # ── 轮级保存 ──
        if epoch % config.save_every_epochs == 0:
            save_checkpoint(
                os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pt'),
                model, optimizer, scheduler, tokenizer, config,
                epoch, global_step, avg_ce, best_ppl,
            )
            print(f"  → 轮级保存: checkpoint_epoch{epoch}.pt")

        # ── 最佳模型（仅权重，~3.3G，推理用）──
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            save_model_only(
                os.path.join(config.checkpoint_dir, 'best_model.pt'),
                model, tokenizer, epoch, global_step, best_ppl,
            )
            print(f"  → 新最佳! PPL={best_ppl:.2f}")

    # ── 训练结束 ──
    save_checkpoint(
        os.path.join(config.checkpoint_dir, 'final_model.pt'),
        model, optimizer, scheduler, tokenizer, config,
        config.epochs, global_step, avg_ce, best_ppl,
    )
    # 额外保存一个轻量权重版方便下载
    save_model_only(
        os.path.join(config.checkpoint_dir, 'model_weights.pt'),
        model, tokenizer, config.epochs, global_step, best_ppl,
    )
    print(f"\n✓ 训练完成! 最佳困惑度: {best_ppl:.2f}")


def _collect_moe_loss(model: TGAILanguageModel) -> torch.Tensor:
    """收集所有 MoE 层的负载均衡损失"""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    if not model.training:
        return total
    for block in model.blocks:
        total = total + block.moe.load_balance_loss
    return total * model.config.moe_load_balance


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TGAI V4 训练')
    parser.add_argument('--data', type=str, default='data/train_all.jsonl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--n_activated', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--resume', type=str, default='', help='续训checkpoint路径 (留空自动找 last_step.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint保存目录 (默认checkpoints/)')
    parser.add_argument('--cache_dir', type=str, default='', help='数据集缓存目录 (默认checkpoints/cache)')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--compile', action='store_true', help='启用 torch.compile 加速 (4090 可提速 20-40%%)')
    parser.add_argument('--save_every', type=int, default=0, help='步级保存间隔 (覆盖默认值，建议 50-500)')
    args = parser.parse_args()

    config = TrainConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_experts=args.n_experts,
        n_activated=args.n_activated,
        dropout=args.dropout,
        warmup_steps=args.warmup,
        grad_accum_steps=args.grad_accum,
        device="cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.resume:
        config._resume_path = args.resume
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.save_every > 0:
        config.save_every_steps = args.save_every
    if args.compile:
        config.compile_model = True
    train(config)