"""
TGAI LoRA SFT 微调
==================
基于 LoRA (Low-Rank Adaptation) 高效微调 TGAI 模型。

特性:
  - LoRA 注入 Attention Q/K/V/O + MoE FFN 层
  - 冻结基座模型，仅训练低秩适配器
  - 支持 SFT 数据加载 (用户:/TGAI? 格式)
  - 梯度检查点 + AMP 混合精度
  - 断点续训

用法:
    python train_lora.py \
        --base_model checkpoints/tgai_sft_v3.pt \
        --data /root/autodl-tmp/TGAI_data/sft/train_all.jsonl \
        --lora_r 16 --lora_alpha 32 \
        --seq_len 512 --batch 4 --grad_accum 4 --epochs 3

LoRA 原理:
    对于预训练权重矩阵 W ∈ R^{d×k}:
      h = Wx + (α/r) · BAx
    其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
    训练时只更新 A 和 B，W 冻结。
"""

import os, sys, json, time, math, hashlib, argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import ChineseTokenizer, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from model import TGAILanguageModel, TGAIConfig, create_model


# ═══════════════════════════════════════════════════════════
# LoRA 模块
# ═══════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """LoRA 包装 nn.Linear: h = Wx + (α/r) · BAx"""

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear  # 冻结的原始权重
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 冻结原始权重
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        out_dim, in_dim = linear.weight.shape
        # A: (in_dim, r) 用 Kaiming 初始化
        self.lora_A = nn.Parameter(torch.zeros(in_dim, r))
        # B: (r, out_dim) 用零初始化 (初始时 LoRA 不改变输出)
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向
        result = self.linear(x)
        # LoRA 增量: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return result + lora_out * self.scaling

    def merge_to_base(self):
        """将 LoRA 权重合并到原始权重中 (推理时使用)"""
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B).T * self.scaling
            self.linear.weight.data += delta
        # 合并后清空 LoRA 参数
        self.lora_A.zero_()
        self.lora_B.zero_()

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def inject_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: List[str] = None,
) -> Tuple[nn.Module, int]:
    """
    递归注入 LoRA 到模型的 Linear 层中。
    
    target_modules: 要注入的模块名列表，如 ['q_proj','k_proj','v_proj','out_proj','w1','w2','w3','router','gate']
                    None 表示注入所有 nn.Linear (不含 bias)
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj',
                          'w1', 'w2', 'w3', 'router', 'gate']

    lora_params = 0

    def _inject(module: nn.Module, prefix: str = ""):
        nonlocal lora_params
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                # 判断是否为目标模块
                should_lora = any(t in name for t in target_modules)
                if should_lora:
                    lora_linear = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_linear)
                    lora_params += child.weight.numel()
            else:
                _inject(child, full_name)

    _inject(model)
    return model, lora_params


def count_lora_params(model: nn.Module) -> Tuple[int, int, int]:
    """统计 LoRA 可训参数 / 总参数 / 冻结参数"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return trainable, total, frozen


def save_lora_weights(model: nn.Module, path: str, global_step: int = 0):
    """只保存 LoRA 权重 (A, B 矩阵)"""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()
    torch.save({
        'lora_config': {
            'r': next(iter(lora_state.values())).shape[1] if lora_state else 0,
            'alpha': 0,
        },
        'lora_state_dict': lora_state,
        'global_step': global_step,
    }, path)
    return len(lora_state) // 2


def load_lora_weights(model: nn.Module, path: str):
    """加载 LoRA 权重到模型"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    lora_state = ckpt.get('lora_state_dict', {})
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state:
                module.lora_A.data = lora_state[f"{name}.lora_A"].to(module.lora_A.device)
                module.lora_B.data = lora_state[f"{name}.lora_B"].to(module.lora_B.device)
    print(f"  LoRA 权重已加载: {len(lora_state)//2} 层")


# ═══════════════════════════════════════════════════════════
# 训练配置
# ═══════════════════════════════════════════════════════════
@dataclass
class LoRATrainConfig:
    # 基座模型
    base_model: str = "checkpoints/tgai_sft_v3.pt"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    # LoRA
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0

    # 数据
    data_path: str = "data/sft_train.jsonl"
    seq_len: int = 512
    cache_dir: str = "/root/autodl-tmp/TGAI_data/cache_lora"

    # 训练
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # 保存
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    checkpoint_dir: str = "checkpoints/lora"

    # 设备
    device: str = "cuda"
    compile_model: bool = False

    # 续训
    resume_lora: str = None
    resume_step: int = 0  # 手动指定 global_step (兼容旧 checkpoint)


# ═══════════════════════════════════════════════════════════
# 数据集 — 最大 packing (填充到 seq_len)
# ═══════════════════════════════════════════════════════════
class SFTTextDataset(Dataset):
    """将 SFT JSONL 文本编码为固定长度序列，支持缓存。"""

    def __init__(self, texts: List[str], tokenizer: ChineseTokenizer, seq_len: int,
                 cache_path: Optional[str] = None):
        self.seq_len = seq_len
        if cache_path and os.path.exists(cache_path):
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            self.inputs = data['inputs']
            self.targets = data['targets']
            print(f"  [数据集] 缓存加载: {len(self.inputs)} 样本")
            return
        self.inputs, self.targets = self._encode(texts, tokenizer)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            torch.save({'inputs': self.inputs, 'targets': self.targets}, cache_path)
            print(f"  [数据集] 缓存保存: {cache_path}")

    def _encode(self, texts, tokenizer):
        inputs, targets = [], []
        print(f"  [数据集] 编码 {len(texts)} 条...")
        for text in texts:
            ids = tokenizer.encode(text, add_special=True)  # BOS + text + EOS
            ids = [min(t, tokenizer.vocab_size_actual - 1) for t in ids]
            if len(ids) < 2:
                continue
            # 截断或填充到 seq_len
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
            else:
                ids = ids + [PAD_ID] * (self.seq_len - len(ids))
            # input = 全序列, target = 左移一位 (预测下一个 token)
            inp = torch.tensor(ids, dtype=torch.long)
            tgt = torch.tensor(ids[1:] + [PAD_ID], dtype=torch.long)
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_sft_texts(data_path: str) -> List[str]:
    """加载 SFT 训练数据的 text 字段"""
    texts = []
    for path in data_path.split(","):
        path = path.strip()
        if not os.path.exists(path):
            print(f"  [警告] 文件不存在: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
    print(f"  加载 {len(texts)} 条 SFT 文本")
    return texts


# ═══════════════════════════════════════════════════════════
# 学习率调度器
# ═══════════════════════════════════════════════════════════
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = lr * (blr / self.base_lrs[0]) if self.base_lrs[0] > 0 else lr

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.base_lrs[0] * (step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════
def train_lora(config: LoRATrainConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == 'cuda'
    if use_gpu:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    else:
        print(f"设备: CPU")

    # ── 1. 加载基座模型 ──
    print("\n" + "=" * 60)
    print("1/5 加载基座模型")
    if not os.path.exists(config.base_model):
        raise FileNotFoundError(f"基座模型未找到: {config.base_model}")
    ckpt = torch.load(config.base_model, map_location='cpu', weights_only=False)
    mc = ckpt.get('model_config', {})
    state_dict = ckpt['model_state_dict']
    ckpt_vocab = state_dict['token_embedding.weight'].shape[0]
    ckpt_d_model = state_dict['token_embedding.weight'].shape[1]

    print(f"  词表: {ckpt_vocab}, d_model: {ckpt_d_model}")
    print(f"  训练步数: {ckpt.get('step', ckpt.get('global_step', '?'))}")

    model = create_model(
        vocab_size=ckpt_vocab,
        d_model=mc.get('d_model', ckpt_d_model),
        n_layers=mc.get('n_layers', 8),
        n_heads=mc.get('n_heads', 8),
        d_ff=mc.get('d_ff', 1024),
        max_seq_len=mc.get('max_seq_len', config.seq_len),
        dropout=mc.get('dropout', 0.0),
        n_experts=mc.get('n_experts', 4),
        n_activated=mc.get('n_activated', 2),
    )

    # 向后兼容: RoPE cos/sin 迁移
    for key in list(state_dict.keys()):
        if '.rope.cos' in key or '.rope.sin' in key:
            if state_dict[key].dim() == 2:
                state_dict[key] = state_dict[key].repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  基座参数: {n_params:,} (~{n_params/1e6:.1f}M)")

    # ── 2. 注入 LoRA ──
    print("\n" + "=" * 60)
    print(f"2/5 注入 LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
    model, _ = inject_lora(model, r=config.lora_r, alpha=config.lora_alpha,
                           dropout=config.lora_dropout)
    trainable, total, frozen = count_lora_params(model)
    print(f"  LoRA 可训: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  冻结: {frozen:,}")
    model.to(device)

    # torch.compile 加速 (可选)
    if config.compile_model and use_gpu:
        try:
            print(f"  编译模型 (torch.compile)...", end=" ", flush=True)
            t_compile = time.time()
            model = torch.compile(model, mode="default")
            print(f"完成 ({time.time()-t_compile:.1f}s)")
        except Exception as e:
            print(f"跳过 ({e})")

    # ── 3. 分词器 & 数据 ──
    print("\n" + "=" * 60)
    print("3/5 加载数据")
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(f"分词器未找到: {config.tokenizer_path}")
    tokenizer = ChineseTokenizer.load(config.tokenizer_path)
    print(f"  词表: {tokenizer.vocab_size_actual}")

    texts = load_sft_texts(config.data_path)
    if len(texts) < 10:
        raise ValueError(f"训练数据太少: {len(texts)} 条")

    # 划分训练/验证
    import random
    random.seed(42)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.95 * len(texts))
    train_texts = [texts[i] for i in indices[:split]]
    val_texts = [texts[i] for i in indices[split:]]

    data_hash = hashlib.md5(config.data_path.encode()).hexdigest()[:8]
    os.makedirs(config.cache_dir, exist_ok=True)
    train_ds = SFTTextDataset(train_texts, tokenizer, config.seq_len,
                               cache_path=os.path.join(config.cache_dir, f"sft_train_s{config.seq_len}_{data_hash}.pt"))
    val_ds = SFTTextDataset(val_texts, tokenizer, config.seq_len,
                             cache_path=os.path.join(config.cache_dir, f"sft_val_s{config.seq_len}_{data_hash}.pt"))

    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True, drop_last=True,
                               num_workers=4, pin_memory=use_gpu, persistent_workers=True,
                               prefetch_factor=2)
    val_loader = DataLoader(val_ds, config.batch_size, shuffle=False, drop_last=True,
                             num_workers=2, pin_memory=use_gpu, persistent_workers=True,
                             prefetch_factor=2)
    print(f"  训练: {len(train_ds)} 样本, {len(train_loader)} batches/epoch")
    print(f"  验证: {len(val_ds)} 样本")

    # ── 4. 优化器 ──
    print("\n" + "=" * 60)
    print("4/5 优化器")
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(0.9, 0.95), fused=use_gpu)
    total_steps = (len(train_loader) // config.grad_accum_steps) * config.epochs
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_steps, total_steps)
    print(f"  总步数: {total_steps}, Warmup: {config.warmup_steps}")

    # 续训
    start_epoch = 0
    global_step = 0
    if config.resume_lora and os.path.exists(config.resume_lora):
        lora_ckpt = torch.load(config.resume_lora, map_location='cpu', weights_only=False)
        load_lora_weights(model, config.resume_lora)
        if 'optimizer' in lora_ckpt:
            optimizer.load_state_dict(lora_ckpt['optimizer'])
        start_epoch = lora_ckpt.get('epoch', 0)
        global_step = config.resume_step or lora_ckpt.get('global_step', 0)
        print(f"  → 续训: epoch={start_epoch}, step={global_step}")

    # ── 5. 训练循环 ──
    print("\n" + "=" * 60)
    print("5/5 开始训练")
    print("=" * 60)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_gpu else None
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs} | 总优化步数: ~{total_steps//config.epochs}/epoch")
        print(f"{'='*60}")
        epoch_loss = 0.0
        accum_loss = 0.0
        epoch_opt_steps = 0
        epoch_tokens = 0
        t0 = time.time()

        import tqdm as _tqdm
        skip_batches = global_step * config.grad_accum_steps if epoch == start_epoch else 0
        if skip_batches > 0:
            print(f"  跳过 {skip_batches} 个已完成的 batch (从 step {global_step} 继续)...")
        pbar = _tqdm.tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx < skip_batches:
                if batch_idx % config.grad_accum_steps == 0:
                    pbar.set_postfix({"step": f"skip→{skip_batches//config.grad_accum_steps}"})
                pbar.update(1)
                continue

            inputs = inputs.to(device, non_blocking=use_gpu)
            targets = targets.to(device, non_blocking=use_gpu)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_gpu else torch.no_grad():
                logits, _ = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=PAD_ID,
                    label_smoothing=0.02,
                )
                loss = loss / config.grad_accum_steps

            if use_gpu:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * config.grad_accum_steps
            epoch_tokens += (targets != PAD_ID).sum().item()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_gpu:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_opt_steps += 1
                epoch_loss += accum_loss
                avg = epoch_loss / max(1, epoch_opt_steps) / config.grad_accum_steps
                lr = scheduler.get_lr()
                elapsed = time.time() - t0
                tokens_per_s = epoch_tokens / max(elapsed, 1)
                gpu_str = f"{torch.cuda.memory_allocated()/1e9:.1f}G" if use_gpu else ""
                warmup_str = "[WARMUP]" if global_step <= config.warmup_steps else ""
                best_ppl_str = f"{math.exp(best_val_loss):.1f}" if best_val_loss < float('inf') else "-"
                pbar.set_postfix({
                    "step": global_step, "loss": f"{avg:.4f}",
                    "lr": f"{lr:.1e}", "gn": f"{grad_norm:.1f}",
                    "tk/s": f"{tokens_per_s:.0f}", "gpu": gpu_str,
                    "bestPPL": best_ppl_str,
                    "phase": warmup_str if warmup_str else "train",
                })
                pbar.update(config.grad_accum_steps)
                accum_loss = 0.0

                # 评估
                if global_step % config.eval_every_steps == 0:
                    model.eval()
                    val_loss = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for vi, (v_inputs, v_targets) in enumerate(val_loader):
                            if vi >= 10:
                                break
                            v_inputs = v_inputs.to(device)
                            v_targets = v_targets.to(device)
                            v_logits, _ = model(v_inputs)
                            v_loss = F.cross_entropy(
                                v_logits.view(-1, v_logits.size(-1)), v_targets.view(-1),
                                ignore_index=PAD_ID,
                            )
                            val_loss += v_loss.item()
                            val_count += 1
                    val_loss /= max(val_count, 1)
                    val_ppl = math.exp(val_loss)
                    lr = scheduler.get_lr()
                    improved = "↓" if val_loss < best_val_loss else ("↑" if best_val_loss < float('inf') else "★")
                    print(f"\n  [EVAL] Step {global_step:6d} | Train Loss: {avg:.4f} | "
                          f"Val PPL: {val_ppl:.1f} {improved} | LR: {lr:.2e} | "
                          f"{epoch_tokens/elapsed:.0f}tk/s | GPU: {torch.cuda.memory_allocated()/1e9:.1f}G")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_lora_weights(model, os.path.join(config.checkpoint_dir, "best_lora.pt"), global_step)
                        print(f"    ✓ 最佳模型已保存 (PPL: {val_ppl:.1f})")

                    model.train()

                # 定期保存
                if global_step % config.save_every_steps == 0:
                    save_lora_weights(model, os.path.join(config.checkpoint_dir, f"lora_step_{global_step}.pt"), global_step)
                    print(f"  [SAVE] lora_step_{global_step}.pt")

        pbar.close()
        avg_loss = epoch_loss / max(epoch_opt_steps, 1) / config.grad_accum_steps
        elapsed = time.time() - t0
        tk_per_s = epoch_tokens / max(elapsed, 1)
        gpu_str = f" | GPU: {torch.cuda.memory_allocated()/1e9:.1f}G/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}G" if use_gpu else ""
        print(f"  Epoch {epoch + 1} 完成 | Loss: {avg_loss:.4f} | Best PPL: {math.exp(best_val_loss):.1f} | "
              f"耗时: {elapsed/60:.0f}min | {tk_per_s:.0f} tk/s{gpu_str}")

        # Epoch 结束保存
        save_lora_weights(model, os.path.join(config.checkpoint_dir, f"lora_epoch_{epoch + 1}.pt"), global_step)
        # 保存训练状态
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(config.checkpoint_dir, "trainer_state.pt"))

    # ── 合并 LoRA 到基座模型 (可选) ──
    print("\n" + "=" * 60)
    print("合并 LoRA 权重到基座模型...")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_to_base()
    # 保存完整模型
    full_path = os.path.join(config.checkpoint_dir, "tgai_lora_full.pt")
    torch.save({
        'model_state_dict': {k: v for k, v in model.state_dict().items() if 'lora_A' not in k and 'lora_B' not in k},
        'model_config': {
            'vocab_size': ckpt_vocab, 'd_model': ckpt_d_model,
            'n_layers': mc.get('n_layers', 8), 'n_heads': mc.get('n_heads', 8),
            'd_ff': mc.get('d_ff', 1024), 'max_seq_len': mc.get('max_seq_len', config.seq_len),
            'n_experts': mc.get('n_experts', 4), 'n_activated': mc.get('n_activated', 2),
        },
        'step': global_step,
        'epoch': config.epochs,
    }, full_path)
    print(f"  完整模型: {full_path}")

    print(f"\n{'=' * 60}")
    print(f"✓ 训练完成! | 最佳 Val PPL: {math.exp(best_val_loss):.1f} | 总步数: {global_step}")
    print(f"  完整模型: {full_path}")
    print(f"  适配器目录: {config.checkpoint_dir}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TGAI LoRA SFT 微调")
    parser.add_argument("--base_model", default="checkpoints/tgai_sft_v3.pt", help="基座模型路径")
    parser.add_argument("--tokenizer", dest="tokenizer_path", default="checkpoints/tokenizer.json")
    parser.add_argument("--data", dest="data_path", default="data/sft_train.jsonl",
                        help="SFT 训练数据 (支持逗号分隔多个文件)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch", dest="batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", dest="grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup", dest="warmup_steps", type=int, default=200)
    parser.add_argument("--save_every", dest="save_every_steps", type=int, default=1000)
    parser.add_argument("--eval_every", dest="eval_every_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", default="checkpoints/lora")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/TGAI_data/cache_lora")
    parser.add_argument("--resume_lora", default=None, help="续训 LoRA checkpoint")
    parser.add_argument("--resume_step", type=int, default=0, help="手动指定续训 global_step (兼容旧 checkpoint)")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = parser.parse_args()

    config = LoRATrainConfig(
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        data_path=args.data_path,
        seq_len=args.seq_len,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_every_steps=args.save_every_steps,
        eval_every_steps=args.eval_every_steps,
        checkpoint_dir=args.checkpoint_dir,
        device="cpu" if args.cpu else "cuda",
        compile_model=args.compile_model,
        resume_lora=args.resume_lora,
        resume_step=args.resume_step,
    )
    train_lora(config)


if __name__ == "__main__":
    main()
