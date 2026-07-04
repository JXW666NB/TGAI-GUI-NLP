"""
TGAI V4 — MoE Transformer 语言模型 + 多模态接口
=================================================
架构: Embedding → N×Block(RMSNorm→FlashAttn(RoPE)→RMSNorm→MoE(SwiGLU)) → RMSNorm → LM Head

新特性:
  - MoE (Mixture of Experts): 4专家, Top-2路由, 负载均衡loss
  - Flash Attention: torch.nn.functional.scaled_dot_product_attention
  - KV Cache: 推理加速, 避免重复计算
  - 多模态接口: ImageEncoder / AudioEncoder / VideoEncoder (预留)
  - Weight Tying: embedding ↔ lm_head 共享权重
  - 8层 × 256维, ~25M参数
"""

import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 特殊token ID（与 tokenizer.py 保持一致）
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
@dataclass
class TGAIConfig:
    vocab_size: int = 16384
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0

    # MoE
    n_experts: int = 4
    n_activated: int = 2       # Top-K 激活专家数
    moe_load_balance: float = 0.01  # 负载均衡loss权重

    # RoPE
    rope_theta: float = 10000.0

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


# ═══════════════════════════════════════════════════════════
# RoPE
# ═══════════════════════════════════════════════════════════
class RotaryPositionEmbedding(nn.Module):
    """RoPE 旋转位置编码。使用标准 rotate_half 实现，避免 float32 升降精度。"""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)  # (max_seq_len, dim//2)
        # 每个频率值重复2次以匹配交错配对
        cos_dup = angles.cos().repeat_interleave(2, dim=-1)  # (max_seq_len, dim)
        sin_dup = angles.sin().repeat_interleave(2, dim=-1)
        # 预广播: (1, 1, max_seq_len, dim)
        self.register_buffer('cos', cos_dup.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin', sin_dup.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """交错取反: x1,x2,x3,x4 → -x2,x1,-x4,x3（ExecuTorch 兼容）"""
        result = torch.empty_like(x)
        result[..., ::2] = -x[..., 1::2]
        result[..., 1::2] = x[..., ::2]
        return result

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if position_ids is not None:
            # 使用指定位置（推理时传入 cache_pos）
            positions = position_ids  # (1, seq_len)
            cos = self.cos[:, :, positions, :].squeeze(2)  # (1, 1, seq, dim)
            sin = self.sin[:, :, positions, :].squeeze(2)
        else:
            # 训练/prefill：按顺序 0,1,2,...
            cos = self.cos[:, :, :seq_len, :]
            sin = self.sin[:, :, :seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)


# ═══════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════
# SwiGLU FFN (专家网络)
# ═══════════════════════════════════════════════════════════
class SwiGLUFFN(nn.Module):
    def __init__(self, config: TGAIConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ═══════════════════════════════════════════════════════════
# MoE 层
# ═══════════════════════════════════════════════════════════
class MoELayer(nn.Module):
    """
    混合专家层。每个 token 通过路由器选择 top-k 个专家处理。
    输出 = Σ(router_prob_i × expert_i(x)) × gate_scale
    """

    def __init__(self, config: TGAIConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_activated = config.n_activated
        self.load_balance_weight = config.moe_load_balance

        # 路由器
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # 专家网络
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(config.n_experts)
        ])
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)  # 门控缩放

        self._load_balance_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        N = x_flat.shape[0]

        # 路由 logits
        router_logits = self.router(x_flat)  # (B*T, n_experts)

        # Top-K 选择
        topk_vals, topk_ids = torch.topk(router_logits, self.n_activated, dim=-1)
        router_probs = F.softmax(topk_vals, dim=-1)  # (B*T, K)

        # 负载均衡 loss
        if self.training:
            gate_logits = self.gate(x_flat)
            expert_probs = F.softmax(gate_logits, dim=-1)
            expert_usage = expert_probs.mean(dim=0)
            target_usage = torch.ones_like(expert_usage) / self.n_experts
            self._load_balance_loss = F.mse_loss(expert_usage, target_usage)

        # ── 高效推理路径: 单 token 直接索引，避免 Python 循环 ──
        if not self.training and N == 1:
            output = torch.zeros_like(x_flat)
            for k in range(self.n_activated):
                eid = topk_ids[0, k].item()
                prob = router_probs[0, k]
                output += self.experts[eid](x_flat) * prob
            return output.view(B, T, D)

        # ── 通用路径: 按专家分组批量计算 ──
        output = torch.zeros_like(x_flat)
        for eid in range(self.n_experts):
            # 收集所有分配给该专家的 (token_idx, activation_idx, prob)
            mask = (topk_ids == eid)  # (N, K)
            if not mask.any():
                continue
            # 对每个激活位置分别处理
            for k in range(self.n_activated):
                tok_mask = mask[:, k]  # (N,)
                if not tok_mask.any():
                    continue
                expert_input = x_flat[tok_mask]
                expert_out = self.experts[eid](expert_input)
                output[tok_mask] += expert_out * router_probs[tok_mask, k].unsqueeze(-1)

        return output.view(B, T, D)

    @property
    def load_balance_loss(self) -> float:
        return self._load_balance_loss


# ═══════════════════════════════════════════════════════════
# Flash Attention
# ═══════════════════════════════════════════════════════════
class FlashSelfAttention(nn.Module):
    """使用 torch.nn.functional.scaled_dot_product_attention 的因果自注意力"""

    def __init__(self, config: TGAIConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryPositionEmbedding(config.d_k, config.max_seq_len, config.rope_theta)
        self.dropout_p = config.dropout
        self._max_seq_len = config.max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        cache_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # RoPE — 解码时传入真实位置
        if kv_cache is not None and kv_cache.get('k') is not None and T == 1:
            # 解码阶段：单 token，使用真实位置
            pos_ids = torch.tensor([[cache_pos]], device=x.device)
            q = self.rope(q, position_ids=pos_ids)
            k = self.rope(k, position_ids=pos_ids)
        else:
            # 训练或 prefill：按顺序计算位置
            q = self.rope(q)
            k = self.rope(k)

        # KV Cache — 预分配 buffer 消除 torch.cat 碎片化
        new_cache = None
        if kv_cache is not None:
            if kv_cache.get('k') is not None:
                # 预分配模式: slice 写入新 KV
                kv_cache['k'][:, :, cache_pos:cache_pos + T] = k
                kv_cache['v'][:, :, cache_pos:cache_pos + T] = v
                kv_len = cache_pos + T
                k_attn = kv_cache['k'][:, :, :kv_len]
                v_attn = kv_cache['v'][:, :, :kv_len]
            else:
                # 首次 prefill: 创建预分配 buffer
                k_buf = torch.zeros(B, self.n_heads, self._max_seq_len, self.d_k,
                                    dtype=k.dtype, device=k.device)
                v_buf = torch.zeros(B, self.n_heads, self._max_seq_len, self.d_k,
                                    dtype=v.dtype, device=v.device)
                k_buf[:, :, :T] = k
                v_buf[:, :, :T] = v
                k_attn, v_attn = k_buf[:, :, :T], v_buf[:, :, :T]
                kv_cache['k'] = k_buf
                kv_cache['v'] = v_buf
            new_cache = kv_cache
        else:
            k_attn, v_attn = k, v

        # Flash Attention (sdpa)
        attn_out = F.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(kv_cache is None),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out), new_cache


# ═══════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    """Pre-Norm: x = x + Attn(RMSNorm(x)), x = x + MoE(RMSNorm(x))"""

    def __init__(self, config: TGAIConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.attn = FlashSelfAttention(config)
        self.moe = MoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        cache_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache, cache_pos)
        x = x + attn_out
        x = x + self.moe(self.ln2(x))
        return x, new_cache


# ═══════════════════════════════════════════════════════════
# 多模态编码器接口 (预留)
# ═══════════════════════════════════════════════════════════
class MultimodalEncoder(nn.Module):
    """多模态编码器基类 —— 后续实现 Image/Audio/Video 继承此类"""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)  # placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入原始信号，输出 (B, T, d_model) 特征"""
        raise NotImplementedError

    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """将模态输入编码为可与文本拼接的 token embeddings"""
        features = self.forward(x)  # (B, T_m, d_model)
        return features


class ImageEncoder(MultimodalEncoder):
    """图片编码器 (预留) —— 计划使用轻量 ViT/CLIP"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("图片编码器待实现")


class AudioEncoder(MultimodalEncoder):
    """音频编码器 (预留) —— 计划使用 Whisper encoder"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("音频编码器待实现")


class VideoEncoder(MultimodalEncoder):
    """视频编码器 (预留) —— 计划使用 TimeSformer"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("视频编码器待实现")


# ═══════════════════════════════════════════════════════════
# TGAI 语言模型
# ═══════════════════════════════════════════════════════════
class TGAILanguageModel(nn.Module):
    """
    V4 Decoder-Only Transformer + MoE + Flash Attention + 多模态接口
    """

    def __init__(self, config: TGAIConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_token_id,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight Tying

        self.dropout = nn.Dropout(config.dropout)

        # 多模态编码器 (预留)
        self.image_encoder: Optional[ImageEncoder] = None
        self.audio_encoder: Optional[AudioEncoder] = None
        self.video_encoder: Optional[VideoEncoder] = None

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / math.sqrt(2 * self.config.n_layers) if 'w2' in name else 0.02
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ─── 多模态接口 ──────────────────────────────────
    def set_image_encoder(self, encoder: ImageEncoder):
        self.image_encoder = encoder

    def set_audio_encoder(self, encoder: AudioEncoder):
        self.audio_encoder = encoder

    def set_video_encoder(self, encoder: VideoEncoder):
        self.video_encoder = encoder

    def embed_multimodal(
        self,
        text_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        将文本和多模态输入融合为 embedding 序列。
        文本: token_embedding(token_ids)
        图片: image_encoder(image) + 位置嵌入
        """
        text_emb = self.token_embedding(text_ids)  # (B, T, D)
        embeddings = [text_emb]

        if image is not None and self.image_encoder is not None:
            img_emb = self.image_encoder.encode_to_tokens(image)
            embeddings.insert(0, img_emb)  # 图片放在文本前面

        if audio is not None and self.audio_encoder is not None:
            aud_emb = self.audio_encoder.encode_to_tokens(audio)
            embeddings.insert(0, aud_emb)

        if video is not None and self.video_encoder is not None:
            vid_emb = self.video_encoder.encode_to_tokens(video)
            embeddings.insert(0, vid_emb)

        if len(embeddings) == 1:
            return text_emb
        return torch.cat(embeddings, dim=1)

    # ─── 前向传播 ────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        cache_pos: int = 0,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[Dict[str, torch.Tensor]]]]]:
        """
        Args:
            input_ids: (B, T)
            kv_caches: KV Cache 列表
            image/audio/video: 多模态输入 (预留)
        Returns:
            logits: (B, T, vocab_size)
            new_kv_caches: 更新的 KV Cache
        """
        # 多模态 embedding
        x = self.embed_multimodal(input_ids, image, audio, video)
        x = self.dropout(x)

        new_caches = [] if kv_caches is not None else None
        load_balance_total = 0.0

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache, cache_pos)
            if new_caches is not None:
                new_caches.append(new_cache)
            if self.training:
                load_balance_total += block.moe.load_balance_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_caches

    # ─── 推理 (KV Cache 加速) ─────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 80,
        top_p: float = 0.9,
        eos_token_id: int = EOS_ID,
        min_new_tokens: int = 5,
        repetition_penalty: float = 1.05,
        frequency_penalty: float = 0.15,
    ) -> torch.Tensor:
        """
        自回归生成，使用 KV Cache 加速。

        Args:
            frequency_penalty: 频率惩罚 (0.1~0.3, 越大越不重复)
            repetition_penalty: 1.0=禁用, 1.05=轻度惩罚
        """
        self.eval()
        prompt_ids = prompt_ids.clamp(0, self.config.vocab_size - 1)
        device = prompt_ids.device
        inv_temp = 1.0 / max(temperature, 0.01)
        vocab_size = self.config.vocab_size

        # Prefill: 一次性处理 prompt
        kv_caches = [{} for _ in range(self.config.n_layers)]
        logits, kv_caches = self(prompt_ids, kv_caches)
        next_logits = logits[:, -1, :] * inv_temp

        generated = prompt_ids.clone()
        new_tokens = 0
        token_counts = torch.zeros(vocab_size, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len // 2:]
                kv_caches = [{} for _ in range(self.config.n_layers)]
                token_counts.zero_()
                for t in generated[0].tolist():
                    if 0 <= t < vocab_size:
                        token_counts[t] += 1

            # 前 min_new_tokens 禁止 EOS
            if new_tokens < min_new_tokens and eos_token_id < vocab_size:
                next_logits[:, eos_token_id] = float('-inf')

            # 轻度频率惩罚 (对欠训练模型更温和)
            if frequency_penalty > 0:
                appeared = token_counts > 0
                if appeared.any():
                    penalty = frequency_penalty * token_counts[appeared].float().unsqueeze(0).clamp(max=3.0)
                    next_logits[:, appeared] -= penalty

            # 中等重复惩罚
            if repetition_penalty > 1.0:
                in_text = token_counts > 0
                if in_text.any():
                    next_logits[:, in_text] /= repetition_penalty

            # Top-K (确保至少保留 EOS)
            if top_k > 0:
                raw_topk = next_logits.clone()  # 备份以恢复 EOS
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k)[0][..., -1, None]
                next_logits[next_logits < threshold] = float('-inf')
                # 如果 EOS 被裁掉了, 恢复它 (让模型有机会结束)
                if eos_token_id < vocab_size and torch.isinf(next_logits[0, eos_token_id]):
                    next_logits[0, eos_token_id] = raw_topk[0, eos_token_id]

            # Top-P
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                keep = cum <= top_p
                keep[..., 0] = True
                next_logits[:, sorted_i[~keep]] = float('-inf')

            # 防止全-inf崩溃: 回退到均匀分布（排除 PAD/UNK/BOS）
            if torch.all(torch.isinf(next_logits)):
                next_logits = torch.ones_like(next_logits) * float('-inf')
                # 只给 non-special tokens 概率
                safe_start = max(eos_token_id + 1, 4)
                next_logits[0, safe_start:safe_start + min(50, vocab_size - safe_start)] = 0.0
                # 确保 EOS 也在选项中
                if eos_token_id < vocab_size:
                    next_logits[0, eos_token_id] = 0.0

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).clamp(0, vocab_size - 1)

            token_id = next_token.item()
            generated = torch.cat([generated, next_token], dim=1)
            new_tokens += 1
            if 0 <= token_id < vocab_size:
                token_counts[token_id] += 1

            if token_id == eos_token_id:
                break

            # Decode step: 只处理新 token
            token_logits, kv_caches = self(next_token, kv_caches, cache_pos=generated.shape[1] - 1)
            next_logits = token_logits[:, -1, :] * inv_temp

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════
def create_model(
    vocab_size: int = 16384,
    d_model: int = 256,
    n_layers: int = 8,
    n_heads: int = 8,
    d_ff: int = 1024,
    max_seq_len: int = 256,
    dropout: float = 0.1,
    n_experts: int = 4,
    n_activated: int = 2,
) -> TGAILanguageModel:
    config = TGAIConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        n_experts=n_experts,
        n_activated=n_activated,
    )
    return TGAILanguageModel(config)