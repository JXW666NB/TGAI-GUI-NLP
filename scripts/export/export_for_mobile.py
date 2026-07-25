#!/usr/bin/env python3
"""
TG CHAT — 模型移动端导出工具
==============================
将 TGAI 模型导出为移动端可用的量化格式。

用法:
    python export_for_mobile.py --input checkpoints/milestone.pt --output tg_chat_model
    python export_for_mobile.py --input checkpoints/milestone.pt --output tg_chat_model --quantize int8
    python export_for_mobile.py --input checkpoints/milestone.pt --output tg_chat_model --quantize int4
"""

import sys
import os
import json
import math
import struct
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tgai_nlp'))
from model import TGAILanguageModel, TGAIConfig, create_model


def load_checkpoint(checkpoint_path: str) -> Tuple[dict, dict]:
    """加载 checkpoint，返回 (state_dict, model_config)"""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt['model_state_dict'], ckpt['model_config']


def export_ggml_format(state_dict: dict, model_config: dict, output_dir: str, quant_bits: int = 32):
    """
    导出为简化的 GGML 兼容格式。
    文件结构:
        tg_chat.tgai  — 二进制权重文件
        tg_chat.json  — 模型配置
        tokenizer.json — 分词器（从原路径复制）
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 保存配置 ──
    config = {
        "vocab_size": state_dict['token_embedding.weight'].shape[0],
        "d_model": model_config.get('d_model', state_dict['token_embedding.weight'].shape[1]),
        "n_layers": model_config.get('n_layers', 8),
        "n_heads": model_config.get('n_heads', 8),
        "d_ff": model_config.get('d_ff', 1024),
        "max_seq_len": model_config.get('max_seq_len', 512),
        "n_experts": model_config.get('n_experts', 4),
        "n_activated": model_config.get('n_activated', 2),
        "rope_theta": model_config.get('rope_theta', 10000.0),
        "quant_bits": quant_bits,
    }

    with open(os.path.join(output_dir, 'tg_chat.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── 2. 量化并写出权重 ──
    bin_path = os.path.join(output_dir, 'tg_chat.tgai')
    header_written = False
    total_bytes = 0

    with open(bin_path, 'wb') as f:
        # 遍历每个 tensor
        key_order = _get_key_order(state_dict, config)
        n_tensors = len(key_order)
        f.write(struct.pack('<I', n_tensors))  # 4 bytes: tensor count

        for name in key_order:
            tensor = state_dict[name].float().numpy()

            if quant_bits == 8:
                # INT8 量化
                q_tensor, scale, zero_point = _quantize_int8(tensor)
                dtype_code = 1  # int8
                f.write(struct.pack('<H', len(name.encode())) + name.encode())
                f.write(struct.pack('<B', dtype_code))  # 1 byte: dtype
                f.write(struct.pack('<I', tensor.ndim))
                for d in tensor.shape:
                    f.write(struct.pack('<I', d))
                f.write(struct.pack('<f', scale))
                f.write(struct.pack('<f', float(zero_point)))
                f.write(q_tensor.tobytes())
                total_bytes += q_tensor.nbytes

            elif quant_bits == 4:
                # INT4 量化（打包2个值到1字节）
                q_tensor, scale, zero_point = _quantize_int4(tensor)
                packed = _pack_int4(q_tensor)
                dtype_code = 2  # int4
                f.write(struct.pack('<H', len(name.encode())) + name.encode())
                f.write(struct.pack('<B', dtype_code))
                f.write(struct.pack('<I', tensor.ndim))
                for d in tensor.shape:
                    f.write(struct.pack('<I', d))
                f.write(struct.pack('<f', scale))
                f.write(struct.pack('<f', float(zero_point)))
                f.write(packed)
                total_bytes += len(packed)

            else:
                # FP32 (不量化)
                dtype_code = 0
                f.write(struct.pack('<H', len(name.encode())) + name.encode())
                f.write(struct.pack('<B', dtype_code))
                f.write(struct.pack('<I', tensor.ndim))
                for d in tensor.shape:
                    f.write(struct.pack('<I', d))
                f.write(tensor.tobytes())
                total_bytes += tensor.nbytes

    size_mb = total_bytes / 1e6
    print(f"  [导出] {bin_path}")
    print(f"  [量化] {quant_bits}bit | 权重: {size_mb:.1f} MB")

    # ── 3. 复制分词器 ──
    import shutil
    tok_src = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
    if os.path.exists(tok_src):
        shutil.copy(tok_src, os.path.join(output_dir, 'tokenizer.json'))
    print(f"  [分词器] tokenizer.json")

    print(f"\n  ✓ 导出完成! 输出目录: {output_dir}/")
    print(f"  📱 将 {output_dir}/ 整个目录放入手机 App 的 models/ 目录即可")


def _get_key_order(state_dict, config) -> List[str]:
    """确定 tensor 的写出顺序（与推理时加载一致）"""
    order = []
    # Embedding
    if 'token_embedding.weight' in state_dict:
        order.append('token_embedding.weight')

    n_layers = config.get('n_layers', 8)
    for i in range(n_layers):
        prefix = f'blocks.{i}.'
        for suffix in [
            'ln1.weight', 'ln2.weight',
            'attn.q_proj.weight', 'attn.k_proj.weight',
            'attn.v_proj.weight', 'attn.out_proj.weight',
        ]:
            order.append(prefix + suffix)
        # MoE
        n_experts = config.get('n_experts', 4)
        order.append(prefix + 'moe.router.weight')
        order.append(prefix + 'moe.gate.weight')
        for eid in range(n_experts):
            for w in ['w1.weight', 'w2.weight', 'w3.weight']:
                order.append(f'{prefix}moe.experts.{eid}.{w}')

    # Final norm
    if 'final_norm.weight' in state_dict:
        order.append('final_norm.weight')
    # LM head (tied)
    if 'lm_head.weight' in state_dict:
        order.append('lm_head.weight')

    # 验证所有 key 都在
    for k in order:
        assert k in state_dict, f"缺少 tensor: {k}"
    return order


def _quantize_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """INT8 对称量化"""
    t_min, t_max = tensor.min(), tensor.max()
    scale = max(abs(t_min), abs(t_max)) / 127.0
    if scale == 0:
        scale = 1.0
    q = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return q, float(scale), 0


def _quantize_int4(tensor: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """INT4 对称量化"""
    t_min, t_max = tensor.min(), tensor.max()
    scale = max(abs(t_min), abs(t_max)) / 7.0
    if scale == 0:
        scale = 1.0
    q = np.clip(np.round(tensor / scale), -7, 7).astype(np.int8)
    return q, float(scale), 0


def _pack_int4(q_tensor: np.ndarray) -> bytes:
    """将 int8 范围的 int4 值打包：2个值 → 1字节"""
    # 转为 0-15 范围
    shifted = (q_tensor + 7).astype(np.uint8).flatten()
    if len(shifted) % 2 == 1:
        shifted = np.append(shifted, 7)
    packed = np.zeros(len(shifted) // 2, dtype=np.uint8)
    for i in range(0, len(shifted), 2):
        packed[i // 2] = (shifted[i] << 4) | shifted[i + 1]
    return packed.tobytes()


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TG CHAT 模型导出')
    parser.add_argument('--input', required=True, help='checkpoint .pt 文件路径')
    parser.add_argument('--output', default='tg_chat_export', help='输出目录')
    parser.add_argument('--quantize', choices=['fp32', 'int8', 'int4'], default='int8',
                        help='量化精度 (默认 int8)')
    args = parser.parse_args()

    print("=" * 60)
    print("  TG CHAT — 移动端模型导出")
    print("=" * 60)

    state_dict, model_config = load_checkpoint(args.input)
    q_bits = {'fp32': 32, 'int8': 8, 'int4': 4}[args.quantize]

    # CLI 修复：传递 checkpoint_path 以便复制 tokenizer
    global checkpoint_path
    checkpoint_path = args.input

    export_ggml_format(state_dict, model_config, args.output, q_bits)
