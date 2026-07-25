"""
将 TGAI PyTorch checkpoint 导出为 PyTorch Mobile Lite Interpreter 格式 (.ptl)。
同时导出 tokenizer 的 vocab 与 merges，供移动端使用。

用法:
    python export_pytorch_mobile.py --checkpoint checkpoints/milestone.pt --out_dir exported/

输出:
    exported/tgai_prefill.ptl
    exported/tgai_decode.ptl
    exported/tokenizer.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制单线程，减少内存
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tgai_nlp'))
from model import TGAILanguageModel, TGAIConfig, MoELayer, RotaryPositionEmbedding, create_model
from tokenizer import ChineseTokenizer


class TraceableMoELayer(nn.Module):
    """可 JIT trace/script 的 MoE 层，行为与原 MoELayer 一致（Top-K 路由）。"""

    def __init__(self, orig: MoELayer):
        super().__init__()
        self.n_experts = orig.n_experts
        self.n_activated = orig.n_activated
        self.router = orig.router
        self.experts = orig.experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        router_logits = self.router(x_flat)

        topk_vals, topk_ids = torch.topk(router_logits, self.n_activated, dim=-1)
        router_probs = F.softmax(topk_vals, dim=-1)

        weights = torch.zeros_like(router_logits)
        weights.scatter_(-1, topk_ids, router_probs)

        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        output = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return output.view(B, T, D)


def make_model_traceable(model: TGAILanguageModel) -> TGAILanguageModel:
    """把原模型中的动态专家选择替换为可 JIT 的形式。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, MoELayer):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], TraceableMoELayer(module))

    # 替换 RoPE forward 使用 index_select，避免动态 tensor 索引
    def traceable_rope_forward(self, x: torch.Tensor, position_ids=None):
        seq_len = x.shape[-2]
        if position_ids is not None:
            positions = position_ids.view(-1)
            cos = self.cos.squeeze(0).squeeze(0).index_select(0, positions).unsqueeze(0).unsqueeze(0)
            sin = self.sin.squeeze(0).squeeze(0).index_select(0, positions).unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos[:, :, :seq_len, :]
            sin = self.sin[:, :, :seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    RotaryPositionEmbedding.forward = traceable_rope_forward
    return model


class TGAIPrefill(nn.Module):
    """一次性处理 prompt，返回 logits 和初始化后的 KV cache。"""

    def __init__(self, model: TGAILanguageModel):
        super().__init__()
        self.model = model
        self.n_layers = model.config.n_layers

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape[:2]
        logits, caches = self.model(input_ids, kv_caches=None)
        if caches is None:
            # 训练模式不会返回 cache，构造全零占位
            n_heads = self.model.config.n_heads
            d_k = self.model.config.d_model // n_heads
            kv_cache = torch.zeros(2 * self.n_layers, B, n_heads, T, d_k)
            return logits, kv_cache
        kvs = []
        for c in caches:
            kvs.append(c['k'])
            kvs.append(c['v'])
        kv_cache = torch.stack(kvs, dim=0)
        return logits, kv_cache


class TGAIDecode(nn.Module):
    """单步 decode：输入新 token 和当前 KV cache，返回 logits 并原地更新 cache。"""

    def __init__(self, model: TGAILanguageModel):
        super().__init__()
        self.model = model
        self.n_layers = model.config.n_layers

    def forward(self, input_ids: torch.Tensor, cache_pos: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        cache_pos_val = int(cache_pos.item())
        kv_caches = []
        for i in range(self.n_layers):
            kv_caches.append({'k': kv_cache[2 * i], 'v': kv_cache[2 * i + 1]})

        logits, caches = self.model(input_ids, kv_caches=kv_caches, cache_pos=cache_pos_val)

        for i in range(self.n_layers):
            kv_cache[2 * i] = caches[i]['k']
            kv_cache[2 * i + 1] = caches[i]['v']
        return logits


def export_tokenizer(tokenizer: ChineseTokenizer, out_path: str):
    """导出 vocab 与 merges 为移动端可用的 JSON。"""
    merges = []
    if tokenizer._hf is not None:
        hf_json = json.loads(tokenizer._hf.to_str())
        merges = hf_json.get('model', {}).get('merges', [])

    data = {
        'vocab_size': tokenizer.vocab_size,
        'token_to_id': tokenizer.token_to_id,
        'id_to_token': {str(k): v for k, v in tokenizer.id_to_token.items()},
        'merges': merges,
        'special_ids': {
            'pad': 0,
            'unk': 1,
            'bos': 2,
            'eos': 3,
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  tokenizer -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='TGAI checkpoint (.pt)')
    parser.add_argument('--out_dir', type=str, default='exported', help='输出目录')
    parser.add_argument('--tokenizer', type=str, default='', help='tokenizer json 路径（留空从 checkpoint 同目录查找）')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    print(f"  config: {cfg}")

    model = create_model(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Float16 量化：移动端用 fp16，体积减半（5GB → ~2.5GB，ZIP 后 ~1.5GB）
    print("  转换为 float16 (移动端量化)...")
    model = model.half()

    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        candidate = Path(args.checkpoint).parent / 'tokenizer.json'
        if candidate.exists():
            tokenizer_path = str(candidate)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        print("[警告] 未找到 tokenizer.json，将只导出模型")
        tokenizer = None
    else:
        tokenizer = ChineseTokenizer.load(tokenizer_path)
        export_tokenizer(tokenizer, str(out_dir / 'tokenizer.json'))

    print("转换为可 trace/script 的模型...")
    make_model_traceable(model)

    prefill = TGAIPrefill(model)
    decode = TGAIDecode(model)

    B, T = 1, cfg['max_seq_len']
    example_prefill = torch.randint(0, cfg['vocab_size'], (B, T), dtype=torch.long)

    example_decode_ids = torch.randint(0, cfg['vocab_size'], (B, 1), dtype=torch.long)
    example_cache_pos = torch.tensor([cfg['max_seq_len'] - 1], dtype=torch.long)
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    d_k = cfg['d_model'] // cfg['n_heads']
    example_kv_cache = torch.zeros(2 * n_layers, B, n_heads, cfg['max_seq_len'], d_k, dtype=torch.float16)

    print("尝试 torch.jit.script...")
    try:
        scripted_prefill = torch.jit.script(prefill)
        scripted_decode = torch.jit.script(decode)
        use_script = True
        print("  script 成功")
    except Exception as e:
        print(f"  script 失败，改用 trace: {e}")
        use_script = False

    if use_script:
        pref = scripted_prefill
        dec = scripted_decode
    else:
        print("  step 1/2: tracing prefill 模型 (可能需要 1-5 分钟)...")
        sys.stdout.flush()
        pref = torch.jit.trace(prefill, example_inputs=(example_prefill,))
        print("  prefill trace 完成")
        print("  step 2/2: tracing decode 模型 (可能需要 1-3 分钟)...")
        sys.stdout.flush()
        dec = torch.jit.trace(decode, example_inputs=(example_decode_ids, example_cache_pos, example_kv_cache))
        print("  decode trace 完成")

    prefill_path = out_dir / 'tgai_prefill.ptl'
    decode_path = out_dir / 'tgai_decode.ptl'

    pref._save_for_lite_interpreter(str(prefill_path))
    dec._save_for_lite_interpreter(str(decode_path))

    print(f"  prefill -> {prefill_path}")
    print(f"  decode  -> {decode_path}")
    print("导出完成。将这两个 .ptl 和 tokenizer.json 一起导入 TG CHAT 即可。")


if __name__ == '__main__':
    main()
