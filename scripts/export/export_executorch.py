"""
将 TGAI PyTorch checkpoint 导出为 ExecuTorch 格式 (.pte)。
默认使用 XNNPACK CPU 后端，适用于移动端推理。

用法:
    python scripts/export_executorch.py --checkpoint checkpoints/milestone.pt --out_dir exported/

输出:
    exported/tgai.pte         (单模型，无 KV cache 分离)
    exported/tokenizer.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple

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
    """可 JIT/export 的 MoE 层，用矩阵运算替代循环路由。"""

    def __init__(self, orig: MoELayer):
        super().__init__()
        self.n_experts = orig.n_experts
        self.n_activated = orig.n_activated
        self.router = orig.router
        self.experts = orig.experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.router(x_flat)

        topk_vals, topk_ids = torch.topk(router_logits, self.n_activated, dim=-1)
        router_probs = F.softmax(topk_vals, dim=-1)

        # 逐专家加权求和（仅循环静态 n_experts/n_activated，不用 scatter/stack/item）
        output = torch.zeros(x_flat.shape[0], D, device=x.device, dtype=x.dtype)
        for e_idx in range(self.n_experts):
            expert_out = self.experts[e_idx](x_flat)
            expert_weight = torch.zeros(x_flat.shape[0], 1, device=x.device, dtype=x.dtype)
            for k in range(self.n_activated):
                mask = (topk_ids[:, k] == e_idx).to(dtype=x.dtype).unsqueeze(-1)
                expert_weight += mask * router_probs[:, k:k + 1]
            output += expert_weight * expert_out

        return output.view(B, T, D)


def make_model_exportable(model: TGAILanguageModel) -> TGAILanguageModel:
    """替换动态 MoE 为可导出版本。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, MoELayer):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], TraceableMoELayer(module))

    return model


def export_tokenizer(tokenizer: ChineseTokenizer, out_path: str):
    """导出 tokenizer 为移动端可用 JSON。"""
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


class TGAIForGeneration(nn.Module):
    """
    完整生成模型：输入 token_ids，返回 logits。
    ExecuTorch 不支持 in-place KV cache 修改，因此每次 forward 处理完整序列。
    后续可通过 ExecuTorch LLM API 添加 KV cache 支持。
    """

    def __init__(self, model: TGAILanguageModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids, kv_caches=None)
        return logits.float()  # 始终输出 float32，与 Kotlin dataAsFloatArray 匹配


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='TGAI checkpoint (.pt)')
    parser.add_argument('--out_dir', type=str, default='exported', help='输出目录')
    parser.add_argument('--tokenizer', type=str, default='', help='tokenizer json 路径')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
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

    # Float16 量化权重（减少内存），运行时由 ExecuTorch 处理 dtype 转换
    print("  转换为 float16...")
    model = model.half()

    # Tokenizer
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        candidate = Path(args.checkpoint).parent / 'tokenizer.json'
        if candidate.exists():
            tokenizer_path = str(candidate)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        print("[警告] 未找到 tokenizer.json，将只导出模型")
    else:
        tokenizer = ChineseTokenizer.load(tokenizer_path)
        export_tokenizer(tokenizer, str(out_dir / 'tokenizer.json'))

    print("转换为可导出模型...")
    make_model_exportable(model)

    # 包装为生成模型
    gen_model = TGAIForGeneration(model)

    # torch.export: 用小示例导出节省内存，动态形状支持运行时扩展
    # 导出用 8 tokens，运行时最多支持 args.max_seq_len
    example_input = torch.randint(0, cfg['vocab_size'], (1, 8), dtype=torch.int)

    # 动态 seq_len，batch 固定为 1
    seq_dim = torch.export.Dim("seq_len", min=1, max=args.max_seq_len)
    dynamic_shapes = ({1: seq_dim},)

    print(f"执行 torch.export（导出示例 8 tokens，运行时动态 {args.max_seq_len}）...")
    try:
        exported = torch.export.export(gen_model, (example_input,), dynamic_shapes=dynamic_shapes)
        print("  torch.export 成功")
    except Exception as e:
        print(f"  torch.export 失败: {e}")
        print("  尝试无动态形状导出...")
        exported = torch.export.export(gen_model, (example_input,))
        print("  torch.export 成功（无动态形状）")

    # 释放原始模型内存，为 XNNPACK lowering 腾空间
    print("释放原始模型...")
    gen_model = gen_model.cpu()
    del model, gen_model
    # 也删除 checkpoint 引用
    del ckpt
    import gc
    gc.collect()
    print("  内存已清理")

    # ---- 断点恢复：如果上次 XNNPACK 委托已完成，跳过重跑 ----
    _et_prog_pkl = out_dir / '_et_program.tmp'
    _skipped_lowering = False
    if _et_prog_pkl.exists():
        print(f"发现中间产物 {_et_prog_pkl}，跳过 XNNPACK 委托...")
        try:
            import pickle
            with open(_et_prog_pkl, 'rb') as _pf:
                et_program = pickle.load(_pf)
            _skipped_lowering = True
            print("  中间产物加载成功，直接写入最终文件...")
        except Exception as _lr:
            print(f"  加载失败 ({_lr})，重新委托...")
            _et_prog_pkl.unlink(missing_ok=True)

    if not _skipped_lowering:
        # 正常流程：XNNPACK 委托
        print("转换为 ExecuTorch 格式（XNNPACK 加速）...")

        try:
            from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

            # per_op_mode 逐算子委托，比整图委托省内存
            edge_manager = to_edge_transform_and_lower(
                exported,
                partitioner=[XnnpackPartitioner(per_op_mode=True)],
                compile_config=EdgeCompileConfig(
                    _core_aten_ops_exception_list=[
                        torch.ops.aten.empty_permuted.default,
                    ],
                ),
            )
            et_program = edge_manager.to_executorch()
            print("  XNNPACK 委托成功")
        except ImportError:
            print("[警告] 未安装 executorch 包。使用简化导出...")
            print("  请运行: pip install executorch")
            return 1
        except MemoryError:
            print("  XNNPACK 委托内存不足，回退到 portable 模式...")
            try:
                from executorch.exir import to_edge
                edge_manager = to_edge(exported)
                et_program = edge_manager.to_executorch()
            except Exception as e2:
                print(f"  to_edge/to_executorch 失败: {e2}")
                return 1
        except Exception as e:
            print(f"  XNNPACK 委托失败 ({e})，回退到 portable 模式...")
            try:
                from executorch.exir import to_edge
                edge_manager = to_edge(exported)
                et_program = edge_manager.to_executorch()
            except Exception as e2:
                print(f"  to_edge/to_executorch 失败: {e2}")
                return 1

        # ---- 断点保存：中间产物 ----
        try:
            print(f"  保存中间产物 ({_et_prog_pkl})...")
            import pickle
            with open(_et_prog_pkl, 'wb') as _pf:
                pickle.dump(et_program, _pf)
            print("  中间产物已保存")
        except Exception as _pkl_err:
            print(f"  (跳过中间保存: {_pkl_err})")

    # 保存（流式写入）
    pte_path = out_dir / 'tgai.pte'
    print(f"  正在写入 {pte_path}...")
    with open(pte_path, 'wb') as f:
        et_program.write_to_file(f)

    print(f"  模型 -> {pte_path}")
    print(f"  文件大小: {pte_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n导出完成。将 tgai.pte 和 tokenizer.json 一起打包为 .TG 文件即可。")

    return 0


if __name__ == '__main__':
    sys.exit(main())
