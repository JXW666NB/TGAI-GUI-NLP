"""
将 TGAI PyTorch checkpoint 导出为 ONNX 格式。
支持 INT8 动态量化（需 8GB+ 内存）。

用法:
    python scripts/export_onnx.py --checkpoint checkpoints/milestone.pt --out_dir exported/
    python scripts/export_onnx.py --checkpoint checkpoints/milestone.pt --out_dir exported/ --int8

输出:
    exported/tgai.onnx
    exported/tokenizer.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import TGAILanguageModel, MoELayer, create_model
from tokenizer import ChineseTokenizer


# ═══════════════════════════════════════════════════════════
# ONNX 兼容 MoE — 纯 tensor 运算，不用 scatter_/stack/item
# ═══════════════════════════════════════════════════════════
class OnnxMoELayer(nn.Module):
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

        output = torch.zeros(x_flat.shape[0], D, device=x.device, dtype=x.dtype)
        for e_idx in range(self.n_experts):
            expert_out = self.experts[e_idx](x_flat)
            expert_weight = torch.zeros(x_flat.shape[0], 1, device=x.device, dtype=x.dtype)
            for k in range(self.n_activated):
                mask = (topk_ids[:, k] == e_idx).to(dtype=x.dtype).unsqueeze(-1)
                expert_weight += mask * router_probs[:, k:k + 1]
            output += expert_weight * expert_out

        return output.view(B, T, D)


def make_model_exportable(model: TGAILanguageModel):
    for name, module in list(model.named_modules()):
        if isinstance(module, MoELayer):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], OnnxMoELayer(module))
    return model


def export_tokenizer(tokenizer: ChineseTokenizer, out_path: str):
    merges = []
    if tokenizer._hf is not None:
        hf_json = json.loads(tokenizer._hf.to_str())
        merges = hf_json.get('model', {}).get('merges', [])

    data = {
        'vocab_size': tokenizer.vocab_size,
        'token_to_id': tokenizer.token_to_id,
        'id_to_token': {str(k): v for k, v in tokenizer.id_to_token.items()},
        'merges': merges,
        'special_ids': {'pad': 0, 'unk': 1, 'bos': 2, 'eos': 3},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  tokenizer -> {out_path}")


class TGAIForGeneration(nn.Module):
    def __init__(self, model: TGAILanguageModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids, kv_caches=None)
        return logits.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exported')
    parser.add_argument('--tokenizer', type=str, default='')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset 版本')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='导出后应用 INT8 动态量化（模型减半，推理加速）')
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

    # FP16 权重省内存
    print("  转换为 float16...")
    model = model.half()

    # Tokenizer
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        candidate = Path(args.checkpoint).parent / 'tokenizer.json'
        if candidate.exists():
            tokenizer_path = str(candidate)
    if tokenizer_path and Path(tokenizer_path).exists():
        tokenizer = ChineseTokenizer.load(tokenizer_path)
        export_tokenizer(tokenizer, str(out_dir / 'tokenizer.json'))
    else:
        print("[警告] 未找到 tokenizer.json")

    print("转换为 ONNX 可导出模型...")
    make_model_exportable(model)
    gen_model = TGAIForGeneration(model)

    # 导出用小示例 + 动态 batch/seq_len
    example_input = torch.randint(0, cfg['vocab_size'], (1, 32), dtype=torch.int)
    del ckpt

    onnx_path = out_dir / 'tgai.onnx'
    print(f"导出 ONNX ({onnx_path})...")

    torch.onnx.export(
        gen_model,
        (example_input,),
        str(onnx_path),
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"  模型 -> {onnx_path}")
    print(f"  文件大小: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 合并外部数据到单个 .onnx 文件
    data_file = out_dir / 'tgai.onnx.data'
    if data_file.exists():
        print("  合并外部权重数据...")
        try:
            import onnx
            from onnx.external_data_helper import convert_model_from_external_data
            model = onnx.load(str(onnx_path), load_external_data=True)
            convert_model_from_external_data(
                model, all_tensors_to_one_file=True,
                location=str(onnx_path), size_threshold=1024, convert_attribute=False
            )
            onnx.save(model, str(onnx_path), save_as_external_data=False)
            data_file.unlink()
            print(f"  合并完成，文件大小: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as merge_err:
            print(f"  [警告] 合并失败 ({merge_err})，请保留 .onnx.data 文件")

    # INT8 动态量化
    if args.int8:
        print(f"  INT8 动态量化中...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quant_path = out_dir / 'tgai.int8.onnx'
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quant_path),
                weight_type=QuantType.QInt8,
                extra_options={'ActivationSymmetric': False},
            )
            quant_size = quant_path.stat().st_size / 1024 / 1024
            onnx_path.unlink()
            quant_path.rename(onnx_path)
            print(f"  INT8 量化完成，文件大小: {quant_size:.1f} MB")
        except ImportError:
            print("  [错误] onnxruntime 未安装，跳过量化。pip install onnxruntime")
        except Exception as e:
            print(f"  [警告] 量化失败 ({e})，保留 FP16 模型")

    print("\n导出完成。将 tgai.onnx 和 tokenizer.json 打包为 .TG 文件即可。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
