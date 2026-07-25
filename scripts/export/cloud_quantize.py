#!/usr/bin/env python3
"""
云端模型量化工具 - 将 PyTorch .pt 转为 TGAI GO .TG 格式
======================================================
在 AutoDL 等云服务器上运行，支持 FP16/INT8/FP32 多种精度。

用法:
    # 3B 模型 INT8 量化（手机部署推荐）
    python scripts/cloud_quantize.py --checkpoint autodl-tmp/TGAI_checkpoints_v3/tgai_sft.pt --out tgai_3b_int8.TG --dtype int8

    # 3B 模型 FP16（平衡）
    python scripts/cloud_quantize.py --checkpoint autodl-tmp/TGAI_checkpoints_v3/tgai_sft.pt --out tgai_3b_fp16.TG --dtype fp16

    # 0.86B 模型 INT8（低端手机）
    python scripts/cloud_quantize.py --checkpoint autodl-tmp/TGAI_checkpoints_v3/tgai_lora_1w.pt --out tgai_086b_int8.TG --dtype int8

精度说明:
    fp32    全精度，无损失。3B ~12GB，0.86B ~3.4GB。桌面端。
    fp16    半精度，几乎无损。3B ~6GB，0.86B ~1.7GB。【推荐桌面/手机】
    q8_0    分组 8-bit，每 32 个权重独立 scale。3B ~4.5GB。精度接近 FP16。
    q4_0    分组 4-bit，极致压缩。3B ~2.1GB，0.86B ~0.6GB。【推荐手机端】
    int8    全局 8-bit（旧），一个 scale 管全部。精度不如 q8_0，建议用 q8_0 替代。
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def estimate_size(checkpoint_path: str, dtype: str) -> str:
    """根据 .pt 文件大小估算各精度下的 .TG 大小"""
    pt_size = os.path.getsize(checkpoint_path)
    pt_gb = pt_size / (1024 ** 3)

    # .pt 文件包含 optimizer state + model weights，实际权重约占 60-70%
    # .TG 不含 optimizer，但 tokenizer 增加少量
    weight_ratio = 0.65

    if dtype == 'fp32':
        tg_gb = pt_gb * weight_ratio
    elif dtype == 'fp16':
        tg_gb = pt_gb * weight_ratio * 0.5
    elif dtype == 'q8_0':
        tg_gb = pt_gb * weight_ratio * 0.37  # 34B per 32 elems ≈ 4.25B/elem vs 4B FP32
    elif dtype == 'q4_0':
        tg_gb = pt_gb * weight_ratio * 0.18  # 18B per 32 elems = 2.25B/elem
    elif dtype == 'int8':
        tg_gb = pt_gb * weight_ratio * 0.25
    else:
        tg_gb = pt_gb * weight_ratio

    return f"~{tg_gb:.1f} GB"


def print_dtype_info(checkpoint_path: str):
    """打印各精度的估算信息"""
    print("\n  ┌───────────────────────────────────────────────┐")
    print("  │  精度       体积 (估算)      适用场景        │")
    print("  ├───────────────────────────────────────────────┤")
    print(f"  │  FP32       {estimate_size(checkpoint_path, 'fp32'):>10s}      桌面端，无精度损失 │")
    print(f"  │  FP16       {estimate_size(checkpoint_path, 'fp16'):>10s}      手机/桌面，推荐     │")
    print(f"  │  Q8_0       {estimate_size(checkpoint_path, 'q8_0'):>10s}      分组 8-bit，高精度  │")
    print(f"  │  Q4_0       {estimate_size(checkpoint_path, 'q4_0'):>10s}      极致压缩，手机首选  │")
    print(f"  │  INT8(旧)   {estimate_size(checkpoint_path, 'int8'):>10s}      全局量化，已被淘汰  │")
    print("  └───────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(
        description='云端模型量化工具 — .pt → .TG 转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # INT8 量化 3B 模型（手机部署）
  python scripts/cloud_quantize.py --checkpoint autodl-tmp/TGAI_checkpoints_v3/tgai_sft.pt --out tgai_3b_int8.TG --dtype int8

  # FP16 转换 0.86B 模型
  python scripts/cloud_quantize.py --checkpoint autodl-tmp/TGAI_checkpoints_v3/tgai_lora_1w.pt --out tgai_086b_fp16.TG
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='.pt checkpoint 路径')
    parser.add_argument('--out', type=str, required=True, help='输出 .TG 文件路径')
    parser.add_argument('--dtype', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'q8_0', 'q4_0', 'int8'],
                        help='精度：fp32/fp16(默认)/q8_0(分组8-bit)/q4_0(极致压缩)/int8(旧)')
    parser.add_argument('--tokenizer', type=str, help='tokenizer.json 路径 (默认自动查找)')
    args = parser.parse_args()

    # ── 验证 checkpoint ──
    if not os.path.exists(args.checkpoint):
        print(f"\n[错误] checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)

    pt_size = os.path.getsize(args.checkpoint)
    print(f"\n  Source: {args.checkpoint}  ({pt_size / 1024**3:.2f} GB)")
    print(f"  Target: {args.dtype.upper()}")
    print(f"  Output: {args.out}")

    print_dtype_info(args.checkpoint)
    print(f"  Estimated output: {estimate_size(args.checkpoint, args.dtype)}")

    # ── 找 converter ──
    project_root = Path(__file__).resolve().parent.parent
    converter = project_root / 'TGAI GO' / 'tools' / 'tgai_convert.py'
    if not converter.exists():
        print(f"\n[错误] 找不到 tgai_convert.py: {converter}")
        sys.exit(1)

    # ── 找 tokenizer ──
    if args.tokenizer:
        tokenizer = args.tokenizer
    else:
        # 自动查找
        candidates = [
            project_root / 'checkpoints' / 'tokenizer.json',
            project_root.parent / 'autodl-tmp' / 'TGAI_checkpoints_v3' / 'tokenizer.json',
            Path(args.checkpoint).parent / 'tokenizer.json',
        ]
        tokenizer = None
        for c in candidates:
            if c.exists():
                tokenizer = str(c)
                break
        if not tokenizer:
            print("\n[错误] 未找到 tokenizer.json，请用 --tokenizer 指定路径")
            print("  已搜索:")
            for c in candidates:
                print(f"    {c}")
            sys.exit(1)

    if not os.path.exists(tokenizer):
        print(f"\n[错误] tokenizer 不存在: {tokenizer}")
        sys.exit(1)

    print(f"\n  Converter: {converter}")
    print(f"  Tokenizer: {tokenizer}")
    print(f"\n{'='*60}")
    print(f"  开始转换... (大模型可能需要 5-15 分钟)")

    t_start = time.time()

    # ── 执行转换 ──
    proc = subprocess.Popen(
        [sys.executable, '-u', str(converter),
         '--checkpoint', args.checkpoint,
         '--output', args.out,
         '--tokenizer', tokenizer,
         '--dtype', args.dtype],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  {line}")

    proc.wait()
    elapsed = time.time() - t_start

    if proc.returncode != 0:
        print(f"\n[失败] 转换出错 (exit={proc.returncode})")
        sys.exit(1)

    if not os.path.exists(args.out):
        print(f"\n[失败] 输出文件未生成: {args.out}")
        sys.exit(1)

    out_size = os.path.getsize(args.out)
    print(f"\n{'='*60}")
    print(f"  转换完成!")
    print(f"  精度:     {args.dtype.upper()}")
    print(f"  输出大小: {out_size / 1024**2:.1f} MB ({out_size / 1024**3:.2f} GB)")
    print(f"  耗时:     {elapsed:.1f} 秒")
    print(f"  输出文件: {args.out}")
    print(f"\n  下载到本地:")
    print(f"    scp user@host:{args.out} ./")
    print(f"  或通过 API 下载后门:")
    print(f"    curl -O http://<server>:<port>/download/{os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
