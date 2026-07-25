"""
剥离训练 checkpoint 中的优化器状态，只保留 model_state_dict + model_config。
输出 FP16 轻量版，供 export_onnx.py 使用。

用法:
    python scripts/strip_optimizer.py milestones4000.pt milestones4000_fp16.pt
"""
import sys
import gc
import torch
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("用法: python strip_optimizer.py <输入.pt> <输出.pt>")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]

    print(f"加载: {src}")
    # 只加载模型权重和配置，不加载完整的 pickle 对象树
    # weights_only=False 是必需的，但我们可以立即丢弃不需要的部分
    ckpt = torch.load(src, map_location='cpu', weights_only=False)

    # 提取配置
    cfg = ckpt.get('model_config')
    if cfg is None:
        print("[错误] checkpoint 中没有 model_config")
        sys.exit(1)

    print(f"  config: {cfg}")
    print("  转换 FP32 → FP16 ...")

    # 逐个转换，立即释放 FP32
    state_fp16 = {}
    raw_state = ckpt['model_state_dict']
    for k in list(raw_state.keys()):
        v = raw_state.pop(k)
        state_fp16[k] = v.half()
    del raw_state
    gc.collect()

    # 彻底清除 ckpt（包含优化器状态等巨大数据）
    ckpt.clear()
    del ckpt
    gc.collect()

    print(f"保存 FP16 精简版: {dst}")
    torch.save({
        'model_config': cfg,
        'model_state_dict': state_fp16,
    }, dst)

    size_mb = Path(dst).stat().st_size / (1024 * 1024)
    print(f"  精简版大小: {size_mb:.1f} MB")
    print("完成！现在可以用精简版导出 ONNX")

if __name__ == '__main__':
    main()
