#!/usr/bin/env python3
"""
TGAI CLI — 云端训练命令行工具
==============================
用于 AutoDL / 云服务器 等无图形环境，通过命令行完成训练、对话、数据管理。

核心命令:
  tgai_cli.py train          启动训练
  tgai_cli.py chat           交互式对话
  tgai_cli.py data stats     训练数据统计
  tgai_cli.py data clean     清理训练数据
  tgai_cli.py data merge     合并多个语料文件
  tgai_cli.py data download     下载公开数据集并转换
  tgai_cli.py code generate     生成编程代码语料
  tgai_cli.py config         生成默认配置文件

示例:
  python tgai_cli.py train --epochs 50 --batch 32 --lr 1e-4 --use_gpu
  python tgai_cli.py train --config autodl_config.yaml
  python tgai_cli.py chat --checkpoint checkpoints/best_model.pt
"""
import os
import sys
import json
import time
import math
import argparse
import shutil
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

# 确保能导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import ChineseTokenizer
from model import create_model
from inference import load_model, interactive_chat


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def _collect_system_stats() -> Dict[str, Any]:
    stats = {'time': _now()}
    try:
        import psutil
        stats['cpu_percent'] = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        stats['ram_used_gb'] = round(mem.used / 1e9, 2)
        stats['ram_total_gb'] = round(mem.total / 1e9, 2)
        stats['ram_percent'] = mem.percent
    except ImportError:
        pass

    if torch.cuda.is_available():
        stats['gpu_name'] = torch.cuda.get_device_name(0)
        stats['gpu_mem_used_gb'] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        stats['gpu_mem_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        stats['gpu_mem_percent'] = round(stats['gpu_mem_used_gb'] / stats['gpu_mem_total_gb'] * 100, 1)
    return stats


# ═══════════════════════════════════════════════════════════
# 配置文件管理
# ═══════════════════════════════════════════════════════════

def cmd_config(args: argparse.Namespace) -> int:
    """生成默认 YAML 配置文件"""
    yaml_text = """# TGAI 云端训练配置文件
# 用法: python tgai_cli.py train --config autodl_config.yaml

# 数据
data_path: tgai_nlp/data/train_all.jsonl
cache_dir: tgai_nlp/checkpoints/cache

# 分词器
vocab_size: 32768      # 1B 模型建议 32k 词表
tokenizer_path: tgai_nlp/checkpoints/tokenizer.json

# 模型 (1B 参数参考配置: d_model=1024, n_layers=24, n_heads=16, d_ff=4096)
# 云端 24G 显存可尝试: d_model=768, n_layers=16, n_heads=12, d_ff=3072 (约 300M)
d_model: 256
n_layers: 8
n_heads: 8
d_ff: 1024
seq_len: 256
dropout: 0.1

# MoE
n_experts: 4
n_activated: 2

# 训练
epochs: 30
batch_size: 16
learning_rate: 0.0003
weight_decay: 0.01
warmup_steps: 500
grad_accum_steps: 1
grad_clip: 1.0

# 保存
checkpoint_dir: tgai_nlp/checkpoints
save_every_epochs: 5
save_every_steps: 500
eval_every_steps: 100

# 硬件
device: cuda           # cuda 或 cpu
use_amp: true
compile_model: false   # PyTorch 2.0+ 可尝试开启
"""
    path = args.output or 'autodl_config.yaml'
    if os.path.exists(path) and not args.force:
        print(f"文件已存在: {path}, 加 --force 覆盖")
        return 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)
    print(f"[OK] 配置文件已生成: {path}")
    return 0


def load_yaml(path: str) -> Dict[str, Any]:
    """简易 YAML 加载器，支持 key: value 格式"""
    cfg = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if val.lower() in ('true', 'false'):
                cfg[key] = val.lower() == 'true'
            else:
                try:
                    if '.' in val:
                        cfg[key] = float(val)
                    else:
                        cfg[key] = int(val)
                except ValueError:
                    cfg[key] = val
    return cfg


# ═══════════════════════════════════════════════════════════
# 训练命令
# ═══════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> int:
    """启动训练"""
    # 1. 加载配置（命令行优先于配置文件）
    config: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        print(f"[配置] 加载 {args.config}")
        config = load_yaml(args.config)

    # 命令行覆盖
    overrides = [
        ('data_path', args.data),
        ('epochs', args.epochs),
        ('batch_size', args.batch),
        ('learning_rate', args.lr),
        ('d_model', args.d_model),
        ('n_layers', args.n_layers),
        ('n_heads', args.n_heads),
        ('d_ff', args.d_ff),
        ('seq_len', args.seq_len),
        ('vocab_size', args.vocab_size),
        ('dropout', args.dropout),
        ('n_experts', args.n_experts),
        ('n_activated', args.n_activated),
        ('warmup_steps', args.warmup),
        ('grad_accum_steps', args.grad_accum),
        ('checkpoint_dir', args.checkpoint_dir),
        ('tokenizer_path', args.tokenizer_path),
        ('device', 'cuda' if args.use_gpu else 'cpu'),
        ('resume', args.resume),
    ]
    for key, val in overrides:
        if val is not None:
            config[key] = val

    # 打印训练计划
    print("=" * 60)
    print("TGAI 云端训练")
    print("=" * 60)
    print(f"时间: {_now()}")
    print(f"设备: {config.get('device', 'cpu')}")
    print(f"模型: d_model={config.get('d_model')}, layers={config.get('n_layers')}, heads={config.get('n_heads')}")
    print(f"数据: {config.get('data_path')}")
    print(f"Epochs: {config.get('epochs')}, Batch: {config.get('batch_size')}, LR: {config.get('learning_rate')}")

    # 估算参数量
    vocab_size = config.get('vocab_size', 16384)
    d_model = config.get('d_model', 256)
    n_layers = config.get('n_layers', 8)
    n_heads = config.get('n_heads', 8)
    d_ff = config.get('d_ff', 1024)
    # 粗略估算（包含 tying）
    emb = vocab_size * d_model
    attn_per_layer = 4 * d_model * d_model
    ffn_per_layer = 3 * d_model * d_ff  # SwiGLU
    moe_extra = (n_layers * 3 * d_model * d_ff * (config.get('n_experts', 4) - 1) *
                 config.get('n_activated', 2) / config.get('n_experts', 4))
    approx_params = emb * 2 + n_layers * (attn_per_layer + ffn_per_layer) + moe_extra
    print(f"估算参数: ~{approx_params/1e6:.1f}M")

    # 显存估算（FP16 训练约 18-20 bytes/param 优化器状态 + 激活）
    if config.get('device') == 'cuda':
        est_mem_gb = approx_params * 18 / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"显存估算: ~{est_mem_gb:.1f}GB / {gpu_total:.1f}GB")
        if est_mem_gb > gpu_total * 0.9:
            print("⚠ 警告: 估算显存接近上限，建议减小 batch 或模型尺寸")

    # 导入训练模块并启动
    try:
        from train import TrainConfig, train
        train_cfg = TrainConfig(
            data_path=config.get('data_path', 'data/train_all.jsonl'),
            seq_len=config.get('seq_len', 256),
            cache_dir=config.get('cache_dir', 'checkpoints/cache'),
            vocab_size=config.get('vocab_size', 16384),
            tokenizer_path=config.get('tokenizer_path', 'checkpoints/tokenizer.json'),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=config.get('dropout', 0.1),
            n_experts=config.get('n_experts', 4),
            n_activated=config.get('n_activated', 2),
            batch_size=config.get('batch_size', 16),
            epochs=config.get('epochs', 30),
            learning_rate=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 500),
            grad_clip=config.get('grad_clip', 1.0),
            grad_accum_steps=config.get('grad_accum_steps', 1),
            checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
            save_every_epochs=config.get('save_every_epochs', 5),
            save_every_steps=config.get('save_every_steps', 500),
            eval_every_steps=config.get('eval_every_steps', 100),
            device=config.get('device', 'cpu'),
        )
        if config.get('resume'):
            train_cfg._resume_path = config['resume']
        train(train_cfg)
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        traceback.print_exc()
        return 1
    return 0


# ═══════════════════════════════════════════════════════════
# 对话命令
# ═══════════════════════════════════════════════════════════

def cmd_chat(args: argparse.Namespace) -> int:
    """启动命令行对话"""
    ckpt = args.checkpoint
    if not ckpt or not os.path.exists(ckpt):
        # 自动寻找最佳模型
        ckpt_dir = 'checkpoints'
        candidates = ['best_model.pt', 'final_model.pt']
        for c in candidates:
            p = os.path.join(ckpt_dir, c)
            if os.path.exists(p):
                ckpt = p
                break
        if not ckpt or not os.path.exists(ckpt):
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and not f.endswith('.tmp')],
                key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
                reverse=True,
            )
            if ckpts:
                ckpt = os.path.join(ckpt_dir, ckpts[0])
    if not ckpt or not os.path.exists(ckpt):
        print("错误: 未找到 checkpoint，请指定 --checkpoint")
        return 1

    device = 'cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
    print(f"[加载] {ckpt} -> {device}")
    try:
        model, tokenizer = load_model(ckpt, args.tokenizer_path, device=device, compile_model=args.compile)
        interactive_chat(model, tokenizer, args.temperature, args.max_tokens, stream=True)
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        return 1
    return 0


# ═══════════════════════════════════════════════════════════
# 数据管理命令
# ═══════════════════════════════════════════════════════════

def cmd_data_stats(args: argparse.Namespace) -> int:
    """统计训练数据"""
    path = args.path
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}")
        return 1

    total = 0
    q_lengths = []
    a_lengths = []
    format_issues = 0
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            text = obj.get('text', '')
            total += 1
            if '用户:' not in text or ('TGAI?' not in text and 'TGAI:' not in text):
                format_issues += 1
                continue
            sep = 'TGAI?' if 'TGAI?' in text else 'TGAI:'
            q, a = text.split(sep, 1)
            q_lengths.append(len(q.replace('用户:', '').strip()))
            a_lengths.append(len(a.strip()))
        except json.JSONDecodeError:
            format_issues += 1

    print("=" * 60)
    print(f"数据文件: {path}")
    print(f"总行数: {total}")
    print(f"格式异常: {format_issues}")
    if q_lengths:
        print(f"问题长度: 平均={sum(q_lengths)/len(q_lengths):.1f}, 最大={max(q_lengths)}, 最小={min(q_lengths)}")
        print(f"回答长度: 平均={sum(a_lengths)/len(a_lengths):.1f}, 最大={max(a_lengths)}, 最小={min(a_lengths)}")
    print("=" * 60)
    return 0


def cmd_data_clean(args: argparse.Namespace) -> int:
    """调用清理脚本"""
    script = Path(__file__).parent.parent / 'scripts' / 'clean_training_data.py'
    if not script.exists():
        print(f"错误: 清理脚本不存在 {script}")
        return 1
    return os.system(f"python {script}") >> 8


def cmd_data_merge(args: argparse.Namespace) -> int:
    """合并多个 jsonl 文件并去重"""
    if not args.inputs:
        print("错误: 请指定至少一个输入文件")
        return 1

    output = args.output or 'data/merged.jsonl'
    seen = set()
    total = 0
    written = 0

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as out_f:
        for inp in args.inputs:
            if not os.path.exists(inp):
                print(f"跳过不存在文件: {inp}")
                continue
            print(f"处理: {inp}")
            for line in open(inp, 'r', encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    if text in seen:
                        continue
                    seen.add(text)
                    out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    written += 1
                except Exception:
                    pass

    print(f"[OK] 合并完成: {written}/{total} 条保留 -> {output}")
    return 0


# ═══════════════════════════════════════════════════════════
# 代码生成命令入口
# ═══════════════════════════════════════════════════════════

def cmd_crawl_baike(args: argparse.Namespace) -> int:
    """调用百度百科爬虫"""
    script = Path(__file__).parent.parent / 'scripts' / 'crawl_baike.py'
    if not script.exists():
        print(f"错误: 爬虫脚本不存在 {script}")
        return 1
    cmd = f"python {script} --output {args.output} --limit {args.limit}"
    if args.category:
        cmd += f" --category {args.category}"
    if args.delay:
        cmd += f" --delay {args.delay}"
    return os.system(cmd) >> 8


def cmd_crawl_wikipedia(args: argparse.Namespace) -> int:
    """调用维基百科爬虫"""
    script = Path(__file__).parent.parent / 'scripts' / 'crawl_wikipedia.py'
    if not script.exists():
        print(f"错误: 爬虫脚本不存在 {script}")
        return 1
    cmd = f"python {script} --mode {args.mode} --output {args.output} --limit {args.limit} --lang {args.lang} --delay {args.delay}"
    if args.mode == 'category' and args.category:
        cmd += f" --category {args.category}"
    if args.mode == 'search' and args.query:
        cmd += f" --query {args.query}"
    if args.mode == 'list' and args.titles:
        cmd += f" --titles {args.titles}"
    if args.proxy:
        cmd += f" --proxy {args.proxy}"
    return os.system(cmd) >> 8


def cmd_crawl_wikipedia_batch(args: argparse.Namespace) -> int:
    """调用维基百科批量爬虫"""
    script = Path(__file__).parent.parent / 'scripts' / 'crawl_wikipedia_batch.py'
    if not script.exists():
        print(f"错误: 批量爬虫脚本不存在 {script}")
        return 1
    cmd = f"python {script} --target {args.target} --output {args.output} --lang {args.lang} --delay {args.delay}"
    if args.proxy:
        cmd += f" --proxy {args.proxy}"
    if args.categories_file:
        cmd += f" --categories-file {args.categories_file}"
    if args.skip_random:
        cmd += " --skip-random"
    return os.system(cmd) >> 8


def cmd_crawl_sogou_baike(args: argparse.Namespace) -> int:
    """调用搜狗百科爬虫"""
    script = Path(__file__).parent.parent / 'scripts' / 'crawl_sogou_baike.py'
    if not script.exists():
        print(f"错误: 搜狗百科爬虫脚本不存在 {script}")
        return 1
    cmd = f"python {script} --target {args.target} --output {args.output} --delay {args.delay} --max-pages {args.max_pages}"
    if args.seeds_file:
        cmd += f" --seeds-file {args.seeds_file}"
    return os.system(cmd) >> 8


def cmd_code_generate(args: argparse.Namespace) -> int:
    """调用代码语料生成器"""
    script = Path(__file__).parent.parent / 'scripts' / 'generate_code_data.py'
    if not script.exists():
        print(f"错误: 代码生成脚本不存在 {script}")
        return 1
    cmd = f"python {script} --output {args.output} --count {args.count}"
    if args.include_markdown:
        cmd += " --include-markdown"
    return os.system(cmd) >> 8


# ═══════════════════════════════════════════════════════════
# 系统状态
# ═══════════════════════════════════════════════════════════

def cmd_status(args: argparse.Namespace) -> int:
    """显示系统状态"""
    stats = _collect_system_stats()
    print("=" * 60)
    print(f"TGAI 系统状态 | {stats['time']}")
    print("=" * 60)
    print(f"CPU: {stats.get('cpu_percent', '-')}%")
    print(f"内存: {stats.get('ram_used_gb', '-')}GB / {stats.get('ram_total_gb', '-')}GB ({stats.get('ram_percent', '-')}%)")
    if 'gpu_name' in stats:
        print(f"GPU: {stats['gpu_name']}")
        print(f"显存: {stats['gpu_mem_used_gb']}GB / {stats['gpu_mem_total_gb']}GB ({stats['gpu_mem_percent']}%)")
    else:
        print("GPU: 未检测到 CUDA")

    # 模型文件
    ckpt_dir = 'checkpoints'
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and not f.endswith('.tmp')]
        print(f"\n检查点文件 ({len(ckpts)}):")
        for f in sorted(ckpts):
            fp = os.path.join(ckpt_dir, f)
            size_mb = os.path.getsize(fp) / 1e6
            print(f"  {f:30s} {size_mb:8.1f} MB")
    return 0


# ═══════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='tgai_cli',
        description='TGAI 云端训练与数据管理 CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 生成云端配置文件
  python tgai_cli.py config --output autodl_config.yaml

  # 按配置训练
  python tgai_cli.py train --config autodl_config.yaml --use_gpu

  # 直接指定参数训练（覆盖配置文件）
  python tgai_cli.py train --d_model 768 --n_layers 16 --batch 32 --epochs 100 --use_gpu

  # 对话
  python tgai_cli.py chat --use_gpu --temperature 0.8

  # 下载公开数据集并转换
  python tgai_cli.py data download --dataset chinese-cosmopedia --output data/cosmopedia.jsonl

  # 生成代码语料
  python tgai_cli.py code generate --count 5000 --output data/code.jsonl

  # 合并所有语料
  python tgai_cli.py data merge data/train_all.jsonl data/baike.jsonl data/code.jsonl --output data/train_all.jsonl
""",
    )
    sub = parser.add_subparsers(dest='command', help='子命令')

    # config
    p_cfg = sub.add_parser('config', help='生成默认云端训练配置文件')
    p_cfg.add_argument('--output', '-o', type=str, default='autodl_config.yaml')
    p_cfg.add_argument('--force', action='store_true')
    p_cfg.set_defaults(func=cmd_config)

    # train
    p_train = sub.add_parser('train', help='启动训练')
    p_train.add_argument('--config', '-c', type=str, help='YAML 配置文件')
    p_train.add_argument('--data', type=str, help='训练数据路径')
    p_train.add_argument('--epochs', type=int, help='训练轮数')
    p_train.add_argument('--batch', type=int, help='批次大小')
    p_train.add_argument('--lr', type=float, help='学习率')
    p_train.add_argument('--d_model', type=int, help='隐藏维度')
    p_train.add_argument('--n_layers', type=int, help='层数')
    p_train.add_argument('--n_heads', type=int, help='注意力头数')
    p_train.add_argument('--d_ff', type=int, help='FFN维度')
    p_train.add_argument('--seq_len', type=int, help='序列长度')
    p_train.add_argument('--vocab_size', type=int, help='词表大小')
    p_train.add_argument('--dropout', type=float, help='dropout')
    p_train.add_argument('--n_experts', type=int, help='MoE专家数')
    p_train.add_argument('--n_activated', type=int, help='激活专家数')
    p_train.add_argument('--warmup', type=int, help='warmup步数')
    p_train.add_argument('--grad_accum', type=int, help='梯度累积')
    p_train.add_argument('--checkpoint_dir', type=str, help='检查点目录')
    p_train.add_argument('--tokenizer_path', type=str, help='分词器路径')
    p_train.add_argument('--resume', type=str, help='续训 checkpoint')
    p_train.add_argument('--use_gpu', action='store_true', help='使用 GPU')
    p_train.set_defaults(func=cmd_train)

    # chat
    p_chat = sub.add_parser('chat', help='交互式对话')
    p_chat.add_argument('--checkpoint', type=str, help='checkpoint 路径')
    p_chat.add_argument('--tokenizer_path', type=str, default='checkpoints/tokenizer.json')
    p_chat.add_argument('--temperature', type=float, default=0.8)
    p_chat.add_argument('--max_tokens', type=int, default=128)
    p_chat.add_argument('--use_gpu', action='store_true')
    p_chat.add_argument('--compile', action='store_true', help='启用 torch.compile')
    p_chat.set_defaults(func=cmd_chat)

    # data
    p_data = sub.add_parser('data', help='数据管理')
    data_sub = p_data.add_subparsers(dest='data_cmd', help='数据子命令')

    p_stats = data_sub.add_parser('stats', help='统计训练数据')
    p_stats.add_argument('--path', type=str, default='tgai_nlp/data/train_all.jsonl')
    p_stats.set_defaults(func=cmd_data_stats)

    p_clean = data_sub.add_parser('clean', help='清理训练数据')
    p_clean.set_defaults(func=cmd_data_clean)

    p_merge = data_sub.add_parser('merge', help='合并多个 jsonl 文件')
    p_merge.add_argument('inputs', nargs='+', help='输入文件')
    p_merge.add_argument('--output', '-o', type=str, help='输出文件')
    p_merge.set_defaults(func=cmd_data_merge)

    # code
    p_code = sub.add_parser('code', help='代码/Markdown 语料')
    code_sub = p_code.add_subparsers(dest='code_cmd', help='代码子命令')

    p_gen = code_sub.add_parser('generate', help='生成代码语料')
    p_gen.add_argument('--output', '-o', type=str, default='tgai_nlp/data/code.jsonl')
    p_gen.add_argument('--count', type=int, default=1000, help='生成条数')
    p_gen.add_argument('--include-markdown', action='store_true', help='包含 Markdown 语料')
    p_gen.set_defaults(func=cmd_code_generate)

    # status
    p_status = sub.add_parser('status', help='显示系统状态')
    p_status.set_defaults(func=cmd_status)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
