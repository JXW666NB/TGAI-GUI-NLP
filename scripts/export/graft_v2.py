"""
TGAI 模型嫁接 (Weights Grafting)
===================================
将 V1 checkpoint 的权重扩展到更大配置的 V2 模型。
扩维部分用 V1 权重填充，新增部分随机初始化。

预设：
    lite1: d_model=896, layers=22, heads=14, experts=6, d_ff=3584  → ~1.4B
    lite2: d_model=896, layers=28, heads=14, experts=8, d_ff=3584  → ~1.8B  
    2b:    d_model=1024, layers=28, heads=16, experts=8, d_ff=4096 → ~2.07B
"""

import os, sys, argparse, torch

# ─── 将项目根目录加入 path ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TGAIConfig, TGAILanguageModel


# ─── 预设配置 ────────────────────────────────────────────
PRESETS = {
    'lite1': dict(d_model=896, n_layers=22, n_heads=14, n_experts=6, d_ff=3584),
    'lite2': dict(d_model=896, n_layers=28, n_heads=14, n_experts=8, d_ff=3584),
    '2b':    dict(d_model=1024, n_layers=28, n_heads=16, n_experts=8, d_ff=4096),
}


def load_ckpt(path: str) -> dict:
    """加载 checkpoint（支持仅权重和完整检查点）"""
    print(f'[加载] {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f'  epoch={ckpt.get("epoch", "?")}, step={ckpt.get("global_step", "?")}, '
          f'ppl={ckpt.get("best_ppl", "?")}')
    return ckpt


def graft_tensor(old_w: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """把 old_w 写入左上角，其余补零"""
    new_t = torch.zeros(new_shape, dtype=old_w.dtype)
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_w.shape, new_shape))
    new_t[slices] = old_w[slices]
    return new_t


def count_params(config_args: dict) -> int:
    """估算参数量"""
    d = config_args
    vocab = d.get('vocab_size', 32768)
    dm = d['d_model']
    layers = d['n_layers']
    heads = d['n_heads']
    experts = d['n_experts']
    dff = d['d_ff']

    # Embedding + LM Head (tied): vocab * d_model
    emb = vocab * dm

    # Per layer:
    #   Attention: 4 * d_model^2 (Q,K,V,O)
    attn = 4 * dm * dm
    #   MoE: n_experts * 3 * d_model * d_ff (w1,w2,w3 per expert) + router + gate
    moe = experts * 3 * dm * dff + 2 * experts * dm
    #   RMSNorm: 2 * d_model (ln1, ln2)
    norm = 2 * dm

    per_layer = attn + moe + norm
    total = emb + layers * per_layer + dm  # + final_norm
    return total


def graft_attention(qkv_weight: torch.Tensor, new_weight: torch.Tensor,
                    old_n_heads: int, new_n_heads: int, old_d_model: int, new_d_model: int):
    """
    Q/K/V 权重嫁接: (n_heads*d_k, d_model)
    前 old_n_heads 个头和前 old_d_model 列从旧权重复制
    """
    d_k = old_d_model // old_n_heads  # same for both
    old_heads_dim = old_n_heads * d_k
    # 前 old_n_heads 个头、前 old_d_model 列
    new_weight[:old_heads_dim, :old_d_model] = qkv_weight[:old_heads_dim, :old_d_model]
    return new_weight


def graft_output(old_weight: torch.Tensor, new_weight: torch.Tensor,
                 old_n_heads: int, new_n_heads: int, old_d_model: int, new_d_model: int):
    """O 投影: (d_model, n_heads*d_k) —— QKV 的转置"""
    d_k = old_d_model // old_n_heads
    old_heads_dim = old_n_heads * d_k
    new_weight[:old_d_model, :old_heads_dim] = old_weight[:old_d_model, :old_heads_dim]
    return new_weight


def graft_v2(ckpt: dict, preset: str, vocab_size: int = None) -> dict:
    """
    主嫁接逻辑。返回新的 state_dict。
    """
    old_config = ckpt.get('model_config', {})
    old_state = ckpt['model_state_dict']

    # 读取旧参数
    old_vocab = old_config.get('vocab_size', 32768)
    old_dm = old_config['d_model']
    old_layers = old_config['n_layers']
    old_heads = old_config['n_heads']
    old_experts = old_config.get('n_experts', 4)
    old_dff = old_config['d_ff']
    old_seq = old_config.get('max_seq_len', 512)

    # 读取新参数
    new_cfg = PRESETS[preset]
    new_dm = new_cfg['d_model']
    new_layers = new_cfg['n_layers']
    new_heads = new_cfg['n_heads']
    new_experts = new_cfg['n_experts']
    new_dff = new_cfg['d_ff']
    new_vocab = vocab_size or old_vocab

    assert old_dm % old_heads == 0, f"old d_model {old_dm} not divisible by n_heads {old_heads}"
    assert new_dm % new_heads == 0, f"new d_model {new_dm} not divisible by n_heads {new_heads}"

    old_dk = old_dm // old_heads
    new_dk = new_dm // new_heads
    assert old_dk == new_dk, (
        f"d_k must match: old={old_dk} ({old_dm}/{old_heads}), "
        f"new={new_dk} ({new_dm}/{new_heads}). Adjust n_heads so {new_dm}/{new_heads}={old_dk}"
    )

    print(f'\n[嫁接] V1 ({old_dm}d, {old_layers}层, {old_heads}头, {old_experts}专家, '
          f'{old_dff}dff, {old_seq}seq)')
    print(f'        → V2 ({new_dm}d, {new_layers}层, {new_heads}头, {new_experts}专家, '
          f'{new_dff}dff, train_seq=8192, yay_n→128K)')

    # 创建 V2 模型（随机初始化）
    # max_seq_len=8192 用于 RoPE 训练缓存，128K 推理通过 self_extend (YaRN) 扩展
    config = TGAIConfig(
        vocab_size=new_vocab, d_model=new_dm, n_layers=new_layers,
        n_heads=new_heads, d_ff=new_dff, max_seq_len=8192,
        n_experts=new_experts, n_activated=2,
        self_extend=True,
        extend_max_seq_len=131072,  # 128K
    )
    model = TGAILanguageModel(config)
    new_state = model.state_dict()

    grafted_keys = 0
    new_keys = 0
    stats = []

    # ── 1. Token Embedding ──
    k = 'token_embedding.weight'
    if k in old_state:
        new_state[k] = graft_tensor(old_state[k], (new_vocab, new_dm))
        grafted_keys += 1
        stats.append(f'{k}: {tuple(old_state[k].shape)} → {tuple(new_state[k].shape)}')

    # ── 2. Final Norm ──
    k = 'final_norm.weight'
    if k in old_state:
        new_state[k] = graft_tensor(old_state[k], (new_dm,))
        grafted_keys += 1

    # ── 3. Layers ──
    for layer_idx in range(new_layers):
        prefix = f'blocks.{layer_idx}.'
        old_prefix = f'blocks.{min(layer_idx, old_layers - 1)}.'

        if layer_idx < old_layers:
            # 嫁接已有层
            # RMSNorm
            for norm_name in ['ln1.weight', 'ln2.weight']:
                k = prefix + norm_name
                ok = f'blocks.{layer_idx}.' + norm_name
                new_state[k] = graft_tensor(old_state[ok], (new_dm,))

            # Attention QKV
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                k = prefix + f'attn.{proj}.weight'
                ok = f'blocks.{layer_idx}.attn.{proj}.weight'
                graft_attention(old_state[ok], new_state[k],
                                old_n_heads=old_heads, new_n_heads=new_heads,
                                old_d_model=old_dm, new_d_model=new_dm)

            # Attention O
            k = prefix + 'attn.out_proj.weight'
            ok = f'blocks.{layer_idx}.attn.out_proj.weight'
            graft_output(old_state[ok], new_state[k],
                         old_n_heads=old_heads, new_n_heads=new_heads,
                         old_d_model=old_dm, new_d_model=new_dm)

            # RoPE cos/sin
            for rope_param in ['attn.rope.cos', 'attn.rope.sin']:
                k = prefix + rope_param
                ok = f'blocks.{layer_idx}.' + rope_param
                # old shape: (1,1,512,64), new shape: (1,1,4096,64)
                new_state[k] = graft_tensor(old_state[ok], new_state[k].shape)

            # MoE Router & Gate
            k = prefix + 'moe.router.weight'
            ok = f'blocks.{layer_idx}.moe.router.weight'
            new_state[k] = graft_tensor(old_state[ok], (new_experts, new_dm))

            k = prefix + 'moe.gate.weight'
            ok = f'blocks.{layer_idx}.moe.gate.weight'
            new_state[k] = graft_tensor(old_state[ok], (new_experts, new_dm))

            # MoE Experts (SwiGLUFFN: w1(d_ff,dm), w2(d_ff,dm), w3(dm,d_ff))
            for eid in range(new_experts):
                if eid < old_experts:
                    # w1, w2: (d_ff, d_model)
                    for wname in ['w1', 'w2']:
                        k = prefix + f'moe.experts.{eid}.{wname}.weight'
                        ok = f'blocks.{layer_idx}.moe.experts.{eid}.{wname}.weight'
                        new_state[k] = graft_tensor(old_state[ok], (new_dff, new_dm))
                    # w3: (d_model, d_ff)
                    k = prefix + f'moe.experts.{eid}.w3.weight'
                    ok = f'blocks.{layer_idx}.moe.experts.{eid}.w3.weight'
                    new_state[k] = graft_tensor(old_state[ok], (new_dm, new_dff))
                else:
                    # 新专家：保持随机初始化
                    pass

            grafted_keys += 1
        else:
            # 全新层：保持随机初始化
            new_keys += 1

    # ── 4. LM Head (weight-tied with embedding) ──
    # 在 TGAILanguageModel 中 lm_head.weight 指向 token_embedding.weight
    # 所以不必单独处理，只需确保 lm_head 的引用正确
    # 但 state_dict 里可能只有一份（如果用了 weight tying）
    if 'lm_head.weight' in new_state and 'lm_head.weight' in old_state:
        new_state['lm_head.weight'] = graft_tensor(
            old_state['lm_head.weight'], (new_vocab, new_dm)
        )
    # 如果 lm_head 不在 old_state 中（weight tied 时 torch 可能不保存），
    # 它会在 model 加载后自动通过 __init__ 的 weight tying 设置

    # ── 构建输出 ──
    old_params = count_params(old_config)
    new_params = count_params({**new_cfg, 'vocab_size': new_vocab})

    print(f'\n  参数: {old_params/1e6:.0f}M → {new_params/1e6:.0f}M '
          f'({new_params/old_params:.1f}x)')
    print(f'  嫁接层数: {min(old_layers, new_layers)}/{new_layers}')
    print(f'  全新层数: {new_keys}/{new_layers}')
    if stats:
        print(f'  关键张量 ({len(stats)}):')
        for s in stats:
            print(f'    {s}')

    return {'model_state_dict': new_state, 'model_config': {
        'vocab_size': new_vocab, 'd_model': new_dm, 'n_layers': new_layers,
        'n_heads': new_heads, 'd_ff': new_dff, 'max_seq_len': 8192,
        'dropout': old_config.get('dropout', 0.1),
        'n_experts': new_experts, 'n_activated': 2,
        'self_extend': True,
        'extend_max_seq_len': 131072,
    }}


def main():
    parser = argparse.ArgumentParser(description='TGAI 模型嫁接 (权重扩展)')
    parser.add_argument('--checkpoint', '-c', required=True, help='V1 checkpoint 路径 (.pt)')
    parser.add_argument('--output', '-o', required=True, help='输出路径')
    parser.add_argument('--preset', '-p', choices=list(PRESETS), default='2b',
                        help='预设配置 (默认: 2b)')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='覆盖词汇表大小 (默认使用原始 checkpoint 的)')
    args = parser.parse_args()

    print('=' * 60)
    print('TGAI 模型嫁接 V2')
    print('=' * 60)
    for name, cfg in PRESETS.items():
        params = count_params({**cfg, 'vocab_size': 32768})
        marker = ' ←' if name == args.preset else ''
        print(f'  [{name}] {cfg["d_model"]}d × {cfg["n_layers"]}层 × '
              f'{cfg["n_heads"]}h × {cfg["n_experts"]}专家 × {cfg["d_ff"]}dff '
              f'→ ~{params/1e6:.1f}M{marker}')
    print()

    # 加载
    ckpt = load_ckpt(args.checkpoint)

    # 嫁接
    new_ckpt = graft_v2(ckpt, args.preset, args.vocab_size)

    # 保存（fp16 减半体积）
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # 模型权重转 fp16
    state = new_ckpt['model_state_dict']
    state_fp16 = {k: v.half() for k, v in state.items()}
    output = {'model_state_dict': state_fp16, 'model_config': new_ckpt['model_config'],
              'epoch': 0, 'global_step': 0, 'best_ppl': 999.0,
              'graft_source': os.path.basename(args.checkpoint)}
    torch.save(output, args.output)
    print(f'\n[完成] 嫁接模型保存至: {args.output}')
    print(f'  文件大小: {os.path.getsize(args.output) / 1024**3:.2f} GiB')
    print(f'  精度: fp16 (加载时自动适配)')


if __name__ == '__main__':
    main()
