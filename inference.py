"""
TGAI V4 推理与对话脚本
======================
基于 KV Cache 加速的自回归生成。

特性:
  - KV Cache: 推理时缓存键值对，避免重复计算
  - 流式生成: yield 逐 token，适合实时 UI
  - 多种采样策略: Temperature + Top-K + Top-P + 重复惩罚
  - 交互式对话 / 单次生成 / 批量评估

用法:
    python inference.py                          # 交互式对话
    python inference.py --mode generate          # 单次生成
    python inference.py --prompt "苹果是什么"     # 指定prompt
    python inference.py --stream                 # 流式输出
"""

import os
import sys
import re
import json
import argparse
import time
from typing import List, Optional, Generator, Tuple

import torch
import torch.nn.functional as F

from tokenizer import ChineseTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID
from model import TGAILanguageModel, TGAIConfig, create_model


# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════
def load_model(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    compile_model: bool = False,
) -> Tuple[TGAILanguageModel, ChineseTokenizer]:
    """加载训练好的模型和分词器"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 未找到: {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"分词器未找到: {tokenizer_path}")

    print(f"分词器: {tokenizer_path}")
    tokenizer = ChineseTokenizer.load(tokenizer_path)
    print(f"  词表: {tokenizer.vocab_size_actual}")

    print(f"模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt.get('model_config', {})
    state_dict = ckpt['model_state_dict']

    embed_weight = state_dict['token_embedding.weight']
    ckpt_vocab = embed_weight.shape[0]
    ckpt_d_model = embed_weight.shape[1]

    print(f"  checkpoint 词表: {ckpt_vocab}, d_model: {ckpt_d_model}")

    # 读取 MoE 配置（V4 新字段，向后兼容）
    n_experts = mc.get('n_experts', 4)
    n_activated = mc.get('n_activated', 2)

    model = create_model(
        vocab_size=ckpt_vocab,
        d_model=mc.get('d_model', ckpt_d_model),
        n_layers=mc.get('n_layers', 8),
        n_heads=mc.get('n_heads', 8),
        d_ff=mc.get('d_ff', 1024),
        max_seq_len=mc.get('max_seq_len', 256),
        dropout=0.0,  # 推理时关闭 dropout
        n_experts=n_experts,
        n_activated=n_activated,
    )
    # 向后兼容: 迁移旧格式 RoPE cos/sin
    for key in list(state_dict.keys()):
        if '.rope.cos' in key or '.rope.sin' in key:
            if state_dict[key].dim() == 2:
                state_dict[key] = state_dict[key].repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 尝试 torch.compile 加速
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile: 已启用 (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile 失败，使用普通模式: {e}")

    n_params = model.count_parameters()
    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', '?')
    print(f"  参数: {n_params:,} | Epoch: {epoch} | Loss: {loss}")
    print(f"  设备: {device}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════
# 流式生成器
# ═══════════════════════════════════════════════════════════
class TextGenerator:
    """封装生成逻辑，支持流式和非流式"""

    def __init__(self, model: TGAILanguageModel, tokenizer: ChineseTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._debug_info: List[dict] = []  # 调试信息收集

    def pop_debug(self) -> List[dict]:
        """取出并清空调试信息（供外部轮询）"""
        info = self._debug_info.copy()
        self._debug_info.clear()
        return info

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        frequency_penalty: float = 0.25,
        repetition_penalty: float = 1.05,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        生成文本。stream=True 返回逐 token 生成器。
        repetition_penalty: 1.0=关闭, 1.05=轻度, 1.2=较强
        """
        # 构造 prompt — 与训练数据格式保持一致
        formatted = f"用户:{prompt}\nTGAI?"
        prompt_ids = [BOS_ID] + self.tokenizer.encode(formatted, add_special=False)
        prompt_ids = [min(tid, self.model.config.vocab_size - 1) for tid in prompt_ids]

        if stream:
            return self._generate_stream(prompt_ids, max_new_tokens, temperature, top_k, top_p, frequency_penalty, repetition_penalty)
        else:
            return self._generate_full(prompt_ids, max_new_tokens, temperature, top_k, top_p, frequency_penalty, repetition_penalty)

    def _generate_full(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        frequency_penalty: float,
        repetition_penalty: float = 1.0,
    ) -> str:
        """非流式: 一次性返回完整回复"""
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=EOS_ID,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
        return self._decode_response(prompt_ids, output_ids[0].tolist())

    def _generate_stream(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        frequency_penalty: float,
        repetition_penalty: float = 1.05,
    ) -> Generator[str, None, None]:
        """流式: 逐个 token yield，使用 KV Cache 加速"""
        model = self.model
        model.eval()
        inv_temp = 1.0 / max(temperature, 0.01)
        vocab_size = model.config.vocab_size
        min_new = 5  # 前5个token禁止EOS

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Prefill
        kv_caches = [{} for _ in range(model.config.n_layers)]
        logits, kv_caches = model(prompt_tensor, kv_caches)
        next_logits = logits[:, -1, :] * inv_temp

        generated_ids = list(prompt_ids)
        token_counts = torch.zeros(vocab_size, dtype=torch.long, device=self.device)
        new_tokens = 0
        t0 = time.time()

        for _ in range(max_new_tokens):
            step_start = time.time()
            # 超长截断
            if len(generated_ids) > model.config.max_seq_len:
                generated_ids = generated_ids[-model.config.max_seq_len // 2:]
                kv_caches = [{} for _ in range(model.config.n_layers)]
                token_counts.zero_()
                for t in generated_ids:
                    if 0 <= t < vocab_size:
                        token_counts[t] += 1

            # 前 min_new tokens 禁止 EOS (防止一上来就结束)
            if new_tokens < min_new and EOS_ID < vocab_size:
                next_logits[:, EOS_ID] = float('-inf')

            # 轻度频率惩罚 (欠训练模型用更低的惩罚)
            freq_penalty_val = 0.0
            if frequency_penalty > 0:
                appeared = token_counts > 0
                if appeared.any():
                    penalty = frequency_penalty * token_counts[appeared].float().unsqueeze(0).clamp(max=2.0)
                    next_logits[:, appeared] -= penalty
                    if token_counts.sum() > 0:
                        freq_penalty_val = float(penalty.max())

            # 重复惩罚 (对已出现 token 降权)
            if repetition_penalty > 1.0 and token_counts.any():
                appeared = token_counts > 0
                next_logits[:, appeared] /= repetition_penalty

            # 采样前记录 top-3 候选 (基于 raw logits 的 softmax)
            raw_logits_snapshot = next_logits.clone()

            # 退化预防: top-1 概率 > 95% 时打压 (用 logit 差值判断，避免一次 softmax)
            sorted_for_check = torch.sort(next_logits, descending=True)[0]
            top1_gap = float(sorted_for_check[0, 0] - sorted_for_check[0, min(1, sorted_for_check.shape[-1]-1)])
            # 用温度校正: 高温下 logit 差距自然小，放宽阈值
            gap_threshold = 0.2 / max(temperature, 0.1)
            if top1_gap > gap_threshold and new_tokens > 0:
                top1_id = int(next_logits.argmax(dim=-1)[0])
                next_logits[0, top1_id] -= 1.5  # 适当打压

            # Top-K + Top-P 采样
            next_logits = self._sample(next_logits.clone(), top_k, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_token = min(next_token, vocab_size - 1)

            # ── 收集调试信息 (基于最终 probs) ──
            token_prob = float(probs[0, next_token])
            # top-5 候选
            top_n = min(5, vocab_size)
            top_vals, top_ids = torch.topk(probs, top_n)
            top_list = []
            for i in range(len(top_ids[0])):
                tid = top_ids[0, i].item()
                tprob = top_vals[0, i].item()
                ttext = self.tokenizer.decode([tid], skip_special=True).strip("'")
                top_list.append((tid, ttext or '∅', tprob))

            # 原始 logits top-5 (采样前)
            probs_before_sample = F.softmax(raw_logits_snapshot, dim=-1)
            raw_top_v, raw_top_i = torch.topk(probs_before_sample, top_n)
            raw_top = []
            for i in range(len(raw_top_i[0])):
                tid = raw_top_i[0, i].item()
                tprob = raw_top_v[0, i].item()
                ttext = self.tokenizer.decode([tid], skip_special=True).strip("'")
                raw_top.append((tid, ttext or '∅', tprob))

            step_ms = (time.time() - step_start) * 1000
            self._debug_info.append({
                'step': new_tokens + 1,
                'token_id': next_token,
                'text': self.tokenizer.decode([next_token], skip_special=True).strip("'"),
                'prob': token_prob,
                'top_sampled': top_list,
                'top_raw': raw_top,
                'freq_penalty': freq_penalty_val,
                'repetition_penalty': repetition_penalty,
                'temp': temperature,
                'step_ms': step_ms,
            })

            generated_ids.append(next_token)
            new_tokens += 1
            if 0 <= next_token < vocab_size:
                token_counts[next_token] += 1

            # 退化检测: 连续8个相同token → 强制结束
            if new_tokens >= 8:
                last8 = generated_ids[-8:]
                if len(set(last8)) == 1:
                    break

            if next_token == EOS_ID:
                break

            # 流式输出
            new_text = self.tokenizer.decode([next_token], skip_special=True)
            if new_text:
                yield new_text

            # Decode step
            token_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            token_logits, kv_caches = model(token_input, kv_caches, cache_pos=len(generated_ids) - 1)
            next_logits = token_logits[:, -1, :] * inv_temp

    def _sample(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Top-K + Top-P 采样，带健全的兜底。合并为单次排序避免冗余。"""
        vocab_size = logits.size(-1)

        if top_k > 0 or top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)

        # Top-K: 只保留前 k 个
        if top_k > 0:
            raw_eos_logit = logits[0, EOS_ID].item() if EOS_ID < vocab_size else None
            k = min(top_k, logits.size(-1))
            threshold = sorted_l[..., k - 1, None]
            logits[logits < threshold] = float('-inf')
            # EOS 被裁掉就恢复 (使用常量而非硬编码)
            if raw_eos_logit is not None and torch.isinf(logits[0, EOS_ID]):
                logits[0, EOS_ID] = raw_eos_logit

        # Top-P: 基于 top-k 过滤后的 logits 再做一次 sort
        if top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            logits[:, sorted_i[~keep]] = float('-inf')

        # 兜底: 全 inf 时给 non-special tokens 均匀概率
        if torch.all(torch.isinf(logits)):
            logits = torch.ones_like(logits) * float('-inf')
            safe_start = EOS_ID + 1  # 跳过 PAD/UNK/BOS/EOS
            safe_end = min(safe_start + 100, vocab_size)
            logits[0, safe_start:safe_end] = 0.0
            if EOS_ID < vocab_size:
                logits[0, EOS_ID] = 0.0  # EOS 也保留

        return logits

    def _decode_response(self, prompt_ids: List[int], all_ids: List[int]) -> str:
        """从生成的 token IDs 中提取回复文本"""
        full_text = self.tokenizer.decode(all_ids, skip_special=True)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special=True)

        if full_text.startswith(prompt_text):
            response = full_text[len(prompt_text):]
        else:
            response = full_text

        # 清理但不过度截断: 只去掉末尾的 "用户:" 和新一轮对话
        # 不再粗暴截断第一行 —— 之前这是导致空回复的主要原因
        for sep in ['\n用户:', '用户:', '用户：']:
            idx = response.find(sep)
            if idx >= 0:
                response = response[:idx]
                break

        # 去掉首尾空白和换行
        response = response.strip().strip('\n').strip()
        # 去掉不可打印字符（保留中文、英文、数字、标点）
        response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', response)

        # 只有完全空且生成了多个 token 才返回提示
        if not response and len(all_ids) > len(prompt_ids) + 1:
            return "[模型未生成有效回复]"
        return response or "[模型未生成回复]"


# ═══════════════════════════════════════════════════════════
# 交互式对话
# ═══════════════════════════════════════════════════════════
def interactive_chat(
    model: TGAILanguageModel,
    tokenizer: ChineseTokenizer,
    temperature: float = 0.8,
    max_new_tokens: int = 128,
    stream: bool = True,
):
    """交互式对话循环"""
    print("\n" + "=" * 60)
    print("  TGAI V4 交互式对话")
    print("  输入 /help 查看命令, /exit 退出")
    print("=" * 60)

    generator = TextGenerator(model, tokenizer)
    history: List[str] = []
    current_temp = temperature
    current_freq_penalty = 0.3

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            cmd = user_input.lower()
            if cmd == '/exit':
                print("再见!")
                break
            elif cmd == '/help':
                print("命令: /exit /temp N /freq N /stream /clear /help")
                print("  /temp N   - 设置温度 (0.1~2.0)")
                print("  /freq N   - 设置频率惩罚 (0.0~1.0)")
                print("  /stream   - 切换流式输出")
                print("  /clear    - 清除对话历史")
            elif cmd == '/clear':
                history = []
                print("[对话历史已清除]")
            elif cmd == '/stream':
                stream = not stream
                print(f"[流式输出: {'ON' if stream else 'OFF'}]")
            elif cmd.startswith('/temp'):
                try:
                    current_temp = max(0.1, min(2.0, float(user_input.split()[-1])))
                    print(f"[温度: {current_temp}]")
                except (ValueError, IndexError):
                    print(f"[当前温度: {current_temp}]")
            elif cmd.startswith('/freq'):
                try:
                    current_freq_penalty = max(0.0, min(1.0, float(user_input.split()[-1])))
                    print(f"[频率惩罚: {current_freq_penalty}]")
                except (ValueError, IndexError):
                    print(f"[当前频率惩罚: {current_freq_penalty}]")
            continue

        try:
            t0 = time.time()
            if stream:
                print("TGAI: ", end="", flush=True)
                response = ""
                for chunk in generator.generate(
                    user_input, max_new_tokens, current_temp,
                    frequency_penalty=current_freq_penalty, stream=True,
                ):
                    print(chunk, end="", flush=True)
                    response += chunk
                print(f"\n  [{time.time() - t0:.1f}s]")
            else:
                response = generator.generate(
                    user_input, max_new_tokens, current_temp,
                    frequency_penalty=current_freq_penalty,
                )
                elapsed = time.time() - t0
                print(f"TGAI: {response}\n  [{elapsed:.1f}s]")

            if not response:
                response = "[模型未生成回复]"

        except Exception as e:
            response = f"[生成错误: {e}]"
            print(f"TGAI: {response}")

        history.append(user_input)
        history.append(response)


# ═══════════════════════════════════════════════════════════
# 单次生成
# ═══════════════════════════════════════════════════════════
def single_generate(
    model: TGAILanguageModel,
    tokenizer: ChineseTokenizer,
    prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    stream: bool = False,
):
    """单次文本生成"""
    print(f"Prompt: {prompt}")
    print("-" * 40)

    generator = TextGenerator(model, tokenizer)

    if stream:
        print("生成: ", end="", flush=True)
        for chunk in generator.generate(prompt, max_new_tokens, temperature, stream=True):
            print(chunk, end="", flush=True)
        print()
    else:
        response = generator.generate(prompt, max_new_tokens, temperature)
        print(f"生成:\n{response}")


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TGAI V4 推理')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--tokenizer', type=str, default='checkpoints/tokenizer.json')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'generate'])
    parser.add_argument('--prompt', type=str, default='你好，请问苹果是什么')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--stream', action='store_true', help='流式输出')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

    # 自动找 checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path) or 'checkpoints'
        if os.path.exists(ckpt_dir):
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and not f.endswith('.tmp')],
                key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
                reverse=True,
            )
            for ck in ckpts:
                if ck in ('last_step.pt',):
                    continue
                ckpt_path = os.path.join(ckpt_dir, ck)
                print(f"自动选择: {ck}")
                break

    try:
        model, tokenizer = load_model(ckpt_path, args.tokenizer, device)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请先运行 train.py 训练模型。")
        sys.exit(1)

    if args.mode == 'interactive':
        interactive_chat(model, tokenizer, args.temperature, args.max_tokens, args.stream)
    else:
        single_generate(model, tokenizer, args.prompt, args.temperature, args.max_tokens, args.stream)