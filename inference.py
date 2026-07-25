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
    self_extend: bool = False,
    extend_max_seq_len: int = 4096,
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
    ckpt_step = ckpt.get('step', ckpt.get('global_step', '?'))
    print(f"  训练步数: {ckpt_step}")
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
        self_extend=self_extend,
        extend_max_seq_len=extend_max_seq_len,
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
        min_new_tokens: int = 5,
        stream: bool = False,
        formatted: bool = False,
        token_bias: dict = None,
        prefix_bias: dict = None,
        force_prefix: str = None,
    ) -> str | Generator[str, None, None]:
        """
        生成文本。stream=True 返回逐 token 生成器。
        repetition_penalty: 1.0=关闭, 1.05=轻度, 1.2=较强
        min_new_tokens: 生成至少N个token后才允许EOS (防止过早结束)
        formatted: True=prompt已经是完整格式(如 "用户:...\\nTGAI?"), False=自动包装
        """
        # 构造 prompt — 与训练数据格式保持一致
        if formatted:
            prompt_text = prompt  # 已经是完整格式，不重新包装
        else:
            prompt_text = f"用户:{prompt}\nTGAI?"
        prompt_ids = [BOS_ID] + self.tokenizer.encode(prompt_text, add_special=False)
        prompt_ids = [min(tid, self.model.config.vocab_size - 1) for tid in prompt_ids]

        if stream:
            return self._generate_stream(prompt_ids, max_new_tokens, temperature, top_k, top_p, frequency_penalty, repetition_penalty, min_new_tokens, token_bias, prefix_bias, force_prefix)
        else:
            return self._generate_full(prompt_ids, max_new_tokens, temperature, top_k, top_p, frequency_penalty, repetition_penalty, min_new_tokens, token_bias, prefix_bias, force_prefix)

    def _generate_full(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        frequency_penalty: float,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 5,
        token_bias: dict = None,
        prefix_bias: dict = None,
        force_prefix: str = None,
    ) -> str:
        """非流式: 一次性返回完整回复（复用流式生成器）"""
        reply = ""
        for token in self._generate_stream(
            prompt_ids, max_new_tokens, temperature, top_k, top_p,
            frequency_penalty, repetition_penalty, min_new_tokens,
            token_bias, prefix_bias, force_prefix,
        ):
            reply += token
        return reply

    def _generate_stream(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        frequency_penalty: float,
        repetition_penalty: float = 1.05,
        min_new_tokens: int = 5,
        token_bias: dict = None,
        prefix_bias: dict = None,
        force_prefix: str = None,
    ) -> Generator[str, None, None]:
        """流式: 逐个 token yield，使用 KV Cache 加速"""
        model = self.model
        model.eval()
        inv_temp = 1.0 / max(temperature, 0.01)
        vocab_size = model.config.vocab_size
        min_new = min_new_tokens  # 允许调用方设置

        # Encode force_prefix (兜底起手式)
        force_ids = None
        force_pos = 0
        if force_prefix:
            force_ids = self.tokenizer.encode(force_prefix, add_special=False)
            force_ids = [min(tid, vocab_size - 1) for tid in force_ids]

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        effective_max = model.config.extend_max_seq_len if model.config.self_extend else model.config.max_seq_len

        # 如果 max_new_tokens 超过上下文窗口，自动缩减并警告
        if max_new_tokens >= effective_max:
            old_max = max_new_tokens
            max_new_tokens = max(effective_max - 16, 16)
            print(f"\n  [警告] max_tokens={old_max} 超出上下文窗口 {effective_max}，自动缩减为 {max_new_tokens}", flush=True)

        # 如果 prompt 超过模型上下文窗口，截断前半部分（保留 BOS + 最近上下文）
        prompt_len = len(prompt_ids)
        max_prompt = max(effective_max - max_new_tokens, 1)  # 至少保留 BOS
        if prompt_len > max_prompt:
            # 跳过 BOS 位置，从头截断
            prompt_ids = [prompt_ids[0]] + prompt_ids[prompt_len - max_prompt + 1:]
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Prefill
        kv_caches = [{} for _ in range(model.config.n_layers)]
        logits, kv_caches = model(prompt_tensor, kv_caches)
        next_logits = logits[:, -1, :] * inv_temp

        # 保存 prefill 阶段的专家使用情况 (用于诊断，不被后续 token 覆盖)
        model._prefill_expert_usage = {}
        for i, block in enumerate(model.blocks):
            usage = block.moe.expert_usage
            if usage is not None:
                model._prefill_expert_usage[i] = usage.detach().clone()

        generated_ids = list(prompt_ids)
        token_counts = torch.zeros(vocab_size, dtype=torch.long, device=self.device)
        new_tokens = 0
        t0 = time.time()
        _reply_so_far = ""
        _last_token_id = -1      # 连续重复检测
        _consecutive_same = 0
        _recent_tids = []        # 最近 12 个 token_id 用于模式检测
        _total_loops = 0          # 总循环次数 (调试用)

        for _ in range(max_new_tokens):
            _total_loops += 1
            step_start = time.time()
            # 超长截断
            if len(generated_ids) >= effective_max:
                generated_ids = generated_ids[-effective_max // 2:]
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

            # 连续重复 token 惩罚（递增打压，防止 "是是是"/"TIATIA" 死循环）
            if _consecutive_same >= 3:
                next_logits[0, _last_token_id] -= _consecutive_same * 3.0
            if _consecutive_same > 12 and new_tokens >= min_new:
                break  # 极端重复直接截断 (但不违反最小输出)

            # 退化预防: top-1 概率 > 95% 时打压 (用 logit 差值判断，避免一次 softmax)
            sorted_for_check = torch.sort(next_logits.detach(), descending=True)[0]
            top1_gap = float(sorted_for_check[0, 0] - sorted_for_check[0, min(1, sorted_for_check.shape[-1]-1)])
            # 用温度校正: 高温下 logit 差距自然小，放宽阈值
            gap_threshold = 0.2 / max(temperature, 0.1)
            if top1_gap > gap_threshold and new_tokens > 0:
                top1_id = int(next_logits.argmax(dim=-1)[0])
                next_logits[0, top1_id] -= 1.5  # 适当打压

            # 兜底起手式：当模型空回复时，跳过采样，直接注入 force_prefix token
            if force_ids is not None and force_pos < len(force_ids):
                next_token = force_ids[force_pos]
                force_pos += 1
                probs = F.softmax(next_logits, dim=-1)  # 用于调试信息
            else:
                # Top-K + Top-P 采样
                next_logits = self._sample(next_logits.clone(), top_k, top_p)
                # _sample fallback 可能恢复 EOS → 重新禁止
                if new_tokens < min_new and EOS_ID < vocab_size:
                    next_logits[:, EOS_ID] = float('-inf')
                # token bias: 惩罚特定 token 但不禁用
                if token_bias:
                    for tid, bias in token_bias.items():
                        next_logits[0, tid] += bias
                # prefix-triggered bias: 特定前缀出现时临时调整 token 概率
                # 空字符串 "" 作为前缀 = 仅第一个 token 生效（用于拦截跑偏开头）
                if prefix_bias:
                    for prefix, biases in prefix_bias.items():
                        if prefix.startswith("_"):
                            continue
                        match = False
                        if prefix == "" and not _reply_so_far:
                            match = True  # 第一个 token
                        elif _reply_so_far and _reply_so_far.rstrip().endswith(prefix):
                            match = True
                        if match:
                            for word, bias_val in biases.items():
                                if word.startswith("_") or not isinstance(bias_val, (int, float)):
                                    continue
                                for tid in self.tokenizer.encode(word, add_special=False):
                                    next_logits[0, tid] += bias_val
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                next_token = min(next_token, vocab_size - 1)

                # "用" 族重试 — 第一token多试几次，后续位置轻量重试
                _tries = 0
                _max_tries = 20 if not _reply_so_far else 3
                while (_tries < _max_tries
                       and self.tokenizer.decode([next_token], skip_special=True).strip() in ("用", "用户", "户")):
                    next_logits[0, next_token] = float('-inf')
                    if torch.all(torch.isinf(next_logits[0])):
                        break  # 全 ban 了，放弃重试
                    probs = F.softmax(next_logits, dim=-1)
                    if torch.isnan(probs).any() or torch.any(probs < 0):
                        break  # NaN，放弃重试
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    next_token = min(next_token, vocab_size - 1)
                    _tries += 1

            # ── 收集调试信息 (基于最终 probs) ──
            token_prob = float(probs[0, next_token].detach())
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

            # ── 连续重复 & 模式检测 ──
            if next_token == _last_token_id:
                _consecutive_same += 1
            else:
                _consecutive_same = 1
            _last_token_id = next_token
            _recent_tids.append(next_token)
            if len(_recent_tids) > 12:
                _recent_tids = _recent_tids[-12:]

            # 2-token 循环: [a,b,a,b,a,b]
            if len(_recent_tids) >= 6 and new_tokens >= min_new:
                if _recent_tids[-2:] == _recent_tids[-4:-2] == _recent_tids[-6:-4]:
                    break
            # 3-token 循环: [a,b,c,a,b,c]
            if len(_recent_tids) >= 9 and new_tokens >= min_new:
                if _recent_tids[-3:] == _recent_tids[-6:-3] == _recent_tids[-9:-6]:
                    break
            # 非中文洪水: 末尾 30 字符中 >60% 是 ASCII 字母 → 英文乱码
            if new_tokens >= min_new and new_tokens >= 15 and len(_reply_so_far) >= 30:
                tail = _reply_so_far[-30:]
                ascii_alpha = sum(1 for c in tail if c.isascii() and c.isalpha())
                if ascii_alpha > 18:
                    break

            if next_token == EOS_ID and new_tokens >= min_new:
                break

            # 流式输出
            new_text = self.tokenizer.decode([next_token], skip_special=True)
            if new_text:
                _reply_so_far += new_text
                # 自问自答检测: 先检查再输出
                if any(m in _reply_so_far for m in ["\n用户", "\n用", "\n户",
                                                        "用户:", "用户：", "用户\n", "用户。",
                                                        "\nTGAI?", "\nTGAI", "\nTG?", "\nTGA?",
                                                        "TGAI?", "TGA?",
                                                        "\nTGA", "\nTGA\n", "TGA\n",
                                                        "\nTG", "\nTGA。",
                                                        "\n用"]):
                    break
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
    top_k: int = 50,
    top_p: float = 0.95,
    frequency_penalty: float = 0.25,
    repetition_penalty: float = 1.05,
    min_new_tokens: int = 5,
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
    current_max = max_new_tokens
    current_topk = top_k
    current_topp = top_p
    current_freq_pen = frequency_penalty
    current_rep_pen = repetition_penalty
    current_min_tokens = min_new_tokens

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
                print("命令: /exit /temp N /maxtokens N /topk N /topp N /freq N /rep N /stream /clear /help")
                print("  /temp N      - 设置温度 (0.1~2.0)")
                print("  /maxtokens N - 设置最大 token 数 (16~1024)")
                print("  /topk N      - 设置 Top-K (1~200)")
                print("  /topp N      - 设置 Top-P (0.0~1.0)")
                print("  /freq N      - 设置频率惩罚 (0.0~1.0)")
                print("  /rep N       - 设置重复惩罚 (0.1~3.0)")
                print("  /mintokens N - 设置最小生成token数 (0~50)")
                print("  /stream      - 切换流式输出")
                print("  /clear       - 清除对话历史")
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
            elif cmd.startswith('/maxtokens'):
                try:
                    current_max = max(16, min(1024, int(user_input.split()[-1])))
                    print(f"[Max Tokens: {current_max}]")
                except (ValueError, IndexError):
                    print(f"[当前 Max Tokens: {current_max}]")
            elif cmd.startswith('/topk'):
                try:
                    current_topk = max(1, min(200, int(user_input.split()[-1])))
                    print(f"[Top-K: {current_topk}]")
                except (ValueError, IndexError):
                    print(f"[当前 Top-K: {current_topk}]")
            elif cmd.startswith('/topp'):
                try:
                    current_topp = max(0.0, min(1.0, float(user_input.split()[-1])))
                    print(f"[Top-P: {current_topp}]")
                except (ValueError, IndexError):
                    print(f"[当前 Top-P: {current_topp}]")
            elif cmd.startswith('/freq'):
                try:
                    current_freq_pen = max(0.0, min(1.0, float(user_input.split()[-1])))
                    print(f"[频率惩罚: {current_freq_pen}]")
                except (ValueError, IndexError):
                    print(f"[当前频率惩罚: {current_freq_pen}]")
            elif cmd.startswith('/rep'):
                try:
                    current_rep_pen = max(0.1, min(3.0, float(user_input.split()[-1])))
                    print(f"[重复惩罚: {current_rep_pen}]")
                except (ValueError, IndexError):
                    print(f"[当前重复惩罚: {current_rep_pen}]")
            elif cmd.startswith('/mintokens'):
                try:
                    current_min_tokens = max(0, min(50, int(user_input.split()[-1])))
                    print(f"[最小Token数: {current_min_tokens}]")
                except (ValueError, IndexError):
                    print(f"[当前最小Token数: {current_min_tokens}]")
            continue

        try:
            t0 = time.time()
            if stream:
                print("TGAI: ", end="", flush=True)
                response = ""
                for chunk in generator.generate(
                    user_input, current_max, current_temp,
                    top_k=current_topk, top_p=current_topp,
                    frequency_penalty=current_freq_pen,
                    repetition_penalty=current_rep_pen,
                    min_new_tokens=current_min_tokens,
                    stream=True,
                ):
                    print(chunk, end="", flush=True)
                    response += chunk
                print(f"\n  [{time.time() - t0:.1f}s]")
            else:
                response = generator.generate(
                    user_input, current_max, current_temp,
                    top_k=current_topk, top_p=current_topp,
                    frequency_penalty=current_freq_pen,
                    repetition_penalty=current_rep_pen,
                    min_new_tokens=current_min_tokens,
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
    top_k: int = 50,
    top_p: float = 0.95,
    frequency_penalty: float = 0.25,
    repetition_penalty: float = 1.05,
    stream: bool = False,
):
    """单次文本生成"""
    print(f"Prompt: {prompt}")
    print("-" * 40)

    generator = TextGenerator(model, tokenizer)

    if stream:
        print("生成: ", end="", flush=True)
        for chunk in generator.generate(
            prompt, max_new_tokens, temperature,
            top_k=top_k, top_p=top_p,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        response = generator.generate(
            prompt, max_new_tokens, temperature,
            top_k=top_k, top_p=top_p,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
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
    parser.add_argument('--top_k', type=int, default=50, help='Top-K 采样')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-P 核采样')
    parser.add_argument('--freq_penalty', type=float, default=0.25, help='频率惩罚')
    parser.add_argument('--rep_penalty', type=float, default=1.05, help='重复惩罚')
    parser.add_argument('--min_tokens', type=int, default=5, help='最小生成长度(EOS禁用token数)')
    parser.add_argument('--stream', action='store_true', help='流式输出')
    parser.add_argument('--extend', type=int, default=0,
                        help='启用 Self-Extend 长上下文扩展，参数为扩展目标长度 (如 4096, 8192, 65536)')
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
        self_extend = args.extend > 0
        extend_len = args.extend if self_extend else 4096
        model, tokenizer = load_model(ckpt_path, args.tokenizer, device,
                                      self_extend=self_extend,
                                      extend_max_seq_len=extend_len)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请先运行 train.py 训练模型。")
        sys.exit(1)

    if args.mode == 'interactive':
        interactive_chat(model, tokenizer, args.temperature, args.max_tokens,
                         args.top_k, args.top_p, args.freq_penalty, args.rep_penalty,
                         args.min_tokens, args.stream)
    else:
        single_generate(model, tokenizer, args.prompt, args.temperature, args.max_tokens,
                        args.top_k, args.top_p, args.freq_penalty, args.rep_penalty,
                        args.stream)