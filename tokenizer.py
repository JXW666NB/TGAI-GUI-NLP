"""
TGAI V4 分词器
=============
基于 HuggingFace tokenizers 的 BPE 分词器。
词表大小: 16384，优化中文编码效率。

设计：
- BPE 子词分词，char-level pre-tokenizer
- 数字/字母/标点保底入词表
- NFKC 归一化
- 缓存高频编码
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple

# ─── CJK 空格清理 ─────────────────────────────────
_CJK_RE = re.compile(
    r'(?<=[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]) '
    r'| (?=[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef])'
)

def _clean_cjk_spacing(text: str) -> str:
    """去除中日韩文字间的多余空格（BPE 解码副作用）"""
    return _CJK_RE.sub('', text)

# ─── 特殊 token ────────────────────────────────────
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
BOS_TOKEN  = "<BOS>"
EOS_TOKEN  = "<EOS>"
IMG_TOKEN  = "<IMG>"    # 多模态预留
AUD_TOKEN  = "<AUD>"    # 多模态预留
VID_TOKEN  = "<VID>"    # 多模态预留

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, IMG_TOKEN, AUD_TOKEN, VID_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
IMG_ID = 4
AUD_ID = 5
VID_ID = 6

HAS_HF_TOKENIZERS = False
try:
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Split
    from tokenizers.normalizers import NFKC
    HAS_HF_TOKENIZERS = True
except ImportError:
    pass


class ChineseTokenizer:
    """V4 BPE分词器，目标词表 16384"""

    def __init__(self, vocab_size: int = 16384):
        self.vocab_size = vocab_size
        self._hf: Optional[Tokenizer] = None
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._encode_cache: Dict[Tuple[str, bool], List[int]] = {}
        self._cache_max = 20000

    # ─── 训练 ───────────────────────────────────────
    def train(self, texts: List[str], min_freq: int = 2, allow_chinese_words: bool = True):
        if not HAS_HF_TOKENIZERS:
            raise ImportError("pip install tokenizers")

        import tempfile

        # 写入临时语料
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for t in texts:
                t = t.strip()
                if t:
                    f.write(t + '\n')
            corpus_path = f.name

        try:
            # 初始化 BPE tokenizer
            tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
            tok.normalizer = NFKC()

            # 中文预切分：允许2-4字符的词组
            if allow_chinese_words:
                pattern = (
                    r'[\u4e00-\u9fff]{1,4}'  # 允许1-4个汉字组成词
                    r'|[\u3400-\u4dbf]'
                    r'|[a-zA-Z]+'
                    r'|\d+'
                    r'|[\s]+'
                    r'|[^\s\w]'
                )
            else:
                pattern = (
                    r'[\u4e00-\u9fff]'
                    r'|[\u3400-\u4dbf]'
                    r'|[a-zA-Z]+'
                    r'|\d+'
                    r'|[\s]+'
                    r'|[^\s\w]'
                )
            tok.pre_tokenizer = Split(pattern=Regex(pattern), behavior='isolated', invert=False)

            # 保底字母表
            initial = list('0123456789')
            initial += list('abcdefghijklmnopqrstuvwxyz')
            initial += list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            initial += list('，。！？、；：""''（）【】《》·…—\n')

            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_freq,
                special_tokens=SPECIAL_TOKENS,
                initial_alphabet=initial,
                show_progress=False,
            )

            tok.train([corpus_path], trainer)
            self._hf = tok

        finally:
            try:
                os.unlink(corpus_path)
            except OSError:
                pass

        self._build_vocab()
        print(f"  [分词器] 训练完成: {len(self.token_to_id)} tokens")

    def _build_vocab(self):
        """构建双向映射，保持特殊token在最前面"""
        if self._hf is None:
            return
        hf_vocab = self._hf.get_vocab()

        self.token_to_id = {}
        self.id_to_token = {}

        # 特殊token固定ID
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

        # 其他token按HF原始ID顺延
        others = [(t, hf_vocab[t]) for t in hf_vocab if t not in SPECIAL_TOKENS]
        others.sort(key=lambda x: x[1])

        for tok, _ in others:
            if tok in self.token_to_id:
                continue
            new_id = len(self.token_to_id)
            self.token_to_id[tok] = new_id
            self.id_to_token[new_id] = tok

    # ─── 编码 ───────────────────────────────────────
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if not text:
            return [BOS_ID, EOS_ID] if add_special else []

        cache_key = (text, add_special)
        if cache_key in self._encode_cache:
            return self._encode_cache[cache_key]

        if self._hf is not None:
            enc = self._hf.encode(text, add_special_tokens=False)
            raw_ids = enc.ids
        else:
            raw_ids = [self.token_to_id.get(ch, UNK_ID) for ch in text]

        vocab_max = len(self.token_to_id)
        ids = [tid if 0 <= tid < vocab_max else UNK_ID for tid in raw_ids]

        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]

        if len(self._encode_cache) < self._cache_max:
            self._encode_cache[cache_key] = ids
        return ids

    # ─── 解码 ───────────────────────────────────────
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if not ids:
            return ""
        vocab_max = len(self.id_to_token)
        safe = [tid for tid in ids if 0 <= tid < vocab_max]
        if skip_special:
            safe = [tid for tid in safe if tid >= len(SPECIAL_TOKENS)]
        if not safe:
            return ""
        if self._hf is not None:
            try:
                text = self._hf.decode(safe, skip_special_tokens=False)
                # BPE 解码后去除 CJK 字符间多余空格
                text = _clean_cjk_spacing(text)
                return text
            except Exception:
                pass
        text = ''.join(self.id_to_token.get(tid, '') for tid in safe)
        return _clean_cjk_spacing(text)

    # ─── 保存/加载 ──────────────────────────────────
    def save(self, path: str):
        if self._hf is not None:
            hf_path = path.replace('.json', '_hf.json')
            self._hf.save(hf_path)
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'has_hf': self._hf is not None,
            'hf_path': path.replace('.json', '_hf.json') if self._hf else None,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ChineseTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        if data.get('has_hf') and data.get('hf_path') and os.path.exists(data['hf_path']):
            try:
                tokenizer._hf = Tokenizer.from_file(data['hf_path'])
            except Exception:
                pass
        return tokenizer

    @property
    def vocab_size_actual(self) -> int:
        return len(self.token_to_id)