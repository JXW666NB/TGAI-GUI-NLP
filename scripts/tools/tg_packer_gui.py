"""
TGAI 模型打包 GUI
==================
双击运行，选择 checkpoint 一键导出并打包为 .TG 文件。

用法:
    python tg_packer_gui.py
"""
import os
import sys
import json
import zipfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# 添加 tgai_nlp 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExportThread(threading.Thread):
    """后台导出线程，避免 GUI 卡死"""
    def __init__(self, checkpoint, out_dir, tokenizer, callback):
        super().__init__(daemon=True)
        self.checkpoint = checkpoint
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.callback = callback
        self.prefill_path = None
        self.decode_path = None
        self.tokenizer_out = None
        self.error = None

    def run(self):
        try:
            import torch
            from model import TGAILanguageModel, TGAIConfig, MoELayer, RotaryPositionEmbedding, create_model
            from tokenizer import ChineseTokenizer

            self.callback('step', '加载 checkpoint...')

            ckpt = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            cfg = ckpt['model_config']

            model = create_model(**cfg)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            # 导出分词器
            if self.tokenizer and os.path.exists(self.tokenizer):
                self.callback('step', '导出分词器...')
                tok = ChineseTokenizer.load(self.tokenizer)
                self._export_tokenizer(tok, os.path.join(self.out_dir, 'tokenizer.json'))

            # 替换 MoE 层
            self.callback('step', '转换模型结构...')
            self._make_traceable(model)

            # Prefill
            self.callback('step', '导出 prefill 模型...')
            prefill = self._TGAIPrefill(model)
            try:
                sp = torch.jit.script(prefill)
            except Exception:
                B, T = 1, cfg['max_seq_len']
                sp = torch.jit.trace(prefill, example_inputs=(torch.randint(0, cfg['vocab_size'], (B, T), dtype=torch.long),))

            self.prefill_path = os.path.join(self.out_dir, 'tgai_prefill.ptl')
            sp._save_for_lite_interpreter(self.prefill_path)

            # Decode
            self.callback('step', '导出 decode 模型...')
            decode = self._TGAIDecode(model)
            try:
                sd = torch.jit.script(decode)
            except Exception:
                B = 1
                n_layers = cfg['n_layers']
                n_heads = cfg['n_heads']
                d_k = cfg['d_model'] // cfg['n_heads']
                sd = torch.jit.trace(decode, example_inputs=(
                    torch.randint(0, cfg['vocab_size'], (B, 1), dtype=torch.long),
                    cfg['max_seq_len'] - 1,
                    torch.zeros(2 * n_layers, B, n_heads, cfg['max_seq_len'], d_k)
                ))

            self.decode_path = os.path.join(self.out_dir, 'tgai_decode.ptl')
            sd._save_for_lite_interpreter(self.decode_path)

            self.tokenizer_out = os.path.join(self.out_dir, 'tokenizer.json')
            self.callback('done', None)
        except Exception as e:
            self.error = str(e)
            self.callback('error', self.error)

    class _TGAIPrefill(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.n_layers = model.config.n_layers
        def forward(self, input_ids):
            logits, caches = self.model(input_ids, kv_caches=None)
            kvs = [c['k'] for c in caches] + [c['v'] for c in caches]
            kv_cache = torch.stack(kvs, dim=0)
            return logits, kv_cache

    class _TGAIDecode(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.n_layers = model.config.n_layers
        def forward(self, input_ids, cache_pos, kv_cache):
            kv_caches = []
            for i in range(self.n_layers):
                kv_caches.append({'k': kv_cache[2 * i], 'v': kv_cache[2 * i + 1]})
            logits, caches = self.model(input_ids, kv_caches=kv_caches, cache_pos=cache_pos)
            for i in range(self.n_layers):
                kv_cache[2 * i] = caches[i]['k']
                kv_cache[2 * i + 1] = caches[i]['v']
            return logits

    def _make_traceable(self, model):
        import torch.nn.functional as F
        for name, module in list(model.named_modules()):
            if isinstance(module, MoELayer):
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], self._TraceableMoE(module))

        def rope_fwd(self, x, position_ids=None):
            seq_len = x.shape[-2]
            if position_ids is not None:
                pos = position_ids.view(-1)
                cos = self.cos.squeeze(0).squeeze(0).index_select(0, pos).unsqueeze(0).unsqueeze(0)
                sin = self.sin.squeeze(0).squeeze(0).index_select(0, pos).unsqueeze(0).unsqueeze(0)
            else:
                cos, sin = self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]
            return (x * cos) + (self._rotate_half(x) * sin)
        RotaryPositionEmbedding.forward = rope_fwd

    class _TraceableMoE(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.n_experts = orig.n_experts
            self.n_activated = orig.n_activated
            self.router = orig.router
            self.experts = orig.experts
        def forward(self, x):
            import torch.nn.functional as F
            B, T, D = x.shape
            x_flat = x.view(-1, D)
            rl = self.router(x_flat)
            vals, ids = torch.topk(rl, self.n_activated, dim=-1)
            probs = F.softmax(vals, dim=-1)
            w = torch.zeros_like(rl)
            w.scatter_(-1, ids, probs)
            out = torch.stack([e(x_flat) for e in self.experts], dim=1)
            return (out * w.unsqueeze(-1)).sum(dim=1).view(B, T, D)

    def _export_tokenizer(self, tokenizer, out_path):
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


def pack_tg(prefill, decode, tokenizer, output, meta=None):
    """打包为 .TG 文件"""
    manifest = {
        'format': 'tgai-mobile-1',
        **(meta or {}),
    }
    if not output.lower().endswith('.tg'):
        output += '.tg'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.write(prefill, 'prefill.ptl')
        zf.write(decode, 'decode.ptl')
        zf.write(tokenizer, 'tokenizer.json')
    return output


class TgPackerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('TGAI 模型打包器')
        self.root.geometry('520x440')
        self.root.resizable(False, False)
        self.root.configure(bg='#1e1e2e')

        self.checkpoint_path = tk.StringVar()
        self.output_path = tk.StringVar(value=os.path.join(os.path.expanduser('~'), 'Desktop'))
        self.model_name = tk.StringVar(value='TGAI-Model