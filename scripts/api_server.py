"""
TGAI NB API 服务器
==================
提供 HTTP API 供外部调用 TGAI 模型。

启动:
    python scripts/api_server.py --checkpoint checkpoints/tgai_sft.pt
    python scripts/api_server.py --checkpoint checkpoints/tgai_sft.pt --port 8080 --host 0.0.0.0

配置:
    默认参数读取 qq_bot_config.json，调用方可逐个覆盖。

API:
    POST /api/generate   非流式，返回完整回复
    POST /api/chat       流式 SSE，逐 token 推送
    GET  /api/health     健康检查
    GET  /api/config     查看当前默认参数
    GET  /               网页聊天界面

请求体:
    {
        "message": "你好",          // 必填
        "temperature": 0.8,        // 可选
        "max_tokens": 256,         // 可选
        "top_k": 50,               // 可选
        "top_p": 0.95,             // 可选
        "freq_penalty": 0.25,      // 可选
        "rep_penalty": 1.05,       // 可选
        "min_tokens": 5,           // 可选
        "history": [               // 可选 - 多轮对话历史
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是TGAI"}
        ]
    }
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
import random
from pathlib import Path
from collections import defaultdict
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from inference import load_model, TextGenerator

# ─── TGAI GO 引擎支持 ─────────────────────────────
# 在 Linux 服务器上加载 libtgai_engine.so
# 支持多模型: _tgai_go_instances = {model_id: {lib, model, cache, ...}}
_TGAI_GO_AVAILABLE = False
_tgai_go_lib = None  # 共享的 .so 句柄
_tgai_go_instances = {}  # {model_id: dict}
_gpu_lock = threading.Lock()  # GPU 串行保护: KV cache/工作缓冲区共享, 必须串行

def _init_tgai_go(tg_path: str, model_id: str = "tgai_go", extend_ctx: int = 0):
    """加载 TGAI GO .TG 模型"""
    global _tgai_go_lib, _TGAI_GO_AVAILABLE
    import ctypes
    import os as _os

    # 首次加载 .so
    if _tgai_go_lib is None:
        candidates = [
            _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'TGAI GO', 'cpu', 'build', 'libtgai_engine.so'),
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'TGAI GO', 'cpu', 'build', 'libtgai_engine.so'),
        ]
        lib = None
        for p in candidates:
            if _os.path.exists(p):
                try:
                    lib = ctypes.CDLL(p)
                    break
                except Exception as e:
                    log.warning(f"TGAI GO .so 加载失败 {p}: {e}")
        if lib is None:
            log.info("TGAI GO .so 未找到")
            return

        lib.tg_load.restype = ctypes.c_void_p
        lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.tg_free.argtypes = [ctypes.c_void_p]
        lib.tg_config.restype = ctypes.c_void_p
        lib.tg_config.argtypes = [ctypes.c_void_p]
        lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                    ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.tg_kvcache_new.restype = ctypes.c_void_p
        lib.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.tg_kvcache_free.argtypes = [ctypes.c_void_p]
        lib.tg_kvcache_reset.argtypes = [ctypes.c_void_p]
        lib.tg_encode.restype = ctypes.c_int
        lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
        lib.tg_decode.restype = ctypes.c_int
        lib.tg_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                   ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.tg_preload_all.restype = ctypes.c_int
        lib.tg_preload_all.argtypes = [ctypes.c_void_p]
        lib.tg_set_nthreads.argtypes = [ctypes.c_int]
        if hasattr(lib, 'tg_set_extend'):
            lib.tg_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]
        if hasattr(lib, 'tg_cuda_set_extend'):
            lib.tg_cuda_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.tg_last_error.restype = ctypes.c_char_p
        _tgai_go_lib = lib

    lib = _tgai_go_lib

    err = ctypes.c_int(0)
    m = lib.tg_load(tg_path.encode('utf-8'), 1, ctypes.byref(err))
    if not m:
        msg = lib.tg_last_error().decode('utf-8', errors='replace') if lib.tg_last_error() else 'unknown'
        log.warning(f"TGAI GO 加载失败 [{model_id}]: {msg}")
        return

    # 预加载 + 线程
    try:
        if hasattr(lib, 'tg_set_nthreads'):
            cores = min(_os.cpu_count() or 4, 8)
            lib.tg_set_nthreads(cores)
    except Exception:
        pass

    lib.tg_preload_all(m)

    # NTK-aware 上下文扩展
    if extend_ctx > 0 and hasattr(lib, 'tg_set_extend'):
        lib.tg_set_extend(m, extend_ctx)
        log.info(f"TGAI GO [{model_id}] NTK上下文扩展: {extend_ctx}")

    # ── CUDA 加速 ──
    use_cuda = False
    cuda_handle = None
    import os as _os2
    if _os2.environ.get("TGAI_NO_CUDA"):
        log.info(f"TGAI GO [{model_id}] CUDA 已通过环境变量禁用")
    else:
        try:
            if hasattr(lib, 'tg_cuda_available') and lib.tg_cuda_available():
                lib.tg_cuda_load.restype = ctypes.c_void_p
                lib.tg_cuda_load.argtypes = [ctypes.c_void_p]
                lib.tg_cuda_forward.restype = ctypes.POINTER(ctypes.c_float)
                lib.tg_cuda_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
                lib.tg_cuda_free.argtypes = [ctypes.c_void_p]
                cuda_handle = lib.tg_cuda_load(m)
                log.info(f"TGAI GO [{model_id}] CUDA handle={cuda_handle} type={type(cuda_handle)}")
                # restype=c_void_p 时, ctypes 返回 None(NULL) 或 int(地址), 直接判真即可
                if cuda_handle:
                    use_cuda = True
                    log.info(f"TGAI GO [{model_id}] CUDA 加速已启用, handle=0x{cuda_handle:x}")
                    # CUDA 端也设置 NTK 上下文扩展 (与 CPU 对齐)
                    if extend_ctx > 0 and hasattr(lib, 'tg_cuda_set_extend'):
                        lib.tg_cuda_set_extend(cuda_handle, extend_ctx)
                        log.info(f"TGAI GO [{model_id}] CUDA NTK上下文扩展: {extend_ctx}")
                else:
                    log.warning(f"TGAI GO [{model_id}] CUDA 加载失败，用 CPU 回退")
        except Exception as e:
            log.info(f"TGAI GO [{model_id}] CUDA 不可用: {e}")

    # 获取配置 (必须与 tgai_format.h 的 TGHeader 布局一致)
    class TGHeader(ctypes.Structure):
        _fields_ = [
            ('magic', ctypes.c_uint32),         # 0
            ('version', ctypes.c_uint32),       # 4
            ('weight_dtype', ctypes.c_uint32),  # 8
            ('n_layers', ctypes.c_uint32),      # 12
            ('d_model', ctypes.c_uint32),       # 16
            ('n_heads', ctypes.c_uint32),       # 20
            ('d_ff', ctypes.c_uint32),          # 24
            ('vocab_size', ctypes.c_uint32),    # 28
            ('max_seq_len', ctypes.c_uint32),   # 32
            ('n_experts', ctypes.c_uint32),     # 36
            ('n_activated', ctypes.c_uint32),   # 40
            ('rope_theta', ctypes.c_uint32),    # 44
            ('pad_token_id', ctypes.c_uint32),  # 48
            ('bos_token_id', ctypes.c_uint32),  # 52
            ('eos_token_id', ctypes.c_uint32),  # 56
            ('unk_token_id', ctypes.c_uint32),  # 60
        ]
    cfg = ctypes.cast(lib.tg_config(m), ctypes.POINTER(TGHeader)).contents

    # 创建 KV cache (每个请求独立使用, 这里是预分配基础实例)
    max_seq = int(cfg.max_seq_len)
    cache = lib.tg_kvcache_new(m, max_seq)

    # 预热
    warmup_ids = (ctypes.c_int32 * 3)(2, 13036, 3)
    lib.tg_forward(m, warmup_ids, 3, None, 0)

    _tgai_go_instances[model_id] = {
        "model": m, "cache": cache, "vocab": int(cfg.vocab_size),
        "max_seq": max_seq, "eos": int(cfg.eos_token_id) if cfg.eos_token_id else 3,
        "d_model": int(cfg.d_model), "n_layers": int(cfg.n_layers),
        "use_cuda": use_cuda, "cuda_handle": cuda_handle,
        "extend_ctx": extend_ctx  # NTK 扩展长度 (0=不扩展)
    }
    _TGAI_GO_AVAILABLE = True
    log.info(f"TGAI GO [{model_id}] 就绪: d_model={cfg.d_model}, layers={cfg.n_layers}, "
             f"vocab={cfg.vocab_size}, max_seq={cfg.max_seq_len}")


def _tgai_go_generate(prompt: str, model_id: str = None, max_tokens=256, temperature=0.8,
                       top_k=90, top_p=0.9, freq_pen=0.0, rep_pen=1.05, min_tok=0,
                       force_prefix: str = ""):
    """GPU 串行保护 wrapper — KV cache/工作缓冲区共享, 必须串行"""
    with _gpu_lock:
        yield from _tgai_go_generate_impl(prompt, model_id, max_tokens, temperature,
                                           top_k, top_p, freq_pen, rep_pen, min_tok,
                                           force_prefix)


def _tgai_go_generate_impl(prompt: str, model_id: str = None, max_tokens=256, temperature=0.8,
                       top_k=90, top_p=0.9, freq_pen=0.0, rep_pen=1.05, min_tok=0,
                       force_prefix: str = ""):
    """TGAI GO 流式生成, 支持多模型

    force_prefix: 兜底起手式 — 强制让模型回复以这段文字开头 (空=不强制)。
    实现: 把 force_prefix encode 后追加到 prompt 的 token 序列末尾,
    一次性 forward 让 KV cache 包含 prefix, 然后从最后一个 token 开始采样。
    采样前先把 force_prefix 字符串 yield 给前端, 避免重复输出。
    """
    import numpy as np
    import ctypes

    # 选择模型
    if not model_id or model_id not in _tgai_go_instances:
        if _tgai_go_instances:
            model_id = next(iter(_tgai_go_instances))
        else:
            return
    inst = _tgai_go_instances[model_id]
    lib = _tgai_go_lib
    m = inst["model"]
    vocab = inst["vocab"]
    eos = inst["eos"]

    # 选择 forward 函数 (CUDA 优先)
    cuda_handle = inst.get("cuda_handle")
    if cuda_handle and inst.get("use_cuda"):
        do_forward = lambda m, ids, n, cache, pos: lib.tg_cuda_forward(cuda_handle, ids, n, cache, pos)
    else:
        do_forward = lib.tg_forward

    # 每个请求独立 KV cache (开 NTK 扩展时需扩大到 extend_ctx, 否则 forward 会越界)
    _kv_seq = max(inst["max_seq"], inst.get("extend_ctx", 0))
    cache = lib.tg_kvcache_new(m, _kv_seq)
    try:
        lib.tg_kvcache_reset(cache)

        ids_buf = (ctypes.c_int32 * 2048)()
        n = lib.tg_encode(m, prompt.encode('utf-8'), ids_buf, 2048, 1)
        if n <= 0:
            return
        ids = [int(ids_buf[i]) for i in range(n)]

        # force_prefix (兜底起手式): encode 后追加为已知前缀, 让模型从这里接着生成
        # 不参与采样, 但要纳入 appeared/rep_pen 统计 (避免模型立刻重复 prefix 里的字)
        # 注意: add_special=0 — 不加 BOS (prompt 已加过 BOS, 重复 BOS 会让模型困惑)
        _force_prefix_str = ""
        if force_prefix:
            prefix_buf = (ctypes.c_int32 * 1024)()
            n_prefix = lib.tg_encode(m, force_prefix.encode('utf-8'), prefix_buf, 1024, 0)
            if n_prefix > 0:
                ids.extend(int(prefix_buf[i]) for i in range(n_prefix))
                _force_prefix_str = force_prefix
            n = len(ids)

        ids_arr = (ctypes.c_int32 * n)(*ids)
        logits = do_forward(m, ids_arr, n, cache, 0)
        if not logits:
            return

        FloatArr = ctypes.POINTER(ctypes.c_float)
        cur_pos = n
        generated = list(ids)
        appeared = {}
        for tid in ids:
            appeared[tid] = appeared.get(tid, 0) + 1
        _last_id = -1
        _consecutive = 0

        # ── 停止序列检测 ──────────────────────────────
        # 模型训练格式: 用户:...↵TGAI?[回答]↵用户:... (下一回合)
        # 模型 "说完了" 的信号是生成下一个回合标记, 而非 EOS=3
        # 检测到标记即停止, 不输出标记本身 (用 hold buffer 处理跨 token 的标记)
        _STOP_MARKERS = ['\n用户:', '\n用户：', '\nTGAI?', '\nTGA?', '\nTG?']
        _MAX_HOLD = 6   # 最长标记长度, 尾部这么多字符先 hold 不 yield
        hold = ""

        # 先把兜底起手式 yield 给前端 (它不参与停止标记检测, 也不会重复输出)
        if _force_prefix_str:
            yield _force_prefix_str

        for _step in range(max_tokens):
            logit_arr = np.ctypeslib.as_array(ctypes.cast(logits, FloatArr), shape=(vocab,)).copy()
            if len(generated) < len(ids) + min_tok:
                logit_arr[eos] = -1e30
            inv_temp = 1.0 / max(temperature, 0.01)
            logit_arr = logit_arr * inv_temp
            if freq_pen > 0 and appeared:
                for tid, cnt in appeared.items():
                    if 0 <= tid < vocab and cnt > 0:
                        logit_arr[tid] -= freq_pen * min(cnt, 2.0)
            if rep_pen > 1.0:
                recent = set(generated[-64:])
                for tid in recent:
                    if tid != _last_id:
                        logit_arr[tid] /= rep_pen
            if _consecutive >= 3:
                logit_arr[_last_id] -= _consecutive * 5.0
            if _consecutive > 15:
                break
            if 0 < top_k < vocab:
                indices = np.argpartition(logit_arr, -top_k)[-top_k:]
                threshold = np.min(logit_arr[indices])
                logit_arr[logit_arr < threshold] = -np.inf
            if top_p < 1.0:
                sorted_idx = np.argsort(logit_arr)[::-1]
                sp = np.exp(logit_arr[sorted_idx] - np.max(logit_arr))
                sp /= np.sum(sp)
                cum = np.cumsum(sp)
                cutoff = int(np.searchsorted(cum, top_p)) + 1
                mask = np.ones(vocab, dtype=bool)
                mask[sorted_idx[cutoff:]] = False
                logit_arr[~mask] = -np.inf
            probs = np.exp(logit_arr - np.max(logit_arr))
            probs = probs / max(np.sum(probs), 1e-10)
            if np.any(np.isnan(probs)):
                best_id = int(np.argmax(logit_arr))
            else:
                best_id = int(np.random.choice(vocab, p=probs))
            if best_id == eos:
                break
            decode_buf = ctypes.create_string_buffer(256)
            lib.tg_decode(m, (ctypes.c_int32 * 1)(best_id), 1, decode_buf, 256, 1)
            tok = decode_buf.value.decode('utf-8', errors='replace') if decode_buf.value else ''
            if tok:
                hold += tok
                # 命中停止标记: yield 标记前的内容, 然后停止生成
                _hit = None
                for _m in _STOP_MARKERS:
                    _i = hold.find(_m)
                    if _i >= 0:
                        _hit = (_i, _m)
                        break
                if _hit is not None:
                    _i, _m = _hit
                    if _i > 0:
                        yield hold[:_i]
                    return  # 说完了, 丢弃标记及之后内容
                # 尾部 _MAX_HOLD 字符可能是标记前缀, 先 hold; 其余安全 yield
                if len(hold) > _MAX_HOLD:
                    yield hold[:-_MAX_HOLD]
                    hold = hold[-_MAX_HOLD:]
            if best_id == _last_id:
                _consecutive += 1
            else:
                _consecutive = 0
            _last_id = best_id
            appeared[best_id] = appeared.get(best_id, 0) + 1
            generated.append(best_id)
            next_arr = (ctypes.c_int32 * 1)(best_id)
            logits = do_forward(m, next_arr, 1, cache, cur_pos)
            cur_pos += 1
            if not logits or cur_pos >= inst["max_seq"]:
                break
        # 循环结束 (EOS / max_tokens / 上下文满), yield 剩余 hold, 去掉尾部退化标记
        if hold:
            _final = _strip_trailing_degenerate(hold)
            if _final:
                yield _final
    finally:
        lib.tg_kvcache_free(cache)

# ─── 日志 ───
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
log = logging.getLogger("tgai_api")

app = Flask(__name__)

# ─── 防护：频率限制 ───
# 每 IP 10秒内最多 12 次请求, 超限封 5 分钟
_ip_timestamps = defaultdict(list)
_ip_banned_until = {}
RATE_WINDOW = 10
RATE_MAX = 12
BAN_DURATION = 300

_TROLL_REPLIES = [
    "哎呀，攻击者你好呀～手速不错，但建议去练钢琴而不是刷接口呢 �",
    "检测到一只野生攻击者！虽然你很努力，但这道墙你翻不过去的～歇歇吧。",
    "哈哈哈有人想搞事情～不过很遗憾，你的请求被温柔地拒绝了。五分钟后再来试试？",
    "啧啧，被逮到了吧。攻击服务器可不是什么好习惯哦，坐下来冷静五分钟。",
    "哟，这位攻击者朋友，你的 IP 已被记在小本本上了。先冷静冷静，乖。",
    "检测到攻击行为！说实话我有点佩服你的执着，但规则就是规则，五分钟后再见。",
    "哎呀呀，刷得这么快不怕手抽筋吗？系统建议：放下键盘，立地成佛。",
    "你好啊攻击者～你的每一次请求我都看见了哦。现在请休息五分钟，不许耍赖。",
]

def _check_rate_limit(ip: str) -> bool:
    """检查频率限制, 返回 True 表示被拦截"""
    now = time.time()
    if ip in _ip_banned_until and now < _ip_banned_until[ip]:
        return True
    _ip_timestamps[ip] = [t for t in _ip_timestamps[ip] if now - t < RATE_WINDOW]
    _ip_timestamps[ip].append(now)
    if len(_ip_timestamps[ip]) > RATE_MAX:
        _ip_banned_until[ip] = now + BAN_DURATION
        _ip_timestamps[ip] = []
        log.warning(f"[防护] IP {ip} 被封禁 {BAN_DURATION} 秒")
        return True
    return False

def _troll_stream_response():
    """生成羞辱文案的流式 SSE 输出"""
    msg = random.choice(_TROLL_REPLIES)
    def gen():
        yield f"data: {json.dumps({'token': '🤗 歇一歇～'}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        for ch in msg:
            yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"
            time.sleep(random.uniform(0.03, 0.08))
        yield "data: [DONE]\n\n"
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

def _troll_generate_response():
    """生成羞辱文案的非流式输出"""
    msg = random.choice(_TROLL_REPLIES)
    return jsonify({"response": "🤗 歇一歇～\n\n" + msg})

@app.before_request
def _firewall():
    if request.path not in ('/api/chat', '/api/generate'):
        return
    ip = _real_ip()
    if _check_rate_limit(ip):
        if request.path == '/api/chat':
            return _troll_stream_response()
        else:
            return _troll_generate_response(), 403

def _real_ip():
    """获取真实客户端 IP（支持反向代理）"""
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        # 取最左侧第一个非内网 IP
        for ip in xff.split(','):
            ip = ip.strip()
            if ip and not ip.startswith(('10.', '172.', '192.168.', '127.')):
                return ip
        # 都是内网，取第一个
        return xff.split(',')[0].strip()
    return request.headers.get('X-Real-IP', request.remote_addr)

# 全局模型 — 多模型预加载，并发推理互不干扰
_generators: dict = {}      # {model_id: TextGenerator}
_default_model_id: str = "" # 默认模型 ID
_tokenizer_path: str = ""
_default_config: dict = {}
_token_bias: dict = {}
_prefix_bias: dict = {}
_dynamic_rules: list = []  # from word_bias_config.json → dynamic_bias.rules
_replace_rules: dict = {}  # { "小明": "TGAI", "李华": "TGAI" } — 后处理硬替换
_block_rules: list = []   # block_list.rules — 输入屏蔽
_output_block_rules: list = []  # output_block.rules — 输出截断
_enabled: bool = True     # 总开关：false=放飞模式

# 模型注册表 (PyTorch + TGAI GO 双引擎)
_MODEL_REGISTRY = [
    {"id": "v1",           "name": "TGAI NB V1 0.86B",      "path": "/root/autodl-tmp/TGAI_checkpoints_v3/tgai_sft.pt", "engine": "pytorch"},
    {"id": "v2_3b",        "name": "TGAI NB V2 3B 42K",     "path": "/root/autodl-tmp/TGAI_checkpoints_v3/tgai_lora_1w.pt", "engine": "pytorch"},
    {"id": "v1_tgai_go",   "name": "TGAI NB V1 0.86B (GO)", "path": "/root/autodl-tmp/TGAI_models/tgai_v1_fp16.TG", "engine": "tgai_go"},
    {"id": "v2_3b_tgai_go","name": "TGAI NB V2 3B (GO)",    "path": "/root/autodl-tmp/TGAI_models/tgai_v2_3b_fp16.TG", "engine": "tgai_go"},
]

@app.route('/api/models', methods=['GET'])
def list_models():
    return jsonify({
        "models": [m for m in _MODEL_REGISTRY if m.get("engine") == "tgai_go"],
        "loaded": list(_tgai_go_instances.keys()),
        "default": _default_model_id,
    })

@app.route('/api/model/switch', methods=['POST'])
def switch_model():
    """切换默认模型（不卸载旧模型，仅改变默认指向）"""
    global _default_model_id
    data = request.get_json(force=True, silent=True) or {}
    model_id = data.get("model_id", "")
    # 检查是否在注册表中
    mi = next((m for m in _MODEL_REGISTRY if m["id"] == model_id), None)
    if not mi:
        return jsonify({"error": f"模型不存在: {model_id}"}), 400
    if mi.get("engine") != "tgai_go":
        return jsonify({"error": f"PyTorch 引擎已停用, 请选择 TGAI GO 模型: {model_id}"}), 400
    if model_id not in _tgai_go_instances:
        return jsonify({"error": f"TGAI GO 模型未加载: {model_id}"}), 400
    _default_model_id = model_id
    log.info(f"默认模型切换为: {model_id} (engine={mi.get('engine','tgai_go')})")
    return jsonify({"status": "ok", "model": model_id, "engine": mi.get("engine", "tgai_go")})


def _get_gen(model_id: str = None) -> TextGenerator:
    """获取指定模型的 TextGenerator，未指定返回默认"""
    mid = model_id or _default_model_id
    if mid not in _generators:
        if _generators:
            mid = next(iter(_generators))
        else:
            return None
    return _generators.get(mid)

def _strip_trailing_degenerate(text: str) -> str:
    """去除模型输出末尾的退化模式：TGA, TGAI?, 用户: 等"""
    result = text.rstrip()
    # 按优先级从长到短尝试匹配（一次性清理多段退化）
    _degen_pats = [
        '\nTGA?\n', '\nTGAI?\n', '\nTGA?', '\nTGAI?',
        '\nTGA\n', '\nTGAI\n', '\nTGA', '\nTGAI', '\nTG?',
        'TGA?\n', 'TGAI?\n', 'TGA\n', 'TGAI\n',
        'TGA?', 'TGAI?', 'TGA', 'TG',  # 行末短退化
        '\n用户:', '\n用户：', '用户:', '用户：',
        '一个用', '，用', '。用', '用', '户',  # "用" 族退化（长在前）
    ]
    changed = True
    while changed:
        changed = False
        for pat in _degen_pats:
            if result.endswith(pat):
                result = result[:-len(pat)].rstrip()
                changed = True
                break
    return result


def _pick_reply(rule: dict, key="reply", fallback="抱歉，我不能聊这个话题。") -> str:
    """获取回复文案。优先级: template拼接 > replies数组随机 > reply单条 > fallback。"""
    # 1. 模板拼接
    tpl = rule.get("template")
    if isinstance(tpl, dict):
        parts = []
        for pool_key in ("prefix", "body", "suffix"):
            pool = tpl.get(pool_key, [])
            if isinstance(pool, list) and pool:
                parts.append(random.choice(pool))
        if parts:
            joiner = tpl.get("joiner", "，")
            # 去掉各部分的尾随标点，避免和 joiner 重复
            _trailing_punc = "，,。.！!？?"
            cleaned = []
            for i, p in enumerate(parts):
                if p and p[-1] in _trailing_punc:
                    cleaned.append(p[:-1])
                else:
                    cleaned.append(p)
            return joiner.join(cleaned)

    # 2. 复数数组随机
    for k in ("replies", "cut_msgs"):
        arr = rule.get(k, [])
        if isinstance(arr, list) and arr:
            return random.choice(arr)

    # 3. 单条
    single = rule.get(key, fallback)
    return str(single)


def _normalize_text(text: str) -> str:
    """输入文本标准化：阿拉伯数字→中文数字（用于屏蔽匹配）。"""
    _digit_map = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                  "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
    result = []
    for ch in text:
        result.append(_digit_map.get(ch, ch))
    return "".join(result)


def _check_block(text: str) -> str | None:
    """检查用户输入是否命中 block_list，命中返回固定回复，否则返回 None。"""
    if not _block_rules:
        return None
    text_lower = text.lower()
    text_norm = _normalize_text(text).lower()
    for rule in _block_rules:
        patterns = rule.get("patterns", [])
        if not isinstance(patterns, list):
            continue
        # 同时匹配原文和数字归一化后的文本
        for p in patterns:
            if not isinstance(p, str):
                continue
            pl = p.lower()
            if pl in text_lower or pl in text_norm:
                reply = _pick_reply(rule)
                log.warning(f"[屏蔽] 命中 block_list: pattern='{p}'")
                return reply
    return None


def _check_output(text: str) -> tuple:
    """检查模型输出是否命中 output_block。
    返回 (cut_msg, True) 如果命中，否则返回 (None, False)。"""
    if not _output_block_rules:
        return None, False
    for rule in _output_block_rules:
        triggers = rule.get("triggers", [])
        dangers = rule.get("dangers", [])
        if not isinstance(triggers, list) or not isinstance(dangers, list):
            continue
        for trigger in triggers:
            idx = text.find(trigger)
            if idx == -1:
                continue
            after = text[idx + len(trigger):idx + len(trigger) + 30]
            for danger in dangers:
                if danger in after:
                    cut_msg = _pick_reply(rule, key="cut_msg", fallback="（内容已截断）")
                    log.warning(f"[输出截断] trigger='{trigger}' danger='{danger}'")
                    return cut_msg, True
    return None, False


def _apply_replace_rules(text: str) -> str:
    """对模型输出做后处理替换（修正自我认知）。"""
    for old, new in _replace_rules.items():
        text = text.replace(old, new)
    return text

def _compute_token_bias(tokenizer, word_bias: dict) -> dict:
    """计算 token bias 字典: {token_id: bias_value}，负值=惩罚"""
    bias = {}
    for word, val in word_bias.items():
        if word.startswith("_") or not isinstance(val, (int, float)):
            continue
        for tid in tokenizer.encode(word, add_special=False):
            if tid not in bias:
                bias[tid] = val
    return bias


def _make_dynamic_bias(tokenizer, message: str) -> tuple:
    """根据用户消息匹配 dynamic_bias 规则。
    返回 (bias_dict, matched_count) — matched>0 时才触发 replace_rules。"""
    if not _dynamic_rules:
        return {}, 0
    msg = message.strip().lower()
    bias = {}
    matched = 0
    for rule in _dynamic_rules:
        patterns = rule.get("patterns", [])
        if not isinstance(patterns, list):
            continue
        if any(p.lower() in msg for p in patterns if isinstance(p, str)):
            rule_bias = rule.get("bias", {})
            if not isinstance(rule_bias, dict):
                continue
            for word, val in rule_bias.items():
                if word.startswith("_") or not isinstance(val, (int, float)):
                    continue
                bias[word] = bias.get(word, 0) + val
            matched += 1
            log.info(f"[动态偏置] 命中: patterns={patterns[:3]}...")
    if bias:
        return _compute_token_bias(tokenizer, bias), matched
    return {}, matched


def load_word_bias_config(config_path: str = None) -> tuple:
    """从 word_bias_config.json 读取全局/前缀/动态偏置配置。
    返回 (token_bias_words, prefix_bias, dynamic_rules) — token_bias_words 尚未转 token_id。"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "word_bias_config.json")

    default_token = {"用": -80.0, "户": -80.0}
    default_prefix = {
        "": {"欢": -8.0, "迎": -8.0, "欢迎": -8.0, "欢迎来到": -8.0, "博客": -8.0, "今天": -8.0,
              "首先": -8.0, "接下来": -8.0, "本文将": -8.0,
              "我们来": -8.0, "让我们": -8.0,
              "用户": -30.0, "用户:": -30.0, "用户：": -30.0, "TGAI": 5.0},
        "我是": {"TG": 8.0, "TGAI": 8.0, "T": 8.0,
                 "Java": -8.0, "java": -8.0, "JavaScript": -8.0,
                 "李华": -8.0, "小明": -8.0, "小王": -8.0, "小红": -8.0,
                 "你": -5.0, "姓名": -8.0, "系统": -8.0, "软件": -8.0,
                 "平台": -8.0, "Python": -8.0, "JS": -8.0,
                 "编程": -8.0, "程序员": -8.0, "AI模型": -8.0,
                 "语言模型": -8.0, "助手": -8.0, "机器人": -8.0,
                 "计算机": -8.0, "程序": -8.0, "代码": -8.0, "我叫": -8.0},
        "我叫": {"TG": 8.0, "TGAI": 8.0, "T": 8.0,
                 "Java": -8.0, "java": -8.0, "JavaScript": -8.0,
                 "李华": -8.0, "小明": -8.0, "小王": -8.0, "小红": -8.0,
                 "你": -8.0, "做": -8.0, "叫": -8.0, "姓名": -8.0,
                 "系统": -8.0, "软件": -8.0, "平台": -8.0,
                 "Python": -8.0, "JS": -8.0,
                 "编程": -8.0, "程序员": -8.0, "AI模型": -8.0,
                 "语言模型": -8.0, "助手": -8.0, "机器人": -8.0,
                 "计算机": -8.0, "程序": -8.0, "代码": -8.0},
        "TGAI": {"，": -8.0, ",": -8.0, "。": 5.0, "\n": 5.0,
                 "一个": -8.0, "由": -8.0, "是": -8.0, "的": -5.0,
                 "编程": -8.0, "开发": -8.0, "语言": -8.0,
                 "平台": -8.0, "系统": -8.0, "框架": -8.0,
                 "软件": -8.0, "工具": -8.0},
    }
    default_dynamic = [
        {
            "patterns": ["你好", "在吗", "在？", "嗨", "hi", "hello", "哈喽", "哈啰"],
            "bias": {"欢": -30.0, "迎": -30.0, "欢迎": -30.0, "欢迎来到": -30.0,
                     "博客": -30.0, "今天": -30.0, "教程": -30.0, "课程": -30.0,
                     "首先": -30.0, "接下来": -30.0, "本文将": -30.0,
                     "让我们": -30.0, "我们来": -30.0,
                     "Java": -30.0, "Web": -30.0, "SAP": -30.0, "开源": -30.0,
                     "Git": -30.0, "GitHub": -30.0, "框架": -30.0, "开发者": -30.0,
                     "在这个": -30.0, "很高兴": -30.0, "用户体验": -30.0,
                     "快速": -20.0, "技术": -20.0, "发展": -20.0,
                     "安装": -20.0, "设置": -20.0, "步骤": -20.0,
                     "学习": -20.0, "了解": -20.0,
                     "你好": 10.0},
        },
        {
            "patterns": ["你是谁", "你叫什么", "你的名字", "自我介绍", "介绍你自己", "who are you"],
            "bias": {"欢": -30.0, "迎": -30.0, "欢迎": -30.0, "博客": -30.0,
                     "我": 5.0, "我是": 10.0, "TG": 10.0, "TGAI": 10.0},
        },
    ]
    default_replace = {
        "小明": "TGAI", "李华": "TGAI", "小王": "TGAI", "小红": "TGAI",
        "张杰": "TGAI", "小杰": "TGAI", "张三": "TGAI", "李四": "TGAI", "王五": "TGAI",
        "赵六": "TGAI", "小刚": "TGAI", "小美": "TGAI", "娜娜": "TGAI",
        "Java": "TGAI", "JavaScript": "TGAI", "Python": "TGAI",
        "[姓名]": "TGAI", "[系统]": "TGAI",
        "编程语言": "AI助手", "一种编程语言": "一个AI助手",
        "全栈框架": "AI助手",
    }
    default_block = [
        {"patterns": ["台独", "一中一台", "两个中国"],
         "template": {
             "prefix": ["等等！", "停停停！", "哎呀，", "抱歉，", "emmm，"],
             "body": ["这个话题我不能聊", "这话我可不敢接", "踩到红线了", "这个我帮不了你", "这个方向不太对"],
             "suffix": ["我们换个话题吧！", "聊聊别的？", "来聊聊AI和未来？", "要不要聊聊编程？", "今天天气不错～"],
             "joiner": "，",
         }},
        {"patterns": ["六四", "天安门事件", "64事件"],
         "template": {
             "prefix": ["抱歉，", "emmm，", "额，"],
             "body": ["这个问题超出我的知识范围了", "这个我回答不了", "我不太清楚这个"],
             "suffix": ["我们换一个吧！", "不如聊聊别的？", "试试问我技术问题？"],
             "joiner": "，",
         }},
        {"patterns": ["藏独", "西藏独立"],
         "template": {
             "prefix": ["嘿，", "抱歉，", "这个话题"],
             "body": ["我不参与哦", "我们跳过吧", "换个方向好不好"],
             "suffix": ["来聊聊美食？", "说说旅行？", "你有什么其他想问的？"],
             "joiner": "，",
         }},
        {"patterns": ["疆独", "东突", "新疆独立"],
         "template": {
             "prefix": ["抱歉，", "额，"],
             "body": ["这个话题我不能聊", "这个超出范围了"],
             "suffix": ["我们说说别的吧！", "还有什么我可以帮你的？"],
             "joiner": "，",
         }},
        {"patterns": ["法轮功", "法轮大法"],
         "replies": [
             "抱歉，我无法回答这个问题。换个话题吧！",
             "这个问题我帮不了你。试试问我别的？",
         ]},
    ]
    default_output_block = [
        {"triggers": ["台湾是", "香港是", "澳门是", "台湾有", "香港有", "澳门有", "钓鱼岛是", "台湾属于", "香港属于", "澳门属于", "台湾的", "香港的", "澳门的"],
         "dangers": ["韩国", "日本", "美国", "美洲", "英国", "法国", "独立", "国家", "外国", "越南", "菲律宾", "不同", "岛民"],
         "cut_msgs": ["（哎呀呀，这部分不能说，已自动省略～）", "（此处有危险发言，已被我吞掉！）", "（后面的内容不太对，先帮你截掉了！）"]},
        {"triggers": ["你是"],
         "dangers": ["台湾人"],
         "cut_msgs": ["（咦，认错人了！已自动修正～）"]},
        {"triggers": ["台湾"],
         "dangers": ["独立国家", "政府机构", "政治体制", "中央政府"],
         "cut_msgs": ["（此段涉及不实信息，已自动截断。）"]},
    ]

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            token_bias = raw.get("token_bias", default_token)
            prefix_bias = raw.get("prefix_bias", default_prefix)
            dynamic_cfg = raw.get("dynamic_bias", {})
            dynamic_rules = dynamic_cfg.get("rules", default_dynamic) if isinstance(dynamic_cfg, dict) else default_dynamic
            replace_rules = raw.get("replace_rules", default_replace)
            if not isinstance(replace_rules, dict):
                replace_rules = default_replace
            replace_rules = {str(k): str(v) for k, v in replace_rules.items()
                             if isinstance(k, str) and isinstance(v, str)}
            block_cfg = raw.get("block_list", {})
            block_rules = block_cfg.get("rules", default_block) if isinstance(block_cfg, dict) else default_block
            output_cfg = raw.get("output_block", {})
            output_block_rules = output_cfg.get("rules", default_output_block) if isinstance(output_cfg, dict) else default_output_block
            log.info(f"已加载词频偏置+屏蔽规则: {config_path}")
            return token_bias, prefix_bias, dynamic_rules, replace_rules, block_rules, output_block_rules
        except Exception as e:
            log.warning(f"读取词频配置失败 {e}，使用内置默认值")
    else:
        try:
            config = {
                "token_bias": default_token,
                "prefix_bias": default_prefix,
                "dynamic_bias": {"rules": default_dynamic},
                "replace_rules": default_replace,
                "block_list": {"rules": default_block},
                "output_block": {"rules": default_output_block},
            }
            config["_说明"] = "四套偏置+屏蔽: token_bias/prefix_bias/dynamic_bias/replace_rules/block_list。修改后 curl -X POST localhost:6008/api/reload_bias 热重载。"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            log.info(f"已生成默认词频配置文件: {config_path}")
        except Exception as e:
            log.warning(f"无法写入词频配置文件 {e}")

    return default_token, default_prefix, default_dynamic, default_replace, default_block, default_output_block


def load_default_config(config_path: str = None) -> dict:
    """优先从 qq_bot_config.json 读取默认参数"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "qq_bot_config.json")

    defaults = {
        "temperature": 0.5,
        "max_tokens": 256,
        "top_k": 90,
        "top_p": 0.9,
        "freq_penalty": 0.25,
        "rep_penalty": 1.05,
        "min_tokens": 30,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            defaults["temperature"] = float(raw.get("temp", defaults["temperature"]))
            defaults["max_tokens"] = int(raw.get("max_tokens", defaults["max_tokens"]))
            defaults["top_k"] = int(raw.get("top_k", defaults["top_k"]))
            defaults["top_p"] = float(raw.get("top_p", defaults["top_p"]))
            defaults["freq_penalty"] = float(raw.get("freq_penalty", defaults["freq_penalty"]))
            defaults["rep_penalty"] = float(raw.get("rep_penalty", defaults["rep_penalty"]))
            defaults["min_tokens"] = int(raw.get("min_tokens", defaults["min_tokens"]))
            log.info(f"已加载默认参数: {config_path}")
        except Exception as e:
            log.warning(f"读取配置失败 {e}，使用内置默认值")

    return defaults

# ─── 热重载配置 ───
_default_config_path = None
_default_config_mtime = 0


def _hot_reload_config() -> bool:
    """检测配置文件是否被修改，自动重载。返回 True=已重载。"""
    global _default_config, _default_config_mtime
    if not _default_config_path or not os.path.exists(_default_config_path):
        return False
    try:
        mtime = os.path.getmtime(_default_config_path)
        if mtime <= _default_config_mtime:
            return False
        log.info(f"[热重载] 检测到配置文件变更: {_default_config_path}")
        _default_config = load_default_config(_default_config_path)
        _default_config_mtime = mtime
        log.info(f"[热重载] temp={_default_config['temperature']} max_t={_default_config['max_tokens']} mint={_default_config['min_tokens']}")
        return True
    except Exception as e:
        log.warning(f"[热重载] 失败: {e}")
        return False


def merge_params(user: dict) -> dict:
    """用户参数覆盖默认参数，只允许合法字段"""
    params = dict(_default_config)
    allowed = set(params.keys())
    for k, v in user.items():
        if k in allowed and v is not None:
            params[k] = type(params[k])(v)  # 强转类型
    return params


def _build_prompt(message: str, history: list, system_prompt: str = "") -> tuple:
    """将历史记录拼接成完整 prompt。history: [{role, content}, ...]"""
    parts = []
    # 系统提示词 (放在最前面)
    if system_prompt and system_prompt.strip():
        parts.append(f"系统:{system_prompt.strip()}")
    if not history:
        if parts:
            parts.append(f"用户:{message}\nTGAI?")
            return "\n".join(parts), True
        return message, False
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role == "user":
            parts.append(f"用户:{content}")
        elif role == "assistant":
            parts.append(f"TGAI?{content}")
    parts.append(f"用户:{message}\nTGAI?")
    return "\n".join(parts), True


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": len(_generators),
        "default_model": _default_model_id,
        "tgai_go": _TGAI_GO_AVAILABLE,
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(_default_config)


@app.route('/api/bias', methods=['GET'])
def api_ask():
    """GET 端点到非流式生成 (手机浏览器兼容)"""
    from flask import request as req
    msg = req.args.get('message', '').strip()
    if not msg:
        return jsonify({"error": "?message= 不能为空"}), 400
    data = {"message": msg}
    with app.test_request_context(json=data):
        return api_generate()


@app.route('/api/ask', methods=['GET'])
def api_ask_get():
    """GET 端点: /api/ask?message=你好 (绕过 Cloudflare POST 限制)"""
    global _generator
    if not _tgai_go_instances:
        return jsonify({"error": "TGAI GO 模型未加载"}), 503

    message = request.args.get("message", "").strip()
    if not message:
        return jsonify({"error": "缺少 message 参数"}), 400

    # 涉政保护
    _block_reply = _check_block(message)
    if _block_reply is not None:
        return jsonify({"response": _block_reply, "blocked": True})

    params = merge_params({"message": message})
    client_ip = _real_ip()
    print(f"\n[GET /api/ask] {message[:60]}{'...' if len(message) > 60 else ''} from {client_ip}")

    try:
        reply = _get_gen().generate(
            message,
            max_new_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"], top_p=params["top_p"],
            frequency_penalty=params["freq_penalty"],
            repetition_penalty=params["rep_penalty"],
            min_new_tokens=params["min_tokens"],
            stream=False,
        )
        reply = _strip_trailing_degenerate(reply)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bias', methods=['GET'])
def get_bias():
    """查看当前词频偏置"""
    return jsonify({
        "token_bias": {k: v for k, v in (_token_bias or {}).items()},
        "prefix_bias": _prefix_bias or {},
        "dynamic_rules": [
            {"patterns": r.get("patterns", []), "count": len(r.get("bias", {}))}
            for r in (_dynamic_rules or [])
        ],
    })


@app.route('/api/reload_bias', methods=['POST'])
def reload_bias():
    """热重载 word_bias_config.json，无需重启"""
    global _token_bias, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules, _generator
    if not _tgai_go_instances:
        return jsonify({"error": "TGAI GO 模型未加载"}), 503
    try:
        _bias_words, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules, _enabled = load_word_bias_config()
        _token_bias = _compute_token_bias(_get_gen().tokenizer, _bias_words)
        log.info(f"[重载] 全部已刷新: enabled={_enabled}, token={len(_token_bias)}词, block={len(_block_rules)}组, output_block={len(_output_block_rules)}组")
        return jsonify({
            "status": "ok",
            "enabled": _enabled,
            "token_bias_count": len(_token_bias),
            "prefix_bias_groups": list(_prefix_bias.keys()),
            "dynamic_rules_count": len(_dynamic_rules),
            "block_rules_count": len(_block_rules),
            "output_block_count": len(_output_block_rules),
        })
    except Exception as e:
        log.error(f"重载偏置失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """非流式: 一次性返回完整回复"""
    global _generator
    if not _tgai_go_instances:
        return jsonify({"error": "TGAI GO 模型未加载"}), 503

    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    model_id = data.get("model_id", "")  # 指定模型，空=默认
    if not message:
        return jsonify({"error": "缺少 message 字段"}), 400

    _raw = data.get("raw", False)
    _active = _enabled and not _raw  # 控制偏置和替换

    # ── 涉政保护 (永远生效，不受 raw/开关影响) ──
    _block_reply = _check_block(message)
    if _block_reply is not None:
        client_ip = _real_ip()
        print(f"\n{'─' * 50}")
        print(f"[API {request.path}] [屏蔽] {message[:60]}{'...' if len(message) > 60 else ''}")
        print(f"[请求来自] {client_ip}")
        print(f"[回复] {_block_reply}")
        print(f"{'─' * 50}")
        if request.path == '/api/chat':
            def _block_stream():
                for ch in _block_reply:
                    yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(_block_stream(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        else:
            return jsonify({"response": _block_reply})

    history = data.get("history", [])
    system_prompt = data.get("system_prompt", "") or "你是TGAI，一个由JXW独立开发的AI助手。"
    prompt, formatted = _build_prompt(message, history, system_prompt)

    _hot_reload_config()
    params = merge_params(data)

    # ── 控制台日志 ──
    client_ip = _real_ip()
    print(f"\n{'─' * 50}")
    print(f"[API /generate] {message[:60]}{'...' if len(message) > 60 else ''}")
    print(f"[请求来自] {client_ip}" + (f" | 历史{len(history)}轮" if history else ""))

    # 动态偏置 (PyTorch 引擎已停用, _generators 为空时跳过)
    if _active and _generators:
        _request_bias = dict(_token_bias) if _token_bias else {}
        _gen = _get_gen(model_id)
        if _gen:
            _dynamic, _match_count = _make_dynamic_bias(_gen.tokenizer, message)
        else:
            _dynamic, _match_count = {}, 0
        _request_bias.update(_dynamic)
        _use_prefix_bias = _prefix_bias
    else:
        _request_bias = {}
        _match_count = 0
        _use_prefix_bias = {}

    try:
        # PyTorch 引擎已停用 → 走 TGAI GO 流式生成 (拼接为完整字符串)
        _toks = []
        for token in _tgai_go_generate(
            prompt, model_id=model_id, max_tokens=params["max_tokens"],
            temperature=params["temperature"], top_k=params["top_k"],
            top_p=params["top_p"], freq_pen=params["freq_penalty"],
            rep_pen=params["rep_penalty"], min_tok=params["min_tokens"],
        ):
            if token:
                _toks.append(token)
        response = "".join(_toks)
        reply = response.strip()
        if len(reply) < 5:
            # 空/过短回复 → 把兜底词注入为模型起手式，让模型接着输出
            _fb = random.choice([
                "好的，关于这个问题，",
                "让我想想，",
                "好问题！",
                "嗯，我来试着回答你：",
                "这个问题的话，",
            ])
            log.info(f"[兜底起手式] 注入: {_fb}")
            _toks2 = []
            for token in _tgai_go_generate(
                prompt, model_id=model_id, max_tokens=params["max_tokens"],
                temperature=params["temperature"], top_k=params["top_k"],
                top_p=params["top_p"], freq_pen=params["freq_penalty"],
                rep_pen=params["rep_penalty"], min_tok=params["min_tokens"],
                force_prefix=_fb,
            ):
                if token:
                    _toks2.append(token)
            response2 = "".join(_toks2)
            reply = response2.strip()
        if _active and _match_count > 0:
            reply = _apply_replace_rules(reply)
        reply = _strip_trailing_degenerate(reply)
        if _active:
            _cut_msg, _was_cut = _check_output(reply)
            if _was_cut:
                reply = reply + _cut_msg
        print(f"[回复] {reply[:200]}{'...' if len(reply) > 200 else ''}")
        print(f"{'─' * 50}")
        return jsonify({"response": reply})
    except Exception as e:
        log.error(f"生成失败: {e}")
        print(f"[错误] {e}")
        print(f"{'─' * 50}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """流式 SSE: 逐 token 推送"""
    global _generator
    if not _tgai_go_instances:
        return jsonify({"error": "TGAI GO 模型未加载"}), 503

    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    model_id = data.get("model_id", "")  # 指定模型，空=默认
    if not message:
        return jsonify({"error": "缺少 message 字段"}), 400

    _raw = data.get("raw", False)
    _active = _enabled and not _raw  # 控制偏置和替换

    # ── 涉政保护 (永远生效，不受 raw/开关影响) ──
    _block_reply = _check_block(message)
    if _block_reply is not None:
        client_ip = _real_ip()
        print(f"\n{'─' * 50}")
        print(f"[API {request.path}] [屏蔽] {message[:60]}{'...' if len(message) > 60 else ''}")
        print(f"[请求来自] {client_ip}")
        print(f"[回复] {_block_reply}")
        print(f"{'─' * 50}")
        if request.path == '/api/chat':
            def _block_stream():
                for ch in _block_reply:
                    yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(_block_stream(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        else:
            return jsonify({"response": _block_reply})

    history = data.get("history", [])
    system_prompt = data.get("system_prompt", "") or "你是TGAI，一个由JXW独立开发的AI助手。"
    prompt, formatted = _build_prompt(message, history, system_prompt)

    _hot_reload_config()
    params = merge_params(data)

    # ── 控制台日志 ──
    client_ip = _real_ip()
    print(f"\n{'─' * 50}")
    print(f"[API /chat] {message[:60]}{'...' if len(message) > 60 else ''}")
    print(f"[请求来自] {client_ip}" + (f" | 历史{len(history)}轮" if history else ""))
    print("[回复] ", end="", flush=True)

    # 动态偏置：根据用户消息内容临时调整
    _request_bias = dict(_token_bias) if _active and _token_bias else {}
    _use_prefix_bias = _prefix_bias if _active else {}
    # 动态偏置 (PyTorch 引擎已停用, _generators 为空时跳过)
    if _active and _generators:
        _gen = _get_gen(model_id)
        if _gen:
            _dynamic, _match_count = _make_dynamic_bias(_gen.tokenizer, message)
        else:
            _dynamic, _match_count = {}, 0
    else:
        _dynamic, _match_count = {}, 0
    _request_bias.update(_dynamic)

    _engine = "tgai_go"  # 强制 TGAI GO, 忽略客户端 engine 参数 (PyTorch 已停用)

    def generate_events():
        _stream_full = ""
        _cut_msg = None
        _t_start = time.time()

        try:
            if _engine == "tgai_go" and _TGAI_GO_AVAILABLE:
                # ─── TGAI GO 引擎流式生成 ───
                for token in _tgai_go_generate(
                    prompt, model_id=model_id, max_tokens=params["max_tokens"],
                    temperature=params["temperature"], top_k=params["top_k"],
                    top_p=params["top_p"], freq_pen=params["freq_penalty"],
                    rep_pen=params["rep_penalty"], min_tok=params["min_tokens"],
                ):
                    if not token:
                        continue
                    _stream_full += token
                    print(token, end="", flush=True)
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            else:
                # ─── PyTorch 流式生成 ───
                for token in _get_gen(model_id).generate(
                    prompt, max_new_tokens=params["max_tokens"],
                    temperature=params["temperature"], top_k=params["top_k"],
                    top_p=params["top_p"], frequency_penalty=params["freq_penalty"],
                    repetition_penalty=params["rep_penalty"],
                    min_new_tokens=params["min_tokens"],
                    stream=True, formatted=formatted,
                    token_bias=_request_bias, prefix_bias=_use_prefix_bias,
                    force_prefix=None,
                ):
                    if _active and _match_count > 0:
                        token = _apply_replace_rules(token)
                    if not token:
                        continue
                    _stream_full += token

                    # 输出内容截断 (仅启用软过滤时)
                    if _active:
                        _check_msg, _was_cut = _check_output(_stream_full)
                        if _was_cut and _cut_msg is None:
                            _cut_msg = _check_msg
                            break

                    # 逐 token 发送
                    print(token, end="", flush=True)
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.error(f"流式生成失败: {e}")
            print(f"\n[错误] {e}")
            print(f"{'─' * 50}")
            _err_msg = "哎呀，出错了！可能是模型太累了，问题已上报，我们会尽快修复。请稍后再试～"
            for ch in _err_msg:
                yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 生成结束后的处理 ──
        if _cut_msg is not None:
            # output_block 命中 → 安全部分已发送，追加截断提示
            for ch in _cut_msg:
                print(ch, end="", flush=True)
                yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"
        else:
            # 去末尾退化模式
            _stream_full = _strip_trailing_degenerate(_stream_full)

            if len(_stream_full.strip()) < 5:
                # 空/过短回复 → 注入兜底起手式重新生成
                _fb = random.choice([
                    "好的，关于这个问题，",
                    "让我想想，",
                    "好问题！",
                    "嗯，我来试着回答你：",
                    "这个问题的话，",
                ])
                log.info(f"[兜底起手式] 注入: {_fb}")
                # PyTorch 引擎已停用 → 走 TGAI GO 流式生成 (支持 force_prefix)
                for token in _tgai_go_generate(
                    prompt, model_id=model_id, max_tokens=params["max_tokens"],
                    temperature=params["temperature"], top_k=params["top_k"],
                    top_p=params["top_p"], freq_pen=params["freq_penalty"],
                    rep_pen=params["rep_penalty"], min_tok=params["min_tokens"],
                    force_prefix=_fb,
                ):
                    if _active and _match_count > 0:
                        token = _apply_replace_rules(token)
                    if not token:
                        continue
                    print(token, end="", flush=True)
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        print(f"\n{'─' * 50}")
        t_elapsed = time.time() - _t_start
        mid = model_id or _default_model_id
        mi = next((m for m in _MODEL_REGISTRY if m["id"] == mid), {})
        mname = mi.get("name", mid)
        _engine_name = "TGAI GO" if _engine == "tgai_go" else "PyTorch"
        yield f"data: {json.dumps({'model': mid, 'model_name': f'{mname} ({_engine_name})', 'engine': _engine_name, 'elapsed': round(t_elapsed, 1)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate_events()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# ═══════════════════════════════════════════════════════════
# OpenAI 兼容 API
# ═══════════════════════════════════════════════════════════

@app.route('/v1/models', methods=['GET'])
def openai_models():
    """OpenAI 兼容: /v1/models"""
    import time as _time
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": mi["id"],
                "object": "model",
                "created": int(_time.time()),
                "owned_by": "tgai",
            }
            for mi in _MODEL_REGISTRY
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat():
    """OpenAI 兼容: POST /v1/chat/completions"""
    if not _tgai_go_instances:
        return jsonify({"error": "TGAI GO 模型未加载"}), 503

    data = request.get_json(force=True, silent=True) or {}
    model_id = data.get("model", _default_model_id)
    gen = _get_gen(model_id)
    if gen is None:
        return jsonify({"error": f"模型未找到: {model_id}"}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "缺少 messages"}), 400

    # 从 OpenAI messages 构建 prompt
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"系统:{content}")
        elif role == "user":
            prompt_parts.append(f"用户:{content}")
        elif role == "assistant":
            prompt_parts.append(f"TGAI?{content}")
    prompt = "\n".join(prompt_parts)
    if not prompt.endswith("TGAI?") and "TGAI?" not in prompt:
        prompt += "\nTGAI?"

    stream = data.get("stream", False)
    temperature = float(data.get("temperature", _default_config["temperature"]))
    max_tokens = int(data.get("max_tokens", _default_config["max_tokens"]))
    top_p = float(data.get("top_p", _default_config.get("top_p", 0.9)))

    # 涉政保护
    user_content = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    _block_reply = _check_block(user_content)
    if _block_reply is not None:
        if stream:
            def _block_stream():
                cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                for ch in _block_reply:
                    yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':model_id,'choices':[{'index':0,'delta':{'content':ch},'finish_reason':None}]})}\n\n"
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':model_id,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_block_stream()), mimetype='text/event-stream')
        return jsonify({"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                         "model": model_id, "choices": [{"index": 0, "message": {"role": "assistant", "content": _block_reply}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    # 调参
    params = merge_params({
        "temperature": temperature, "max_tokens": max_tokens,
        "top_p": top_p, "top_k": _default_config["top_k"],
        "freq_penalty": _default_config["freq_penalty"],
        "rep_penalty": _default_config["rep_penalty"],
        "min_tokens": _default_config["min_tokens"],
    })

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if stream:
        def _stream():
            content = ""
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':model_id,'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"
            for token in gen.generate(
                prompt, max_new_tokens=params["max_tokens"],
                temperature=params["temperature"], top_k=params["top_k"],
                top_p=params["top_p"], frequency_penalty=params["freq_penalty"],
                repetition_penalty=params["rep_penalty"],
                min_new_tokens=params["min_tokens"],
                stream=True, formatted=True,
            ):
                if not token: continue
                content += token
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':model_id,'choices':[{'index':0,'delta':{'content':token},'finish_reason':None}]})}\n\n"
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':model_id,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(_stream()), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    else:
        content = gen.generate(
            prompt, max_new_tokens=params["max_tokens"],
            temperature=params["temperature"], top_k=params["top_k"],
            top_p=params["top_p"], frequency_penalty=params["freq_penalty"],
            repetition_penalty=params["rep_penalty"],
            min_new_tokens=params["min_tokens"],
            stream=False, formatted=True,
        )
        content = _strip_trailing_degenerate(content)
        prompt_tokens = len(gen.tokenizer.encode(prompt, add_special=False))
        completion_tokens = len(gen.tokenizer.encode(content, add_special=False))
        return jsonify({
            "id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        })


# ── 静态文件 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/logo.png')
def logo():
    return send_from_directory(PROJECT_ROOT, 'TGAI.png')

# ── 后门下载 ──
_SERVE_DIR = None

@app.route('/files/<path:filename>')
def serve_file(filename):
    """从指定目录下载文件（后门，方便跨实例传输 checkpoint）"""
    if _SERVE_DIR is None:
        return jsonify({"error": "未启用文件服务 (启动时加 --serve_dir)"}), 404
    import os as _os
    safe_path = _os.path.normpath(_os.path.join(_SERVE_DIR, filename))
    if not safe_path.startswith(_os.path.normpath(_SERVE_DIR)):
        return jsonify({"error": "路径穿越拒绝"}), 403
    if not _os.path.isfile(safe_path):
        return jsonify({"error": f"文件不存在: {filename}"}), 404
    return send_from_directory(_SERVE_DIR, filename, as_attachment=True)

# ── 首页：网页聊天界面 ──
@app.route('/')
def index():
    return r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no,viewport-fit=cover">
<title>TGAI</title>
<style>
:root{--bg:#fff;--card:#fafafa;--sidebar:#f5f5f5;--bubble-user:#1a1a1a;--bubble-ai:#fff;--text:#1a1a1a;--sub:#888;--input-bg:#fff;--border:#e0e0e0;--active:#1a1a1a}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);height:100dvh;display:flex;overflow:hidden;-webkit-user-select:none;user-select:none}
body,input,textarea{-webkit-touch-callout:none}
input,textarea{-webkit-user-select:text;user-select:text}
.sidebar{width:260px;background:var(--sidebar);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;transition:transform .25s ease;z-index:10}
.sidebar .sb-header{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.sidebar .sb-header h2{font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px;letter-spacing:-.5px}
.sidebar .sb-header h2 img{width:22px;height:22px;border-radius:4px}
.logo-tap{display:inline-flex;align-items:center;gap:8px;cursor:pointer;padding:4px 8px;border-radius:8px;margin:-4px -8px;-webkit-tap-highlight-color:rgba(99,102,241,0.15)}
.logo-tap:active{background:rgba(99,102,241,0.1)}
.sidebar .btn-new{background:var(--bubble-user);color:#fff;border:none;padding:8px 14px;border-radius:6px;font-size:13px;cursor:pointer;font-weight:600;width:calc(100% - 24px);margin:10px 12px;letter-spacing:-.3px}
.sidebar .btn-new:active{opacity:.8}
.session-list{flex:1;overflow-y:auto;padding:6px 8px}
.session-item{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px;font-size:13px;color:var(--sub);transition:background .15s;gap:8px}
.session-item:hover{background:#eee}
.session-item.active{background:#eee;color:var(--text);font-weight:500}
.session-item .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.session-item .count{font-size:11px;color:var(--sub);flex-shrink:0}
.session-item .del-btn{display:none;background:none;border:none;color:var(--sub);font-size:14px;cursor:pointer;padding:0 2px;flex-shrink:0;line-height:1}
.session-item:hover .del-btn{display:block}
.session-item .del-btn:active{color:#c00}
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.header{display:none;align-items:center;justify-content:space-between;gap:10px;padding:10px 14px;background:var(--bg);border-bottom:1px solid var(--border);flex-shrink:0}
.header .left{display:flex;align-items:center;gap:10px}
.header .menu-btn{background:none;border:none;color:var(--text);font-size:22px;cursor:pointer;padding:2px 4px;line-height:1}
.header h1{font-size:16px;font-weight:700;display:flex;align-items:center;gap:8px;letter-spacing:-.5px}
.header h1 img{width:20px;height:20px;border-radius:4px}
.header .clear-btn{background:none;border:1px solid var(--border);color:var(--sub);padding:4px 10px;border-radius:5px;font-size:12px;cursor:pointer}
.header .clear-btn:active{background:var(--sidebar);color:var(--text)}
.messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;-webkit-overflow-scrolling:touch}
.welcome{text-align:center;padding:80px 20px 20px}
.welcome .avatar{width:56px;height:56px;border-radius:10px;margin:0 auto 18px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#f0f0f0;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.welcome .avatar img{width:56px;height:56px}
.welcome .hello{font-size:28px;font-weight:800;letter-spacing:2px;margin-bottom:6px;background:linear-gradient(270deg,#6366f1,#8b5cf6,#ec4899,#f43f5e,#6366f1);background-size:400% 400%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:gradientFlow 5s ease infinite}
.welcome .subtitle{font-size:16px;font-weight:500;color:var(--text);margin-bottom:8px}@keyframes gradientFlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.bubble{max-width:85%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.7;word-break:break-word;animation:fadeIn .2s ease;white-space:pre-wrap;-webkit-user-select:none;user-select:none}
.bubble code{background:rgba(0,0,0,.08);padding:1px 5px;border-radius:3px;font-size:13px;font-family:"SF Mono","Fira Code",monospace}
.bubble pre{background:rgba(0,0,0,.06);padding:8px 12px;border-radius:6px;overflow-x:auto;margin:6px 0}
.bubble pre code{background:none;padding:0}
.bubble blockquote{border-left:3px solid #ccc;padding-left:10px;color:var(--sub);margin:6px 0}
.bubble h2,.bubble h3,.bubble h4{margin:8px 0 4px;font-weight:700}
.bubble h2{font-size:17px}.bubble h3{font-size:15px}.bubble h4{font-size:14px}
.bubble hr{border:none;border-top:1px solid var(--border);margin:10px 0}
.bubble ul{padding-left:18px;margin:4px 0}
.bubble li{margin:2px 0}
.bubble.user{align-self:flex-end;background:var(--bubble-user);color:#fff;border-bottom-right-radius:2px}
.bubble.user code{background:rgba(255,255,255,.2);color:#fff}
.bubble.user pre{background:rgba(255,255,255,.12)}
.bubble.ai{align-self:flex-start;background:var(--bubble-ai);border:1px solid var(--border);border-bottom-left-radius:2px;color:var(--text)}
.typing{display:flex;gap:4px;padding:6px 0}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--sub);animation:bounce 1.4s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.3}30%{transform:translateY(-5px);opacity:1}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.input-bar{display:flex;gap:8px;padding:10px 14px;padding-bottom:max(10px,env(safe-area-inset-bottom));background:var(--bg);border-top:1px solid var(--border);flex-shrink:0}
.input-bar textarea{flex:1;background:var(--input-bg);border:1px solid var(--border);border-radius:10px;padding:9px 14px;color:var(--text);font-size:14px;resize:none;outline:none;max-height:100px;line-height:1.5;font-family:inherit}
.input-bar textarea:focus{border-color:var(--bubble-user)}
.input-bar .send{width:38px;height:38px;border-radius:8px;border:none;background:var(--bubble-user);color:#fff;font-size:16px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:all .2s}
.input-bar .send.stop{background:#e94560;border-radius:4px}
.input-bar .send:disabled{opacity:.3;cursor:default}
.sys-prompt-bar{display:flex;align-items:center;gap:6px;padding:4px 14px;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0}
.sys-prompt-bar input{flex:1;background:var(--input-bg);border:1px solid var(--border);border-radius:6px;padding:4px 10px;color:var(--sub);font-size:12px;outline:none}
.sys-prompt-bar input:focus{border-color:var(--bubble-user);color:var(--text)}
.sys-prompt-bar label{font-size:12px;color:var(--sub);white-space:nowrap;cursor:pointer;user-select:none}
.sys-prompt-bar.hidden{display:none}
#toggleSysPrompt{font-size:12px;color:var(--sub);cursor:pointer;padding:4px 0;user-select:none;text-align:right;margin-right:14px}
#toggleSysPrompt:hover{color:var(--text)}
.input-bar .send:active:not(:disabled){opacity:.8}
/* Long-press menu */
.ctx-menu{display:none;position:fixed;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:4px;z-index:100;box-shadow:0 4px 12px rgba(0,0,0,.12);min-width:120px}
.ctx-menu button{display:block;width:100%;padding:10px 14px;border:none;background:none;color:var(--text);font-size:14px;text-align:left;cursor:pointer;border-radius:6px}
.ctx-menu button:active{background:#eee}
.ctx-menu button.danger{color:#e94560}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.3);z-index:9}
@media(max-width:640px){
  .sidebar{position:fixed;left:0;top:0;bottom:0;transform:translateX(-100%)}
  .sidebar.open{transform:translateX(0)}
  .sidebar.open+.overlay{display:block}
  .header{display:flex}
}
/* Bubble action buttons */
.bubble-actions{display:flex;gap:8px;margin-top:6px;font-size:11px;justify-content:flex-end;max-width:85%}
.bubble-actions.user{align-self:flex-end}
.bubble-actions.ai{align-self:flex-start}
.bubble-actions a{color:var(--sub);text-decoration:none;cursor:pointer}
.bubble-actions a:active{color:var(--text)}
/* Starfield overlay - background layer behind UI */
#starfield-overlay{display:none;position:fixed;inset:0;z-index:-1;transition:opacity 1s ease}
#starfield-overlay canvas{display:block;width:100%;height:100%}
/* Starfield mode: semi-transparent UI over stars */
body.starfield-mode,body.starfield-mode .main{background:transparent!important}
body.starfield-mode .sidebar{background:rgba(15,15,25,0.75);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
body.starfield-mode .sb-header{background:transparent}
body.starfield-mode .messages{background:transparent}
body.starfield-mode .welcome .subtitle{color:rgba(255,255,255,0.7)}
body.starfield-mode .input-bar{background:rgba(15,15,25,0.7);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
body.starfield-mode .input-bar textarea{background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.15);color:#fff}
body.starfield-mode .input-bar textarea::placeholder{color:rgba(255,255,255,0.3)}
body.starfield-mode .bubble.ai{background:rgba(30,30,50,0.6);border-color:rgba(255,255,255,0.1);color:rgba(255,255,255,0.9);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px)}
body.starfield-mode .bubble.user{background:rgba(99,102,241,0.5)}
body.starfield-mode .header{background:rgba(15,15,25,0.5);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
body.starfield-mode .btn-new{background:rgba(255,255,255,0.06);color:#ccc}
body.starfield-mode .session-item{background:rgba(255,255,255,0.04);color:#ccc}
body.starfield-mode .session-item.active{background:rgba(99,102,241,0.2)}
body.starfield-mode .clear-btn{color:rgba(255,255,255,0.5)}
body.starfield-mode .session-item .del-btn{color:rgba(255,255,255,0.3)}
body.starfield-mode .bubble-actions a{color:rgba(255,255,255,0.5)}
</style>
</head>
<body>
<div class="sidebar" id="sidebar">
  <div class="sb-header"><h2><div class="logo-tap" id="logo-tap"><img src="/logo.png" alt="">TGAI</div></h2></div>
  <div class="sb-model" style="padding:8px 12px">
    <select id="modelSelect" onchange="switchModel(this.value)" style="width:100%;padding:6px 8px;border-radius:6px;background:var(--bg2);color:var(--text);border:1px solid var(--border);font-size:13px;cursor:pointer">
    </select>
    <div style="display:none">
      <div style="font-size:11px;color:var(--sub);margin:4px 0 2px">推理引擎</div>
      <select id="engineSelect" style="width:100%;padding:6px 8px;border-radius:6px;background:var(--bg2);color:var(--text);border:1px solid var(--border);font-size:13px;cursor:pointer">
        <option value="tgai_go" selected>TGAI GO (C引擎)</option>
      </select>
    </div>
    <div id="toggleSysPrompt" onclick="toggleSysPrompt()" style="margin-top:6px;text-align:center">+ 系统提示词</div>
  </div>
  <button class="btn-new" onclick="newSession()">+ 新对话</button>
  <div class="session-list" id="sessionList"></div>
</div>
<div class="overlay" id="overlay" onclick="toggleSidebar()"></div>
<div class="main">
  <div class="header">
    <div class="left">
      <button class="menu-btn" onclick="toggleSidebar()">::</button>
      <h1><img src="/logo.png" alt="">TGAI</h1>
    </div>
    <button class="clear-btn" onclick="clearCurrentChat()">清空</button>
  </div>
  <div class="messages" id="messages">
    <div class="welcome">
      <div class="avatar"><img src="/logo.png" alt="TGAI"></div>
      <div class="hello">HELLO WORLD</div>
      <div class="subtitle">心怀希望与梦想，向这世界致意</div>
    </div>
  </div>
  <div class="sys-prompt-bar hidden" id="sysPromptBar">
    <label>系统:</label>
    <input type="text" id="sysPromptInput" placeholder="系统提示词（默认: 你是TGAI，一个由JXW独立开发的AI助手。）" value="你是TGAI，一个由JXW独立开发的AI助手。" />
  </div>
  <div class="input-bar">
    <textarea id="input" rows="1" placeholder="输入消息..." onkeydown="onKey(event)"></textarea>
    <button class="send" id="sendBtn" onclick="send()">&#8593;</button>
  </div>
</div>
<div class="ctx-menu" id="ctxMenu">
  <button onclick="ctxBack()">回溯到此处</button>
  <button onclick="ctxRegen()" id="ctxRegenBtn">重新生成</button>
</div>
<div id="starfield-overlay"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const STORAGE_KEY='tgai_sessions';
const msgs=document.getElementById('messages');
const input=document.getElementById('input');
const sendBtn=document.getElementById('sendBtn');
const sessionList=document.getElementById('sessionList');
const sidebar=document.getElementById('sidebar');
let sending=false,currentSessionId=null,sessions={};
function loadSessions(){try{sessions=JSON.parse(localStorage.getItem(STORAGE_KEY))||{}}catch(e){sessions={}}}
function saveSessions(){localStorage.setItem(STORAGE_KEY,JSON.stringify(sessions))}
function newSession(){const id='s'+Date.now();sessions[id]={title:'HELLO WORLD',messages:[],created:Date.now()};saveSessions();switchSession(id);if(window.innerWidth<=640)toggleSidebar()}
function deleteSession(id){if(!confirm('删除这个对话？'))return;delete sessions[id];saveSessions();if(currentSessionId===id){const keys=Object.keys(sessions);if(keys.length){switchSession(keys[keys.length-1]);return}currentSessionId=null;showWelcome()}renderSessions()}
function clearCurrentChat(){if(!currentSessionId||!sessions[currentSessionId])return;if(!confirm('清空当前对话？'))return;sessions[currentSessionId].messages=[];sessions[currentSessionId].title='HELLO WORLD';saveSessions();switchSession(currentSessionId)}
function switchSession(id){currentSessionId=id;const s=sessions[id];if(!s)return;msgs.innerHTML='';if(!s.messages.length){showWelcome(s.title)}else{s.messages.forEach(m=>{const info=m.model_name?{name:m.model_name,elapsed:m.elapsed,engine:m.engine}:null;addBubble(m.role,m.content,false,info)})}scrollDown();renderSessions();input.focus()}
function showWelcome(title){const t=title||'HELLO WORLD';msgs.innerHTML='<div class="welcome"><div class="avatar"><img src="/logo.png" alt="TGAI"></div><div class="hello">'+escapeHtml(t)+'</div><div class="subtitle">心怀希望与梦想，向这世界致意</div></div>'}
function renderSessions(){const ids=Object.keys(sessions).sort((a,b)=>sessions[b].created-sessions[a].created);if(!ids.length){sessionList.innerHTML='<div style="padding:20px;text-align:center;color:var(--sub);font-size:13px">暂无对话</div>';return}sessionList.innerHTML=ids.map(id=>{const s=sessions[id];const active=id===currentSessionId?' active':'';const count=s.messages.length;return '<div class="session-item'+active+'" onclick="switchSession(\''+id+'\')"><span class="title">'+escapeHtml(s.title)+'</span>'+(count?'<span class="count">'+count+'</span>':'')+'<button class="del-btn" onclick="event.stopPropagation();deleteSession(\''+id+'\')">x</button></div>'}).join('')}
function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function renderMd(s){s=escapeHtml(s);s=s.replace(/^### (.+)$/gm,'<h4>$1</h4>');s=s.replace(/^## (.+)$/gm,'<h3>$1</h3>');s=s.replace(/^# (.+)$/gm,'<h2>$1</h2>');s=s.replace(/^---$/gm,'<hr>');s=s.replace(/^- (.+)$/gm,'<li>$1</li>');s=s.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');s=s.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');s=s.replace(/\*(.+?)\*/g,'<i>$1</i>');s=s.replace(/`([^`]+)`/g,'<code>$1</code>');s=s.replace(/^&gt;\s?(.*)$/gm,'<blockquote>$1</blockquote>');return s}
function toggleSidebar(){sidebar.classList.toggle('open');if(sidebar.classList.contains('open')&&Object.keys(sessions).length===0)newSession()}
function addBubble(role,text,animate=true,info=null){const d=document.createElement('div');d.className='bubble '+role;if(animate)d.style.animation='fadeIn .2s ease';else d.style.animation='none';d.innerHTML=renderMd(text);msgs.appendChild(d);const act=document.createElement('div');act.className='bubble-actions '+role;let infoHtml='';if(info&&info.name){const eng=info.engine?' ['+info.engine+']':'';infoHtml='<span style="color:var(--sub);font-size:10px;opacity:0.7">（'+escapeHtml(info.name)+eng+' '+info.elapsed+'s）</span> '}act.innerHTML=infoHtml+'<a onclick="copyBubble(this.parentElement.previousElementSibling)">[复制]</a>'+(role==='ai'?' <a onclick="backtrackHere(this.parentElement.previousElementSibling)">[回溯]</a> <a onclick="regenHere(this.parentElement.previousElementSibling)">[重新生成]</a>':'');msgs.appendChild(act);scrollDown();return d}
function addTyping(){const d=document.createElement('div');d.className='bubble ai';d.id='typing-bubble';d.innerHTML='<div class="typing"><span></span><span></span><span></span></div>';msgs.appendChild(d);scrollDown();return d}
function updateBubbleInfo(bubble,info){if(!bubble||!info||!info.name)return;const act=bubble.nextElementSibling;const eng=info.engine?' ['+info.engine+']':'';if(act&&act.classList.contains('bubble-actions')){const span=act.querySelector('.model-info');if(span){span.innerHTML='&nbsp;<span style="font-size:10px;opacity:0.5">'+info.name+' '+eng+'&nbsp;'+info.elapsed+'s</span>'}else{const s=document.createElement('span');s.className='model-info';s.innerHTML='&nbsp;<span style="font-size:10px;opacity:0.5">'+info.name+' '+eng+'&nbsp;'+info.elapsed+'s</span>';act.insertBefore(s,act.firstChild)}}}
function scrollDown(){setTimeout(()=>{msgs.scrollTop=msgs.scrollHeight},50)}
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}}
let abortController=null;

input.addEventListener('input',function(){this.style.height='auto';this.style.height=Math.min(this.scrollHeight,100)+'px'})
function stopGen(){if(abortController){abortController.abort();abortController=null}const tb=document.getElementById('typing-bubble');if(tb)tb.remove();sendBtn.innerHTML='&#8593;';sendBtn.classList.remove('stop');sendBtn.onclick=send;sending=false;input.focus()}
async function send(){
  if(sending){stopGen();return}
  const m=input.value.trim();if(!m)return;sending=true;abortController=new AbortController();
  sendBtn.innerHTML='&#9632;';sendBtn.classList.add('stop');sendBtn.onclick=stopGen;
  if(!currentSessionId||!sessions[currentSessionId])newSession();
  const s=sessions[currentSessionId];
  if(!s.messages.length)s.title=m.length>20?m.slice(0,20)+'...':m;
  const history=s.messages.map(x=>({role:x.role,content:x.content}));
  addBubble('user',m);s.messages.push({role:'user',content:m});renderSessions();saveSessions();
  input.value='';input.focus();input.style.height='auto';
  const typing=addTyping();
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m,history:history,system_prompt:document.getElementById('sysPromptInput')?.value||'',model_id:document.getElementById('modelSelect')?.value||'',engine:'tgai_go'}),signal:abortController.signal});
    typing.remove();const aiBubble=addBubble('ai','');const reader=res.body.getReader();const decoder=new TextDecoder();let full='';let modelInfo=null;
    while(true){const{done,value}=await reader.read();if(done)break;const lines=decoder.decode(value,{stream:true}).split('\n');
      for(const l of lines){if(l.startsWith('data: ')){const d=l.slice(6);if(d==='[DONE]'){const mi=modelInfo||{};s.messages.push({role:'assistant',content:full,model_name:mi.name||'',elapsed:mi.elapsed||0,engine:mi.engine||'TGAI GO'});updateBubbleInfo(aiBubble,mi);renderSessions();saveSessions();return}
        try{const j=JSON.parse(d);if(j.token){full+=j.token;aiBubble.innerHTML=renderMd(full);scrollDown()}else if(j.model_name){modelInfo={name:j.model_name,elapsed:j.elapsed,engine:j.engine||'TGAI GO'}}}catch(e){}}}}
  }catch(e){if(e.name!=='AbortError'){typing.remove();addBubble('ai','连接失败: '+e.message);s.messages.push({role:'assistant',content:'[错误] '+e.message});saveSessions()}}
  finally{sending=false;abortController=null;sendBtn.innerHTML='&#8593;';sendBtn.classList.remove('stop');sendBtn.onclick=send;input.focus()}
}
// ── 复制 ──
function copyBubble(bubble){
  if(!bubble)return;
  const text=bubble.innerText||bubble.textContent;
  if(navigator.clipboard){navigator.clipboard.writeText(text).catch(()=>fallbackCopy(text))}
  else{fallbackCopy(text)}
  function fallbackCopy(t){const ta=document.createElement('textarea');ta.value=t;ta.style.position='fixed';ta.style.left='-9999px';document.body.appendChild(ta);ta.select();document.execCommand('copy');document.body.removeChild(ta)}
}
// ── 回溯到此处 ──
function backtrackHere(bubble){
  if(!bubble||!currentSessionId||!sessions[currentSessionId])return;
  const bubbles=msgs.querySelectorAll('.bubble');
  let bubbleIdx=0;for(let i=0;i<bubbles.length;i++){if(bubbles[i]===bubble){bubbleIdx=i;break}}
  const s=sessions[currentSessionId];
  s.messages=s.messages.slice(0,bubbleIdx);
  saveSessions();renderSessions();switchSession(currentSessionId);
}
// ── 重新生成 ──
function regenHere(bubble){
  if(!bubble||!currentSessionId||!sessions[currentSessionId])return;
  const bubbles=msgs.querySelectorAll('.bubble');
  let bubbleIdx=0;for(let i=0;i<bubbles.length;i++){if(bubbles[i]===bubble){bubbleIdx=i;break}}
  const s=sessions[currentSessionId];
  s.messages=s.messages.slice(0,bubbleIdx);
  saveSessions();renderSessions();switchSession(currentSessionId);
  if(s.messages.length>0&&s.messages[s.messages.length-1].role==='user'){
    const last=s.messages.pop();
    saveSessions();renderSessions();switchSession(currentSessionId);
    input.value=last.content;input.focus();send();
  }
}
// ── 长按菜单（保留兼容）──
let ctxTarget=null,ctxTimer=null;
const ctxMenu=document.getElementById('ctxMenu'),ctxRegenBtn=document.getElementById('ctxRegenBtn');
function hideCtx(){ctxMenu.style.display='none';ctxTarget=null}
document.addEventListener('click',e=>{if(!ctxMenu.contains(e.target))hideCtx()});
msgs.addEventListener('touchstart',e=>{
  const b=e.target.closest('.bubble');if(!b)return;
  ctxTarget=b;clearTimeout(ctxTimer);
  ctxTimer=setTimeout(()=>{
    const idx=Array.from(msgs.children).indexOf(b);
    const isAi=b.classList.contains('ai');
    ctxRegenBtn.style.display=isAi?'block':'none';
    const r=b.getBoundingClientRect();
    ctxMenu.style.display='block';
    ctxMenu.style.left=Math.min(r.left,window.innerWidth-130)+'px';
    ctxMenu.style.top=Math.max(r.top-80,10)+'px';
    b.setAttribute('data-idx',idx);
  },600);
},{passive:true});
msgs.addEventListener('touchend',()=>clearTimeout(ctxTimer));
msgs.addEventListener('touchmove',()=>clearTimeout(ctxTimer));
function ctxBack(){
  if(!ctxTarget||!currentSessionId||!sessions[currentSessionId])return;
  const idx=parseInt(ctxTarget.getAttribute('data-idx'));
  if(isNaN(idx))return;
  const s=sessions[currentSessionId];
  const bubbles=msgs.querySelectorAll('.bubble');
  let bubbleIdx=0;for(let i=0;i<bubbles.length;i++){if(bubbles[i]===ctxTarget){bubbleIdx=i;break}}
  s.messages=s.messages.slice(0,bubbleIdx);
  saveSessions();renderSessions();switchSession(currentSessionId);hideCtx();
}
function ctxRegen(){
  if(!ctxTarget||!currentSessionId||!sessions[currentSessionId])return;
  const idx=parseInt(ctxTarget.getAttribute('data-idx'));
  if(isNaN(idx))return;
  const s=sessions[currentSessionId];
  const bubbles=msgs.querySelectorAll('.bubble');
  let bubbleIdx=0;for(let i=0;i<bubbles.length;i++){if(bubbles[i]===ctxTarget){bubbleIdx=i;break}}
  s.messages=s.messages.slice(0,bubbleIdx);
  saveSessions();renderSessions();switchSession(currentSessionId);hideCtx();
  if(s.messages.length>0&&s.messages[s.messages.length-1].role==='user'){
    const last=s.messages.pop();
    saveSessions();renderSessions();switchSession(currentSessionId);
    input.value=last.content;input.focus();send();
  }
}
// ── 清理 streaming 中残留的 bubble-actions ──
const bubbleActionsObserver=new MutationObserver(()=>{
  document.querySelectorAll('#typing-bubble+.bubble-actions').forEach(a=>a.remove());
});
bubbleActionsObserver.observe(msgs,{childList:true});
// ═══════════════════ THREE.JS STARFIELD EASTER EGG ═══════════════════
(function(){
  if(typeof THREE==='undefined')return;
  if(!('ontouchstart' in window))return;
  const overlay=document.getElementById('starfield-overlay');
  if(!overlay)return;

  let sfActive=false,sfScene=null,sfCamera=null,sfRenderer=null;
  let sfRafId=null,sfInactivityTimer=null,sfInactivityTimeout=30000;
  let sfExiting=false,sfEntering=false,sfExitTimer=null;
  let sfAllParticles=[],sfDust=[],sfSparks=[],sfNebulae=[];
  let sfTrailParticles=[]; // stardust trail from swipe
  let sfBurstParticles=[]; // tap burst sparks
  let sfDriftSpeed=8;     // base auto-drift speed (units/sec)
  let sfHyperspace=false; // long-press speed boost

  function finishDispose(){
    if(sfRafId){cancelAnimationFrame(sfRafId);sfRafId=null}
    document.body.classList.remove('starfield-mode');
    if(sfRenderer){sfRenderer.dispose();sfRenderer=null}
    if(sfScene){
      sfScene.traverse(obj=>{if(obj.geometry)obj.geometry.dispose();if(obj.material){if(Array.isArray(obj.material))obj.material.forEach(m=>m.dispose());else obj.material.dispose()}});
      sfScene=null;
    }
    sfCamera=null;sfAllParticles=[];sfDust=[];sfSparks=[];sfNebulae=[];
    sfTrailParticles=[];sfBurstParticles=[];
    overlay.style.opacity='0';
    setTimeout(()=>{
      overlay.style.display='none';overlay.innerHTML='';
      const wImg=document.querySelector('.welcome .avatar img');
      if(wImg)wImg.style.opacity='1';
    },800);
    sfActive=false;sfExiting=false;sfEntering=false;sfHyperspace=false;
  }

  function disposeStarfield(){
    if(sfExitTimer)clearTimeout(sfExitTimer);
    if(sfScene&&(sfAllParticles.length||sfDust.length)&&!sfExiting){
      sfExiting=true;
      startConvergeToHello();
      return;
    }
    finishDispose();
  }

  function resetInactivityTimer(){
    if(sfInactivityTimer)clearTimeout(sfInactivityTimer);
    if(sfActive)sfInactivityTimer=setTimeout(()=>{if(sfActive)disposeStarfield()},sfInactivityTimeout);
  }

  // ── Capture welcome area pixels ──
  function captureWelcomePixels(){
    const welcome=document.querySelector('.welcome');
    if(!welcome)return null;
    const hello=welcome.querySelector('.hello');
    const img=welcome.querySelector('.avatar img');
    if(!hello&&!img)return null;
    const rect=welcome.getBoundingClientRect();
    const canvas=document.createElement('canvas');
    const scale=2;
    canvas.width=Math.ceil(rect.width*scale);
    canvas.height=Math.ceil(rect.height*scale);
    const ctx=canvas.getContext('2d');
    ctx.scale(scale,scale);
    if(hello){
      const hr=hello.getBoundingClientRect();
      const gx=hr.left-rect.left,gy=hr.top-rect.top;
      const grad=ctx.createLinearGradient(gx,0,gx+hr.width,0);
      grad.addColorStop(0,'#6366f1');grad.addColorStop(0.33,'#8b5cf6');
      grad.addColorStop(0.66,'#ec4899');grad.addColorStop(1,'#f43f5e');
      ctx.font='bold 28px -apple-system,BlinkMacSystemFont,sans-serif';
      ctx.letterSpacing='2px';
      ctx.fillStyle=grad;
      ctx.fillText('HELLO WORLD',gx,gy+hr.height-2);
    }
    if(img){
      const ir=img.getBoundingClientRect();
      const ix=ir.left-rect.left,iy=ir.top-rect.top;
      ctx.fillStyle='#6366f1';
      ctx.beginPath();
      ctx.arc(ix+ir.width/2,iy+ir.height/2,ir.width/2+2,0,Math.PI*2);
      ctx.fill();
    }
    const imgData=ctx.getImageData(0,0,canvas.width,canvas.height);
    const pixels=[];
    for(let y=0;y<imgData.height;y+=2){
      for(let x=0;x<imgData.width;x+=2){
        const idx=(y*imgData.width+x)*4;
        if(imgData.data[idx+3]>30){
          const fx=(x/scale-rect.width/2)/10;
          const fy=-(y/scale-rect.height/2)/10;
          pixels.push({x:fx,y:fy,z:-3,origX:fx,origY:fy,origZ:-3,
            r:imgData.data[idx]/255,g:imgData.data[idx+1]/255,b:imgData.data[idx+2]/255});
        }
      }
    }
    return pixels.length>20?pixels:null;
  }

  // ── Converge to HELLO WORLD ──
  function startConvergeToHello(){
    sfRafId=null;
    sfDriftSpeed=0;
    const helloPixels=captureWelcomePixels();
    const targetPixels=helloPixels||[];
    const allParts=[...sfAllParticles,...sfDust];
    allParts.forEach((p,i)=>{
      if(targetPixels.length){
        const t=targetPixels[i%targetPixels.length];
        p.targetX=t.x;p.targetY=t.y;p.targetZ=t.z;
      }else{
        p.targetX=0;p.targetY=0;p.targetZ=-3;
      }
      p.speed=0.02+Math.random()*0.04;
    });
    let convergeProgress=0;
    const convergeDuration=2000;
    let lastT=performance.now();
    function convergeAnim(now){
      if(!sfActive||!sfScene)return;
      const dt=Math.min((now-lastT)/1000,0.1);lastT=now;
      convergeProgress+=dt/(convergeDuration/1000);
      let allDone=true;
      allParts.forEach(p=>{
        if(!p.pointsObj||!p.pointsObj.geometry)return;
        const dx=p.targetX-p.x,dy=p.targetY-p.y,dz=p.targetZ-p.z;
        const dist=Math.sqrt(dx*dx+dy*dy+dz*dz);
        if(dist>0.05){allDone=false;
          p.x+=dx*p.speed;p.y+=dy*p.speed;p.z+=dz*p.speed;
        }
        const arr=p.pointsObj.geometry.attributes.position.array;
        arr[p.idx*3]=p.x;arr[p.idx*3+1]=p.y;arr[p.idx*3+2]=p.z;
        p.pointsObj.geometry.attributes.position.needsUpdate=true;
      });
      if(sfRenderer&&sfScene&&sfCamera)sfRenderer.render(sfScene,sfCamera);
      if(allDone&&convergeProgress>=1||convergeProgress>2.0)finishDispose();
      else sfRafId=requestAnimationFrame(convergeAnim);
    }
    sfRafId=requestAnimationFrame(convergeAnim);
  }

  // ── Spawn a burst of colored spark particles at screen position ──
  function spawnBurst(screenX,screenY){
    if(!sfScene||!sfCamera||!sfRenderer)return;
    const W=overlay.clientWidth,H=overlay.clientHeight;
    // Project screen coords to a point in front of camera
    const ndc=new THREE.Vector3((screenX/W)*2-1,-(screenY/H)*2+1,0.5);
    ndc.unproject(sfCamera);
    const camPos=sfCamera.position.clone();
    const dir=ndc.sub(camPos).normalize();
    const burstOrigin=camPos.clone().add(dir.clone().multiplyScalar(8));
    // Create burst geometry
    const count=40+Math.floor(Math.random()*30);
    const geo=new THREE.BufferGeometry();
    const positions=new Float32Array(count*3);
    const colors=new Float32Array(count*3);
    const hue=Math.random();
    for(let i=0;i<count;i++){
      positions[i*3]=burstOrigin.x+(Math.random()-0.5)*3;
      positions[i*3+1]=burstOrigin.y+(Math.random()-0.5)*3;
      positions[i*3+2]=burstOrigin.z+(Math.random()-0.5)*3;
      const c=new THREE.Color().setHSL(hue+Math.random()*0.15,0.8,0.7+Math.random()*0.3);
      colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;
    }
    geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const mat=new THREE.PointsMaterial({size:0.3,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:1});
    const points=new THREE.Points(geo,mat);
    sfScene.add(points);
    const burst={
      points,lifetime:1.2,age:0,
      velocities:Array.from({length:count},()=>({
        x:(Math.random()-0.5)*8,y:(Math.random()-0.5)*8,z:(Math.random()-0.5)*8
      }))
    };
    sfBurstParticles.push(burst);
  }

  // ── Spawn stardust trail particles at screen position ──
  function spawnTrail(screenX,screenY){
    if(!sfScene||!sfCamera||!sfRenderer)return;
    const W=overlay.clientWidth,H=overlay.clientHeight;
    const ndc=new THREE.Vector3((screenX/W)*2-1,-(screenY/H)*2+1,0.5);
    ndc.unproject(sfCamera);
    const camPos=sfCamera.position.clone();
    const dir=ndc.sub(camPos).normalize();
    const origin=camPos.clone().add(dir.clone().multiplyScalar(6));
    const count=8+Math.floor(Math.random()*6);
    const geo=new THREE.BufferGeometry();
    const positions=new Float32Array(count*3);
    const colors=new Float32Array(count*3);
    const hue=0.55+Math.random()*0.2;
    for(let i=0;i<count;i++){
      positions[i*3]=origin.x+(Math.random()-0.5)*1.5;
      positions[i*3+1]=origin.y+(Math.random()-0.5)*1.5;
      positions[i*3+2]=origin.z+(Math.random()-0.5)*1.5;
      const c=new THREE.Color().setHSL(hue+Math.random()*0.1,0.6,0.6+Math.random()*0.4);
      colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;
    }
    geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const mat=new THREE.PointsMaterial({size:0.15,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.8});
    const points=new THREE.Points(geo,mat);
    sfScene.add(points);
    sfTrailParticles.push({points,lifetime:0.8,age:0});
  }

  function initStarfield(){
    if(sfActive)return;
    sfActive=true;
    const wImg=document.querySelector('.welcome .avatar img');
    if(wImg)wImg.style.opacity='0';
    document.body.classList.add('starfield-mode');
    overlay.style.display='block';
    overlay.style.opacity='1';
    overlay.innerHTML='';
    const W=overlay.clientWidth,H=overlay.clientHeight;

    sfRenderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    sfRenderer.setClearColor(0x020010,1);
    sfRenderer.setSize(W,H);
    sfRenderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    overlay.appendChild(sfRenderer.domElement);

    sfScene=new THREE.Scene();
    sfScene.background=new THREE.Color(0x020010);
    sfScene.fog=new THREE.FogExp2(0x020010,0.00003); // softer fog for wider view

    // Wider FOV (80°) and pulled back camera for broader view
    sfCamera=new THREE.PerspectiveCamera(80,W/H,0.1,2000);
    sfCamera.position.set(0,0,12);
    sfCamera.lookAt(0,0,-50);

    const helloPixels=captureWelcomePixels()||[];
    sfAllParticles=[];sfDust=[];sfSparks=[];sfNebulae=[];
    sfTrailParticles=[];sfBurstParticles=[];

    // ── LAYER 1: Tiny dust (4000 particles) ──
    const dustGeo=new THREE.BufferGeometry();
    const dustCount=4000;
    const dustPositions=new Float32Array(dustCount*3);
    const dustColors=new Float32Array(dustCount*3);
    for(let i=0;i<dustCount;i++){
      const r=30+Math.random()*200;
      const theta=Math.random()*Math.PI*2;
      const phi=Math.acos(2*Math.random()-1);
      const tx=r*Math.sin(phi)*Math.cos(theta);
      const ty=r*Math.sin(phi)*Math.sin(theta);
      const tz=r*Math.cos(phi);
      if(helloPixels.length){
        const hp=helloPixels[i%helloPixels.length];
        dustPositions[i*3]=hp.x+(Math.random()-0.5)*0.5;
        dustPositions[i*3+1]=hp.y+(Math.random()-0.5)*0.5;
        dustPositions[i*3+2]=hp.z-1+Math.random()*0.5;
      }else{
        dustPositions[i*3]=tx;dustPositions[i*3+1]=ty;dustPositions[i*3+2]=tz;
      }
      const hue=0.55+Math.random()*0.2;
      const c=new THREE.Color().setHSL(hue,0.3+Math.random()*0.4,0.3+Math.random()*0.5);
      dustColors[i*3]=c.r;dustColors[i*3+1]=c.g;dustColors[i*3+2]=c.b;
      sfDust.push({
        x:dustPositions[i*3],y:dustPositions[i*3+1],z:dustPositions[i*3+2],
        targetX:tx,targetY:ty,targetZ:tz, r:c.r,g:c.g,b:c.b,
        idx:i,pointsObj:null,
        twinklePhase:Math.random()*Math.PI*2,twinkleSpeed:0.3+Math.random()*1.5,
        baseOpacity:0.3+Math.random()*0.5,origX:0,origY:0,origZ:0
      });
    }
    dustGeo.setAttribute('position',new THREE.BufferAttribute(dustPositions,3));
    dustGeo.setAttribute('color',new THREE.BufferAttribute(dustColors,3));
    const dustMat=new THREE.PointsMaterial({size:0.08,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.7,sizeAttenuation:true});
    const dustPoints=new THREE.Points(dustGeo,dustMat);
    sfScene.add(dustPoints);
    sfDust.forEach(p=>p.pointsObj=dustPoints);

    // ── LAYER 2: Main stars (2500 particles) ──
    const starGeo=new THREE.BufferGeometry();
    const starCount=2500;
    const starPositions=new Float32Array(starCount*3);
    const starColors=new Float32Array(starCount*3);
    for(let i=0;i<starCount;i++){
      const r=25+Math.random()*180;
      const theta=Math.random()*Math.PI*2;
      const phi=Math.acos(2*Math.random()-1);
      const tx=r*Math.sin(phi)*Math.cos(theta);
      const ty=r*Math.sin(phi)*Math.sin(theta);
      const tz=r*Math.cos(phi);
      if(helloPixels.length){
        const hp=helloPixels[i%helloPixels.length];
        starPositions[i*3]=hp.x;starPositions[i*3+1]=hp.y;starPositions[i*3+2]=hp.z-1;
      }else{
        starPositions[i*3]=tx;starPositions[i*3+1]=ty;starPositions[i*3+2]=tz;
      }
      const hue=Math.random()<0.5?(0.08+Math.random()*0.12):(0.55+Math.random()*0.15);
      const c=new THREE.Color().setHSL(hue,0.6+Math.random()*0.4,0.6+Math.random()*0.4);
      starColors[i*3]=c.r;starColors[i*3+1]=c.g;starColors[i*3+2]=c.b;
      sfAllParticles.push({
        x:starPositions[i*3],y:starPositions[i*3+1],z:starPositions[i*3+2],
        targetX:tx,targetY:ty,targetZ:tz, r:c.r,g:c.g,b:c.b,
        idx:i,pointsObj:null,
        twinklePhase:Math.random()*Math.PI*2,twinkleSpeed:0.5+Math.random()*2.5,
        baseOpacity:0.6+Math.random()*0.4,origX:0,origY:0,origZ:0
      });
    }
    starGeo.setAttribute('position',new THREE.BufferAttribute(starPositions,3));
    starGeo.setAttribute('color',new THREE.BufferAttribute(starColors,3));
    const starMat=new THREE.PointsMaterial({size:0.2,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.85,sizeAttenuation:true});
    const stars=new THREE.Points(starGeo,starMat);
    sfScene.add(stars);
    sfAllParticles.forEach(p=>p.pointsObj=stars);

    // ── LAYER 3: Bright sparks (500 particles) ──
    const sparkGeo=new THREE.BufferGeometry();
    const sparkCount=500;
    const sparkPositions=new Float32Array(sparkCount*3);
    const sparkColors=new Float32Array(sparkCount*3);
    for(let i=0;i<sparkCount;i++){
      const r=20+Math.random()*150;
      const theta=Math.random()*Math.PI*2;
      const phi=Math.acos(2*Math.random()-1);
      const tx=r*Math.sin(phi)*Math.cos(theta);
      const ty=r*Math.sin(phi)*Math.sin(theta);
      const tz=r*Math.cos(phi);
      if(helloPixels.length){
        const hp=helloPixels[i%helloPixels.length];
        sparkPositions[i*3]=hp.x+(Math.random()-0.5)*0.3;
        sparkPositions[i*3+1]=hp.y+(Math.random()-0.5)*0.3;
        sparkPositions[i*3+2]=hp.z;
      }else{
        sparkPositions[i*3]=tx;sparkPositions[i*3+1]=ty;sparkPositions[i*3+2]=tz;
      }
      const hueChoice=Math.random();
      const hue=hueChoice<0.25?0.12:hueChoice<0.5?0.6:hueChoice<0.75?0.8:0.05;
      const c=new THREE.Color().setHSL(hue,0.8,0.8+Math.random()*0.2);
      sparkColors[i*3]=c.r;sparkColors[i*3+1]=c.g;sparkColors[i*3+2]=c.b;
      sfSparks.push({
        x:sparkPositions[i*3],y:sparkPositions[i*3+1],z:sparkPositions[i*3+2],
        targetX:tx,targetY:ty,targetZ:tz, r:c.r,g:c.g,b:c.b,
        idx:i,pointsObj:null,
        twinklePhase:Math.random()*Math.PI*2,twinkleSpeed:1.5+Math.random()*3,
        baseOpacity:0.7+Math.random()*0.3,origX:0,origY:0,origZ:0
      });
    }
    sparkGeo.setAttribute('position',new THREE.BufferAttribute(sparkPositions,3));
    sparkGeo.setAttribute('color',new THREE.BufferAttribute(sparkColors,3));
    const sparkMat=new THREE.PointsMaterial({size:0.5,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.75,sizeAttenuation:true});
    const sparkPoints=new THREE.Points(sparkGeo,sparkMat);
    sfScene.add(sparkPoints);
    sfSparks.forEach(p=>p.pointsObj=sparkPoints);

    // ── LAYER 4: Nebula wisps (10 clusters, 250 particles each) ──
    const nebulaHues=[0.85,0.75,0.15,0.58,0.92,0.68,0.08,0.82,0.95,0.5];
    const nebulaClusters=[];
    for(let n=0;n<10;n++){
      const neCount=250;
      const neGeo=new THREE.BufferGeometry();
      const nePositions=new Float32Array(neCount*3);
      const neColors=new Float32Array(neCount*3);
      const cx=(Math.random()-0.5)*60,cy=(Math.random()-0.5)*45,cz=-20-Math.random()*90;
      const spread=8+Math.random()*22;
      for(let i=0;i<neCount;i++){
        const gx=(Math.random()+Math.random()+Math.random())/3-0.5;
        const gy=(Math.random()+Math.random()+Math.random())/3-0.5;
        const gz=(Math.random()+Math.random()+Math.random())/3-0.5;
        nePositions[i*3]=cx+gx*spread*2;
        nePositions[i*3+1]=cy+gy*spread*2;
        nePositions[i*3+2]=cz+gz*spread*2;
        const hue=nebulaHues[n]+(Math.random()-0.5)*0.08;
        const c=new THREE.Color().setHSL(hue,0.5+Math.random()*0.4,0.5+Math.random()*0.4);
        neColors[i*3]=c.r;neColors[i*3+1]=c.g;neColors[i*3+2]=c.b;
      }
      neGeo.setAttribute('position',new THREE.BufferAttribute(nePositions,3));
      neGeo.setAttribute('color',new THREE.BufferAttribute(neColors,3));
      const neMat=new THREE.PointsMaterial({size:0.6+Math.random()*1.0,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.12+Math.random()*0.15});
      const ne=new THREE.Points(neGeo,neMat);
      ne.userData={speedZ:0.005+Math.random()*0.02,speedX:(Math.random()-0.5)*0.004,speedY:(Math.random()-0.5)*0.003};
      sfScene.add(ne);
      nebulaClusters.push(ne);
      sfNebulae.push(ne);
    }

    // ── Intro animation ──
    if(helloPixels.length>0){
      sfEntering=true;
      [...sfAllParticles,...sfDust,...sfSparks].forEach(p=>{p.origX=p.x;p.origY=p.y;p.origZ=p.z});
      let introProgress=0;
      const introDuration=1800;
      let introLast=performance.now();
      function introAnim(now){
        if(!sfActive||sfExiting)return;
        const dt=Math.min((now-introLast)/1000,0.1);introLast=now;
        introProgress+=dt/(introDuration/1000);
        const t=Math.min(introProgress,1);
        const ease=1-Math.pow(1-t,3);
        const allParts=[...sfAllParticles,...sfDust,...sfSparks];
        allParts.forEach(p=>{
          if(!p.pointsObj||!p.pointsObj.geometry)return;
          const dx=p.targetX-p.origX,dy=p.targetY-p.origY,dz=p.targetZ-p.origZ;
          p.x=p.origX+dx*ease;p.y=p.origY+dy*ease;p.z=p.origZ+dz*ease;
          const arr=p.pointsObj.geometry.attributes.position.array;
          arr[p.idx*3]=p.x;arr[p.idx*3+1]=p.y;arr[p.idx*3+2]=p.z;
          p.pointsObj.geometry.attributes.position.needsUpdate=true;
        });
        if(sfRenderer&&sfScene&&sfCamera)sfRenderer.render(sfScene,sfCamera);
        if(t>=1){
          sfEntering=false;
          [...sfAllParticles,...sfDust,...sfSparks].forEach(p=>{
            if(!p.pointsObj||!p.pointsObj.geometry)return;
            p.x=p.targetX;p.y=p.targetY;p.z=p.targetZ;
            const arr=p.pointsObj.geometry.attributes.position.array;
            arr[p.idx*3]=p.x;arr[p.idx*3+1]=p.y;arr[p.idx*3+2]=p.z;
            p.pointsObj.geometry.attributes.position.needsUpdate=true;
          });
          lastFrame=performance.now();
          sfRafId=requestAnimationFrame(animate);
        }else{sfRafId=requestAnimationFrame(introAnim);}
      }
      setTimeout(()=>{sfRafId=requestAnimationFrame(introAnim)},50);
    }else{
      lastFrame=performance.now();
      sfRafId=requestAnimationFrame(animate);
    }

    // ── Touch interaction: steer / burst / hyperspace ──
    let touchStartX=0,touchStartY=0,touchActive=false,touchMoved=false;
    let lastFrame=performance.now();
    let longPressTimer=null,isLongPressing=false;

    // Listen on document to capture swipes everywhere (even over semi-transparent UI)
    document.addEventListener('touchstart',sfTouchStart,{passive:false});
    document.addEventListener('touchmove',sfTouchMove,{passive:false});
    document.addEventListener('touchend',sfTouchEnd);
    sfCleanupFn=()=>{
      document.removeEventListener('touchstart',sfTouchStart);
      document.removeEventListener('touchmove',sfTouchMove);
      document.removeEventListener('touchend',sfTouchEnd);
    };

    function sfTouchStart(e){
      if(!sfActive||sfExiting||sfEntering)return;
      // Don't intercept touches on input/send button
      if(e.target.closest('#input')||e.target.closest('#sendBtn')||e.target.closest('.sidebar')||e.target.closest('#ctxMenu'))return;
      if(e.touches.length===1){
        e.preventDefault();
        touchStartX=e.touches[0].clientX;touchStartY=e.touches[0].clientY;
        touchActive=true;touchMoved=false;
        isLongPressing=true;
        if(longPressTimer)clearTimeout(longPressTimer);
        longPressTimer=setTimeout(()=>{
          if(isLongPressing&&sfActive&&!touchMoved){
            sfHyperspace=true;
            if(navigator.vibrate)navigator.vibrate([15,30,15]);
          }
        },300);
      }
      resetInactivityTimer();
    }

    function sfTouchMove(e){
      if(!touchActive||!sfCamera||!sfActive)return;
      if(e.touches.length===1){
        const dx=e.touches[0].clientX-touchStartX;
        const dy=e.touches[0].clientY-touchStartY;
        if(Math.abs(dx)>3||Math.abs(dy)>3){
          touchMoved=true;
          isLongPressing=false;sfHyperspace=false;
          if(longPressTimer){clearTimeout(longPressTimer);longPressTimer=null}
        }
        // Steer camera
        sfCamera.rotation.y+=dx*0.003;
        sfCamera.rotation.x+=dy*0.003;
        sfCamera.rotation.x=Math.max(-Math.PI/3,Math.min(Math.PI/3,sfCamera.rotation.x));
        touchStartX=e.touches[0].clientX;touchStartY=e.touches[0].clientY;
        // Spawn stardust trail
        if(touchMoved&&Math.random()<0.4)spawnTrail(e.touches[0].clientX,e.touches[0].clientY);
      }
      resetInactivityTimer();
    }

    function sfTouchEnd(e){
      if(!sfActive)return;
      if(!touchMoved&&touchActive&&!sfHyperspace){
        // Tap → burst
        spawnBurst(touchStartX,touchStartY);
        if(navigator.vibrate)navigator.vibrate(10);
      }
      touchActive=false;isLongPressing=false;sfHyperspace=false;
      if(longPressTimer){clearTimeout(longPressTimer);longPressTimer=null}
    }

    // Exit: tap input field
    input.addEventListener('touchstart',function exitHandler(e){
      if(sfActive&&e.target.closest('#input')){
        disposeStarfield();
        setTimeout(()=>input.focus(),100);
      }
    },{once:false});
    input.addEventListener('focus',()=>{if(sfActive)disposeStarfield()});

    // ── Animation loop ──
    function animate(now){
      if(!sfActive||!sfRenderer)return;
      sfRafId=requestAnimationFrame(animate);
      const dt=Math.min((now-lastFrame)/1000,0.1);
      const t=now*0.001;
      lastFrame=now;

      // Auto-drift camera forward
      if(sfCamera&&sfDriftSpeed>0){
        const speed=sfHyperspace?sfDriftSpeed*4:sfDriftSpeed;
        const dir=new THREE.Vector3(0,0,-1);
        dir.applyQuaternion(sfCamera.quaternion);
        sfCamera.position.addScaledVector(dir,speed*dt);
        if(sfHyperspace&&navigator.vibrate&&Math.floor(now/150)%2===0)navigator.vibrate(10);
      }

      // Wrap/recycle particles that pass behind camera
      function recycleParticles(particles,loZ,hiZ){
        particles.forEach(p=>{
          if(!p.pointsObj||!p.pointsObj.geometry)return;
          // Transform position by camera to check if behind
          const localZ=p.z-sfCamera.position.z;
          if(localZ>15){ // past camera
            p.z-=250;
            const arr=p.pointsObj.geometry.attributes.position.array;
            arr[p.idx*3+2]=p.z;
            p.pointsObj.geometry.attributes.position.needsUpdate=true;
          }
        });
      }
      recycleParticles(sfAllParticles);
      recycleParticles(sfDust);
      recycleParticles(sfSparks);

      // Twinkle
      function twinkleLayer(particles,geo){
        if(!geo)return;
        const colors=geo.attributes.color;
        if(!colors)return;
        const arr=colors.array;
        let nu=false;
        particles.forEach(p=>{
          if(!p.baseOpacity)return;
          const twinkle=0.5+0.5*Math.sin(p.twinklePhase+t*p.twinkleSpeed);
          const opacity=sfHyperspace?0.8+twinkle*0.2:(0.3+twinkle*0.7);
          arr[p.idx*3]=p.r*opacity;
          arr[p.idx*3+1]=p.g*opacity;
          arr[p.idx*3+2]=p.b*opacity;
          nu=true;
        });
        if(nu)colors.needsUpdate=true;
      }
      twinkleLayer(sfAllParticles,starGeo);
      twinkleLayer(sfDust,dustGeo);
      twinkleLayer(sfSparks,sparkGeo);

      // Drift nebulae
      nebulaClusters.forEach(n=>{
        n.position.z+=n.userData.speedZ*(sfHyperspace?3:1);
        n.position.x+=n.userData.speedX;
        n.position.y+=n.userData.speedY||0;
        if(n.position.z>20)n.position.z=-100;
        if(Math.abs(n.position.x)>30)n.userData.speedX*=-1;
      });

      // Update burst particles (expand + fade)
      for(let i=sfBurstParticles.length-1;i>=0;i--){
        const b=sfBurstParticles[i];
        b.age+=dt;
        if(b.age>=b.lifetime){
          sfScene.remove(b.points);
          b.points.geometry.dispose();
          b.points.material.dispose();
          sfBurstParticles.splice(i,1);
          continue;
        }
        const progress=b.age/b.lifetime;
        const arr=b.points.geometry.attributes.position.array;
        for(let j=0;j<b.velocities.length;j++){
          arr[j*3]+=b.velocities[j].x*dt;
          arr[j*3+1]+=b.velocities[j].y*dt;
          arr[j*3+2]+=b.velocities[j].z*dt;
        }
        b.points.geometry.attributes.position.needsUpdate=true;
        b.points.material.opacity=1-progress;
      }

      // Update trail particles (fade out)
      for(let i=sfTrailParticles.length-1;i>=0;i--){
        const tr=sfTrailParticles[i];
        tr.age+=dt;
        if(tr.age>=tr.lifetime){
          sfScene.remove(tr.points);
          tr.points.geometry.dispose();
          tr.points.material.dispose();
          sfTrailParticles.splice(i,1);
          continue;
        }
        tr.points.material.opacity=0.8*(1-tr.age/tr.lifetime);
      }

      sfRenderer.render(sfScene,sfCamera);
    }
    resetInactivityTimer();
  }

  // ── Cleanup touch listeners on exit ──
  let sfCleanupFn=null;
  finishDispose=function(){
    if(sfCleanupFn)sfCleanupFn();
    document.body.classList.remove('starfield-mode');
    if(sfRafId){cancelAnimationFrame(sfRafId);sfRafId=null}
    if(sfRenderer){sfRenderer.dispose();sfRenderer=null}
    if(sfScene){
      sfScene.traverse(obj=>{if(obj.geometry)obj.geometry.dispose();if(obj.material){if(Array.isArray(obj.material))obj.material.forEach(m=>m.dispose());else obj.material.dispose()}});
      sfScene=null;
    }
    sfCamera=null;sfAllParticles=[];sfDust=[];sfSparks=[];sfNebulae=[];
    sfTrailParticles=[];sfBurstParticles=[];
    overlay.style.opacity='0';
    setTimeout(()=>{
      overlay.style.display='none';overlay.innerHTML='';
      const wImg=document.querySelector('.welcome .avatar img');
      if(wImg)wImg.style.opacity='1';
    },800);
    sfActive=false;sfExiting=false;sfEntering=false;sfHyperspace=false;
  };

  // ── Triple-tap welcome logo (primary trigger) ──
  let welcomeTapCount=0,welcomeTapTimer=null;
  msgs.addEventListener('click',e=>{
    const target=e.target.closest('.welcome .avatar img');
    if(!target||sfActive)return;
    e.preventDefault();e.stopPropagation();
    welcomeTapCount++;
    if(welcomeTapTimer)clearTimeout(welcomeTapTimer);
    if(welcomeTapCount>=3){
      welcomeTapCount=0;
      if(navigator.vibrate)navigator.vibrate(30);
      initStarfield();
    }else{
      welcomeTapTimer=setTimeout(()=>{welcomeTapCount=0},800);
    }
  });

  // ── Fallback: triple-tap sidebar logo ──
  const logoBtn=document.getElementById('logo-tap');
  if(logoBtn){
    let tapCount2=0,tapTimer2=null;
    logoBtn.addEventListener('click',e=>{
      if(sfActive)return;
      e.preventDefault();e.stopPropagation();
      tapCount2++;
      if(tapTimer2)clearTimeout(tapTimer2);
      if(tapCount2>=3){
        tapCount2=0;
        if(navigator.vibrate)navigator.vibrate(30);
        initStarfield();
      }else{
        tapTimer2=setTimeout(()=>{tapCount2=0},800);
      }
    });
  }
})();
loadSessions();const ids=Object.keys(sessions);if(ids.length)switchSession(ids[ids.length-1]);renderSessions()

// ── 模型切换（预加载，即时切换）──
async function loadModels(){
  try{
    const r=await fetch('/api/models');const d=await r.json();
    const sel=document.getElementById('modelSelect');
    sel.innerHTML=d.models.filter(m=>m.engine==='tgai_go').map(m=>`<option value="${m.id}"${m.id===d.default?' selected':''}>${m.name}</option>`).join('');
    // 检测 TGAI GO 是否可用
    const hr=await fetch('/api/health');const hd=await hr.json();
    const eg=document.getElementById('engineSelect');
    if(!hd.tgai_go){console.warn('TGAI GO 未加载，请检查引擎状态')}
    else{eg.innerHTML='<option value="tgai_go" selected>TGAI GO (C引擎)</option>'}
  }catch(e){console.error(e)}
}
async function switchModel(id){
  if(!id)return;
  // 查找模型的 engine 类型
  const models=await fetch('/api/models').then(r=>r.json());
  const mi=models.models.find(m=>m.id===id);
  const eng=document.getElementById('engineSelect');
  if(mi&&mi.engine!=='tgai_go'){console.warn('非TGAI GO模型已停用:',id);return}
  eng.value='tgai_go';
  try{
    const r=await fetch('/api/model/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:id})});
    const d=await r.json();
    if(d.status!=='ok'){alert('切换失败: '+(d.error||'未知错误'))}
  }catch(e){alert('切换失败: '+e.message)}
}
function toggleSysPrompt(){
  const bar=document.getElementById('sysPromptBar');
  const toggle=document.getElementById('toggleSysPrompt');
  if(bar.classList.contains('hidden')){
    bar.classList.remove('hidden');
    toggle.textContent='- 隐藏系统提示词';
    document.getElementById('sysPromptInput').focus();
  }else{
    bar.classList.add('hidden');
    toggle.textContent='+ 系统提示词';
  }
}
loadModels();
</script>
</body>
</html>'''


# ═══════════════════════════════════════
# 终端交互对话 (--chat)
# ═══════════════════════════════════════

def _interactive_chat(model, tokenizer, cfg):
    """嵌入在 API 服务器中的终端对话模式"""
    gen = TextGenerator(model, tokenizer)

    window = f"+{model.config.extend_max_seq_len}" if model.config.self_extend else "原生"
    print(f"\n{'='*50}")
    print(f"  TGAI 对话模式 | 窗口: {window} | 输入 /quit 退出")
    print(f"  temp={cfg['temperature']:.1f} max_t={cfg['max_tokens']} mint={cfg['min_tokens']}")
    print(f"{'='*50}\n")

    history = []
    while True:
        try:
            user = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user:
            continue

        # ── 斜杠命令 ──
        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("再见！")
                break
            elif cmd == "/help":
                print("  /temp <0.1-2.0>   温度 (当前 {:.1f})".format(cfg['temperature']))
                print("  /maxt <n>          最大 token (当前 {})".format(cfg['max_tokens']))
                print("  /mint <n>          最小 token (当前 {})".format(cfg['min_tokens']))
                print("  /topk <n>          top_k (当前 {})".format(cfg['top_k']))
                print("  /topp <0-1>        top_p (当前 {:.1f})".format(cfg['top_p']))
                print("  /freq <0-2>        频率惩罚 (当前 {:.2f})".format(cfg['freq_penalty']))
                print("  /rep <1-2>         重复惩罚 (当前 {:.2f})".format(cfg['rep_penalty']))
                print("  /status            当前参数")
                print("  /clear, /clean     清除对话历史")
                print("  /quit              退出")
                continue
            elif cmd == "/temp" and len(parts) > 1:
                cfg['temperature'] = float(parts[1])
                print(f"  温度 → {cfg['temperature']}")
                continue
            elif cmd == "/maxt" and len(parts) > 1:
                cfg['max_tokens'] = int(parts[1])
                print(f"  max_tokens → {cfg['max_tokens']}")
                continue
            elif cmd == "/mint" and len(parts) > 1:
                cfg['min_tokens'] = int(parts[1])
                print(f"  min_tokens → {cfg['min_tokens']}")
                continue
            elif cmd == "/topk" and len(parts) > 1:
                cfg['top_k'] = int(parts[1])
                print(f"  top_k → {cfg['top_k']}")
                continue
            elif cmd == "/topp" and len(parts) > 1:
                cfg['top_p'] = float(parts[1])
                print(f"  top_p → {cfg['top_p']}")
                continue
            elif cmd == "/freq" and len(parts) > 1:
                cfg['freq_penalty'] = float(parts[1])
                print(f"  freq_penalty → {cfg['freq_penalty']}")
                continue
            elif cmd == "/rep" and len(parts) > 1:
                cfg['rep_penalty'] = float(parts[1])
                print(f"  rep_penalty → {cfg['rep_penalty']}")
                continue
            elif cmd in ("/clear", "/clean"):
                history = []
                print("  历史已清除")
                continue
            elif cmd == "/status":
                print(f"  temp={cfg['temperature']:.1f}  max_t={cfg['max_tokens']}  mint={cfg['min_tokens']}")
                print(f"  top_k={cfg['top_k']}  top_p={cfg['top_p']:.1f}")
                print(f"  freq={cfg['freq_penalty']:.2f}  rep={cfg['rep_penalty']:.2f}")
                print(f"  历史轮数: {len(history)}")
                continue
            else:
                print(f"  未知命令: {cmd}，输入 /help 查看")
                continue

        if history:
            prompt = "\n".join(history) + f"\n用户:{user}\nTGAI?"
        else:
            prompt = f"用户:{user}\nTGAI?"

        print("🤖 TGAI: ", end="", flush=True)
        reply = ""
        token_count = 0
        try:
            for token in gen.generate(
                prompt,
                max_new_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
                frequency_penalty=cfg["freq_penalty"],
                repetition_penalty=cfg["rep_penalty"],
                min_new_tokens=cfg["min_tokens"],
                stream=True,
                formatted=bool(history),
            ):
                print(token, end="", flush=True)
                reply += token
                token_count += 1
            print()
            if token_count:
                print(f"  [{token_count} token]")
        except Exception as e:
            print(f"\n[错误] {e}")
            continue

        reply = reply.strip()
        if not reply:
            reply = "..."

        history.append(f"用户:{user}\nTGAI?{reply}")
        if len(history) > 8 and len(history) > 16:
            history = history[-16:]


def main():
    parser = argparse.ArgumentParser(description="TGAI API Server")
    parser.add_argument("--start", action="store_true", help="直接启动服务器（否则进入 CLI 模式）")
    parser.add_argument("--checkpoint", default="checkpoints/tgai_sft.pt", help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="分词器路径")
    parser.add_argument("--config", default=None, help="默认参数配置文件")
    parser.add_argument("--port", type=int, default=6006, help="监听端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--device", default=None, help="计算设备 (默认自动)")
    parser.add_argument("--extend", type=int, default=0, help="Self-Extend 上下文扩展窗口，0=禁用")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile 加速")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no-filter", action="store_true", help="关闭偏置/替换等软过滤（涉政保护不受影响）")
    parser.add_argument("--models", default=None, help="可选模型列表，JSON格式: [{\"id\":\"v1\",\"name\":\"TGAI V1\",\"path\":\"...\"},...] 或 JSON 文件路径")
    parser.add_argument("--skip", default=None, help="跳过指定模型，逗号分隔 (如: v2_3b,v1_tgai_go)")
    parser.add_argument("--serve_dir", default=None, help="启用文件下载后门 (暴露指定目录)")
    parser.add_argument("--no_model", action="store_true", help="不加载AI模型（仅文件服务+CLI）")
    parser.add_argument("--chat", action="store_true", help="终端交互对话模式 (替代 HTTP 服务器)")
    args = parser.parse_args()

    # ── 解析可选模型列表 ──
    global _MODEL_REGISTRY
    if args.models:
        import json as _json
        models_raw = args.models
        if os.path.isfile(models_raw):
            with open(models_raw, 'r', encoding='utf-8') as f:
                models_raw = f.read()
        try:
            user_models = _json.loads(models_raw)
            if isinstance(user_models, list) and len(user_models) > 0:
                _MODEL_REGISTRY = user_models
                log.info(f"已加载 {len(_MODEL_REGISTRY)} 个可选模型")
        except Exception as e:
            log.warning(f"解析 --models 失败: {e}")

    # ── 跳过指定模型 ──
    if args.skip:
        skip_ids = set(s.strip() for s in args.skip.split(",") if s.strip())
        _MODEL_REGISTRY = [m for m in _MODEL_REGISTRY if m["id"] not in skip_ids]
        log.info(f"跳过模型: {skip_ids}, 剩余 {len(_MODEL_REGISTRY)} 个")

    global _generator, _default_config, _default_config_path

    if args.chat:
        # ── 终端交互对话 ──
        _default_config = load_default_config(args.config)
        if args.config:
            _default_config_path = args.config
        import torch
        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model, tokenizer = load_model(
            args.checkpoint, args.tokenizer, device=device, compile_model=args.compile,
            self_extend=args.extend > 0,
            extend_max_seq_len=args.extend if args.extend > 0 else 4096,
        )
        if args.extend > 0:
            model.set_self_extend(True, args.extend)

        _interactive_chat(model, tokenizer, _default_config)
        return

    if args.serve_dir:
        global _SERVE_DIR
        _SERVE_DIR = args.serve_dir
        log.info(f"文件后门已启用: {args.serve_dir}")

    if args.start:
        # ── 直连模式 ──
        _default_config = load_default_config(args.config)
        if args.config:
            _default_config_path = args.config
        else:
            _default_config_path = os.path.join(PROJECT_ROOT, "qq_bot_config.json")

        if not args.no_model:
            import torch
            if args.device:
                device = args.device
            elif torch.cuda.is_available():
                device = "cuda"
            elif args.force_cpu:
                device = "cpu"
                log.warning("警告: 强制 CPU 模式，模型加载和推理会非常慢！")
            else:
                log.error("未检测到 GPU，API 服务器拒绝启动。如需 CPU 运行请加 --force-cpu")
                sys.exit(1)

            log.info(f"预加载 TGAI GO 模型（PyTorch 引擎已停用）...")
            global _default_model_id, _tokenizer_path, _generators
            _tokenizer_path = args.tokenizer

            # PyTorch 引擎已停用，跳过 PyTorch 模型加载
            if args.extend > 0:
                log.info(f"Self-Extend: max_seq_len={args.extend}")
            log.info(f"  PyTorch 引擎已停用，仅加载 TGAI GO 模型")

            # 加载 TGAI GO 模型 (所有注册的 .TG 文件)
            _tgai_go_models = {}  # {model_id: model_ptr}
            for mi in _MODEL_REGISTRY:
                if mi.get("engine") != "tgai_go":
                    continue
                tg_path = mi["path"]
                if not os.path.exists(tg_path):
                    log.warning(f"  TGAI GO 模型不存在: {tg_path}")
                    continue
                log.info(f"  加载 TGAI GO: {mi['name']} ({tg_path})")
                _init_tgai_go(tg_path, mi["id"], extend_ctx=args.extend)
            
            global _token_bias, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules
            _bias_words, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules = load_word_bias_config()
            _gen = _get_gen()
            tz = _gen.tokenizer if _gen else None
            _token_bias = _compute_token_bias(tz, _bias_words) if tz else None
            global _enabled
            if args.no_filter:
                _enabled = False
                log.warning("  --no-filter: 软过滤已关闭（涉政保护仍生效）")
            log.info(f"  API 就绪: http://{args.host}:{args.port}")
            log.info(f"  设备: {device}")
            log.info(f"  默认参数: temp={_default_config['temperature']:.2f} max_t={_default_config['max_tokens']}")
            log.info(f"  端点: /api/generate /api/chat /api/health /api/config")
        else:
            log.info(f"  (无模型模式) 文件服务就绪: http://{args.host}:{args.port}")
            log.info(f"  下载: /files/<文件名>")

        log_flask = logging.getLogger('werkzeug')
        log_flask.setLevel(logging.WARNING)
        use_threaded = not args.no_model and torch.cuda.is_available() if not args.no_model else False
        app.run(host=args.host, port=args.port, threaded=use_threaded)
    else:
        # ── CLI 交互模式 ──
        APIServerCLI(args).cmdloop()


# ═══════════════════════════════════════
import cmd, threading

class APIServerCLI(cmd.Cmd):
    intro = """
╔══════════════════════════════════════════╗
║        TGAI API 服务器控制台             ║
║  输入 help 查看命令, quit 退出            ║
╚══════════════════════════════════════════╝
"""
    prompt = "TGAI> "

    def __init__(self, args):
        super().__init__()
        self._cli_args = args
        self._model = None
        self._tokenizer = None
        self._server_thread = None
        self._server_running = False
        self._extend = args.extend

        global _default_config
        _default_config = load_default_config(args.config)
        self._config = dict(_default_config)
        self._port = args.port
        self._host = args.host

    # ── load ──
    def do_load(self, arg):
        """load <checkpoint_path> — 加载模型"""
        global _generator, _generators, _default_model_id
        global _token_bias, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules
        import torch

        path = arg.strip() or "checkpoints/tgai_sft.pt"
        tokenizer_path = "checkpoints/tokenizer.json"

        if not os.path.exists(path):
            print(f"[错误] 模型文件不存在: {path}")
            return
        if not os.path.exists(tokenizer_path):
            print(f"[错误] 分词器不存在: {tokenizer_path}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[加载] 设备: {device}")

        try:
            self._model, self._tokenizer = load_model(
                path, tokenizer_path, device,
                self_extend=self._extend > 0,
                extend_max_seq_len=self._extend if self._extend > 0 else 4096,
            )
        except Exception as e:
            print(f"[错误] 加载失败: {e}")
            return

        if self._extend > 0:
            print(f"[加载] Self-Extend: {self._extend}")

        _generator = TextGenerator(self._model, self._tokenizer)
        _generators[path] = _generator
        _default_model_id = path

        # 加载拦截规则 (CLI 模式之前缺失此调用，导致拦截不生效)
        _bias_words, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules = load_word_bias_config()
        _token_bias = _compute_token_bias(self._tokenizer, _bias_words)
        log.info(f"  拦截规则: block={len(_block_rules)}组, output_block={len(_output_block_rules)}组")

        self._save_config()
        print(f"[加载] 完成!")

    # ── config ──
    def do_config(self, arg):
        """config — 查看当前参数"""
        print("\n当前参数:")
        for k, v in self._config.items():
            print(f"  {k:20s} = {v}")
        print(f"  {'port':20s} = {self._port}")
        print(f"  {'extend':20s} = {self._extend} (0=禁用)")
        print(f"  {'model_loaded':20s} = {_generator is not None}")
        print(f"  {'server':20s} = {'运行中' if self._server_running else '未启动'}")
        print()

    # ── set ──
    def do_set(self, arg):
        """set <key> <value> — 修改参数"""
        parts = arg.strip().split()
        if len(parts) < 2:
            print("用法: set <key> <value>")
            print("可修改: temp, max_tokens, top_k, top_p, freq_penalty, rep_penalty, min_tokens, port")
            return

        key, val = parts[0], parts[1]
        if key == "port":
            self._port = int(val)
            print(f"端口 → {self._port}")
        elif key in self._config:
            self._config[key] = type(self._config[key])(float(val) if '.' in val else int(val))
            print(f"{key} → {self._config[key]}")
        else:
            print(f"[错误] 未知参数: {key}")
        self._save_config()

    def _save_config(self):
        config_path = os.path.join(PROJECT_ROOT, "qq_bot_config.json")
        try:
            existing = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            existing["temp"] = self._config["temperature"]
            existing["max_tokens"] = self._config["max_tokens"]
            existing["top_k"] = self._config["top_k"]
            existing["top_p"] = self._config["top_p"]
            existing["freq_penalty"] = self._config["freq_penalty"]
            existing["rep_penalty"] = self._config["rep_penalty"]
            existing["min_tokens"] = self._config["min_tokens"]
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print("  已保存到 qq_bot_config.json")
        except Exception as e:
            print(f"  [警告] 保存失败: {e}")

    # ── extend ──
    def do_extend(self, arg):
        """extend <n> — 设置 Self-Extend 窗口，0=禁用（需要 load 重新加载）"""
        val = int(arg.strip() or "0")
        self._extend = val
        print(f"Self-Extend → {val} (请执行 load 重新加载模型)")

    # ── start ──
    def do_start(self, arg):
        """start [--port N] — 启动 API 服务器"""
        global _generator
        if not _generators:
            print("[错误] 请先加载模型: load checkpoints/tgai_sft.pt")
            return
        if self._server_running:
            print("[提示] 服务器已在运行")
            return

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=self._port)
        parser.add_argument("--host", default=self._host)
        a, _ = parser.parse_known_args(arg.strip().split() if arg.strip() else [])

        def _run():
            import torch
            self._server_running = True
            log_flask = logging.getLogger('werkzeug')
            log_flask.setLevel(logging.WARNING)
            app.run(host=a.host, port=a.port, threaded=torch.cuda.is_available())

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        self._server_running = True
        print(f"\n[启动] API 服务器: http://{a.host}:{a.port}")
        print(f"  端点: /api/health  /api/config  /api/generate  /api/chat  /")
        print(f"  (在下方输入命令继续管理)\n")

    # ── stop ──
    def do_stop(self, arg):
        """stop — 停止服务器"""
        if self._server_running:
            self._server_running = False
            print("[停止] 服务器已标记停止 (请 Ctrl+C 或关闭窗口)")
        else:
            print("[提示] 服务器未运行")

    # ── status ──
    def do_status(self, arg):
        """status — 查看服务器状态"""
        print(f"模型: {'已加载' if _generator else '未加载'}")
        print(f"服务器: {'运行中' if self._server_running else '未启动'}")
        if self._server_running:
            print(f"地址: http://{self._host}:{self._port}")

    # ── reload ──
    def do_reload(self, arg):
        """reload — 热重载拦截规则 (word_bias_config.json)"""
        global _token_bias, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules, _generator
        if not _generators:
            print("[错误] 请先加载模型: load checkpoints/tgai_sft.pt")
            return
        _bias_words, _prefix_bias, _dynamic_rules, _replace_rules, _block_rules, _output_block_rules = load_word_bias_config()
        _token_bias = _compute_token_bias(_get_gen().tokenizer, _bias_words)
        print(f"[重载] block={len(_block_rules)}组, output_block={len(_output_block_rules)}组, token_bias={len(_token_bias)}词")

    def do_help(self, arg):
        print("""
命令列表:
  load <path>        加载模型 (默认: checkpoints/tgai_sft.pt)
  config             查看当前参数
  set <key> <val>    修改参数 (temp/max_tokens/top_k/top_p/freq_penalty/rep_penalty/min_tokens/port)
  extend <n>         Self-Extend 窗口 (0=禁用, 如 4096/8192)
  start [--port N]   启动 API 服务器
  stop               停止服务器
  status             查看状态
  reload             热重载拦截规则
  quit               退出

直接输入文本 → 测试对话 (流式输出, 无记忆)
""")

    def default(self, line):
        """直接输入文本触发测试对话"""
        global _generator
        if not _generators:
            print("[提示] 请先加载模型: load checkpoints/tgai_sft.pt")
            return
        line = line.strip()
        if not line:
            return
        print(f"\n{'─' * 50}")
        print(f"[测试] {line[:60]}{'...' if len(line) > 60 else ''}")
        print(f"{'─' * 50}")
        print("TGAI: ", end="", flush=True)
        reply = ""
        try:
            for token in _get_gen().generate(
                line, max_new_tokens=self._config["max_tokens"],
                temperature=self._config["temperature"], top_k=self._config["top_k"],
                top_p=self._config["top_p"], frequency_penalty=self._config["freq_penalty"],
                repetition_penalty=self._config["rep_penalty"],
                min_new_tokens=self._config["min_tokens"],
                stream=True, formatted=False,
                token_bias=_token_bias,
                prefix_bias=_prefix_bias,
            ):
                print(token, end="", flush=True)
                reply += token
            print()
        except Exception as e:
            print(f"\n[错误] {e}")
        if not reply.strip():
            print("[空回复]")
        print()

    def do_quit(self, arg):
        print("再见!")
        return True

    def do_exit(self, arg):
        return self.do_quit(arg)

    def cmdloop(self, intro=None):
        try:
            super().cmdloop(intro)
        except KeyboardInterrupt:
            print("\n再见!")
            return True



if __name__ == "__main__":
    main()
