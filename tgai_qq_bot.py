"""
TGAI QQ 机器人 — 基于 NapCat OneBot 协议
==========================================
启动前确保 NapCat 已配置并运行（默认 ws://127.0.0.1:6099）

用法:
    python tgai_qq_bot.py                          # CPU 推理
    python tgai_qq_bot.py --cuda                   # GPU 推理
    python tgai_qq_bot.py --port 6099 --bot-qq 123456  # 自定义配置
"""

import os
import sys
import json
import time
import asyncio
import argparse
import threading
from typing import Optional

import requests
import websocket

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ChineseTokenizer, BOS_ID, EOS_ID
from model import create_model


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
class BotConfig:
    napcat_ws_url: str = "ws://127.0.0.1:6099"      # NapCat WebSocket 地址
    napcat_http_url: str = "http://127.0.0.1:6099"   # NapCat HTTP API 地址
    bot_qq: int = 0          # 机器人 QQ 号（0=自动获取）
    whitelist: list = []      # 白名单 QQ 号（空=不限制）
    max_reply_tokens: int = 256
    temperature: float = 0.8
    max_context: int = 10     # 上下文保留轮数


# ═══════════════════════════════════════════════════════════
# TGAI 推理引擎
# ═══════════════════════════════════════════════════════════
class TGAIEngine:
    """加载 TGAI 模型，提供 generate() 接口"""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
        print(f"[TGAI] 加载模型: {checkpoint_path}")
        self.tokenizer = ChineseTokenizer.load(tokenizer_path)
        print(f"[TGAI] 词表: {self.tokenizer.vocab_size_actual}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mc = ckpt.get('model_config', {})
        state_dict = ckpt['model_state_dict']
        print(f"[TGAI] Epoch={ckpt.get('epoch','?')}, Step={ckpt.get('global_step','?')}")

        self.model = create_model(
            vocab_size=state_dict['token_embedding.weight'].shape[0],
            d_model=mc.get('d_model', 256),
            n_layers=mc.get('n_layers', 8),
            n_heads=mc.get('n_heads', 8),
            d_ff=mc.get('d_ff', 1024),
            max_seq_len=mc.get('max_seq_len', 512),
            dropout=0.0,
            n_experts=mc.get('n_experts', 4),
            n_activated=mc.get('n_activated', 2),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[TGAI] 参数: {n_params/1e6:.1f}M, 设备: {device}")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.8) -> str:
        """根据 prompt 生成回复"""
        formatted = f"用户:{prompt}\nTGAI?"
        prompt_ids = [BOS_ID] + self.tokenizer.encode(formatted, add_special=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=80,
            top_p=0.9,
            eos_token_id=EOS_ID,
            min_new_tokens=3,
            repetition_penalty=1.05,
            frequency_penalty=0.15,
        )

        full_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special=True)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special=True)

        if full_text.startswith(prompt_text):
            reply = full_text[len(prompt_text):]
        else:
            reply = full_text

        # 截断多余的对话轮次
        for sep in ['\n用户:', '用户:', 'TGAI?']:
            idx = reply.find(sep)
            if idx > 0:
                reply = reply[:idx]
        return reply.strip() or "[空回复]"


# ═══════════════════════════════════════════════════════════
# QQ 机器人
# ═══════════════════════════════════════════════════════════
class TGAIQQBot:
    def __init__(self, engine: TGAIEngine, config: BotConfig):
        self.engine = engine
        self.config = config
        self.context: dict = {}  # user_id -> list of (role, text)
        self.ws: Optional[websocket.WebSocketApp] = None

    # ── HTTP API 封装 ────────────────────────────────
    def _api(self, action: str, params: dict) -> dict:
        """调用 NapCat HTTP API"""
        try:
            resp = requests.post(
                f"{self.config.napcat_http_url}/{action}",
                json=params, timeout=10
            )
            return resp.json()
        except Exception as e:
            print(f"[API] {action} 失败: {e}")
            return {}

    def send_private_msg(self, user_id: int, text: str):
        """发送私聊消息（长文本自动分段）"""
        for chunk in self._split_long(text):
            self._api("send_private_msg", {
                "user_id": user_id,
                "message": [{"type": "text", "data": {"text": chunk}}]
            })

    def send_group_msg(self, group_id: int, text: str, reply_msg_id: int = 0):
        """发送群聊消息"""
        for chunk in self._split_long(text):
            params = {
                "group_id": group_id,
                "message": [{"type": "text", "data": {"text": chunk}}]
            }
            if reply_msg_id:
                params["message"].insert(0, {
                    "type": "reply",
                    "data": {"id": str(reply_msg_id)}
                })
            self._api("send_group_msg", params)

    def _split_long(self, text: str, max_len: int = 800) -> list:
        """QQ 文本长度限制，分段发送"""
        if len(text) <= max_len:
            return [text]
        chunks = []
        for i in range(0, len(text), max_len):
            chunks.append(text[i:i+max_len])
        return chunks

    # ── 消息处理 ──────────────────────────────────────
    def _check_whitelist(self, user_id: int) -> bool:
        if not self.config.whitelist:
            return True
        return user_id in self.config.whitelist

    def _on_message(self, ws, raw: str):
        """WebSocket 收到 QQ 消息"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # 仅处理消息事件
        post_type = data.get("post_type")
        meta = data.get("meta_event_type", "")
        if post_type != "message" and post_type != "message_sent":
            # 处理心跳/元事件（获取 bot_qq）
            if meta == "lifecycle" and not self.config.bot_qq:
                self.config.bot_qq = data.get("self_id", 0)
                print(f"[QQ] 获取到机器人 QQ: {self.config.bot_qq}")
            return

        msg_type = data.get("message_type")
        user_id = data.get("user_id", 0)
        raw_msg = data.get("raw_message", "").strip()
        message_id = data.get("message_id", 0)

        if not raw_msg:
            return
        if not self._check_whitelist(user_id):
            return

        # 过滤自己发的消息
        if user_id == self.config.bot_qq:
            return

        # ── 群聊处理 ──
        if msg_type == "group":
            group_id = data.get("group_id", 0)
            # 检查是否 @机器人
            at_bot = f"[CQ:at,qq={self.config.bot_qq}]" if self.config.bot_qq else "@"
            if at_bot not in raw_msg and not raw_msg.startswith("TGAI") and not raw_msg.startswith("tgai"):
                return  # 没 @ 就不回
            clean_msg = raw_msg.replace(at_bot, "").strip()
            print(f"[群聊] {group_id}/{user_id}: {clean_msg}")
            reply = self._get_reply(user_id, clean_msg)
            self.send_group_msg(group_id, reply, message_id)

        # ── 私聊处理 ──
        elif msg_type == "private":
            print(f"[私聊] {user_id}: {raw_msg}")
            reply = self._get_reply(user_id, raw_msg)
            self.send_private_msg(user_id, reply)

    def _get_reply(self, user_id: int, msg: str) -> str:
        """用 TGAI 生成回复"""
        t0 = time.time()
        reply = self.engine.generate(
            msg,
            max_tokens=self.config.max_reply_tokens,
            temperature=self.config.temperature,
        )
        elapsed = time.time() - t0
        print(f"[TGAI] {elapsed:.1f}s → {reply[:80]}...")
        return reply

    # ── 启动 ──────────────────────────────────────────
    def run(self):
        """启动 WebSocket 连接"""
        print(f"[QQ] 连接 NapCat: {self.config.napcat_ws_url}")
        self.ws = websocket.WebSocketApp(
            self.config.napcat_ws_url,
            on_message=self._on_message,
            on_open=lambda ws: print("[QQ] 已连接，等待消息..."),
            on_error=lambda ws, e: print(f"[QQ] 错误: {e}"),
            on_close=lambda ws, code, msg: print(f"[QQ] 断开: {code}"),
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TGAI QQ 机器人')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--tokenizer', default='checkpoints/tokenizer.json')
    parser.add_argument('--cuda', action='store_true', help='GPU 推理')
    parser.add_argument('--port', type=int, default=6099, help='NapCat 端口')
    parser.add_argument('--bot-qq', type=int, default=0, help='机器人 QQ 号')
    parser.add_argument('--whitelist', type=str, default='', help='白名单(逗号分隔)')
    parser.add_argument('--temp', type=float, default=0.8, help='温度')
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 加载模型
    engine = TGAIEngine(args.checkpoint, args.tokenizer, device)

    # 配置
    config = BotConfig()
    config.napcat_ws_url = f"ws://127.0.0.1:{args.port}"
    config.napcat_http_url = f"http://127.0.0.1:{args.port}"
    config.bot_qq = args.bot_qq
    config.temperature = args.temp
    if args.whitelist:
        config.whitelist = [int(x.strip()) for x in args.whitelist.split(',')]

    # 启动
    bot = TGAIQQBot(engine, config)
    bot.run()
