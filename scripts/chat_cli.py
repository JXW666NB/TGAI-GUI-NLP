"""
TGAI 交互对话 CLI
===============
用法:
    python3 scripts/chat_cli.py \
        --checkpoint /root/autodl-tmp/TGAI_checkpoints_v3/milestone_26k.pt \
        --tokenizer /root/autodl-tmp/TGAI_checkpoints_v3/tokenizer.json \
        --extend 8192
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_model, TextGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="qq_bot_config.json")
    parser.add_argument("--extend", type=int, default=0, help="self_extend max_seq_len")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    if args.extend:
        model.set_self_extend(True, args.extend)

    gen = TextGenerator(model, tokenizer)

    # 加载参数
    cfg = {"temperature": 0.8, "max_tokens": 256, "top_k": 90,
           "top_p": 0.9, "freq_penalty": 0.25, "rep_penalty": 1.05, "min_tokens": 10}
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                raw = json.load(f)
            for k, key in [("temperature", "temp"), ("max_tokens", "max_tokens"),
                           ("top_k", "top_k"), ("top_p", "top_p"),
                           ("freq_penalty", "freq_penalty"),
                           ("rep_penalty", "rep_penalty"), ("min_tokens", "min_tokens")]:
                if key in raw:
                    cfg[k] = type(cfg[k])(raw[key])
    except Exception:
        pass

    window = f"+{args.extend}" if args.extend else "原生"
    print(f"\n{'='*50}")
    print(f"  TGAI 对话模式 | 窗口: {window} | 输入 /quit 退出")
    print(f"  temp={cfg['temperature']} max_t={cfg['max_tokens']}")
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
                print(f"  temp={cfg['temperature']:.1f}  max_t={cfg['max_tokens']}  "
                      f"top_k={cfg['top_k']}  top_p={cfg['top_p']:.1f}  "
                      f"freq={cfg['freq_penalty']:.2f}  rep={cfg['rep_penalty']:.2f}")
                print(f"  历史轮数: {len(history)}")
                continue
            else:
                print(f"  未知命令: {cmd}，输入 /help 查看")
                continue

        # 构建格式化 prompt
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
                print(f"  [生成了 {token_count} 个 token | mint={cfg['min_tokens']} | maxt={cfg['max_tokens']}]")
        except Exception as e:
            print(f"\n[错误] {e}")
            continue

        reply = reply.strip()
        if not reply:
            reply = "..."

        history.append(f"用户:{user}\nTGAI?{reply}")

        # 只保留最近 8 轮
        if len(history) > 8:
            # 8轮 = 16行
            if len(history) > 16:
                history = history[-16:]


if __name__ == "__main__":
    main()
