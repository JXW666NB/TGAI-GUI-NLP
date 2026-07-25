"""
TGAI WebUI — 基于 Flask + Socket.IO 的远程管理界面
=====================================================
支持:
  - 手机/平板远程访问
  - 训练监控 (进度/loss/启停)
  - 对话交互 (流式输出 + 调试)
  - 分词器测试
  - 模型加载/切换

启动: python webui.py --port 5000
访问: http://你的电脑IP:5000
"""

import os
import sys
import json
import time
import math
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from collections import deque

from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit

# 导入项目模块
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import ChineseTokenizer, BOS_ID, EOS_ID
from model import TGAILanguageModel, TGAIConfig, create_model
from inference import TextGenerator

# ——— 配置 ———
HOST = '0.0.0.0'
PORT = 5000
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'train_all.jsonl')
TOKENIZER_PATH = os.path.join(CKPT_DIR, 'tokenizer.json')

# ——— Flask 应用 ———
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tgai-webui-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ——— 全局状态 ———
_g_model: Optional[TGAILanguageModel] = None
_g_tokenizer: Optional[ChineseTokenizer] = None
_g_device: str = 'cpu'
_g_chat_generator: Optional[TextGenerator] = None

_g_train_thread: Optional[threading.Thread] = None
_g_train_stop = threading.Event()
_g_train_status: Dict[str, Any] = {
    'running': False,
    'epoch': 0, 'total_epochs': 0,
    'global_step': 0, 'total_steps': 0,
    'loss': 0.0, 'ppl': 0.0,
    'lr': 0.0,
    'message': '等待开始',
}
_g_loss_history: List[tuple] = []  # (step, loss)


# ——— 工具函数 ———
def _model_forward(model, input_ids):
    result = model(input_ids)
    if isinstance(result, tuple):
        return result[0]
    return result


def _collect_moe_loss(model):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    if not model.training:
        return total
    for block in model.blocks:
        total = total + block.moe.load_balance_loss
    return total


def _get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'


# ═══════════════════════════════════════════════════════════
# 训练线程
# ═══════════════════════════════════════════════════════════
def _train_worker(config: dict):
    global _g_model, _g_tokenizer, _g_device, _g_train_status, _g_loss_history
    try:
        import random
        from torch.utils.data import DataLoader
        from train import TextDataset, CosineWarmupScheduler

        use_gpu = config.get('use_gpu', False) and torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu else 'cpu')
        _g_device = str(device)
        batch_size = config.get('batch_size', 16)

        if not use_gpu:
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(min(cpu_count, 4))
            if batch_size > 8:
                batch_size = 8
        else:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < 2.5 and batch_size > 8:
                batch_size = 8
            elif gpu_mem < 5 and batch_size > 16:
                batch_size = 16

        _g_train_status['message'] = f'设备: {device} | batch={batch_size}'

        # 数据加载
        data_path = config.get('data_path', DATA_PATH)
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [json.loads(line)['text'] for line in f if line.strip()]
        else:
            from train import _get_demo_texts
            texts = _get_demo_texts()

        # 分词器
        tok_path = config.get('tokenizer_path', TOKENIZER_PATH)
        if os.path.exists(tok_path):
            tokenizer = ChineseTokenizer.load(tok_path)
        else:
            tokenizer = ChineseTokenizer(vocab_size=config.get('vocab_size', 16384))
            tokenizer.train(texts)
            os.makedirs(os.path.dirname(tok_path) or '.', exist_ok=True)
            tokenizer.save(tok_path)
        _g_tokenizer = tokenizer

        # 数据集
        random.seed(42)
        indices = list(range(len(texts)))
        random.shuffle(indices)
        split = int(0.9 * len(texts))
        train_texts = [texts[i] for i in indices[:split]]
        val_texts = [texts[i] for i in indices[split:]]

        seq_len = config.get('seq_len', 256)
        train_dataset = TextDataset(train_texts, tokenizer, seq_len)
        val_dataset = TextDataset(val_texts, tokenizer, seq_len)

        n_workers = min(2, os.cpu_count() or 1)
        dl_kw = dict(pin_memory=use_gpu, num_workers=n_workers,
                      persistent_workers=n_workers > 0)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, **dl_kw)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True, **dl_kw)

        socketio.emit('train_log', {'msg': f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}'})

        # 模型
        model = create_model(
            vocab_size=tokenizer.vocab_size_actual,
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 8),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 1024),
            max_seq_len=seq_len,
            dropout=config.get('dropout', 0.1),
            n_experts=config.get('n_experts', 4),
            n_activated=config.get('n_activated', 2),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        socketio.emit('train_log', {'msg': f'参数: {n_params:,} (~{n_params/1e6:.1f}M)'})

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 3e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95),
        )
        epochs = config.get('epochs', 30)
        total_steps = len(train_loader) * epochs
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=config.get('warmup_steps', 500),
            total_steps=total_steps,
        )

        ckpt_dir = config.get('checkpoint_dir', CKPT_DIR)
        os.makedirs(ckpt_dir, exist_ok=True)
        watchdog_path = os.path.join(ckpt_dir, 'last_step.pt')

        # 续训检测
        start_epoch = 1
        global_step = 0
        best_ppl = float('inf')
        resume_path = config.get('resume_from')
        if not resume_path and os.path.exists(watchdog_path):
            resume_path = watchdog_path
        if resume_path and os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'], strict=True)
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 1)
                global_step = ckpt.get('global_step', 0)
                best_ppl = ckpt.get('best_ppl', float('inf'))
                for _ in range(global_step):
                    scheduler.step()
                socketio.emit('train_log', {'msg': f'续训: epoch={start_epoch}, step={global_step}'})
            except Exception as e:
                socketio.emit('train_log', {'msg': f'续训失败: {e}, 从头训练'})
                start_epoch = 1
                global_step = 0

        save_every_steps = 500
        next_save_at = ((global_step // save_every_steps) + 1) * save_every_steps
        scaler = torch.amp.GradScaler('cuda') if use_gpu else None
        _g_model = model
        _g_train_status['total_epochs'] = epochs
        _g_train_status['total_steps'] = total_steps
        _g_loss_history = []

        grad_accum = config.get('grad_accum_steps', 1)
        _g_train_status['running'] = True
        _g_train_status['message'] = '训练中...'

        for epoch in range(start_epoch, epochs + 1):
            if _g_train_stop.is_set():
                break

            model.train()
            epoch_loss = 0.0
            accum_loss = 0.0
            t0 = time.time()

            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                if _g_train_stop.is_set():
                    break

                vocab_size = model.config.vocab_size
                input_ids = input_ids.clamp(0, vocab_size - 1).to(device, non_blocking=use_gpu)
                target_ids = target_ids.clamp(0, vocab_size - 1).to(device, non_blocking=use_gpu)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = _model_forward(model, input_ids)
                        ce_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            target_ids.view(-1), ignore_index=0,
                        )
                        moe_loss = _collect_moe_loss(model) if model.training else torch.tensor(0.0, device=device)
                        loss = (ce_loss + moe_loss) / grad_accum
                    scaler.scale(loss).backward()
                    accum_loss += loss.item() * grad_accum
                else:
                    logits = _model_forward(model, input_ids)
                    ce_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1), ignore_index=0,
                    )
                    moe_loss = _collect_moe_loss(model) if model.training else torch.tensor(0.0, device=device)
                    loss = (ce_loss + moe_loss) / grad_accum
                    loss.backward()
                    accum_loss += loss.item() * grad_accum

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    epoch_loss += accum_loss
                    accum_loss = 0.0

                    # 看门狗保存
                    _save_checkpoint_atomic(watchdog_path, model, optimizer, config['checkpoint_dir'],
                                            epoch, global_step, best_ppl)

                    # 步级保存
                    if global_step >= next_save_at:
                        step_ckpt = os.path.join(ckpt_dir, f'checkpoint_step{global_step}.pt')
                        _save_checkpoint(step_ckpt, model, optimizer, epoch, global_step, best_ppl)
                        next_save_at += save_every_steps
                        socketio.emit('train_log', {'msg': f'→ 步级保存: checkpoint_step{global_step}.pt'})

                    # 实时推送进度
                    _g_train_status['global_step'] = global_step
                    _g_train_status['lr'] = scheduler.get_lr()
                    _g_loss_history.append((global_step, loss.item() * grad_accum))
                    if len(_g_loss_history) > 500:
                        _g_loss_history = _g_loss_history[-500:]
                    socketio.emit('train_progress', {
                        'epoch': epoch,
                        'total_epochs': epochs,
                        'step': global_step,
                        'total_steps': total_steps,
                        'loss': round(loss.item() * grad_accum, 4),
                        'lr': round(scheduler.get_lr(), 6),
                    })

            if _g_train_stop.is_set():
                break

            num_updates = (len(train_loader) + grad_accum - 1) // grad_accum
            avg_loss = epoch_loss / max(num_updates, 1)
            _g_train_status['epoch'] = epoch
            _g_train_status['loss'] = round(avg_loss, 4)

            # 验证
            val_ppl = 0
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0.0
                val_tokens = 0
                with torch.no_grad():
                    for i, (vi, vt) in enumerate(val_loader):
                        if i >= 3:
                            break
                        vi, vt = vi.to(device), vt.to(device)
                        logits = _model_forward(model, vi)
                        val_loss += F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            vt.view(-1), ignore_index=0, reduction='sum',
                        ).item()
                        val_tokens += (vt != 0).sum().item()
                val_ppl = math.exp(val_loss / max(val_tokens, 1))
                _g_train_status['ppl'] = round(val_ppl, 2)

            elapsed = time.time() - t0
            socketio.emit('train_log', {
                'msg': f'Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | '
                       f'PPL: {val_ppl:.2f} | Time: {elapsed:.0f}s'
            })

            # 轮级保存
            if epoch % max(1, epochs // 4) == 0:
                ep_ckpt = os.path.join(ckpt_dir, f'checkpoint_epoch{epoch}.pt')
                _save_checkpoint(ep_ckpt, model, optimizer, epoch, global_step, best_ppl)

            if val_ppl > 0 and val_ppl < best_ppl:
                best_ppl = val_ppl
                best_ckpt = os.path.join(ckpt_dir, 'best_model.pt')
                _save_checkpoint(best_ckpt, model, optimizer, epoch, global_step, best_ppl)

        _g_train_status['running'] = False
        _g_train_status['message'] = f'训练完成! 最佳PPL: {best_ppl:.2f}'
        socketio.emit('train_done', {'best_ppl': round(best_ppl, 2)})

    except Exception as e:
        _g_train_status['running'] = False
        _g_train_status['message'] = f'错误: {e}'
        socketio.emit('train_error', {'error': str(e)})
        traceback.print_exc()


def _save_checkpoint_atomic(path, model, optimizer, ckpt_dir, epoch, global_step, best_ppl):
    tmp = path + '.tmp'
    mc = model.config
    data = {
        'epoch': epoch, 'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': mc.vocab_size, 'd_model': mc.d_model,
            'n_layers': mc.n_layers, 'n_heads': mc.n_heads,
            'd_ff': mc.d_ff, 'max_seq_len': mc.max_seq_len,
            'dropout': mc.dropout, 'n_experts': mc.n_experts,
            'n_activated': mc.n_activated,
        },
    }
    torch.save(data, tmp)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


def _save_checkpoint(path, model, optimizer, epoch, global_step, best_ppl):
    mc = model.config
    torch.save({
        'epoch': epoch, 'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0, 'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': mc.vocab_size, 'd_model': mc.d_model,
            'n_layers': mc.n_layers, 'n_heads': mc.n_heads,
            'd_ff': mc.d_ff, 'max_seq_len': mc.max_seq_len,
            'dropout': mc.dropout, 'n_experts': mc.n_experts,
            'n_activated': mc.n_activated,
        },
    }, path)


# ═══════════════════════════════════════════════════════════
# REST API
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def api_status():
    return jsonify({
        **{k: v for k, v in _g_train_status.items()},
        'model_loaded': _g_model is not None,
        'device': _g_device,
    })


@app.route('/api/models')
def api_models():
    ckpts = []
    if os.path.exists(CKPT_DIR):
        for f in sorted(os.listdir(CKPT_DIR), reverse=True):
            if f.endswith('.pt') and not f.endswith('.tmp'):
                fp = os.path.join(CKPT_DIR, f)
                size_mb = os.path.getsize(fp) / 1e6
                ckpts.append({
                    'name': f,
                    'size': round(size_mb, 1),
                    'path': fp,
                })
    return jsonify(ckpts)


@app.route('/api/model/load', methods=['POST'])
def api_model_load():
    global _g_model, _g_tokenizer, _g_device, _g_chat_generator
    data = request.get_json()
    path = data.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({'ok': False, 'error': '文件不存在'}), 400

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = ChineseTokenizer.load(TOKENIZER_PATH)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        mc = ckpt.get('model_config', {})
        sd = ckpt['model_state_dict']

        model = create_model(
            vocab_size=sd['token_embedding.weight'].shape[0],
            d_model=mc.get('d_model', 256),
            n_layers=mc.get('n_layers', 8),
            n_heads=mc.get('n_heads', 8),
            d_ff=mc.get('d_ff', 1024),
            max_seq_len=mc.get('max_seq_len', 256),
            dropout=0.0, n_experts=mc.get('n_experts', 4),
            n_activated=mc.get('n_activated', 2),
        )
        model.load_state_dict(sd)
        model.to(device)
        model.eval()

        _g_model = model
        _g_tokenizer = tokenizer
        _g_device = device
        _g_chat_generator = TextGenerator(model, tokenizer)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return jsonify({
            'ok': True,
            'name': os.path.basename(path),
            'epoch': ckpt.get('epoch', '?'),
            'params': n_params,
            'device': device,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# Socket.IO 事件
# ═══════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    emit('status', {**{k: v for k, v in _g_train_status.items()},
                    'model_loaded': _g_model is not None})
    if _g_loss_history:
        emit('loss_history', [{'step': s, 'loss': l} for s, l in _g_loss_history[-100:]])


@socketio.on('start_train')
def on_start_train(config):
    global _g_train_thread, _g_train_stop
    if _g_train_status['running']:
        emit('train_log', {'msg': '训练已在运行中'})
        return

    _g_train_stop.clear()
    _g_train_status['running'] = True
    _g_train_status['message'] = '准备训练...'
    _g_loss_history = []

    cfg = {
        'd_model': config.get('d_model', 256),
        'n_layers': config.get('n_layers', 8),
        'n_heads': config.get('n_heads', 8),
        'd_ff': config.get('d_ff', 1024),
        'dropout': config.get('dropout', 0.1),
        'seq_len': config.get('seq_len', 256),
        'vocab_size': config.get('vocab_size', 16384),
        'n_experts': config.get('n_experts', 4),
        'n_activated': config.get('n_activated', 2),
        'epochs': config.get('epochs', 30),
        'batch_size': config.get('batch_size', 16),
        'lr': config.get('lr', 0.0003),
        'weight_decay': config.get('weight_decay', 0.01),
        'warmup_steps': config.get('warmup_steps', 500),
        'checkpoint_dir': CKPT_DIR,
        'tokenizer_path': TOKENIZER_PATH,
        'data_path': DATA_PATH,
        'grad_accum_steps': 1,
        'use_gpu': config.get('use_gpu', True) and torch.cuda.is_available(),
        'save_every_steps': 500,
    }

    _g_train_thread = threading.Thread(target=_train_worker, args=(cfg,), daemon=True)
    _g_train_thread.start()
    emit('train_log', {'msg': '训练已启动'})


@socketio.on('stop_train')
def on_stop_train():
    _g_train_stop.set()
    _g_train_status['running'] = False
    _g_train_status['message'] = '已停止'
    emit('train_log', {'msg': '正在停止... (当前epoch完成后停止)'})


@socketio.on('send_message')
def on_send_message(data):
    if _g_model is None or _g_tokenizer is None:
        emit('chat_error', {'error': '请先加载模型'})
        return

    prompt = data.get('text', '').strip()
    if not prompt:
        return

    temperature = data.get('temperature', 0.8)
    max_tokens = data.get('max_tokens', 128)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    frequency_penalty = data.get('frequency_penalty', 0.25)
    repetition_penalty = data.get('repetition_penalty', 1.05)
    min_new_tokens = data.get('min_new_tokens', 5)
    show_debug = data.get('debug', False)

    try:
        generator = TextGenerator(_g_model, _g_tokenizer)
        response = ''
        debug_info = []

        for chunk in generator.generate(
            prompt, max_new_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens, stream=True,
        ):
            response += chunk
            emit('chat_chunk', {'token': chunk})
            if show_debug:
                dbg = generator.pop_debug()
                if dbg:
                    debug_info.extend(dbg)

        if show_debug:
            dbg = generator.pop_debug()
            if dbg:
                debug_info.extend(dbg)
            if debug_info:
                # 精简调试信息
                simple_debug = []
                for d in debug_info:
                    candidates = d['top_sampled']
                    cand_str = ' | '.join(f"{c[1]}({c[2]*100:.0f}%)" for c in candidates)
                    simple_debug.append(
                        f"#{d['step']} {d['text']}({d['prob']*100:.0f}%) "
                        f"[{cand_str}] {d['step_ms']:.0f}ms"
                    )
                emit('chat_debug', {'lines': simple_debug})

        emit('chat_done', {'response': response or '[模型未生成回复]'})

    except Exception as e:
        emit('chat_error', {'error': str(e)})


@socketio.on('tokenize')
def on_tokenize(data):
    if _g_tokenizer is None:
        try:
            _tokenizer_tmp = ChineseTokenizer.load(TOKENIZER_PATH)
        except:
            emit('tokenize_result', {'error': '分词器未加载'})
            return
    else:
        _tokenizer_tmp = _g_tokenizer

    text = data.get('text', '').strip()
    if not text:
        return

    ids = _tokenizer_tmp.encode(text, add_special=True)
    ids_ns = _tokenizer_tmp.encode(text, add_special=False)
    decoded = _tokenizer_tmp.decode(ids)

    tokens = []
    for tid in ids:
        if tid in _tokenizer_tmp.id_to_token:
            tok = _tokenizer_tmp.id_to_token[tid]
        else:
            tok = '<UNK>'
        tokens.append({'id': tid, 'token': tok})

    emit('tokenize_result', {
        'text': text,
        'ids': ids,
        'ids_no_special': ids_ns,
        'decoded': decoded,
        'tokens': tokens,
        'vocab_size': _tokenizer_tmp.vocab_size_actual,
    })


# ═══════════════════════════════════════════════════════════
# HTML 模板 (移动端适配深色主题)
# ═══════════════════════════════════════════════════════════

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>TGAI WebUI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#1a1a2e;color:#cdd6f4;max-width:800px;margin:0 auto;min-height:100vh}
.tabs{display:flex;background:#16213e;position:sticky;top:0;z-index:10}
.tab{flex:1;text-align:center;padding:14px 8px;cursor:pointer;font-size:14px;color:#888;border-bottom:2px solid transparent;transition:all .2s}
.tab.active{color:#89b4fa;border-bottom-color:#89b4fa}
.panel{display:none;padding:12px;flex:1}
.panel.active{display:block}
/* 训练面板 */
.train-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px}
.btn{padding:8px 18px;border:none;border-radius:6px;font-size:14px;font-weight:bold;cursor:pointer;transition:all .2s}
.btn-green{background:#40a02b;color:#fff}
.btn-red{background:#d20f39;color:#fff}
.btn-blue{background:#1e66f5;color:#fff}
.btn-gray{background:#45475a;color:#cdd6f4}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-sm{padding:4px 12px;font-size:12px}
.progress-bar{height:20px;background:#313244;border-radius:10px;overflow:hidden;margin:8px 0}
.progress-bar div{height:100%;background:linear-gradient(90deg,#89b4fa,#74c7ec);border-radius:10px;transition:width.3s;display:flex;align-items:center;justify-content:center;font-size:11px;color:#1e1e2e}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.stat{background:#313244;padding:10px;border-radius:8px;text-align:center}
.stat .val{font-size:22px;font-weight:bold;color:#89b4fa}
.stat .lbl{font-size:11px;color:#888;margin-top:4px}
#train-log{background:#11111b;border-radius:8px;padding:10px;height:200px;overflow-y:auto;font-family:Consolas,monospace;font-size:12px;line-height:1.6;margin-top:8px;white-space:pre-wrap}
/* 对话面板 */
.chat-area{display:flex;flex-direction:column;height:calc(100vh - 60px)}
#chat-messages{flex:1;overflow-y:auto;padding:8px 0}
.chat-msg{margin-bottom:10px;line-height:1.6}
.chat-user{color:#89b4fa;font-weight:bold}
.chat-bot{color:#f9e2af;font-weight:bold}
.chat-text{color:#cdd6f4;word-break:break-word}
.chat-debug{font-size:11px;color:#6c7086;margin-top:4px;line-height:1.4}
.chat-time{font-size:10px;color:#585b70;float:right}
.chat-input-area{display:flex;gap:8px;padding:10px 0;border-top:1px solid #313244;margin-top:8px}
.chat-input-area input{flex:1;background:#313244;border:1px solid #45475a;color:#cdd6f4;padding:10px 14px;border-radius:8px;font-size:14px;outline:none}
.chat-controls{display:flex;align-items:center;gap:10px;padding:4px 0}
.chat-controls label{font-size:12px;color:#888}
.chat-controls input[type=range]{width:80px}
/* 分词器 */
#tokenize-result{margin-top:10px;background:#11111b;border-radius:8px;padding:10px;font-size:12px;line-height:1.8;overflow-x:auto}
/* 配置面板 */
.config-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.config-item{background:#313244;padding:8px 10px;border-radius:6px;display:flex;justify-content:space-between;align-items:center;font-size:13px}
.config-item input,.config-item select{width:70px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;padding:3px 6px;border-radius:4px;text-align:center;font-size:13px}
.notification{position:fixed;top:12px;right:12px;background:#40a02b;color:#fff;padding:10px 18px;border-radius:8px;font-size:13px;z-index:100;box-shadow:0 4px 12px rgba(0,0,0,.3);transition:opacity .3s;opacity:0;pointer-events:none}
.notification.show{opacity:1}
</style>
</head>
<body>

<div class="tabs">
  <div class="tab active" onclick="switchTab('train')">🎯 训练</div>
  <div class="tab" onclick="switchTab('chat')">💬 对话</div>
  <div class="tab" onclick="switchTab('tokenizer')">🔤 分词</div>
</div>

<!-- 训练面板 -->
<div id="panel-train" class="panel active">
  <div class="train-header">
    <span id="train-status" style="font-size:13px;color:#888">等待开始</span>
    <div>
      <button class="btn btn-green" id="btn-start" onclick="startTrain()">▶ 开始</button>
      <button class="btn btn-red" id="btn-stop" onclick="stopTrain()" disabled>⏹ 停止</button>
    </div>
  </div>
  <div class="progress-bar"><div id="train-progress" style="width:0%">0%</div></div>
  <div class="stats">
    <div class="stat"><div class="val" id="stat-loss">-</div><div class="lbl">Loss</div></div>
    <div class="stat"><div class="val" id="stat-ppl">-</div><div class="lbl">PPL</div></div>
    <div class="stat"><div class="val" id="stat-step">0</div><div class="lbl">Step</div></div>
    <div class="stat"><div class="val" id="stat-lr">-</div><div class="lbl">LR</div></div>
  </div>
  <details style="margin:8px 0">
    <summary style="cursor:pointer;color:#888;font-size:13px">⚙ 训练参数</summary>
    <div class="config-grid" style="margin-top:6px">
      <div class="config-item">隐藏维度 <input id="cfg-d_model" value="256"></div>
      <div class="config-item">层数 <input id="cfg-layers" value="8"></div>
      <div class="config-item">注意力头 <input id="cfg-heads" value="8"></div>
      <div class="config-item">FFN维度 <input id="cfg-dff" value="1024"></div>
      <div class="config-item">Dropout <input id="cfg-dropout" value="0.1"></div>
      <div class="config-item">序列长度 <input id="cfg-seq" value="256"></div>
      <div class="config-item">训练轮数 <input id="cfg-epochs" value="100"></div>
      <div class="config-item">批次大小 <input id="cfg-batch" value="16"></div>
      <div class="config-item">学习率 <input id="cfg-lr" value="0.0003"></div>
      <div class="config-item">权重衰减 <input id="cfg-wd" value="0.01"></div>
      <div class="config-item">Warmup <input id="cfg-warmup" value="500"></div>
      <div class="config-item">MoE专家 <input id="cfg-experts" value="4"></div>
    </div>
  </details>
  <div id="train-log"></div>
</div>

<!-- 对话面板 -->
<div id="panel-chat" class="panel">
  <div class="chat-controls">
    <button class="btn btn-gray btn-sm" id="btn-load-model" onclick="showModelList()">📂 加载模型</button>
    <span id="chat-model-info" style="font-size:11px;color:#888"></span>
    <details style="margin-left:auto">
      <summary style="cursor:pointer;color:#888;font-size:12px">⚙ 参数</summary>
      <div class="config-grid" style="margin-top:6px;gap:4px">
        <div class="config-item">温度 <input id="chat-temp" type="number" min="0.1" max="2" step="0.1" value="0.9" style="width:55px"></div>
        <div class="config-item">Max Token <input id="chat-max-tokens" type="number" min="16" max="1024" step="16" value="128" style="width:55px"></div>
        <div class="config-item">Top-K <input id="chat-topk" type="number" min="1" max="200" step="1" value="50" style="width:55px"></div>
        <div class="config-item">Top-P <input id="chat-topp" type="number" min="0" max="1" step="0.05" value="0.95" style="width:55px"></div>
        <div class="config-item">频率惩罚 <input id="chat-freq-pen" type="number" min="0" max="1" step="0.05" value="0.25" style="width:55px"></div>
        <div class="config-item">重复惩罚 <input id="chat-rep-pen" type="number" min="0.1" max="3" step="0.05" value="1.05" style="width:55px"></div>
        <div class="config-item">最小长度 <input id="chat-min-tokens" type="number" min="0" max="50" step="1" value="5" style="width:55px"></div>
      </div>
    </details>
    <label><input type="checkbox" id="chat-debug"> 调试</label>
  </div>
  <div class="chat-area">
    <div id="chat-messages"></div>
    <div class="chat-input-area">
      <input id="chat-input" placeholder="输入消息..." onkeydown="if(event.key==='Enter')sendMessage()">
      <button class="btn btn-blue" onclick="sendMessage()">发送</button>
    </div>
  </div>
  <!-- 模型列表弹出 -->
  <div id="model-overlay" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:20;align-items:center;justify-content:center" onclick="closeModelList()">
    <div style="background:#1e1e2e;border-radius:12px;padding:16px;max-width:400px;width:90%;max-height:70vh;overflow-y:auto" onclick="event.stopPropagation()">
      <h3 style="margin-bottom:12px">选择模型</h3>
      <div id="model-list" style="font-size:13px">加载中...</div>
      <button class="btn btn-gray btn-sm" style="margin-top:10px" onclick="closeModelList()">关闭</button>
    </div>
  </div>
</div>

<!-- 分词器面板 -->
<div id="panel-tokenizer" class="panel">
  <input id="tok-input" placeholder="输入文本..." style="width:100%;background:#313244;border:1px solid #45475a;color:#cdd6f4;padding:10px;border-radius:8px;font-size:14px" onkeydown="if(event.key==='Enter')doTokenize()">
  <button class="btn btn-blue btn-sm" style="margin-top:8px" onclick="doTokenize()">分词</button>
  <div id="tokenize-result"></div>
</div>

<div id="notification" class="notification"></div>

<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
const socket = io();
let currentModel = null;

// Tab 切换
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['train','chat','tokenizer'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

// 通知
function notify(msg, duration=2000) {
  const n = document.getElementById('notification');
  n.textContent = msg; n.classList.add('show');
  setTimeout(() => n.classList.remove('show'), duration);
}

// ——— 训练 ———
function startTrain() {
  if (socket) socket.emit('start_train', {
    d_model: +document.getElementById('cfg-d_model').value,
    n_layers: +document.getElementById('cfg-layers').value,
    n_heads: +document.getElementById('cfg-heads').value,
    d_ff: +document.getElementById('cfg-dff').value,
    dropout: +document.getElementById('cfg-dropout').value,
    seq_len: +document.getElementById('cfg-seq').value,
    vocab_size: 16384,
    n_experts: +document.getElementById('cfg-experts').value,
    n_activated: 2,
    epochs: +document.getElementById('cfg-epochs').value,
    batch_size: +document.getElementById('cfg-batch').value,
    lr: +document.getElementById('cfg-lr').value,
    weight_decay: +document.getElementById('cfg-wd').value,
    warmup_steps: +document.getElementById('cfg-warmup').value,
  });
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  document.getElementById('train-status').textContent = '训练中...';
}

function stopTrain() {
  socket.emit('stop_train');
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('train-status').textContent = '已停止';
}

socket.on('train_progress', data => {
  document.getElementById('stat-step').textContent = data.step;
  document.getElementById('stat-loss').textContent = data.loss;
  document.getElementById('stat-lr').textContent = data.lr;
  document.getElementById('stat-ppl').textContent = data.ppl || '-';
  document.getElementById('train-status').textContent =
    `Epoch ${data.epoch}/${data.total_epochs} | Step ${data.step}`;
  const pct = Math.round(data.step / data.total_steps * 100);
  const bar = document.getElementById('train-progress');
  bar.style.width = pct + '%';
  bar.textContent = pct + '%';
  document.getElementById('btn-stop').disabled = false;
});

socket.on('train_log', data => {
  const log = document.getElementById('train-log');
  log.textContent += data.msg + '\n';
  log.scrollTop = log.scrollHeight;
});

socket.on('train_done', data => {
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('train-status').textContent =
    '完成! 最佳PPL: ' + data.best_ppl;
  notify('训练完成!');
});

socket.on('train_error', data => {
  document.getElementById('train-log').textContent += '\n[错误] ' + data.error;
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
});

// ——— 对话 ———
function showModelList() {
  document.getElementById('model-overlay').style.display = 'flex';
  fetch('/api/models').then(r => r.json()).then(models => {
    const html = models.map(m =>
      `<div style="padding:8px;border-bottom:1px solid #313244;cursor:pointer;display:flex;justify-content:space-between" onclick="loadModel('${m.path}','${m.name}')">
        <span>${m.name}</span><span style="color:#888">${m.size}MB</span>
      </div>`
    ).join('') || '<div style="color:#888">没有找到模型文件</div>';
    document.getElementById('model-list').innerHTML = html;
  });
}

function closeModelList() {
  document.getElementById('model-overlay').style.display = 'none';
}

function loadModel(path, name) {
  closeModelList();
  notify('加载中...');
  fetch('/api/model/load', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path})
  }).then(r => r.json()).then(data => {
    if (data.ok) {
      currentModel = data;
      document.getElementById('chat-model-info').textContent =
        `${data.name} | Epoch:${data.epoch} | ${(data.params/1e6).toFixed(1)}M | ${data.device}`;
      notify('模型已加载');
    } else {
      notify('加载失败: ' + data.error, 3000);
    }
  });
}

function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  const msgs = document.getElementById('chat-messages');
  msgs.innerHTML += `<div class="chat-msg"><span class="chat-user">你:</span> <span class="chat-text">${esc(text)}</span></div>`;

  // 创建 bot 消息容器
  const botDiv = document.createElement('div');
  botDiv.className = 'chat-msg';
  botDiv.innerHTML = '<span class="chat-bot">TGAI:</span> <span class="chat-text" id="bot-stream"></span>';
  msgs.appendChild(botDiv);
  const streamEl = botDiv.querySelector('#bot-stream');

  const debug = document.getElementById('chat-debug').checked;
  const temp = parseFloat(document.getElementById('chat-temp').value);
  const maxTokens = parseInt(document.getElementById('chat-max-tokens').value) || 128;
  const topk = parseInt(document.getElementById('chat-topk').value) || 50;
  const topp = parseFloat(document.getElementById('chat-topp').value) || 0.95;
  const freqPen = parseFloat(document.getElementById('chat-freq-pen').value) || 0.25;
  const repPen = parseFloat(document.getElementById('chat-rep-pen').value) || 1.05;
  const minTokens = parseInt(document.getElementById('chat-min-tokens').value) || 5;

  socket.emit('send_message', {text, temperature: temp, max_tokens: maxTokens,
    top_k: topk, top_p: topp, frequency_penalty: freqPen, repetition_penalty: repPen,
    min_new_tokens: minTokens, debug});

  // 移除旧监听器
  socket.off('chat_chunk');
  socket.off('chat_debug');
  socket.off('chat_done');

  socket.on('chat_chunk', data => {
    streamEl.textContent += data.token;
    msgs.scrollTop = msgs.scrollHeight;
  });

  socket.on('chat_debug', data => {
    if (data.lines) {
      const debugEl = document.createElement('div');
      debugEl.className = 'chat-debug';
      debugEl.textContent = data.lines.join('\n');
      botDiv.appendChild(debugEl);
      msgs.scrollTop = msgs.scrollHeight;
    }
  });

  socket.on('chat_done', data => {
    streamEl.id = '';  // 移除id防止冲突
  });
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ——— 分词器 ———
function doTokenize() {
  const text = document.getElementById('tok-input').value.trim();
  if (!text) return;
  socket.emit('tokenize', {text});
}

socket.on('tokenize_result', data => {
  let html = `<div style="color:#888">词表: ${data.vocab_size} | 编码: ${data.ids.join(' ')}</div>`;
  html += `<div style="color:#a6e3a1">解码: ${data.decoded}</div>`;
  html += '<div style="margin-top:6px">';
  data.tokens.forEach(t => {
    const special = ['<PAD>','<UNK>','<BOS>','<EOS>'].includes(t.token);
    const color = special ? '#f38ba8' : '#89b4fa';
    html += `<span style="color:${color};margin-right:4px">[${t.id}]${t.token}</span>`;
  });
  html += '</div>';
  document.getElementById('tokenize-result').innerHTML = html;
});

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/status').then(r => r.json()).then(s => {
    if (s.running) {
      document.getElementById('btn-start').disabled = true;
      document.getElementById('btn-stop').disabled = false;
      document.getElementById('train-status').textContent = s.message;
    }
    if (s.model_loaded) document.getElementById('chat-model-info').textContent =
      '已有模型加载 (' + s.device + ')';
  });
});
</script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--host', type=str, default=HOST)
    args = ap.parse_args()

    local_ip = _get_local_ip()
    print(f"""
╔══════════════════════════════════════╗
║       TGAI WebUI  v4               ║
╠══════════════════════════════════════╣
║  本机访问: http://127.0.0.1:{args.port}
║  手机访问: http://{local_ip}:{args.port}
╚══════════════════════════════════════╝
""")
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
