"""
TGAI NLP - PyQt6 图形界面
==========================
功能标签页:
  1. 训练 - 配置模型参数、启动训练、实时查看 loss 曲线
  2. 对话 - 加载模型后交互式聊天
  3. 分词器 - 测试分词效果、查看词表
  4. 数据 - 编辑/查看训练语料
"""

import sys
import os
import json
import time
import shutil
import ctypes
import threading
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

import requests
import websocket

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QGridLayout,
    QProgressBar, QFileDialog, QMessageBox, QSplitter, QFrame,
    QScrollArea, QSlider, QCheckBox, QListWidget, QListWidgetItem,
    QPlainTextEdit, QSizePolicy, QStackedWidget,
    QDialog, QFormLayout, QDialogButtonBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer,
)
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QPalette, QPainter, QPen, QBrush,
)

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(__file__))


# ─── V4 前向传播辅助 ─────────────────────────────────────
def _migrate_rope_buffers(state_dict: dict) -> dict:
    """向后兼容：将旧格式 RoPE cos/sin (seq_len, d_k//2) 转换为新格式 (1,1,seq_len,d_k)"""
    for key in list(state_dict.keys()):
        if '.rope.cos' in key or '.rope.sin' in key:
            old = state_dict[key]
            if old.dim() == 2:
                # 旧格式 (max_seq_len, d_k//2) → 新格式 (1,1,max_seq_len,d_k)
                new = old.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
                state_dict[key] = new
    return state_dict

def _model_forward(model, input_ids):
    """V4 model.forward() 返回 (logits, kv_caches)，安全解包"""
    result = model(input_ids)
    if isinstance(result, tuple):
        return result[0]
    return result

def _collect_moe_loss(model):
    """收集所有 MoE 层的负载均衡损失"""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    if not model.training:
        return total
    for block in model.blocks:
        total = total + block.moe.load_balance_loss
    return total

# ─── 知识蒸馏损失计算 ────────────────────────────────────
def _compute_distill_loss(logits, target_ids, teacher_logits, distill_alpha, distill_temp, model=None):
    """计算混合损失：硬标签交叉熵 + 软标签 KL 散度 + MoE 负载均衡"""
    import torch.nn.functional as F
    vocab_size = logits.size(-1)
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = target_ids.view(-1)

    hard_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=0)

    if teacher_logits is None or distill_alpha <= 0:
        # 加上 MoE 负载均衡损失
        if model is not None and model.training:
            moe_loss = _collect_moe_loss(model)
            if isinstance(moe_loss, torch.Tensor) and moe_loss.item() != 0:
                hard_loss = hard_loss + moe_loss * model.config.moe_load_balance
        return hard_loss

    flat_teacher = teacher_logits.view(-1, vocab_size)
    mask = flat_targets != 0
    if mask.sum() == 0:
        return hard_loss

    student_log_probs = F.log_softmax(flat_logits[mask] / distill_temp, dim=-1)
    teacher_probs = F.softmax(flat_teacher[mask] / distill_temp, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    kl_loss = kl_loss * (distill_temp ** 2)

    total = (1 - distill_alpha) * hard_loss + distill_alpha * kl_loss
    if model is not None and model.training:
        moe_loss = _collect_moe_loss(model)
        if isinstance(moe_loss, torch.Tensor) and moe_loss.item() != 0:
            total = total + moe_loss * model.config.moe_load_balance
    return total


# ─── Loss 曲线图组件 ────────────────────────────────────
class LossChart(QWidget):
    """简单的 Loss 曲线图，使用 QPainter 绘制"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loss_history = []  # [(train_loss, val_ppl), ...]
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)
        
    def add_loss(self, train_loss: float, val_ppl: float):
        """添加新的 loss 数据点"""
        self.loss_history.append((train_loss, val_ppl))
        self.update()  # 触发重绘
        
    def clear(self):
        """清空历史数据"""
        self.loss_history.clear()
        self.update()
        
    def paintEvent(self, event):
        """绘制曲线图"""
        if not self.loss_history:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 获取绘图区域
        w = self.width()
        h = self.height()
        margin = 50  # 边距
        chart_w = w - 2 * margin
        chart_h = h - 2 * margin
        
        if chart_w <= 0 or chart_h <= 0:
            return
            
        # 背景
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        # 绘制边框
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(margin, margin, chart_w, chart_h)
        
        # 获取数据范围
        train_losses = [x[0] for x in self.loss_history]
        val_ppls = [x[1] for x in self.loss_history if x[1] > 0]
        
        if not train_losses:
            return
            
        min_loss = min(train_losses)
        max_loss = max(train_losses)
        
        # 避免除零
        if max_loss == min_loss:
            max_loss = min_loss + 1
            
        # 绘制网格线和标签
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))
        for i in range(5):
            y = margin + int(chart_h * i / 4)
            painter.drawLine(margin, y, margin + chart_w, y)
            
        # 绘制 Y 轴标签
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Consolas", 8))
        for i in range(5):
            y = margin + int(chart_h * i / 4)
            val = max_loss - (max_loss - min_loss) * i / 4
            painter.drawText(5, y + 5, f"{val:.2f}")
            
        # 绘制 X 轴标签
        n = len(self.loss_history)
        for i in range(0, n, max(1, n // 5)):
            x = margin + int(chart_w * i / max(1, n - 1))
            painter.drawText(x - 10, h - 10, str(i + 1))
            
        # 绘制标题
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(margin, margin - 10, "训练 Loss 曲线")
        
        # 绘制 Train Loss 曲线（蓝色）
        if len(train_losses) > 1:
            painter.setPen(QPen(QColor(66, 133, 244), 2))
            for i in range(len(train_losses) - 1):
                x1 = margin + int(chart_w * i / max(1, n - 1))
                y1 = margin + int(chart_h * (max_loss - train_losses[i]) / (max_loss - min_loss))
                x2 = margin + int(chart_w * (i + 1) / max(1, n - 1))
                y2 = margin + int(chart_h * (max_loss - train_losses[i + 1]) / (max_loss - min_loss))
                painter.drawLine(x1, y1, x2, y2)
                
        # 绘制 Val PPL 曲线（绿色）
        if val_ppls and len(val_ppls) > 1:
            min_ppl = min(val_ppls)
            max_ppl = max(val_ppls)
            if max_ppl > min_ppl:
                painter.setPen(QPen(QColor(52, 168, 83), 2))
                val_idx = 0
                for i in range(len(self.loss_history)):
                    if self.loss_history[i][1] > 0:
                        x = margin + int(chart_w * i / max(1, n - 1))
                        y = margin + int(chart_h * (max_ppl - self.loss_history[i][1]) / (max_ppl - min_ppl))
                        if val_idx > 0:
                            prev_i = i - 1
                            while prev_i >= 0 and self.loss_history[prev_i][1] <= 0:
                                prev_i -= 1
                            if prev_i >= 0:
                                x_prev = margin + int(chart_w * prev_i / max(1, n - 1))
                                y_prev = margin + int(chart_h * (max_ppl - self.loss_history[prev_i][1]) / (max_ppl - min_ppl))
                                painter.drawLine(x_prev, y_prev, x, y)
                        val_idx += 1
                        
        # 图例
        painter.setPen(QColor(66, 133, 244))
        painter.drawLine(w - 120, 20, w - 90, 20)
        painter.drawText(w - 85, 25, "Train Loss")
        
        painter.setPen(QColor(52, 168, 83))
        painter.drawLine(w - 120, 35, w - 90, 35)
        painter.drawText(w - 85, 40, "Val PPL")
        
        painter.end()


# ─── 保存工具函数 ────────────────────────────────────────
def _save_checkpoint_atomic(path, model, optimizer, tokenizer, ckpt_dir, config, epoch, global_step, best_ppl):
    """原子写入：先写 .tmp 再重命名，防止写一半崩溃损坏文件"""
    tmp = path + '.tmp'
    mc = model.config
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': mc.vocab_size,
            'd_model': mc.d_model,
            'n_layers': mc.n_layers,
            'n_heads': mc.n_heads,
            'd_ff': mc.d_ff,
            'max_seq_len': mc.max_seq_len,
            'dropout': mc.dropout,
            'n_experts': mc.n_experts,
            'n_activated': mc.n_activated,
        },
    }, tmp)
    # Windows 文件锁重试（最多等 3 秒）
    for attempt in range(10):
        try:
            # 先删目标，再重命名
            if os.path.exists(path):
                os.remove(path)
            os.replace(tmp, path)
            return
        except (PermissionError, OSError):
            if attempt < 9:
                time.sleep(0.1 + attempt * 0.05)
            else:
                # 最终兜底：直接写入目标
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_ppl': best_ppl,
                    'model_config': {
                        'vocab_size': mc.vocab_size,
                        'd_model': mc.d_model,
                        'n_layers': mc.n_layers,
                        'n_heads': mc.n_heads,
                        'd_ff': mc.d_ff,
                        'max_seq_len': mc.max_seq_len,
                        'dropout': mc.dropout,
                        'n_experts': mc.n_experts,
                        'n_activated': mc.n_activated,
                    },
                }, path)


def _save_checkpoint_full(path, model, optimizer, tokenizer, ckpt_dir, config, epoch, global_step, avg_loss, best_ppl):
    """完整 checkpoint（用于步级/轮级保存）"""
    mc = model.config
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_ppl': best_ppl,
        'model_config': {
            'vocab_size': mc.vocab_size,
            'd_model': mc.d_model,
            'n_layers': mc.n_layers,
            'n_heads': mc.n_heads,
            'd_ff': mc.d_ff,
            'max_seq_len': mc.max_seq_len,
            'dropout': mc.dropout,
            'n_experts': mc.n_experts,
            'n_activated': mc.n_activated,
        },
    }, path)


# ─── 导出线程 ────────────────────────────────────────────
class ExportThread(QThread):
    """后台导出线程：将 .pt → .pte → 打包 .TG"""

    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str, str)  # success, path, error

    def __init__(self, checkpoint: str, out_dir: str, name: str, model_type: str = "auto", int8: bool = True):
        super().__init__()
        self.checkpoint = checkpoint
        self.out_dir = out_dir
        self.name = name or os.path.splitext(os.path.basename(checkpoint))[0]
        self.model_type = model_type  # "auto", "tgai", "yuaz"
        self.int8 = int8

    def run(self):
        try:
            self._do_export()
        except Exception as e:
            self.log_signal.emit(f"[错误] {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.done_signal.emit(False, "", str(e))

    @staticmethod
    def _find_scripts_dir():
        """找到 scripts 目录（优先同级，回退上级）"""
        base = os.path.dirname(os.path.abspath(__file__))
        # 优先：tgai_nlp/scripts/（repo clone 结构）
        scripts = os.path.join(base, 'scripts')
        if os.path.isdir(scripts):
            return scripts
        # 回退：../scripts/（用户原有结构）
        return os.path.join(base, '..', 'scripts')

    def _do_export(self):
        import subprocess
        import sys
        import threading

        scripts_dir = self._find_scripts_dir()

        # 选择导出脚本
        model_type = self.model_type
        if model_type == "auto":
            # 自动检测：尝试从文件名或模型结构判断
            ckpt_name = os.path.basename(self.checkpoint).lower()
            if "yuaz" in ckpt_name:
                model_type = "yuaz"
            else:
                model_type = "tgai"

        if model_type == "yuaz":
            # YUAZ 模型 → 用 YUAZ 仓库里自带的导出脚本
            self.log_signal.emit("检测到 YUAZ 模型，使用 Yuaz-inference 导出脚本...")
            script_export = os.path.join(os.path.dirname(os.path.abspath(self.checkpoint)), 'export_onnx.py')
            if not os.path.isfile(script_export):
                # 回退：在 E:\YUAZ LLM\yuaz-inference 查找
                yuaz_dir = r'E:\YUAZ LLM\yuaz-inference'
                script_export = os.path.join(yuaz_dir, 'export_onnx.py')
            if not os.path.isfile(script_export):
                self.log_signal.emit(f"[错误] 找不到 YUAZ 导出脚本: {script_export}")
                self.done_signal.emit(False, "", "找不到 YUAZ export_onnx.py")
                return
        else:
            script_export = os.path.join(scripts_dir, 'export', 'export_onnx.py')

        script_pack = os.path.join(scripts_dir, 'export', 'pack_tg.py')

        # 步骤1：导出 ONNX
        self.log_signal.emit("━" * 50)
        self.log_signal.emit("步骤 1/2: 导出 ONNX 模型 (.onnx)")
        self.log_signal.emit(f"  检查点: {self.checkpoint}")
        self.log_signal.emit(f"  输出目录: {self.out_dir}")
        self.log_signal.emit("  提示: torch.onnx.export 需要几分钟，请耐心等待...")

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'  # 防止 Windows GBK 乱码导致 torch.onnx 崩溃
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')
        env.setdefault('OPENBLAS_NUM_THREADS', '1')
        env.setdefault('NUMEXPR_NUM_THREADS', '1')
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

        cmd = [sys.executable, '-u', script_export, '--checkpoint', self.checkpoint, '--out_dir', self.out_dir]
        if self.int8:
            cmd.append('--int8')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, encoding='utf-8', errors='replace', bufsize=1,
        )

        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line:
                self.log_signal.emit(f"  {line}")

        proc.wait(timeout=600)

        if proc.returncode != 0:
            self.done_signal.emit(False, "", "导出 .onnx 失败")
            return

        # 查找生成的模型文件
        model_path = None
        tokenizer_path = None
        for f in os.listdir(self.out_dir):
            if f.endswith('.onnx'):
                model_path = os.path.join(self.out_dir, f)
            elif f == 'tokenizer.json':
                tokenizer_path = os.path.join(self.out_dir, f)

        if not model_path:
            self.done_signal.emit(False, "", f"未找到生成的 .onnx 文件（目录: {self.out_dir}）")
            return
        if not tokenizer_path:
            self.done_signal.emit(False, "", f"未找到 tokenizer.json（目录: {self.out_dir}）")
            return

        self.log_signal.emit(f"  模型: {os.path.basename(model_path)}")
        self.log_signal.emit(f"  分词器: tokenizer.json")

        # 步骤2：打包 .TG
        self.log_signal.emit("━" * 50)
        self.log_signal.emit("步骤 2/2: 打包为 .TG 文件")
        tg_output = os.path.join(self.out_dir, f"{self.name}.tg")

        pack_proc = subprocess.Popen(
            [sys.executable, '-u', script_pack,
             '--model', model_path, '--tokenizer', tokenizer_path,
             '--out', tg_output, '--name', self.name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, encoding='utf-8', errors='replace', bufsize=1,
        )

        for line in iter(pack_proc.stdout.readline, ''):
            line = line.strip()
            if line:
                self.log_signal.emit(f"  {line}")

        pack_proc.wait(timeout=120)

        if pack_proc.returncode != 0:
            self.done_signal.emit(False, "", "打包 .TG 失败")
            return

        self.done_signal.emit(True, tg_output, "")


# ─── 快速打包线程 ────────────────────────────────────────
class QuickPackThread(QThread):
    """快速打包：已有 ONNX + tokenizer → .TG"""

    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str, str)

    def __init__(self, onnx_path: str, tokenizer_path: str, name: str):
        super().__init__()
        self.onnx_path = onnx_path
        self.tokenizer_path = tokenizer_path
        self.name = name or os.path.splitext(os.path.basename(onnx_path))[0]

    def run(self):
        try:
            self._do_pack()
        except Exception as e:
            self.log_signal.emit(f"[错误] {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.done_signal.emit(False, "", str(e))

    def _do_pack(self):
        import subprocess
        import sys

        scripts_dir = ExportThread._find_scripts_dir()
        script_pack = os.path.join(scripts_dir, 'export', 'pack_tg.py')
        out_dir = os.path.dirname(os.path.abspath(self.onnx_path))
        out_tg = os.path.join(out_dir, f"{self.name}.tg")

        self.log_signal.emit(f"模型文件: {self.onnx_path}")
        self.log_signal.emit(f"分词器: {self.tokenizer_path}")
        self.log_signal.emit(f"输出: {out_tg}")
        self.log_signal.emit("━" * 40)

        # 如果 onnx 有同名的 .onnx.data 文件，一并打包
        model_paths = [self.onnx_path]
        data_path = os.path.splitext(self.onnx_path)[0] + '.onnx.data'
        if os.path.isfile(data_path):
            model_paths.append(data_path)
            self.log_signal.emit(f"附带外部数据: {os.path.basename(data_path)}")

        cmd = [sys.executable, '-u', script_pack,
               '--model', self.onnx_path,
               '--tokenizer', self.tokenizer_path,
               '--name', self.name,
               '--output', out_tg]

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, encoding='utf-8', errors='replace', bufsize=1,
        )

        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line:
                self.log_signal.emit(f"  {line}")

        proc.wait(timeout=300)

        if proc.returncode != 0:
            self.done_signal.emit(False, "", "打包失败")
            return

        if os.path.isfile(out_tg):
            size_mb = os.path.getsize(out_tg) / (1024 * 1024)
            self.log_signal.emit(f"\n✓ 打包完成: {out_tg}")
            self.log_signal.emit(f"  大小: {size_mb:.1f} MB")
            self.done_signal.emit(True, out_tg, "")
        else:
            self.done_signal.emit(False, "", f"打包成功但找不到输出文件: {out_tg}")


# ─── 训练线程 ────────────────────────────────────────────
class TrainingThread(QThread):
    """后台训练线程，支持断点续训 + 看门狗自动保存"""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)       # (epoch, total_epochs)
    batch_progress_signal = pyqtSignal(int, int) # (step, total_steps)
    loss_signal = pyqtSignal(float, float)       # (train_loss, val_ppl)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop_flag = False
        self.resume_from = config.get('resume_from', None)

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._do_train()
        except Exception as e:
            self.log_signal.emit(f"[错误] {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

    def _do_train(self):
        import torch
        from tokenizer import ChineseTokenizer
        from model import create_model
        from train import TextDataset, CosineWarmupScheduler
        import torch.nn.functional as F
        import math
        import random

        # ── 硬件检测 & 自动调参 ──
        use_gpu = self.config.get('use_gpu', False) and torch.cuda.is_available()
        if self.config.get('use_gpu', False) and not torch.cuda.is_available():
            self.log_signal.emit("  ⚠ 已勾选 GPU 但未检测到 CUDA，回退到 CPU")
            self.log_signal.emit("  提示: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        device = torch.device('cuda' if use_gpu else 'cpu')
        batch_size = self.config.get('batch_size', 16)

        if use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.log_signal.emit(f"  设备: {gpu_name} ({gpu_mem_gb:.1f}GB 显存)")
            # 小显存自动降 batch，但不要降太多（实际占用远低于上限）
            if gpu_mem_gb < 2.5 and batch_size > 8:
                batch_size = 8
                self.log_signal.emit(f"  ⚠ 小显存，自动降 batch → {batch_size}")
            elif gpu_mem_gb < 5.0 and batch_size > 16:
                batch_size = 16
                self.log_signal.emit(f"  ⚠ 中等显存，自动降 batch → {batch_size}")
        else:
            # CPU 模式：用满全核（留1核给 GUI），不比当年旧电脑了
            cpu_count = os.cpu_count() or 4
            n_threads = max(4, cpu_count - 1)
            torch.set_num_threads(n_threads)
            if batch_size > 8:
                batch_size = 8
                self.log_signal.emit(f"  CPU 模式，batch=8 (线程={n_threads})")
            else:
                self.log_signal.emit(f"  设备: CPU (线程={n_threads})")

        # 报告系统内存状态（Windows）
        try:
            import psutil
            ram = psutil.virtual_memory()
            self.log_signal.emit(f"  系统 RAM: {ram.total / 1e9:.1f}GB (可用: {ram.available / 1e9:.1f}GB)")
            if ram.available < 2 * 1024 ** 3:  # 可用少于 2GB
                self.log_signal.emit(f"  ⚠ 内存紧张! 建议关闭其他程序后重试")
        except ImportError:
            pass  # psutil 未安装就跳过

        scaler = torch.amp.GradScaler('cuda') if use_gpu else None

        # ── 加载数据 ──
        self.log_signal.emit("加载数据...")
        data_path = self.config.get('data_path', '')
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    texts = [json.loads(line)['text'] for line in f if line.strip()]
                else:
                    data = json.load(f)
                    texts = data if isinstance(data, list) else data.get('texts', [])
        elif os.path.exists('data/train_qa.jsonl'):
            self.log_signal.emit("  自动检测到 data/train_qa.jsonl")
            with open('data/train_qa.jsonl', 'r', encoding='utf-8') as f:
                texts = [json.loads(line)['text'] for line in f if line.strip()]
        else:
            from train import _get_demo_texts
            texts = _get_demo_texts()
            self.log_signal.emit(f"  使用内置数据 ({len(texts)} 条)")

        # ── 分词器 ──
        tok_path = self.config.get('tokenizer_path', 'checkpoints/tokenizer.json')
        if os.path.exists(tok_path):
            tokenizer = ChineseTokenizer.load(tok_path)
            self.log_signal.emit(f"  加载分词器: 词表 {tokenizer.vocab_size_actual}")
        else:
            tokenizer = ChineseTokenizer(
                vocab_size=self.config.get('vocab_size', 16384),
            )
            tokenizer.train(texts)
            os.makedirs(os.path.dirname(tok_path) or '.', exist_ok=True)
            tokenizer.save(tok_path)

        # ── 提前检测续训checkpoint，获取模型词表大小 ──
        ckpt_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        watchdog_path = os.path.join(ckpt_dir, 'last_step.pt')
        resume_path = None
        if self.resume_from and os.path.exists(self.resume_from):
            resume_path = self.resume_from
        elif os.path.exists(watchdog_path):
            resume_path = watchdog_path
        if resume_path is None and not self.resume_from:
            custom_ckpt = self.config.get('custom_checkpoint', '')
            if custom_ckpt and os.path.exists(custom_ckpt):
                resume_path = custom_ckpt

        # 确定模型词表大小（续训时使用checkpoint的词表大小）
        model_vocab_size = tokenizer.vocab_size_actual
        if resume_path:
            try:
                ckpt_temp = torch.load(resume_path, map_location='cpu', weights_only=False)
                if 'model_config' in ckpt_temp:
                    ckpt_cfg = ckpt_temp['model_config']
                    model_vocab_size = ckpt_cfg.get('vocab_size', model_vocab_size)
                    self.log_signal.emit(f"  [续训] 检测到checkpoint词表大小: {model_vocab_size}")
                del ckpt_temp
            except Exception as e:
                self.log_signal.emit(f"  [续训] 读取checkpoint配置失败: {e}")

        # ── 数据集 ──
        random.seed(42)
        indices = list(range(len(texts)))
        random.shuffle(indices)
        split = int(0.9 * len(texts))
        train_texts = [texts[i] for i in indices[:split]]
        val_texts = [texts[i] for i in indices[split:]]

        seq_len = self.config.get('seq_len', 256)
        # 注意：batch_size 已在开头根据硬件自动调整

        train_dataset = TextDataset(train_texts, tokenizer, seq_len)
        val_dataset = TextDataset(val_texts, tokenizer, seq_len)

        from torch.utils.data import DataLoader
        # Windows: num_workers=0 避免多进程内存开销（8.5GB内存不够支撑多worker）
        n_workers = 0
        _dl_kwargs = dict(
            pin_memory=use_gpu,
            num_workers=n_workers,
            persistent_workers=n_workers > 0,
            prefetch_factor=2 if n_workers > 0 else None,
        )
        train_loader = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True, **_dl_kwargs,
        )
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, drop_last=True, **_dl_kwargs,
        )

        self.log_signal.emit(f"  训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)} | 实际 batch={batch_size}")

        # ── 模型 ──
        # 使用之前已经确定的 model_vocab_size（已考虑续训checkpoint）
        model_d_model = self.config.get('d_model', 256)
        model_n_layers = self.config.get('n_layers', 4)
        model_n_heads = self.config.get('n_heads', 8)
        model_d_ff = self.config.get('d_ff', 1024)
        model_max_seq_len = seq_len

        # 如果有续训 checkpoint，从 checkpoint 读取完整模型配置
        if resume_path:
            try:
                ckpt_temp = torch.load(resume_path, map_location='cpu', weights_only=False)
                if 'model_config' in ckpt_temp:
                    ckpt_cfg = ckpt_temp['model_config']
                    model_vocab_size = ckpt_cfg.get('vocab_size', model_vocab_size)
                    model_d_model = ckpt_cfg.get('d_model', model_d_model)
                    model_n_layers = ckpt_cfg.get('n_layers', model_n_layers)
                    model_n_heads = ckpt_cfg.get('n_heads', model_n_heads)
                    model_d_ff = ckpt_cfg.get('d_ff', model_d_ff)
                    model_max_seq_len = ckpt_cfg.get('max_seq_len', model_max_seq_len)
                    self.log_signal.emit(f"  [续训] 使用 checkpoint 模型配置: vocab={model_vocab_size}, d_model={model_d_model}, layers={model_n_layers}")
                del ckpt_temp
            except Exception as e:
                self.log_signal.emit(f"  [续训] 读取 checkpoint 配置失败: {e}")

        model = create_model(
            vocab_size=model_vocab_size,
            d_model=model_d_model,
            n_layers=model_n_layers,
            n_heads=model_n_heads,
            d_ff=model_d_ff,
            max_seq_len=model_max_seq_len,
            dropout=self.config.get('dropout', 0.1),
            n_experts=self.config.get('n_experts', 4),
            n_activated=self.config.get('n_activated', 2),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_signal.emit(f"  参数: {n_params:,} (~{n_params/1e6:.1f}M)")

        # ── 加载教师模型（知识蒸馏）─────────────────────
        teacher_model = None
        distill_alpha = self.config.get('distill_alpha', 0.0)
        distill_temp = self.config.get('distill_temp', 2.0)
        teacher_path = self.config.get('teacher_model_path', '')
        teacher_device = self.config.get('teacher_device', 'cpu')

        if distill_alpha > 0 and teacher_path:
            if os.path.exists(teacher_path):
                try:
                    teacher_ckpt = torch.load(teacher_path, map_location=teacher_device)
                    teacher_cfg = teacher_ckpt.get('model_config', {})
                    teacher_model = create_model(
                        vocab_size=teacher_cfg.get('vocab_size', tokenizer.vocab_size_actual),
                        d_model=teacher_cfg.get('d_model', self.config.get('d_model', 256)),
                        n_layers=teacher_cfg.get('n_layers', self.config.get('n_layers', 8)),
                        n_heads=teacher_cfg.get('n_heads', self.config.get('n_heads', 8)),
                        d_ff=teacher_cfg.get('d_ff', self.config.get('d_ff', 1024)),
                        max_seq_len=teacher_cfg.get('max_seq_len', seq_len),
                        dropout=0.0,
                        n_experts=teacher_cfg.get('n_experts', self.config.get('n_experts', 4)),
                        n_activated=teacher_cfg.get('n_activated', self.config.get('n_activated', 2)),
                    )
                    teacher_model.load_state_dict(
                        _migrate_rope_buffers(teacher_ckpt['model_state_dict']))
                    teacher_model.to(teacher_device)
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad = False
                    self.log_signal.emit(f"  [蒸馏] 教师模型已加载: {teacher_path}")
                    self.log_signal.emit(f"  [蒸馏] α={distill_alpha}, T={distill_temp}")
                except Exception as e:
                    self.log_signal.emit(f"  [蒸馏] 教师模型加载失败: {e}")
                    teacher_model = None
            else:
                self.log_signal.emit(f"  [蒸馏] 找不到教师模型: {teacher_path}")

        # ── 优化器 ──
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('lr', 3e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95),
        )
        epochs = self.config.get('epochs', 30)
        total_steps = len(train_loader) * epochs
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.config.get('warmup_steps', 500),
            total_steps=total_steps,
        )

        ckpt_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        # ── 断点续训：优先加载看门狗 last_step.pt ──
        start_epoch = 1
        global_step = 0
        best_ppl = float('inf')

        # resume_path 已在模型创建前检测过，这里直接使用
        if resume_path:
            try:
                self.log_signal.emit(f"  ↻ 续训: 加载 {os.path.basename(resume_path)}")
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                _migrate_rope_buffers(ckpt['model_state_dict'])
                model.load_state_dict(ckpt['model_state_dict'], strict=True)
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 1)
                global_step = ckpt.get('global_step', 0)
                best_ppl = ckpt.get('best_ppl', float('inf'))
                # 恢复 scheduler 步数（直接设置，避免万次循环）
                scheduler.current_step = global_step
                # 立即应用当前步数对应的 LR
                scheduler.step()
                scheduler.current_step = global_step
                self.log_signal.emit(f"  → 从 epoch {start_epoch}, step {global_step} 继续训练")
            except RuntimeError as e:
                self.log_signal.emit(f"  ⚠ 续训失败: {str(e)[:120]}")
                self.log_signal.emit("  → 架构不匹配，从头开始训练")
                start_epoch = 1
                global_step = 0
                best_ppl = float('inf')
                resume_path = None

        # ── 步级保存配置 ──
        save_every_steps = self.config.get('save_every_steps', 500)
        # 对齐: 下次保存在 global_step 之后的下一个 500 倍数
        next_save_at = ((global_step // save_every_steps) + 1) * save_every_steps

        # ── 训练循环 ──
        self.log_signal.emit(f"开始训练 ({epochs} epochs, {len(train_loader)} steps/epoch)")
        if start_epoch > 1:
            self.log_signal.emit(f"  (续训模式: 从 epoch {start_epoch} 开始)")
        grad_accum = self.config.get('grad_accum_steps', 1)

        # 优化：尝试 torch.compile（PyTorch 2.0+）加速训练
        # MX150 等旧 GPU (compute capability < 7.0) 不支持 Triton，跳过
        use_compile = False
        if use_gpu and hasattr(torch, 'compile'):
            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            if cap[0] >= 7:
                try:
                    self.log_signal.emit("  [优化] 启用 torch.compile 加速...")
                    model = torch.compile(model)
                    use_compile = True
                except Exception:
                    self.log_signal.emit("  [优化] torch.compile 失败，跳过")
            else:
                self.log_signal.emit(f"  [优化] GPU compute capability {cap[0]}.{cap[1]} < 7.0，跳过 torch.compile")

        for epoch in range(start_epoch, epochs + 1):
            if self._stop_flag:
                self.log_signal.emit("训练已手动停止")
                break

            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            accum_loss = 0.0
            t0 = time.time()

            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                if self._stop_flag:
                    break

                # 确保 input_ids 不超出模型词表范围
                vocab_size = model.config.vocab_size if hasattr(model, 'config') else model_vocab_size
                input_ids = input_ids.clamp(0, vocab_size - 1).to(device, non_blocking=use_gpu)
                target_ids = target_ids.clamp(0, vocab_size - 1).to(device, non_blocking=use_gpu)

                # 知识蒸馏：获取教师 logits
                teacher_logits = None
                if teacher_model is not None:
                    with torch.no_grad():
                        if teacher_device != str(device):
                            t_input = input_ids.to(teacher_device)
                            teacher_logits = _model_forward(teacher_model, t_input).to(device)
                        else:
                            teacher_logits = _model_forward(teacher_model, input_ids)

                # GPU: AMP 混合精度前向
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = _model_forward(model, input_ids)
                        loss = _compute_distill_loss(
                            logits, target_ids, teacher_logits,
                            distill_alpha, distill_temp, model=model,
                        )
                        loss = loss / grad_accum
                    scaler.scale(loss).backward()
                    accum_loss += loss.item() * grad_accum
                else:
                    logits = _model_forward(model, input_ids)
                    loss = _compute_distill_loss(
                        logits, target_ids, teacher_logits,
                        distill_alpha, distill_temp, model=model,
                    )
                    loss = loss / grad_accum
                    loss.backward()
                    accum_loss += loss.item() * grad_accum

                # 梯度累积：每 grad_accum 步或最后一步更新参数
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
                    epoch_steps += 1
                    accum_loss = 0.0

                    # 每个参数更新步都发射进度
                    self.batch_progress_signal.emit(global_step, total_steps)

                    # 每100步更新一次loss（step级别，ppl=0标记为步级）
                    if global_step % 100 == 0:
                        step_avg = epoch_loss / max(epoch_steps, 1)
                        self.loss_signal.emit(step_avg, 0)

                    # ── 看门狗: 每50步原子写 last_step.pt（避免每步序列化拖慢训练）──
                    if global_step % 50 == 0:
                        _save_checkpoint_atomic(
                            watchdog_path, model, optimizer, tokenizer,
                            ckpt_dir, self.config, epoch, global_step, best_ppl,
                        )

                    # ── 步级自动保存 (对齐500倍数) ──
                    if global_step >= next_save_at:
                        step_ckpt = os.path.join(ckpt_dir, f'checkpoint_step{global_step}.pt')
                        _save_checkpoint_full(
                            step_ckpt, model, optimizer, tokenizer,
                            ckpt_dir, self.config, epoch, global_step, avg_loss=0, best_ppl=best_ppl,
                        )
                        next_save_at += save_every_steps
                        self.log_signal.emit(f"  → 步级保存 checkpoint_step{global_step}.pt")

            if self._stop_flag:
                break

            num_updates = (len(train_loader) + grad_accum - 1) // grad_accum
            avg_loss = epoch_loss / max(num_updates, 1)
            mem_info = ""
            if use_gpu:
                allocated = torch.cuda.memory_allocated() / 1e9
                mem_info = f" | GPU: {allocated:.2f}GB"
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                try:
                    import psutil
                    proc = psutil.Process()
                    ram_mb = proc.memory_info().rss / 1024 / 1024
                    mem_info = f" | RAM: {ram_mb:.0f}MB"
                except ImportError:
                    pass
            self.log_signal.emit(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_lr():.2e}{mem_info}")

            # 验证（轻量：只跑 3 个 batch，每 5 个 epoch 才验证）
            if epoch % 5 == 0:
                val_ppl = float('inf')
                try:
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
                                vt.view(-1),
                                ignore_index=0,
                                reduction='sum',
                            ).item()
                            val_tokens += (vt != 0).sum().item()

                    val_ppl = math.exp(min(val_loss / max(val_tokens, 1), 20))
                    self.log_signal.emit(f"  → Val PPL: {val_ppl:.2f}")
                except Exception as ve:
                    self.log_signal.emit(f"  → 验证失败: {ve}")
                    if use_gpu:
                        torch.cuda.empty_cache()
            else:
                val_ppl = float('inf')  # 本轮不验证，不参与 best_model 判断

            elapsed = time.time() - t0
            self.log_signal.emit(
                f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            self.progress_signal.emit(epoch, epochs)
            self.loss_signal.emit(avg_loss, val_ppl)

            # 轮级保存
            mc = model.config
            if epoch % self.config.get('save_every', 5) == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'best_ppl': best_ppl,
                    'model_config': {
                        'vocab_size': mc.vocab_size,
                        'd_model': mc.d_model,
                        'n_layers': mc.n_layers,
                        'n_heads': mc.n_heads,
                        'd_ff': mc.d_ff,
                        'max_seq_len': mc.max_seq_len,
                        'dropout': mc.dropout,
                        'n_experts': mc.n_experts,
                        'n_activated': mc.n_activated,
                    },
                }, os.path.join(ckpt_dir, f'checkpoint_epoch{epoch}.pt'))
                self.log_signal.emit(f"  → 已保存 checkpoint_epoch{epoch}.pt")

            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_ppl': best_ppl,
                    'model_config': {
                        'vocab_size': mc.vocab_size,
                        'd_model': mc.d_model,
                        'n_layers': mc.n_layers,
                        'n_heads': mc.n_heads,
                        'd_ff': mc.d_ff,
                        'max_seq_len': mc.max_seq_len,
                        'dropout': mc.dropout,
                        'n_experts': mc.n_experts,
                        'n_activated': mc.n_activated,
                    },
                }, os.path.join(ckpt_dir, 'best_model.pt'))

        self.finished_signal.emit(True, f"完成! 最佳困惑度: {best_ppl:.2f}")


# ─── 聊天生成线程 (支持流式+调试) ─────────────────────────
class ChatThread(QThread):
    """后台生成线程，逐 token 流式输出"""
    chunk_signal = pyqtSignal(str)      # 每个 token chunk
    response_signal = pyqtSignal(str)   # 最终完整回复
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()      # 生成完成
    debug_signal = pyqtSignal(list)     # 调试信息列表

    def __init__(self, model, tokenizer, prompt, temperature, max_tokens, device, debug=False,
                 repetition_penalty=1.05, top_k=80, top_p=0.90, frequency_penalty=0.15,
                 min_new_tokens=5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self._debug = debug
        self._repetition_penalty = repetition_penalty
        self._top_k = top_k
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._min_new_tokens = min_new_tokens
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from inference import TextGenerator

            generator = TextGenerator(self.model, self.tokenizer)
            response = ""
            all_debug = []  # 累积全部调试信息，最后一次性发送

            for chunk in generator.generate(
                self.prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                frequency_penalty=self._frequency_penalty,
                repetition_penalty=self._repetition_penalty,
                min_new_tokens=self._min_new_tokens,
                stream=True,
                formatted=True,  # prompt 已经由 _send_chat 格式化为 "用户:...\nTGAI?"
            ):
                if self._stop:
                    break
                response += chunk
                self.chunk_signal.emit(chunk)
                # 调试: 只收集，不发信号（避免打断正文流）
                if self._debug:
                    dbg = generator.pop_debug()
                    if dbg:
                        all_debug.extend(dbg)

            # 拉取最后的调试信息
            if self._debug:
                dbg = generator.pop_debug()
                if dbg:
                    all_debug.extend(dbg)
                if all_debug:
                    self.debug_signal.emit(all_debug)

            self.response_signal.emit(response or "[模型未生成回复]")
            self.finished_signal.emit()

        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")


# ─── TGAI GO 引擎 (ctypes → libtgai_engine.dll) ──────────────────────────
class TGEngine:
    """TGAI GO C 引擎的 ctypes 包装 (单例, 全局加载一次)"""
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.lib = None
        self.model = None
        self.tg_path = None
        self.vocab_size = 0
        self.max_seq = 0
        self._load_dll()

    def _load_dll(self):
        """查找并加载 libtgai_engine.dll"""
        project_root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(project_root, 'TGAI GO', 'cpu', 'build', 'libtgai_engine.dll'),
            r'D:\TGAI\TGAI GO\cpu\build\libtgai_engine.dll',
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    self.lib = ctypes.CDLL(p)
                    self._bind_functions()
                    return
                except Exception as e:
                    raise RuntimeError(f"加载 DLL 失败 {p}: {e}")
        raise FileNotFoundError(
            "找不到 libtgai_engine.dll\n"
            "请运行: cd \"TGAI GO\\cpu\" && .\\build.ps1"
        )

    def _bind_functions(self):
        L = self.lib
        L.tg_load.restype = ctypes.c_void_p
        L.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.tg_free.argtypes = [ctypes.c_void_p]
        L.tg_config.restype = ctypes.c_void_p
        L.tg_config.argtypes = [ctypes.c_void_p]
        L.tg_encode.restype = ctypes.c_int
        L.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
        L.tg_decode.restype = ctypes.c_int
        L.tg_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.tg_token_to_str.restype = ctypes.c_int
        L.tg_token_to_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        L.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        L.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        L.tg_kvcache_new.restype = ctypes.c_void_p
        L.tg_kvcache_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.tg_kvcache_free.argtypes = [ctypes.c_void_p]
        L.tg_kvcache_reset.argtypes = [ctypes.c_void_p]
        L.tg_preload_all.restype = ctypes.c_int
        L.tg_preload_all.argtypes = [ctypes.c_void_p]
        L.tg_preload_auto.restype = ctypes.c_int
        L.tg_preload_auto.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.tg_preload_memory_estimate.restype = ctypes.c_uint64
        L.tg_preload_memory_estimate.argtypes = [ctypes.c_void_p]
        L.tg_last_error.restype = ctypes.c_char_p
        L.tg_set_nthreads.argtypes = [ctypes.c_int]
        L.tg_get_nthreads.restype = ctypes.c_int
        if hasattr(L, 'tg_set_extend'):
            L.tg_set_extend.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def load_model(self, tg_path: str):
        """加载 .TG 模型文件"""
        if self.model is not None:
            self.lib.tg_free(self.model)
            self.model = None
        err = ctypes.c_int(0)
        m = self.lib.tg_load(tg_path.encode('utf-8'), 1, ctypes.byref(err))
        if not m:
            msg = self.lib.tg_last_error()
            raise RuntimeError(f"tg_load 失败 (err={err.value}): {msg}")
        self.model = m
        self.tg_path = tg_path
        # 强制预加载所有权重 + 预热 forward (避免首 token 慢)
        import numpy as np
        preload_err = self.lib.tg_preload_all(m)
        if preload_err == 0:
            fp32_mb = self.lib.tg_preload_memory_estimate(m) / 1024 / 1024
            print(f"[TG] 预加载完成: {fp32_mb:.0f} MB FP32 缓存")
        else:
            print(f"[TG] 预加载失败 (err={preload_err}), 将使用懒加载")
        # 预热: 跑一次小 forward 触发所有权重解码 (避免首 token 卡顿)
        try:
            warmup_ids = (ctypes.c_int32 * 3)(2, 13036, 3)  # BOS + 你好 + EOS
            lib = self.lib
            if not hasattr(lib, 'tg_forward'):
                lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
                lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                           ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            lib.tg_forward(m, warmup_ids, 3, None, 0)
            print("[TG] 预热完成")
        except Exception as e:
            print(f"[TG] 预热异常 (可忽略): {e}")
        # 读取配置 (v2 header: arch_type at cfg[28], num_kv_heads at cfg[29])
        cfg_ptr = self.lib.tg_config(m)
        cfg = ctypes.cast(cfg_ptr, ctypes.POINTER(ctypes.c_uint32))
        self.vocab_size = cfg[7]
        self.max_seq = cfg[8]
        n_layers = cfg[3]
        d_model = cfg[4]
        n_heads = cfg[5]
        n_experts = cfg[9]
        arch_type = cfg[28] if cfg[0] >= 2 else 0
        num_kv_heads = cfg[29] if cfg[0] >= 2 else 0
        if num_kv_heads == 0:
            num_kv_heads = n_heads

        arch_names = {0: 'TGAI MoE', 1: 'Llama', 2: 'Mistral', 3: 'Qwen2', 4: 'Gemma', 5: 'Phi-3', 6: 'ChatGLM'}
        arch_name = arch_names.get(arch_type, f'Unknown({arch_type})')
        gqa_info = ''
        if num_kv_heads < n_heads:
            gqa_info = f' GQA(n_kv={num_kv_heads})'

        cfg_result = {
            'n_layers': n_layers, 'd_model': d_model,
            'vocab_size': self.vocab_size, 'max_seq': self.max_seq,
            'n_heads': n_heads, 'n_experts': n_experts,
            'arch_type': arch_type, 'arch_name': arch_name,
            'num_kv_heads': num_kv_heads, 'gqa_info': gqa_info,
        }
        print(f"[TG] arch={arch_name}{gqa_info} layers={n_layers} d_model={d_model} "
              f"heads={n_heads} experts={n_experts}")
        return cfg_result

    def set_extend(self, extend_ctx: int):
        """NTK-aware 上下文扩展 (CPU + CUDA 通用)
        设 0 关闭, 设 > max_seq 启用。
        必须在 load_model 之后调用。"""
        if self.model is None or not hasattr(self.lib, 'tg_set_extend'):
            return
        self.lib.tg_set_extend(self.model, extend_ctx)
        print(f"[TG] 上下文扩展: {extend_ctx}")

    def encode(self, text: str, add_special: bool = True):
        """BPE 编码: 文本 → token ids"""
        if not self.model:
            raise RuntimeError("模型未加载")
        text_bytes = text.encode('utf-8')
        max_out = len(text_bytes) + 8
        out = (ctypes.c_int32 * max_out)()
        n = self.lib.tg_encode(self.model, text_bytes, out, max_out, 1 if add_special else 0)
        if n < 0:
            raise RuntimeError("tg_encode 失败")
        return [out[i] for i in range(n)]

    def decode(self, ids, skip_special: bool = True):
        """BPE 解码"""
        if not self.model:
            raise RuntimeError("模型未加载")
        n = len(ids)
        arr = (ctypes.c_int32 * n)(*ids)
        buf = ctypes.create_string_buffer(4096)
        m = self.lib.tg_decode(self.model, arr, n, buf, 4096, 1 if skip_special else 0)
        if m < 0:
            return ""
        return buf.value.decode('utf-8', errors='replace')

    def token_str(self, tid: int):
        if not self.model:
            return ""
        buf = ctypes.create_string_buffer(128)
        n = self.lib.tg_token_to_str(self.model, tid, buf, 128)
        if n <= 0:
            return ""
        return buf.value.decode('utf-8', errors='replace')


class ChatThreadGO(QThread):
    """TGAI GO 引擎对话线程, 流式输出"""
    chunk_signal = pyqtSignal(str)
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, engine: TGEngine, prompt: str, temperature: float,
                 max_tokens: int, repetition_penalty: float = 1.0,
                 top_k: int = 90, top_p: float = 0.9,
                 freq_penalty: float = 0.25, min_tokens: int = 10,
                 tokenizer=None):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.temperature = max(temperature, 0.01)
        self.max_tokens = max_tokens
        self.rep_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.min_tokens = min_tokens
        self.tokenizer = tokenizer  # 统一分词方式
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            lib = self.engine.lib
            model = self.engine.model
            vocab = self.engine.vocab_size
            eos_id = 3  # 默认 eos

            # 1. 编码 prompt
            # .TG 模型: 用引擎内置 tokenizer (与模型一一对应)
            #   .pt 模型: 用 self.tokenizer (与 PyTorch 模型配套)
            if self.tokenizer:
                ids = self.tokenizer.encode(self.prompt, add_special=True)
                ids = [min(tid, vocab - 1) for tid in ids]
            else:
                try:
                    ids = self.engine.encode(self.prompt, add_special=True)
                except Exception:
                    # 回退: 引擎编码失败时用默认 TGAI tokenizer
                    from tokenizer import ChineseTokenizer
                    tk = ChineseTokenizer.load('checkpoints/tokenizer.json')
                    ids = tk.encode(self.prompt, add_special=True)
                    ids = [min(tid, vocab - 1) for tid in ids]
            if not ids:
                self.error_signal.emit("编码 prompt 失败")
                return

            # 2. 创建 KV cache 并 prefill
            cache = lib.tg_kvcache_new(model, self.engine.max_seq)
            if not cache:
                self.error_signal.emit("创建 KV cache 失败")
                return

            try:
                ids_arr = (ctypes.c_int32 * len(ids))(*ids)
                logits = lib.tg_forward(model, ids_arr, len(ids), cache, 0)
                if not logits:
                    self.error_signal.emit("prefill forward 失败")
                    return

                cur_pos = len(ids)
                response = ""

                # numpy 加速
                import numpy as np
                FloatArr = ctypes.POINTER(ctypes.c_float)

                # 3. 自回归生成 (temperature + top-k + top-p 采样)
                _last_id = -1
                _consecutive = 0
                for step in range(self.max_tokens):
                    if self._stop:
                        break

                    logit_ptr = ctypes.cast(logits, FloatArr)
                    logit_arr = np.ctypeslib.as_array(logit_ptr, shape=(vocab,)).copy()

                    # 连续重复 > 3 次 → 强制降权
                    if _consecutive >= 3:
                        logit_arr[_last_id] -= _consecutive * 5.0
                    if _consecutive > 15:
                        break  # 极端循环截断

                    # temperature 缩放
                    if self.temperature > 0.01:
                        logit_arr = logit_arr / self.temperature

                    # top-k 截断
                    if self.top_k > 0 and self.top_k < vocab:
                        indices = np.argpartition(logit_arr, -self.top_k)[-self.top_k:]
                        threshold = np.min(logit_arr[indices])
                        logit_arr[logit_arr < threshold] = -np.inf

                    # top-p 截断
                    if self.top_p < 1.0:
                        sorted_idx = np.argsort(logit_arr)[::-1]
                        sorted_probs = np.exp(logit_arr[sorted_idx] - np.max(logit_arr))
                        sorted_probs /= np.sum(sorted_probs)
                        cumsum = np.cumsum(sorted_probs)
                        cutoff = np.searchsorted(cumsum, self.top_p) + 1
                        mask = np.ones(vocab, dtype=bool)
                        mask[sorted_idx[cutoff:]] = False
                        logit_arr[~mask] = -np.inf

                    # softmax + 采样
                    probs = np.exp(logit_arr - np.max(logit_arr))
                    probs = probs / np.sum(probs)
                    if np.any(np.isnan(probs)):
                        best_id = int(np.argmax(logit_arr))
                    else:
                        best_id = int(np.random.choice(vocab, p=probs))

                    # 遇到 EOS 停止
                    if best_id == eos_id:
                        break

                    # 连续重复跟踪
                    if best_id == _last_id:
                        _consecutive += 1
                    else:
                        _consecutive = 0
                    _last_id = best_id

                    # decode
                    if self.tokenizer:
                        tok_str = self.tokenizer.id_to_token.get(best_id, '')
                    else:
                        tok_str = self.engine.token_str(best_id)
                    if tok_str:
                        response += tok_str
                        self.chunk_signal.emit(tok_str)

                    # forward 下一步
                    next_arr = (ctypes.c_int32 * 1)(best_id)
                    logits = lib.tg_forward(model, next_arr, 1, cache, cur_pos)
                    if not logits:
                        break
                    cur_pos += 1

                    if cur_pos >= self.engine.max_seq:
                        break

                self.response_signal.emit(response or "[模型未生成回复]")
            finally:
                lib.tg_kvcache_free(cache)

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")


# ─── QQ 机器人后台线程 ────────────────────────────────────
class QQBotThread(QThread):
    """后台运行 NapCat WebSocket，支持记忆存储和指令系统"""
    log_signal = pyqtSignal(str)
    reply_signal = pyqtSignal(str, str)

    def __init__(self, generate_fn, napcat_ws, napcat_http, bot_qq=0, token="", memory_dir="qq_memory"):
        super().__init__()
        self.generate_fn = generate_fn
        self.napcat_ws = napcat_ws
        self.napcat_http = napcat_http
        self.bot_qq = bot_qq
        self.token = token
        self.memory_dir = memory_dir
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=2)  # 独立线程生成，不阻塞 WebSocket ping
        self._chat_temp: dict = {}

        os.makedirs(memory_dir, exist_ok=True)

    def _api(self, action, params):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.post(f"{self.napcat_http}/{action}", json=params, headers=headers, timeout=10)
            data = resp.json()
            if data.get("status") != "ok":
                self.log_signal.emit(f"[API] {action} 失败: {data}")
            return data
        except Exception as e:
            self.log_signal.emit(f"[API] {action} 异常: {e}")
            return {}

    def _send_private(self, uid, text):
        for chunk in self._chunk(text):
            self._api("send_private_msg", {"user_id": uid, "message": [{"type":"text","data":{"text":chunk}}]})

    def _send_group(self, gid, text, uid):
        for chunk in self._chunk(text):
            msg = [
                {"type": "at", "data": {"qq": str(uid)}},
                {"type": "text", "data": {"text": chunk}},
            ]
            self._api("send_group_msg", {"group_id": int(gid), "message": msg})

    def _chunk(self, text, size=600):
        return [text[i:i+size] for i in range(0, len(text), size)]

    # ── 记忆系统 ──────────────────────────────────
    def _mem_path(self, chat_type, chat_id):
        return os.path.join(self.memory_dir, f"{chat_type}_{chat_id}.json")

    def _load_memory(self, chat_type, chat_id):
        path = self._mem_path(chat_type, chat_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_memory(self, chat_type, chat_id, memory):
        path = self._mem_path(chat_type, chat_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def _clear_memory(self, chat_type, chat_id):
        path = self._mem_path(chat_type, chat_id)
        if os.path.exists(path):
            os.remove(path)

    def _get_chat_id(self, msg_type, uid, gid):
        return "group", str(gid) if msg_type == "group" else ("private", str(uid))

    def _build_context(self, memory, max_rounds=4):
        if not memory:
            return ""
        ctx = ""
        for m in memory[-max_rounds:]:
            ctx += f"用户:{m['q']}\nTGAI?{m['a']}\n"
        return ctx

    # ── 指令处理 ──────────────────────────────────
    def _handle_command(self, msg_type, uid, gid, cmd_line):
        parts = cmd_line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        chat_type, chat_id = self._get_chat_id(msg_type, uid, gid)
        chat_name = f"{'群聊' if chat_type == 'group' else '私聊'}{chat_id}"

        if cmd == "/help":
            return (
                "TGAI QQ 机器人指令:\n"
                "/help     - 查看此帮助\n"
                "/clean    - 清空当前对话记忆\n"
                "/temp N   - 设置回复温度 (0.3-2.0)\n"
                "/status   - 查看模型状态\n"
                "/forget   - 删除最后一轮记忆\n"
                "/memory   - 显示当前记忆条数"
            )

        elif cmd == "/clean":
            self._clear_memory(chat_type, chat_id)
            return f"[{chat_name}] 对话记忆已清空 ✓"

        elif cmd == "/temp":
            try:
                t = float(arg)
                t = max(0.3, min(2.0, t))
                self._chat_temp[chat_id] = t
                return f"[{chat_name}] 温度设为 {t}"
            except ValueError:
                cur = self._chat_temp.get(chat_id, "默认")
                return f"当前温度: {cur}。用法: /temp 0.8"

        elif cmd == "/status":
            return (
                f"TGAI 0.5B (864M 参数)\n"
                f"MoE: 4专家 × 2激活\n"
                f"上下文: 512 tokens\n"
                f"记忆目录: {self.memory_dir}/"
            )

        elif cmd == "/forget":
            mem = self._load_memory(chat_type, chat_id)
            if mem:
                mem.pop()
                self._save_memory(chat_type, chat_id, mem)
                return f"[{chat_name}] 已删除最后一轮记忆 (剩余 {len(mem)} 轮)"
            return f"[{chat_name}] 没有可删除的记忆"

        elif cmd == "/memory":
            mem = self._load_memory(chat_type, chat_id)
            return f"[{chat_name}] 当前记忆: {len(mem)} 轮"

        return None  # 不是指令

    # ── 消息处理 ──────────────────────────────────
    def _on_msg(self, ws, raw):
        try:
            data = json.loads(raw)
        except:
            return

        if data.get("post_type") != "message":
            if data.get("meta_event_type") == "lifecycle" and not self.bot_qq:
                self.bot_qq = data.get("self_id", 0)
                self.log_signal.emit(f"[QQ] 机器人 QQ: {self.bot_qq}")
            return

        msg_type = data.get("message_type")
        uid = str(data.get("user_id", ""))
        gid = data.get("group_id")
        message_segments = data.get("message", [])

        if uid == str(self.bot_qq):
            return

        # 提取文本
        raw_msg = ""
        for seg in message_segments:
            if seg.get("type") == "text":
                raw_msg += seg.get("data", {}).get("text", "")

        if not raw_msg.strip():
            return

        # ── 群聊处理 ──
        if msg_type == "group":
            is_at = False
            for seg in message_segments:
                if seg.get("type") == "at" and str(seg.get("data", {}).get("qq")) == str(self.bot_qq):
                    is_at = True
                    break
            if not is_at and not raw_msg.lower().startswith("tgai"):
                return

            # 从被 @ 后的文本中提取内容
            clean = raw_msg.replace(f"[CQ:at,qq={self.bot_qq}]", "").strip()
            self.log_signal.emit(f"[群聊] {gid}/{uid}: {clean[:80]}")

            # 检查指令
            cmd_reply = self._handle_command(msg_type, uid, gid, clean)
            if cmd_reply:
                self._send_group(gid, cmd_reply, uid)
                self.reply_signal.emit(f"群聊{gid}", f"[指令] {clean}")
                return

            # 异步生成回复（不阻塞 WebSocket ping）
            mem = self._load_memory("group", str(gid))
            ctx = self._build_context(mem)
            self._executor.submit(self._generate_and_reply_group, gid, uid, clean, mem, ctx)

        # ── 私聊处理 ──
        elif msg_type == "private":
            self.log_signal.emit(f"[私聊] {uid}: {raw_msg[:80]}")

            # 检查指令
            cmd_reply = self._handle_command(msg_type, uid, None, raw_msg)
            if cmd_reply:
                self._send_private(uid, cmd_reply)
                self.reply_signal.emit(f"私聊{uid}", f"[指令] {raw_msg}")
                return

            # 异步生成回复
            mem = self._load_memory("private", str(uid))
            ctx = self._build_context(mem)
            self._executor.submit(self._generate_and_reply_private, uid, raw_msg, mem, ctx)

    def _generate_and_reply_group(self, gid, uid, clean, mem, ctx):
        """在线程池中生成并发送群聊回复"""
        full_prompt = f"{ctx}用户:{clean}\nTGAI?"
        reply = self.generate_fn(full_prompt)
        self.log_signal.emit(f"[TGAI] → {reply[:60]}...")
        mem.append({"q": clean, "a": reply})
        if len(mem) > 12:
            mem = mem[-8:]
        self._save_memory("group", str(gid), mem)
        self._send_group(gid, reply, uid)
        self.reply_signal.emit(f"群聊{gid}", reply[:80])

    def _generate_and_reply_private(self, uid, raw_msg, mem, ctx):
        """在线程池中生成并发送私聊回复"""
        full_prompt = f"{ctx}用户:{raw_msg}\nTGAI?"
        reply = self.generate_fn(full_prompt)
        self.log_signal.emit(f"[TGAI] → {reply[:60]}...")
        mem.append({"q": raw_msg, "a": reply})
        if len(mem) > 12:
            mem = mem[-8:]
        self._save_memory("private", str(uid), mem)
        self._send_private(uid, reply)
        self.reply_signal.emit(f"私聊{uid}", reply[:80])

    def run(self):
        self.log_signal.emit(f"[QQ] 连接: {self.napcat_ws}")
        ws = websocket.WebSocketApp(
            self.napcat_ws,
            on_message=self._on_msg,
            on_open=lambda w: self.log_signal.emit("[QQ] 已连接，等待消息..."),
            on_error=lambda w, e: self.log_signal.emit(f"[QQ] 错误: {e}"),
            on_close=lambda w, c, m: self.log_signal.emit(f"[QQ] 断开 ({c})"),
        )
        while self._running:
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                self.log_signal.emit(f"[QQ] 重连中... ({e})")
                time.sleep(5)

    def stop(self):
        self._running = False


# ─── 主窗口 ──────────────────────────────────────────────
class TGAIWindow(QMainWindow):
    # WebUI → GUI 对话桥接信号 (text, temperature, max_tokens, top_k, top_p, freq_pen, rep_penalty, min_new_tokens, debug)
    _web_chat_request = pyqtSignal(str, float, int, int, float, float, float, int, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TGAI NLP - 计算语言小模型")
        self.resize(1000, 680)

        self.model = None
        self.tokenizer = None
        self._model_device = 'cpu'
        self.training_thread: Optional[TrainingThread] = None
        self.chat_thread: Optional[ChatThread] = None
        self.qq_bot_thread: Optional[QQBotThread] = None
        self.chat_memory: list = []  # 对话记忆 [(q, a), ...]
        self.tg_engine: Optional[TGEngine] = None  # TGAI GO C 引擎
        self.use_go_engine = False  # 是否使用 GO 引擎
        self.use_cloud_api = False  # 是否使用云端 API
        self.cloud_api_url = "http://127.0.0.1:6008"  # 云端 API 地址
        self.cloud_model_id = ""  # 云端模型 ID（空=默认）
        self._cloud_abort = False  # 云端请求中断标志
        self.loss_history = []
        self._train_log_history = []
        self._dark_mode = True

        self._setup_ui()
        self._apply_theme()
        self._web_chat_request.connect(self._handle_web_chat)

    # ─── UI 构建 ──────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 侧边栏 ──
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 12, 8, 12)
        side_layout.setSpacing(2)

        logo = QLabel("TGAI NLP")
        logo.setObjectName("sidebarTitle")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(logo)
        side_layout.addSpacing(8)

        self.sidebar_btns = []
        nav_items = [("💬", "对话", 0), ("📦", "导出", 1), ("🖥", "训练", 2), ("⚙️", "设置", 3), ("🤖", "QQ机器人", 4)]
        for icon, label, idx in nav_items:
            btn = QPushButton(f"  {icon}  {label}")
            btn.setObjectName("sidebarBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self._switch_panel(i))
            side_layout.addWidget(btn)
            self.sidebar_btns.append(btn)

        side_layout.addSpacing(10)

        # 快捷操作
        sep1 = QLabel("快捷操作"); sep1.setObjectName("sidebarSep")
        side_layout.addWidget(sep1)

        btn_server = QPushButton("  🌐  启动API服务")
        btn_server.setObjectName("sidebarBtnSmall")
        btn_server.clicked.connect(self._start_api_server)
        side_layout.addWidget(btn_server)

        btn_apk = QPushButton("  📱  编译APK")
        btn_apk.setObjectName("sidebarBtnSmall")
        btn_apk.clicked.connect(self._build_apk)
        side_layout.addWidget(btn_apk)

        side_layout.addStretch()

        # 性能监控
        self.perf_label = QLabel("CPU: --  RAM: --")
        self.perf_label.setObjectName("perfLabel")
        self.perf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.perf_label.setWordWrap(True)
        side_layout.addWidget(self.perf_label)

        sep2 = QLabel("系统")
        sep2.setObjectName("sidebarSep")
        side_layout.addWidget(sep2)

        self.btn_theme = QPushButton("  🌙  深色模式")
        self.btn_theme.setObjectName("sidebarBtnSmall")
        self.btn_theme.clicked.connect(self._toggle_theme)
        side_layout.addWidget(self.btn_theme)

        layout.addWidget(sidebar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("sidebarSep")
        layout.addWidget(sep)

        # ── 内容区 ──
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self._create_chat_panel())
        self.content_stack.addWidget(self._create_export_panel())
        self.content_stack.addWidget(self._create_train_panel())
        self.content_stack.addWidget(self._create_settings_panel())
        self.content_stack.addWidget(self._create_qq_bot_panel())
        layout.addWidget(self.content_stack, 1)
        outer.addLayout(layout)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusBar")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(26)
        outer.addWidget(self.status_label)
        self.sidebar_btns[0].setChecked(True)

        self._perf_timer = QTimer()
        self._perf_timer.timeout.connect(self._update_perf_monitor)
        self._perf_timer.start(3000)

    def _switch_panel(self, idx):
        self.content_stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.sidebar_btns):
            btn.setChecked(i == idx)

    # ─── 训练面板 ──────────────────────────────────
    def _create_train_panel(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # 上半部分：配置 + 按钮
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(top)

        # 模型参数组
        model_group = QGroupBox("模型参数")
        g = QGridLayout(model_group)
        g.setSpacing(4)

        self.cfg_d_model = self._add_spin(g, "隐藏维度:", 0, 64, 1024, 256, 64)
        self.cfg_n_layers = self._add_spin(g, "层数:", 1, 1, 24, 8)
        self.cfg_n_heads = self._add_spin(g, "注意力头:", 2, 1, 16, 8)
        self.cfg_d_ff = self._add_spin(g, "FFN维度:", 3, 128, 4096, 1024, 128)
        self.cfg_dropout = self._add_double(g, "Dropout:", 4, 0.0, 0.5, 0.1, 0.05)
        self.cfg_seq_len = self._add_spin(g, "序列长度:", 5, 64, 512, 256, 64)
        self.cfg_vocab = self._add_spin(g, "目标词表:", 6, 1024, 32768, 16384, 1024)
        self.cfg_n_experts = self._add_spin(g, "MoE专家数:", 7, 1, 16, 4)
        self.cfg_n_activated = self._add_spin(g, "激活专家:", 8, 1, 4, 2)
        top_layout.addWidget(model_group)

        # 训练参数组
        train_group = QGroupBox("训练参数")
        g2 = QGridLayout(train_group)
        g2.setSpacing(4)

        self.cfg_epochs = self._add_spin(g2, "训练轮数:", 0, 1, 200, 100)
        self.cfg_batch = self._add_spin(g2, "批次大小:", 1, 1, 128, 16)
        self.cfg_lr = QDoubleSpinBox()
        self.cfg_lr.setRange(0.00001, 0.1)
        self.cfg_lr.setValue(0.0003)
        self.cfg_lr.setDecimals(5)
        self.cfg_lr.setSingleStep(0.0001)
        g2.addWidget(QLabel("学习率:"), 2, 0)
        g2.addWidget(self.cfg_lr, 2, 1)

        self.cfg_wd = self._add_double(g2, "权重衰减:", 3, 0.0, 0.5, 0.01, 0.01)
        self.cfg_warmup = self._add_spin(g2, "Warmup步:", 4, 0, 5000, 100, 100)

        # 语料路径
        self.cfg_data_path = QLineEdit()
        self.cfg_data_path.setText("data/train_all.jsonl")
        self.cfg_data_path.setPlaceholderText("训练数据 .jsonl 路径")
        btn_browse_data = QPushButton("浏览...")
        btn_browse_data.clicked.connect(self._browse_data_path)
        g2.addWidget(QLabel("训练数据:"), 5, 0)
        g2.addWidget(self.cfg_data_path, 5, 1)
        g2.addWidget(btn_browse_data, 5, 2)

        top_layout.addWidget(train_group)

        # 知识蒸馏参数组
        distill_group = QGroupBox("知识蒸馏 (软标签)")
        dg = QGridLayout(distill_group)
        dg.setSpacing(4)

        self.cfg_distill_enable = QCheckBox("启用软标签蒸馏")
        self.cfg_distill_enable.setChecked(False)
        dg.addWidget(self.cfg_distill_enable, 0, 0, 1, 2)

        self.cfg_distill_alpha = self._add_double(dg, "蒸馏权重α:", 1, 0.0, 1.0, 0.3, 0.1)
        self.cfg_distill_temp = self._add_double(dg, "蒸馏温度:", 2, 1.0, 10.0, 2.0, 0.5)

        self.cfg_teacher_path = QLineEdit()
        self.cfg_teacher_path.setPlaceholderText("教师模型 checkpoint 路径 (可选)")
        dg.addWidget(QLabel("教师模型:"), 3, 0)
        dg.addWidget(self.cfg_teacher_path, 3, 1)

        self.btn_browse_teacher = QPushButton("浏览...")
        self.btn_browse_teacher.clicked.connect(self._browse_teacher)
        dg.addWidget(self.btn_browse_teacher, 3, 2)

        top_layout.addWidget(distill_group)

        # 续训参数组
        resume_group = QGroupBox("断点续训")
        rg = QGridLayout(resume_group)
        rg.setSpacing(4)

        self.cfg_custom_ckpt = QLineEdit()
        self.cfg_custom_ckpt.setPlaceholderText("自定义 checkpoint 路径 (留空自动找 last_step.pt)")
        rg.addWidget(QLabel("Checkpoint:"), 0, 0)
        rg.addWidget(self.cfg_custom_ckpt, 0, 1)

        self.btn_browse_ckpt = QPushButton("浏览...")
        self.btn_browse_ckpt.clicked.connect(self._browse_checkpoint)
        rg.addWidget(self.btn_browse_ckpt, 0, 2)

        top_layout.addWidget(resume_group)

        # 按钮组
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout(btn_group)

        # GPU 开关
        has_cuda = torch.cuda.is_available()
        self.cfg_use_gpu = QCheckBox("启用 GPU 加速")
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.cfg_use_gpu.setToolTip(f"检测到: {gpu_name} ({gpu_mem:.1f}GB)")
            self.cfg_use_gpu.setChecked(True)
        else:
            self.cfg_use_gpu.setToolTip(
                "未检测到 CUDA GPU。请安装 CUDA 版 PyTorch:\n"
                "国内镜像: pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118\n"
                "或官方源: pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )
            self.cfg_use_gpu.setStyleSheet("color: #c90;")
        btn_layout.addWidget(self.cfg_use_gpu)

        # Self-Extend 长上下文扩展
        self.cfg_self_extend = QCheckBox("Self-Extend 长上下文扩展")
        self.cfg_self_extend.setToolTip(
            "动态位置分组，零训练扩展上下文。\n"
            "模型原窗口 512 token，启用后可扩展至 4096+ tokens\n"
            "短输入 (<512) 完全等于原模型，无质量损失"
        )
        self.cfg_self_extend.setChecked(False)
        btn_layout.addWidget(self.cfg_self_extend)

        extend_layout = QHBoxLayout()
        extend_layout.addWidget(QLabel("扩展窗口:"))
        self.cfg_extend_len = QSpinBox()
        self.cfg_extend_len.setRange(1024, 131072)
        self.cfg_extend_len.setSingleStep(1024)
        self.cfg_extend_len.setValue(4096)
        self.cfg_extend_len.setToolTip("扩展后的最大上下文长度 (KV 缓存大小)\n注意: 越大显存占用越多")
        extend_layout.addWidget(self.cfg_extend_len)
        extend_layout.addWidget(QLabel("tokens"))
        btn_layout.addLayout(extend_layout)

        self.btn_start_train = QPushButton("▶ 开始训练")
        self.btn_start_train.clicked.connect(self._start_training)
        self.btn_start_train.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_start_train)

        self.btn_stop_train = QPushButton("⏹ 停止训练")
        self.btn_stop_train.clicked.connect(self._stop_training)
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_stop_train)

        self.btn_load_model = QPushButton("📂 加载已有模型")
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        self.btn_load_model.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_load_model)

        self.cfg_webui = QCheckBox("🌐 启动 Web 服务 (端口5000)")
        self.cfg_webui.setToolTip("启动后可用手机/平板通过浏览器访问\nhttp://你的电脑IP:5000")
        self.cfg_webui.stateChanged.connect(self._toggle_webui)
        btn_layout.addWidget(self.cfg_webui)

        btn_layout.addStretch()
        top_layout.addWidget(btn_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 日志区域
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFont(QFont("Consolas", 10))
        self.train_log.setMinimumHeight(150)
        main_layout.addWidget(self.train_log, 1)

        # Loss 曲线图
        self.loss_chart = LossChart()
        self.loss_chart.setMinimumHeight(200)
        main_layout.addWidget(self.loss_chart, 1)

        return tab

    # ─── 对话标签页 (新) ──────────────────────────
    def _create_chat_panel(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── 顶部快捷栏 ──
        top = QHBoxLayout()
        top.setSpacing(8)

        # 模型选择器
        top.addWidget(QLabel("模型:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(160)
        self.model_selector.setToolTip("切换已加载的模型")
        self.model_selector.addItem("（未加载模型）", "")
        self.model_selector.currentIndexChanged.connect(self._on_model_switch)
        top.addWidget(self.model_selector)

        self.btn_load_model = QPushButton("📂 加载")
        self.btn_load_model.setToolTip("加载 .pt 或 .TG 模型")
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        self.btn_load_model.setFixedWidth(70)
        top.addWidget(self.btn_load_model)

        # 云端开关
        self.cloud_check = QCheckBox("☁️ 云端")
        self.cloud_check.setToolTip("启用云端 API 推理")
        self.cloud_check.stateChanged.connect(self._on_cloud_toggle)
        top.addWidget(self.cloud_check)

        top.addStretch()

        self.btn_clear_chat = QPushButton("清空")
        self.btn_clear_chat.clicked.connect(self._clear_chat_memory)
        self.btn_clear_chat.setFixedWidth(60)
        top.addWidget(self.btn_clear_chat)

        self.debug_check = QCheckBox("调试")
        self.debug_check.setToolTip("显示每步采样详情")
        self.debug_check.stateChanged.connect(self._toggle_debug)
        top.addWidget(self.debug_check)

        layout.addLayout(top)

        # ── 云端配置行 (始终显示) ──
        cloud_row = QHBoxLayout()
        cloud_row.setSpacing(6)

        cloud_row.addWidget(QLabel("URL:"))
        self.cloud_url_input = QLineEdit("http://127.0.0.1:6008")
        self.cloud_url_input.setPlaceholderText("http://IP:端口")
        self.cloud_url_input.setToolTip("云端 API 地址")
        cloud_row.addWidget(self.cloud_url_input, 1)

        cloud_row.addWidget(QLabel("Key:"))
        self.cloud_api_key = QLineEdit()
        self.cloud_api_key.setPlaceholderText("API Key（可选）")
        self.cloud_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.cloud_api_key.setFixedWidth(140)
        self.cloud_api_key.setToolTip("OpenAI 兼容 API 需要 Key，TGAI 私有协议留空")
        cloud_row.addWidget(self.cloud_api_key)

        cloud_row.addWidget(QLabel("协议:"))
        self.cloud_protocol = QComboBox()
        self.cloud_protocol.addItem("TGAI", "tgai")
        self.cloud_protocol.addItem("OpenAI", "openai")
        self.cloud_protocol.setFixedWidth(80)
        self.cloud_protocol.setToolTip("TGAI: 私有协议 / OpenAI: 兼容 /v1/chat/completions")
        cloud_row.addWidget(self.cloud_protocol)

        cloud_row.addWidget(QLabel("模型:"))
        self.cloud_model_combo = QComboBox()
        self.cloud_model_combo.addItem("默认", "")
        self.cloud_model_combo.addItem("TGAI NB V1", "v1")
        self.cloud_model_combo.addItem("TGAI NB V2 42K", "v2_preview")
        self.cloud_model_combo.setFixedWidth(130)
        cloud_row.addWidget(self.cloud_model_combo)

        layout.addLayout(cloud_row)

        # ── 聊天历史 ──
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Microsoft YaHei", 11))
        layout.addWidget(self.chat_history, 1)

        # ── 输入栏 ──
        input_bar = QHBoxLayout()
        input_bar.setSpacing(6)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入消息... (Enter 发送)")
        self.chat_input.returnPressed.connect(self._send_chat)
        self.chat_input.setMinimumHeight(34)
        input_bar.addWidget(self.chat_input, 1)

        self.btn_send = QPushButton("发送")
        self.btn_send.clicked.connect(self._send_chat)
        self.btn_send.setMinimumHeight(34); self.btn_send.setMinimumWidth(70)
        input_bar.addWidget(self.btn_send)

        self.btn_stop_chat = QPushButton("⏹ 停止")
        self.btn_stop_chat.clicked.connect(self._stop_chat)
        self.btn_stop_chat.setEnabled(False)
        self.btn_stop_chat.setMinimumHeight(34); self.btn_stop_chat.setMinimumWidth(70)
        self.btn_stop_chat.setStyleSheet("color: #c44;")
        input_bar.addWidget(self.btn_stop_chat)

        layout.addLayout(input_bar)

        # 状态行
        self.chat_status = QLabel("就绪 — 请加载模型")
        self.chat_status.setStyleSheet("color: #888; font-size: 10pt; padding: 2px;")
        layout.addWidget(self.chat_status)

        return tab

    def _on_cloud_toggle(self, state):
        """云端模式开关"""
        self.use_cloud_api = self.cloud_check.isChecked()
        self.cloud_url_input.setEnabled(self.use_cloud_api)
        self.cloud_api_key.setEnabled(self.use_cloud_api)
        self.cloud_model_combo.setEnabled(self.use_cloud_api)
        self.cloud_protocol.setEnabled(self.use_cloud_api)
        if self.use_cloud_api:
            proto = self.cloud_protocol.currentData()
            self.chat_status.setText(f"状态: ☁️ 云端模式 ({proto.upper()}) — 无需本地模型")
            self.chat_status.setStyleSheet("color: #6af; padding: 4px;")
        else:
            self.chat_status.setText("状态: 未加载模型 — 请先加载 .TG 或 .pt 模型")
            self.chat_status.setStyleSheet("color: #888; padding: 4px;")

    def _cloud_chat_stream(self, prompt: str, temp=0.9, max_tokens=128, top_k=80,
                           top_p=0.9, freq_penalty=0.15, rep_penalty=1.05,
                           min_tokens=5, history=None):
        """通过云端 API 流式生成，yield 每个 token (支持 TGAI 和 OpenAI 协议)"""
        import urllib.request
        import urllib.error
        import json

        url = self.cloud_url_input.text().strip().rstrip("/")
        protocol = self.cloud_protocol.currentData()
        model_id = self.cloud_model_combo.currentData() or ""
        api_key = self.cloud_api_key.text().strip()

        if protocol == "openai":
            # OpenAI 兼容协议: POST /v1/chat/completions
            endpoint = url + "/v1/chat/completions"
            msgs = [{"role": "user", "content": prompt}]
            if history:
                msgs = []
                for h in history:
                    msgs.append({"role": "user", "content": h.get("user", "")})
                    msgs.append({"role": "assistant", "content": h.get("assistant", "")})
                msgs.append({"role": "user", "content": prompt})
            body = {
                "messages": msgs,
                "model": model_id or "tgai",
                "temperature": temp,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": True,
            }
        else:
            # TGAI 私有协议: POST /api/chat
            endpoint = url + "/api/chat"
            body = {
                "message": prompt,
                "model_id": model_id,
                "temperature": temp,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "freq_penalty": freq_penalty,
                "rep_penalty": rep_penalty,
                "min_tokens": min_tokens,
                "history": history or [],
            }

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(endpoint, data=data, headers=headers)
        req.timeout = 120

        try:
            resp = urllib.request.urlopen(req)
            while not self._cloud_abort:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    return
                try:
                    j = json.loads(d)
                    if protocol == "openai":
                        # OpenAI SSE: {"choices":[{"delta":{"content":"text"}}]}
                        choices = j.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    else:
                        if j.get("token"):
                            yield j["token"]
                except json.JSONDecodeError:
                    pass
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"云端 API 错误 ({e.code}): {err_body[:200]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接云端 ({e.reason})")
        except Exception as e:
            raise RuntimeError(f"云端请求失败: {e}")
        finally:
            self._cloud_abort = False

    # ─── 分词器标签页 ──────────────────────────────────
    def _create_tokenizer_panel(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 输入
        layout.addWidget(QLabel("输入文本:"))
        self.tok_input = QPlainTextEdit()
        self.tok_input.setPlaceholderText("输入要分词的文本...")
        self.tok_input.setMaximumHeight(80)
        self.tok_input.setFont(QFont("Microsoft YaHei", 11))
        layout.addWidget(self.tok_input)

        btn_layout = QHBoxLayout()
        self.btn_tokenize = QPushButton("🔍 分词")
        self.btn_tokenize.clicked.connect(self._do_tokenize)
        btn_layout.addWidget(self.btn_tokenize)

        self.btn_show_vocab = QPushButton("📋 查看词表")
        self.btn_show_vocab.clicked.connect(self._show_vocab)
        btn_layout.addWidget(self.btn_show_vocab)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 结果
        self.tok_result = QTextEdit()
        self.tok_result.setReadOnly(True)
        self.tok_result.setFont(QFont("Consolas", 10))
        layout.addWidget(self.tok_result, 1)

        return tab

    # ─── QQ 机器人标签页 ─────────────────────────────
    def _create_qq_bot_panel(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hint = QLabel("使用前请先加载模型，并确保 NapCat 已运行。\n"
                       "群聊中 @机器人 自动回复（4轮记忆），私聊直接回复。\n"
                       "指令: /help /clean /temp /status /forget /memory")
        hint.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(hint)

        # NapCat 配置
        group = QGroupBox("NapCat 连接配置")
        g = QGridLayout(group)
        g.setSpacing(4)

        g.addWidget(QLabel("WebSocket 地址:"), 0, 0)
        self.qq_ws_url = QLineEdit("ws://127.0.0.1:6099")
        g.addWidget(self.qq_ws_url, 0, 1)

        g.addWidget(QLabel("HTTP API 地址:"), 1, 0)
        self.qq_http_url = QLineEdit("http://127.0.0.1:6099")
        g.addWidget(self.qq_http_url, 1, 1)

        g.addWidget(QLabel("访问令牌:"), 2, 0)
        self.qq_token = QLineEdit("")
        self.qq_token.setPlaceholderText("NapCat 设置的 token")
        self.qq_token.setEchoMode(QLineEdit.EchoMode.Password)
        g.addWidget(self.qq_token, 2, 1)

        g.addWidget(QLabel("机器人 QQ:"), 3, 0)
        self.qq_bot_uin = QLineEdit("")
        self.qq_bot_uin.setPlaceholderText("留空自动获取")
        g.addWidget(self.qq_bot_uin, 3, 1)

        # 参数说明
        param_hint = QLabel("📌 生成参数沿用「对话」标签页的设置")
        param_hint.setStyleSheet("color: #89b4fa; padding: 4px; font-size: 11px;")
        g.addWidget(param_hint, 4, 0, 1, 2)

        layout.addWidget(group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_qq_start = QPushButton("▶ 启动")
        self.btn_qq_start.clicked.connect(self._start_qq_bot)
        self.btn_qq_start.setMinimumWidth(100)
        btn_layout.addWidget(self.btn_qq_start)

        self.btn_qq_stop = QPushButton("⏹ 停止")
        self.btn_qq_stop.clicked.connect(self._stop_qq_bot)
        self.btn_qq_stop.setEnabled(False)
        self.btn_qq_stop.setMinimumWidth(100)
        btn_layout.addWidget(self.btn_qq_stop)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 日志
        self.qq_log = QTextEdit()
        self.qq_log.setReadOnly(True)
        self.qq_log.setFont(QFont("Consolas", 10))
        self.qq_log.setMaximumHeight(200)
        layout.addWidget(self.qq_log)

        # 最近回复
        self.qq_reply_preview = QLabel("最近回复: —")
        self.qq_reply_preview.setStyleSheet("color: #4a9; padding: 4px;")
        layout.addWidget(self.qq_reply_preview)

        layout.addStretch()

        # 加载保存的配置
        self._load_qq_config()

        return tab

    def _qq_config_path(self):
        return os.path.join(os.path.dirname(__file__), "qq_bot_config.json")

    def _load_qq_config(self):
        path = self._qq_config_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.qq_ws_url.setText(cfg.get("ws", "ws://127.0.0.1:6099"))
                self.qq_http_url.setText(cfg.get("http", "http://127.0.0.1:6099"))
                self.qq_token.setText(cfg.get("token", ""))
                self.qq_bot_uin.setText(cfg.get("bot_qq", ""))
                # 加载共享的聊天参数
                self.chat_temp.setValue(cfg.get("temp", 0.9))
                self.chat_max_tokens.setValue(cfg.get("max_tokens", 128))
                self.chat_topk.setValue(cfg.get("top_k", 80))
                self.chat_topp.setValue(cfg.get("top_p", 0.9))
                self.chat_freq_pen.setValue(cfg.get("freq_penalty", 0.15))
                self.rep_penalty_spin.setValue(cfg.get("rep_penalty", 1.05))
                self.chat_min_tokens.setValue(cfg.get("min_tokens", 5))
                # 云端 API 设置
                if cfg.get("cloud_api_url"):
                    self.cloud_url_input.setText(cfg["cloud_api_url"])
                cloud_check = cfg.get("use_cloud_api", False)
                self.cloud_check.setChecked(cloud_check)
                self.use_cloud_api = cloud_check
                self.cloud_url_input.setEnabled(cloud_check)
                self.cloud_model_combo.setEnabled(cloud_check)
            except Exception:
                pass

    def _save_qq_config(self):
        cfg = {
            "ws": self.qq_ws_url.text().strip(),
            "http": self.qq_http_url.text().strip(),
            "token": self.qq_token.text().strip(),
            "bot_qq": self.qq_bot_uin.text().strip(),
            # 聊天生成参数 (共用)
            "temp": self.chat_temp.value(),
            "max_tokens": self.chat_max_tokens.value(),
            "top_k": self.chat_topk.value(),
            "top_p": self.chat_topp.value(),
            "freq_penalty": self.chat_freq_pen.value(),
            "rep_penalty": self.rep_penalty_spin.value(),
            "min_tokens": self.chat_min_tokens.value(),
            # 云端 API
            "use_cloud_api": self.use_cloud_api,
            "cloud_api_url": self.cloud_url_input.text().strip(),
            "cloud_model_id": self.cloud_model_combo.currentData() or "",
        }
        with open(self._qq_config_path(), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def _start_qq_bot(self):
        # 云端模式 — 不需要本地模型
        if not self.use_cloud_api:
            if self.use_go_engine and self.tg_engine and self.tg_engine.model:
                pass  # GO 引擎已就绪
            elif self.model is not None:
                pass  # PyTorch 模型已就绪
            else:
                QMessageBox.warning(self, "提示",
                    "请先加载模型 (.TG 或 .pt)\n"
                    "或勾选「☁️ 云端调用」使用云端 API")
                return

        self._save_qq_config()

        ws = self.qq_ws_url.text().strip()
        http = self.qq_http_url.text().strip()
        bot_qq = int(self.qq_bot_uin.text()) if self.qq_bot_uin.text().strip() else 0
        token = self.qq_token.text().strip()
        mem_dir = os.path.join(os.path.dirname(__file__), "qq_memory")

        if self.use_cloud_api:
            # ── 云端 API 路径 ──
            cloud_url = self.cloud_url_input.text().strip().rstrip("/")
            cloud_model = self.cloud_model_combo.currentData() or ""
            temp = self.chat_temp.value()
            max_tok = self.chat_max_tokens.value()
            top_k = self.chat_topk.value()
            top_p = self.chat_topp.value()
            freq_pen = self.chat_freq_pen.value()
            rep_pen = self.rep_penalty_spin.value()
            min_tok = self.chat_min_tokens.value()

            def generate_fn(prompt):
                import urllib.request
                import json
                url = cloud_url + "/api/generate"
                body = {
                    "message": prompt,
                    "model_id": cloud_model,
                    "system_prompt": "你是TGAI，一个由JXW独立开发的AI助手。",
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "top_k": top_k,
                    "top_p": top_p,
                    "freq_penalty": freq_pen,
                    "rep_penalty": rep_pen,
                    "min_tokens": min_tok,
                }
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(url, data=data,
                    headers={"Content-Type": "application/json"})
                req.timeout = 60
                try:
                    resp = urllib.request.urlopen(req)
                    result = json.loads(resp.read().decode("utf-8"))
                    return result.get("response", "[空回复]")
                except Exception as e:
                    return f"[云端错误: {e}]"

        elif self.use_go_engine and self.tg_engine and self.tg_engine.model:
            # ── GO 引擎路径: 同步推理 ──
            engine = self.tg_engine
            temp = self.chat_temp.value()
            top_k = self.chat_topk.value()
            top_p = self.chat_topp.value()
            freq_pen = self.chat_freq_pen.value()
            min_tok = self.chat_min_tokens.value()

            def generate_fn(prompt):
                import math, random
                lib = engine.lib
                ids = engine.encode(prompt, add_special=True)
                if not ids:
                    return "[空回复]"
                cache = lib.tg_kvcache_new(engine.model, engine.max_seq)
                try:
                    ids_arr = (ctypes.c_int32 * len(ids))(*ids)
                    logits = lib.tg_forward(engine.model, ids_arr, len(ids), cache, 0)
                    if not logits:
                        return "[推理失败]"
                    cur_pos = len(ids)
                    response = ""
                    appeared = {}
                    import numpy as np
                    FloatArr = ctypes.POINTER(ctypes.c_float)
                    vocab = engine.vocab_size
                    _last_id = -1
                    _consecutive_same = 0
                    for _ in range(256):
                        logit_ptr = ctypes.cast(logits, FloatArr)
                        logit_arr = np.ctypeslib.as_array(logit_ptr, shape=(vocab,)).copy()

                        # ★ 温度缩放
                        inv_temp = 1.0 / max(temp, 0.01)
                        logit_arr = logit_arr * inv_temp

                        # ★ 前 min_tokens 个 token 禁止 EOS
                        if len(appeared) < min_tok:
                            logit_arr[3] = -1e30

                        # ★ 频率惩罚
                        if freq_pen > 0 and appeared:
                            for tid, cnt in appeared.items():
                                if 0 <= tid < vocab and cnt > 0:
                                    logit_arr[tid] -= freq_pen * min(cnt, 2.0)

                        # ★ 连续重复惩罚
                        if _consecutive_same >= 3:
                            logit_arr[_last_id] -= _consecutive_same * 3.0
                        if _consecutive_same > 12:
                            break

                        # ★ top-1 过于自信时打压
                        sorted_check = np.sort(logit_arr)[-2:]
                        top1_gap = sorted_check[-1] - sorted_check[-2]
                        gap_threshold = 0.2 / max(temp, 0.1)
                        if top1_gap > gap_threshold:
                            logit_arr[int(np.argmax(logit_arr))] -= 1.5

                        # GO引擎: temperature + top-k + top-p 采样
                        k = min(top_k, vocab)
                        if k > 0:
                            top_idx = np.argpartition(logit_arr, -k)[-k:]
                            top_logits = logit_arr[top_idx]
                        else:
                            top_idx = np.arange(vocab)
                            top_logits = logit_arr
                        top_logits = top_logits - np.max(top_logits)
                        probs = np.exp(top_logits)
                        probs = probs / np.maximum(np.sum(probs), 1e-10)
                        # top-p 截断
                        if top_p < 1.0:
                            sorted_order = np.argsort(probs)[::-1]
                            cumsum = np.cumsum(probs[sorted_order])
                            keep = cumsum <= top_p
                            keep[0] = True
                            cutoff = np.argmin(keep)
                            probs[sorted_order[cutoff:]] = 0
                            probs = probs / max(np.sum(probs), 1e-10)
                        if np.any(np.isnan(probs)):
                            best_id = int(np.argmax(logit_arr))
                        else:
                            choice = np.random.choice(len(top_idx), p=probs)
                            best_id = int(top_idx[choice])
                        if best_id == 3:  # EOS
                            break

                        if best_id == _last_id:
                            _consecutive_same += 1
                        else:
                            _consecutive_same = 0
                        _last_id = best_id
                        appeared[best_id] = appeared.get(best_id, 0) + 1
                        response += engine.token_str(best_id)
                        next_arr = (ctypes.c_int32 * 1)(best_id)
                        logits = lib.tg_forward(engine.model, next_arr, 1, cache, cur_pos)
                        if not logits:
                            break
                        cur_pos += 1
                        if cur_pos >= engine.max_seq:
                            break
                    for sep in ['\n用户:', '用户:']:
                        idx = response.find(sep)
                        if idx > 0:
                            response = response[:idx]
                    return response.strip() or "[空回复]"
                finally:
                    lib.tg_kvcache_free(cache)
        else:
            # ── PyTorch 路径 (原有逻辑) ──
            device = getattr(self, '_model_device', 'cpu')
            model = self.model
            tokenizer = self.tokenizer

            def generate_fn(prompt):
                """prompt 已经是 "用户:...\\nTGAI?" 格式"""
                from inference import TextGenerator
                # 智能截断: 只截历史上下文, 保留当前问题
                max_gen = self.chat_max_tokens.value()
                effective_max = model.config.extend_max_seq_len if model.config.self_extend else model.config.max_seq_len
                max_prompt_tokens = effective_max - max_gen - 5
                prompt_ids = tokenizer.encode(prompt, add_special=False)
                if len(prompt_ids) > max_prompt_tokens:
                    # 找到最后一个 "用户:" 的位置(当前问题开头)
                    current_q_marker = tokenizer.encode("用户:", add_special=False)
                    # 从后往前找当前问题的起始位置
                    cut_point = len(prompt_ids) - max_prompt_tokens
                    # 保证不切到当前问题内: 找到 cut_point 之后第一个 "用户:"
                    for i in range(cut_point, len(prompt_ids) - len(current_q_marker)):
                        if prompt_ids[i:i+len(current_q_marker)] == current_q_marker:
                            cut_point = i
                            break
                    prompt_ids = prompt_ids[cut_point:]
                    prompt = tokenizer.decode(prompt_ids, skip_special=True)
                    print(f"[TGAI] 上下文过长, 截断至 {len(prompt_ids)} tokens (保留当前问题)")

                generator = TextGenerator(model, tokenizer)
                response = ""
                token_count = 0
                print(f"\n[TGAI] 生成中... (prompt {len(prompt_ids)} tokens, 窗口 {effective_max}, 预留 {max_gen+5})")
                try:
                    for chunk in generator.generate(
                        prompt,
                        max_new_tokens=self.chat_max_tokens.value(),
                        temperature=self.chat_temp.value(),
                        top_k=self.chat_topk.value(),
                        top_p=self.chat_topp.value(),
                        frequency_penalty=self.chat_freq_pen.value(),
                        repetition_penalty=self.rep_penalty_spin.value(),
                        min_new_tokens=self.chat_min_tokens.value(),
                        formatted=True, stream=True,
                    ):
                        print(chunk, end="", flush=True)
                        response += chunk
                        token_count += 1
                except Exception as e:
                    print(f"\n[TGAI] 错误: {e}")
                    return f"[生成错误: {e}]"
                print()
                if token_count == 0:
                    print("[TGAI] 未生成任何token, 可能是温度过低或模型退化")
                return response.strip() or "[空回复]"

        self.qq_bot_thread = QQBotThread(generate_fn, ws, http, bot_qq, token=token, memory_dir=mem_dir)
        self.qq_bot_thread.log_signal.connect(lambda t: self.qq_log.append(t))
        self.qq_bot_thread.reply_signal.connect(lambda info, prev: self.qq_reply_preview.setText(f"最近回复 [{info}]: {prev}"))
        self.qq_bot_thread.start()

        self.btn_qq_start.setEnabled(False)
        self.btn_qq_stop.setEnabled(True)
        self.qq_log.append("[QQ] 机器人已启动")

    def _stop_qq_bot(self):
        if self.qq_bot_thread:
            self.qq_bot_thread.stop()
            self.qq_bot_thread.quit()
            self.qq_bot_thread.wait(2000)
            self.qq_bot_thread = None
        self.btn_qq_start.setEnabled(True)
        self.btn_qq_stop.setEnabled(False)
        self.qq_log.append("[QQ] 机器人已停止")

    # ─── 数据标签页 ──────────────────────────────────
    def _create_data_panel(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("训练语料编辑器（每行一条数据）:"))

        self.data_editor = QPlainTextEdit()
        self.data_editor.setFont(QFont("Microsoft YaHei", 11))
        self.data_editor.setPlaceholderText(
            "苹果是食物\n苹果手机是苹果公司的产品\n我是TGAI，你好\n..."
        )
        layout.addWidget(self.data_editor, 1)

        btn_layout = QHBoxLayout()

        self.btn_load_demo = QPushButton("📥 加载内置数据")
        self.btn_load_demo.clicked.connect(self._load_demo_data)
        btn_layout.addWidget(self.btn_load_demo)

        self.btn_save_data = QPushButton("💾 保存为训练数据")
        self.btn_save_data.clicked.connect(self._save_training_data)
        btn_layout.addWidget(self.btn_save_data)

        self.btn_load_data = QPushButton("📂 从文件加载")
        self.btn_load_data.clicked.connect(self._load_data_file)
        btn_layout.addWidget(self.btn_load_data)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return tab

    # ─── 导出面板 ──────────────────────────────────
    def _create_export_panel(self):
        """导出面板 — pt→TG 转换 + ONNX 导出"""
        import os

        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Card 1: pt / GGUF → TG 转换 (主力)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        tg_card = QGroupBox("📱 pt / GGUF → TG 转换  （TGAI GO 引擎）")
        tg_card.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        tg_layout = QVBoxLayout(tg_card)
        tg_layout.setSpacing(10)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("源模型 (.pt / .gguf):"))
        self.tg_export_pt = QLineEdit()
        self.tg_export_pt.setPlaceholderText("选择 .pt checkpoint 或 .gguf 文件...")
        self.tg_export_pt.setReadOnly(True)
        row1.addWidget(self.tg_export_pt, 1)
        btn_pt = QPushButton("浏览"); btn_pt.setFixedWidth(60)
        btn_pt.clicked.connect(lambda: self._pick_tg_export_pt())
        row1.addWidget(btn_pt)
        tg_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出 .TG:"))
        self.tg_export_out = QLineEdit()
        self.tg_export_out.setPlaceholderText("自动填充，或手动选择...")
        row2.addWidget(self.tg_export_out, 1)
        btn_out = QPushButton("浏览"); btn_out.setFixedWidth(60)
        btn_out.clicked.connect(lambda: self._pick_tg_export_out())
        row2.addWidget(btn_out)
        tg_layout.addLayout(row2)

        row_tok = QHBoxLayout()
        row_tok.addWidget(QLabel("分词器 (.pt 必需):"))
        self.tg_export_tokenizer = QLineEdit()
        self.tg_export_tokenizer.setPlaceholderText("自动检测 tokenizer.json...")
        row_tok.addWidget(self.tg_export_tokenizer, 1)
        btn_tok = QPushButton("浏览"); btn_tok.setFixedWidth(60)
        btn_tok.clicked.connect(lambda: self._pick_tg_export_tokenizer())
        row_tok.addWidget(btn_tok)
        tg_layout.addLayout(row_tok)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("量化精度:"))
        self.tg_export_dtype = QComboBox()
        self.tg_export_dtype.addItem("FP16 半精度 — 平衡，推荐", "fp16")
        self.tg_export_dtype.addItem("Q8_0 分组 8-bit — 高精度", "q8_0")
        self.tg_export_dtype.addItem("Q4_0 分组 4-bit — 极致压缩，手机首选", "q4_0")
        self.tg_export_dtype.addItem("INT8 全局 (旧)", "int8")
        self.tg_export_dtype.addItem("FP32 全精度", "fp32")
        row3.addWidget(self.tg_export_dtype, 1)
        row3.addWidget(QLabel("  架构:"))
        self.tg_export_arch = QComboBox()
        self.tg_export_arch.addItem("TGAI MoE (默认)", "tgai_moe")
        self.tg_export_arch.addItem("Llama 3", "llama")
        self.tg_export_arch.addItem("Mistral", "mistral")
        self.tg_export_arch.addItem("Qwen2", "qwen2")
        self.tg_export_arch.addItem("Gemma 2", "gemma")
        self.tg_export_arch.addItem("Phi-3", "phi3")
        self.tg_export_arch.setToolTip("模型架构 (.pt 需要选择, .gguf 自动检测)")
        row3.addWidget(self.tg_export_arch)
        row3.addStretch()
        self.btn_tg_export = QPushButton("🔄 开始转换")
        self.btn_tg_export.setMinimumHeight(34); self.btn_tg_export.setMinimumWidth(120)
        self.btn_tg_export.clicked.connect(self._start_tg_export)
        row3.addWidget(self.btn_tg_export)
        tg_layout.addLayout(row3)

        self.tg_export_progress = QProgressBar()
        self.tg_export_progress.setVisible(False)
        self.tg_export_progress.setTextVisible(True)
        self.tg_export_progress.setFormat("转换中... %p%")
        tg_layout.addWidget(self.tg_export_progress)

        self.tg_export_log = QTextEdit()
        self.tg_export_log.setReadOnly(True)
        self.tg_export_log.setFont(QFont("Consolas", 9))
        self.tg_export_log.setMaximumHeight(200)
        self.tg_export_log.setPlaceholderText("转换日志将显示在这里...")
        tg_layout.addWidget(self.tg_export_log)

        layout.addWidget(tg_card)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Card 2: ONNX 导出 (可折叠)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.onnx_collapse_btn = QPushButton("📦 传统 ONNX 导出 + 快速打包 ▼")
        self.onnx_collapse_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 8px 12px; font-size: 11pt; "
            "background: transparent; border: 1px solid #3e4452; border-radius: 6px; }"
            "QPushButton:hover { border-color: #61afef; }"
        )
        self.onnx_collapse_btn.clicked.connect(self._toggle_onnx_section)
        layout.addWidget(self.onnx_collapse_btn)

        self.onnx_section = QWidget()
        self.onnx_section.setVisible(False)
        onnx_outer = QVBoxLayout(self.onnx_section)
        onnx_outer.setContentsMargins(0, 0, 0, 0)
        onnx_outer.setSpacing(12)

        onnx_card = QGroupBox("ONNX 导出配置")
        cg = QGridLayout(onnx_card)
        cg.setSpacing(8)
        cg.addWidget(QLabel("检查点 (.pt):"), 0, 0)
        self.export_ckpt_path = QLineEdit()
        self.export_ckpt_path.setPlaceholderText("选择 milestone.pt 或 last_step.pt")
        self.export_ckpt_path.setReadOnly(True)
        cg.addWidget(self.export_ckpt_path, 0, 1)
        btn = QPushButton("浏览"); btn.clicked.connect(self._browse_export_ckpt); cg.addWidget(btn, 0, 2)

        cg.addWidget(QLabel("输出目录:"), 1, 0)
        self.export_out_dir = QLineEdit()
        self.export_out_dir.setPlaceholderText("导出 .onnx / .TG 存放位置")
        self.export_out_dir.setReadOnly(True)
        cg.addWidget(self.export_out_dir, 1, 1)
        btn = QPushButton("浏览"); btn.clicked.connect(self._browse_export_out); cg.addWidget(btn, 1, 2)

        cg.addWidget(QLabel("模型名:"), 2, 0)
        self.export_model_name = QLineEdit()
        self.export_model_name.setPlaceholderText("可选，TG 包显示名")
        cg.addWidget(self.export_model_name, 2, 1, 1, 2)

        cg.addWidget(QLabel("类型:"), 3, 0)
        self.export_model_type = QComboBox()
        self.export_model_type.addItems(["自动检测", "TGAI (MoE)", "YUAZ (Llama)"])
        cg.addWidget(self.export_model_type, 3, 1, 1, 2)

        self.export_int8 = QCheckBox("INT8 动态量化")
        self.export_int8.setChecked(True)
        cg.addWidget(self.export_int8, 4, 1, 1, 2)
        onnx_outer.addWidget(onnx_card)

        onnx_btns = QHBoxLayout()
        self.btn_do_export = QPushButton("🚀 导出 ONNX + 打包 .TG")
        self.btn_do_export.setMinimumHeight(34)
        self.btn_do_export.clicked.connect(self._do_export)
        onnx_btns.addWidget(self.btn_do_export)
        self.btn_export_cancel = QPushButton("取消")
        self.btn_export_cancel.clicked.connect(self._cancel_export)
        self.btn_export_cancel.setEnabled(False)
        onnx_btns.addWidget(self.btn_export_cancel); onnx_btns.addStretch()
        onnx_outer.addLayout(onnx_btns)

        self.export_progress = QProgressBar(); self.export_progress.setVisible(False)
        onnx_outer.addWidget(self.export_progress)
        self.export_log = QTextEdit(); self.export_log.setReadOnly(True)
        self.export_log.setFont(QFont("Consolas", 9)); self.export_log.setMaximumHeight(150)
        onnx_outer.addWidget(self.export_log)

        quick_card = QGroupBox("⚡ 快速打包（已有 ONNX）")
        qg = QGridLayout(quick_card); qg.setSpacing(6)
        qg.addWidget(QLabel("ONNX:"), 0, 0)
        self.quick_onnx_path = QLineEdit(); self.quick_onnx_path.setPlaceholderText("xxx.onnx 文件")
        self.quick_onnx_path.setReadOnly(True); qg.addWidget(self.quick_onnx_path, 0, 1)
        btn = QPushButton("浏览"); btn.clicked.connect(self._browse_quick_onnx); qg.addWidget(btn, 0, 2)

        qg.addWidget(QLabel("分词器:"), 1, 0)
        self.quick_tokenizer_path = QLineEdit(); self.quick_tokenizer_path.setPlaceholderText("tokenizer.json")
        self.quick_tokenizer_path.setReadOnly(True); qg.addWidget(self.quick_tokenizer_path, 1, 1)
        btn = QPushButton("浏览"); btn.clicked.connect(self._browse_quick_tok); qg.addWidget(btn, 1, 2)

        qg.addWidget(QLabel("名称:"), 2, 0)
        self.quick_model_name = QLineEdit(); self.quick_model_name.setPlaceholderText("可选")
        qg.addWidget(self.quick_model_name, 2, 1, 1, 2)
        self.btn_quick_pack = QPushButton("⚡ 快速打包 .TG"); self.btn_quick_pack.setMinimumHeight(34)
        self.btn_quick_pack.clicked.connect(self._quick_pack); qg.addWidget(self.btn_quick_pack, 3, 0, 1, 3)
        self.quick_progress = QProgressBar(); self.quick_progress.setVisible(False)
        qg.addWidget(self.quick_progress, 4, 0, 1, 3)
        self.quick_log = QTextEdit(); self.quick_log.setReadOnly(True)
        self.quick_log.setFont(QFont("Consolas", 9)); self.quick_log.setMaximumHeight(100)
        self.quick_log.setPlaceholderText("打包日志..."); qg.addWidget(self.quick_log, 5, 0, 1, 3)
        onnx_outer.addWidget(quick_card)
        layout.addWidget(self.onnx_section)

        default_out = os.path.join(os.path.dirname(__file__), '..', 'tg_chat', 'tg_mobile_models')
        self.export_out_dir.setText(os.path.abspath(default_out))
        self._auto_detect_checkpoint()

        layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        return tab

    def _auto_detect_checkpoint(self):
        import os
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            return

        candidates = []
        for f in sorted(os.listdir(ckpt_dir), reverse=True):
            if f.endswith('.pt'):
                full = os.path.join(ckpt_dir, f)
                mtime = os.path.getmtime(full)
                candidates.append((mtime, full))

        if candidates:
            candidates.sort(reverse=True)
            best = candidates[0][1]
            self.export_ckpt_path.setText(best)
            # 自动填充模型名
            stem = os.path.splitext(os.path.basename(best))[0]
            self.export_model_name.setText(stem)

    # ─── pt→TG 内联导出辅助 ───────────────────────
    def _pick_tg_export_pt(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择源模型", "",
                                            "模型文件 (*.pt *.gguf);;PyTorch (*.pt);;GGUF (*.gguf);;All (*.*)")
        if p:
            self.tg_export_pt.setText(p)
            if not self.tg_export_out.text():
                self.tg_export_out.setText(os.path.splitext(p)[0] + '.TG')
            # Auto-detect tokenizer for .pt files
            if p.lower().endswith('.pt') and not self.tg_export_tokenizer.text():
                # Look for tokenizer.json near the checkpoint
                ckpt_dir = os.path.dirname(p)
                for attempt in [
                    os.path.join(ckpt_dir, 'tokenizer.json'),
                    os.path.join(ckpt_dir, '..', 'tokenizer.json'),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'tokenizer.json'),
                ]:
                    if os.path.exists(attempt):
                        self.tg_export_tokenizer.setText(attempt)
                        break
            # Auto-select GGUF hint
            if p.lower().endswith('.gguf'):
                self.tg_export_tokenizer.setText("(GGUF 自动提取, 无需手动指定)")
                self.tg_export_tokenizer.setEnabled(False)
            else:
                self.tg_export_tokenizer.setEnabled(True)

    def _pick_tg_export_tokenizer(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 tokenizer.json", "",
                                            "JSON (*.json);;All (*.*)")
        if p:
            self.tg_export_tokenizer.setText(p)

    def _pick_tg_export_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "保存 .TG 文件",
                                            self.tg_export_out.text() or "model.TG",
                                            "TGAI Model (*.TG)")
        if p:
            self.tg_export_out.setText(p)

    def _start_tg_export(self):
        import subprocess
        pt_path = self.tg_export_pt.text().strip()
        tg_path = self.tg_export_out.text().strip()
        dtype = self.tg_export_dtype.currentData()
        is_gguf = pt_path.lower().endswith('.gguf')

        if not pt_path or not os.path.exists(pt_path):
            QMessageBox.critical(self, "错误", "请选择有效的源模型文件"); return
        if not tg_path:
            QMessageBox.critical(self, "错误", "请指定输出 .TG 路径"); return

        project_root = os.path.dirname(os.path.abspath(__file__))
        tokenizer = self.tg_export_tokenizer.text().strip()

        if is_gguf:
            converter = os.path.join(project_root, 'TGAI GO', 'tools', 'gguf_to_tg.py')
            if not os.path.exists(converter):
                QMessageBox.critical(self, "错误", f"找不到 GGUF 转换脚本:\n{converter}"); return
            # GGUF 转换: --dtype 仅支持 fp16/fp32
            if dtype not in ('fp16', 'fp32'):
                dtype = 'fp16'
            cmd = [sys.executable, '-u', converter,
                   '--input', pt_path, '--output', tg_path, '--dtype', dtype]
            log_prefix = f"<b>GGUF→TG: {os.path.basename(pt_path)} → {dtype.upper()}</b>"
        else:
            converter = os.path.join(project_root, 'TGAI GO', 'tools', 'tgai_convert.py')
            if not os.path.exists(converter):
                QMessageBox.critical(self, "错误", f"找不到转换脚本:\n{converter}"); return
            if not tokenizer or tokenizer.startswith('('):
                # Auto-detect tokenizer
                for attempt in [
                    os.path.join(os.path.dirname(pt_path), 'tokenizer.json'),
                    os.path.join(project_root, 'checkpoints', 'tokenizer.json'),
                ]:
                    if os.path.exists(attempt):
                        tokenizer = attempt
                        break
            if not tokenizer or not os.path.exists(tokenizer):
                QMessageBox.critical(self, "错误", f"找不到 tokenizer.json!\n请手动指定或放在 checkpoints/ 目录下"); return
            arch = self.tg_export_arch.currentData()
            cmd = [sys.executable, '-u', converter,
                   '--checkpoint', pt_path, '--output', tg_path,
                   '--tokenizer', tokenizer, '--dtype', dtype,
                   '--arch', arch]
            log_prefix = f"<b>.pt→TG: {os.path.basename(pt_path)} → {dtype.upper()} (arch={arch})</b>"

        self.btn_tg_export.setEnabled(False)
        self.tg_export_progress.setVisible(True); self.tg_export_progress.setValue(0)
        self.tg_export_log.clear()
        self.tg_export_log.append(log_prefix)

        class TGExportThread(QThread):
            log_signal = pyqtSignal(str); progress_signal = pyqtSignal(int)
            done_signal = pyqtSignal(bool, str)
            def run(self):
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line: self.log_signal.emit(line)
                    proc.wait()
                    if proc.returncode == 0 and os.path.exists(tg_path):
                        self.progress_signal.emit(100)
                        sz = os.path.getsize(tg_path) / 1024 / 1024
                        self.done_signal.emit(True, f"完成! {sz:.1f} MB ∙ {tg_path}")
                    else:
                        self.done_signal.emit(False, f"失败 (exit={proc.returncode})")
                except Exception as e:
                    self.done_signal.emit(False, str(e))

        self._tg_thread = TGExportThread()
        self._tg_thread.log_signal.connect(
            lambda s: self.tg_export_log.append(f"<span style='color:#888;'>{s}</span>"))
        self._tg_thread.progress_signal.connect(self.tg_export_progress.setValue)
        self._tg_thread.done_signal.connect(self._on_tg_export_done)
        self._tg_thread.start()

    def _on_tg_export_done(self, success, msg):
        self.btn_tg_export.setEnabled(True); self.tg_export_progress.setVisible(False)
        color = "#98c379" if success else "#e44"
        self.tg_export_log.append(f"<br><b style='color:{color};'>{'✅' if success else '❌'} {msg}</b>")

    def _toggle_onnx_section(self):
        visible = not self.onnx_section.isVisible()
        self.onnx_section.setVisible(visible)
        arrow = "▲" if visible else "▼"
        self.onnx_collapse_btn.setText(f"📦 传统 ONNX 导出 + 快速打包 {arrow}")

    def _browse_export_ckpt(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型检查点", "", "PyTorch 检查点 (*.pt);;所有文件 (*)"
        )
        if path:
            self.export_ckpt_path.setText(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            if not self.export_model_name.text():
                self.export_model_name.setText(stem)

    def _browse_export_out(self):
        from PyQt6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.export_out_dir.setText(path)

    def _do_export(self):
        ckpt = self.export_ckpt_path.text().strip()
        out_dir = self.export_out_dir.text().strip()
        name = self.export_model_name.text().strip()

        if not ckpt:
            QMessageBox.warning(self, "提示", "请先选择模型检查点文件")
            return
        if not os.path.isfile(ckpt):
            QMessageBox.warning(self, "提示", f"检查点文件不存在:\n{ckpt}")
            return

        os.makedirs(out_dir, exist_ok=True)

        self.btn_do_export.setEnabled(False)
        self.btn_export_cancel.setEnabled(True)
        self.export_progress.setVisible(True)
        self.export_progress.setRange(0, 0)  # 不确定进度

        model_type = self.export_model_type.currentText()
        if model_type == "自动检测":
            model_type = "auto"
        elif "TGAI" in model_type:
            model_type = "tgai"
        elif "YUAZ" in model_type:
            model_type = "yuaz"

        self._export_thread = ExportThread(ckpt, out_dir, name, model_type, int8=self.export_int8.isChecked())
        self._export_thread.log_signal.connect(self._export_log_append)
        self._export_thread.done_signal.connect(self._export_done)
        self._export_thread.start()

    def _cancel_export(self):
        if hasattr(self, '_export_thread') and self._export_thread.isRunning():
            self._export_thread.terminate()
        self._export_reset_ui()
        self.export_log.append("[已取消]")

    def _export_log_append(self, msg):
        self.export_log.append(msg)

    def _export_done(self, success: bool, result_path: str, error: str):
        self._export_reset_ui()
        if success:
            self.export_log.append(f"\n✅ 导出成功!")
            self.export_log.append(f"  文件位置: {result_path}")
            self.export_log.append(f"\n将此文件传到手机后，在 TG CHAT「模型」页点击「导入 .TG」即可。")
        else:
            self.export_log.append(f"\n❌ 导出失败: {error}")

    def _export_reset_ui(self):
        self.btn_do_export.setEnabled(True)
        self.btn_export_cancel.setEnabled(False)
        self.export_progress.setVisible(False)
        self.export_progress.setRange(0, 100)

    # ─── 快速打包 ────────────────────────────────
    def _browse_quick_onnx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ONNX/PTE 模型文件", "",
            "模型文件 (*.onnx *.pte);;所有文件 (*)")
        if path:
            self.quick_onnx_path.setText(path)

    def _browse_quick_tok(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 tokenizer.json", "",
            "Tokenizer (*.json);;所有文件 (*)")
        if path:
            self.quick_tokenizer_path.setText(path)

    def _quick_pack(self):
        onnx_path = self.quick_onnx_path.text().strip()
        tok_path = self.quick_tokenizer_path.text().strip()
        name = self.quick_model_name.text().strip()

        if not onnx_path:
            QMessageBox.warning(self, "提示", "请先选择 ONNX 模型文件")
            return
        if not os.path.isfile(onnx_path):
            QMessageBox.warning(self, "提示", f"模型文件不存在:\n{onnx_path}")
            return
        if not tok_path:
            QMessageBox.warning(self, "提示", "请先选择 tokenizer.json")
            return
        if not os.path.isfile(tok_path):
            QMessageBox.warning(self, "提示", f"tokenizer 文件不存在:\n{tok_path}")
            return

        self.btn_quick_pack.setEnabled(False)
        self.quick_progress.setVisible(True)
        self.quick_progress.setRange(0, 0)
        self.quick_log.clear()

        self._quick_thread = QuickPackThread(onnx_path, tok_path, name)
        self._quick_thread.log_signal.connect(self._quick_log_append)
        self._quick_thread.done_signal.connect(self._quick_done)
        self._quick_thread.start()

    def _quick_log_append(self, msg):
        self.quick_log.append(msg)

    def _quick_done(self, success: bool, result_path: str, error: str):
        self.btn_quick_pack.setEnabled(True)
        self.quick_progress.setVisible(False)
        self.quick_progress.setRange(0, 100)
        if success:
            self.quick_log.append(f"\n✅ 打包完成！传到手机后在 TG CHAT 导入即可。")
        else:
            self.quick_log.append(f"\n❌ 打包失败: {error}")

    # ─── 辅助控件方法 ────────────────────────────────
    def _add_spin(self, grid, label, row, min_v, max_v, default, step=1):
        grid.addWidget(QLabel(label), row, 0)
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setSingleStep(step)
        grid.addWidget(spin, row, 1)
        return spin

    def _add_double(self, grid, label, row, min_v, max_v, default, step):
        grid.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        grid.addWidget(spin, row, 1)
        return spin

    # ─── Web 服务 ────────────────────────────────────
    _web_thread = None
    _web_stop = None

    def _toggle_webui(self, state: int):
        """启动或停止内嵌 Web 服务"""
        if state == Qt.CheckState.Checked.value:
            self._start_web_server()
        else:
            self._stop_web_server()

    def _start_web_server(self):
        """在后台线程启动 Flask+SocketIO 服务"""
        import threading
        import socket as _socket

        # 获取局域网 IP
        ip = '127.0.0.1'
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
        except:
            pass

        try:
            from flask import Flask, request, jsonify, render_template_string
            from flask_socketio import SocketIO, emit
        except ImportError:
            QMessageBox.warning(self, "缺少依赖",
                "Web 服务需要安装 flask 和 flask-socketio:\n"
                "pip install flask flask-socketio")
            self.cfg_webui.setChecked(False)
            return

        window = self  # 闭包捕获

        app_flask = Flask(__name__)
        app_flask.config['SECRET_KEY'] = 'tgai-web'
        socketio = SocketIO(app_flask, cors_allowed_origins='*', async_mode='threading')

        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')

        # --- HTML ---
        html = r'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>TGAI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei",sans-serif;background:#1a1a2e;color:#cdd6f4;max-width:800px;margin:0 auto;min-height:100vh}
.tabs{display:flex;background:#16213e;position:sticky;top:0;z-index:10}
.tab{flex:1;text-align:center;padding:12px 6px;cursor:pointer;font-size:13px;color:#888;border-bottom:2px solid transparent}
.tab.active{color:#89b4fa;border-bottom-color:#89b4fa}
.panel{display:none;padding:10px}.panel.active{display:block}
.btn{padding:7px 16px;border:none;border-radius:6px;font-size:13px;font-weight:bold;cursor:pointer}
.btn-green{background:#40a02b;color:#fff}.btn-red{background:#d20f39;color:#fff}.btn-blue{background:#1e66f5;color:#fff}.btn-gray{background:#45475a;color:#cdd6f4}
.btn:disabled{opacity:.5}.btn-sm{padding:4px 10px;font-size:11px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.stat{background:#313244;padding:8px;border-radius:8px;text-align:center}
.stat .v{font-size:20px;font-weight:bold;color:#89b4fa}.stat .l{font-size:10px;color:#888;margin-top:2px}
#log{background:#11111b;border-radius:8px;padding:8px;height:180px;overflow-y:auto;font-family:Consolas,monospace;font-size:11px;line-height:1.6;white-space:pre-wrap}
#chat-msgs{flex:1;overflow-y:auto;padding:6px 0}.chat-msg{margin-bottom:8px;line-height:1.5}
.cu{color:#89b4fa;font-weight:bold}.cb{color:#f9e2af;font-weight:bold}.ct{color:#cdd6f4;word-break:break-word}
.cd{font-size:10px;color:#6c7086;margin-top:2px;line-height:1.3}
.ci{display:flex;gap:6px;padding:8px 0;border-top:1px solid #313244}
.ci input{flex:1;background:#313244;border:1px solid #45475a;color:#cdd6f4;padding:8px 12px;border-radius:8px;font-size:13px;outline:none}
#tok-r{margin-top:8px;background:#11111b;border-radius:8px;padding:8px;font-size:11px;line-height:1.8;overflow-x:auto}
.toast{position:fixed;top:10px;right:10px;background:#40a02b;color:#fff;padding:8px 14px;border-radius:8px;font-size:12px;z-index:100;opacity:0;transition:opacity .3s;pointer-events:none}
.toast.show{opacity:1}
.progress{height:16px;background:#313244;border-radius:8px;overflow:hidden;margin:6px 0}
.progress div{height:100%;background:linear-gradient(90deg,#89b4fa,#74c7ec);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e1e2e}
</style></head><body>
<div class="tabs">
<div class="tab active" onclick="sw('train')">训练</div>
<div class="tab" onclick="sw('chat')">对话</div>
<div class="tab" onclick="sw('tok')">分词</div></div>
<div id="p-train" class="panel active">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px">
<span id="ts" style="font-size:12px;color:#888">等待开始</span>
<div><button class="btn btn-green" id="bs" onclick="start()">开始</button>
<button class="btn btn-red" id="bp" onclick="stop()" disabled>停止</button></div></div>
<div class="progress"><div id="pb" style="width:0">0%</div></div>
<div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0">
<span id="eta" style="font-size:12px;color:#89b4fa">预计剩余: --:--</span>
<span id="stept" style="font-size:11px;color:#888">步速: -</span></div>
<div class="stats">
<div class="stat"><div class="v" id="sl">-</div><div class="l">Loss</div></div>
<div class="stat"><div class="v" id="sp">-</div><div class="l">PPL</div></div>
<div class="stat"><div class="v" id="ss">0</div><div class="l">Step</div></div>
<div class="stat"><div class="v" id="sr">-</div><div class="l">LR</div></div></div>
<div style="font-size:11px;color:#888;margin:6px 0 2px">系统状态</div>
<div class="stats" style="grid-template-columns:repeat(5,1fr)">
<div class="stat"><div class="v" id="scpu">-</div><div class="l">CPU</div></div>
<div class="stat"><div class="v" id="sram">-</div><div class="l">内存</div></div>
<div class="stat"><div class="v" id="sgpu">-</div><div class="l">GPU</div></div>
<div class="stat"><div class="v" id="svram">-</div><div class="l">显存</div></div>
<div class="stat"><div class="v" id="stemp">-</div><div class="l">温度</div></div></div>
<details><summary style="cursor:pointer;color:#888;font-size:12px">参数</summary>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:6px 0;font-size:12px">
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">隐藏维度<input id="c1" value="256" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">层数<input id="c2" value="8" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">注意力头<input id="c3" value="8" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">FFN维度<input id="c4" value="1024" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">Dropout<input id="c5" value="0.1" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">序列长度<input id="c6" value="256" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">轮数<input id="c7" value="100" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">批次<input id="c8" value="16" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">学习率<input id="c9" value="0.0003" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">衰减<input id="ca" value="0.01" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
<div style="background:#313244;padding:6px;border-radius:4px;display:flex;justify-content:space-between">Warmup<input id="cb" value="500" style="width:60px;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;text-align:center"></div>
</div></details>
<div id="log" onclick="this.scrollTop=this.scrollHeight"></div></div>
<div id="p-chat" class="panel">
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px">
<button class="btn btn-gray btn-sm" onclick="lm()">加载模型</button>
<span id="mi" style="font-size:10px;color:#888"></span>
<label style="font-size:11px;color:#888;margin-left:auto">温度<span id="tv">0.9</span>
<input type="range" min="0.1" max="2" step="0.1" value="0.9" style="width:70px" oninput="document.getElementById('tv').textContent=this.value"></label>
<label style="font-size:11px;color:#888">MaxToken<input id="mtn" type="number" min="16" max="1024" step="16" value="128" style="width:48px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 3px"></label>
<label style="font-size:11px;color:#888">重复惩罚<input id="rpn" type="number" min="0.1" max="3.0" step="0.05" value="1.05" style="width:55px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 4px"></label>
<label style="font-size:11px;color:#888"><input type="checkbox" id="dbg">调试</label></div>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px">
<label style="font-size:11px;color:#888">Top-K<input id="tkv" type="number" min="1" max="200" value="80" style="width:48px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 3px"></label>
<label style="font-size:11px;color:#888">Top-P<input id="tpv" type="number" min="0" max="1" step="0.05" value="0.90" style="width:48px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 3px"></label>
<label style="font-size:11px;color:#888">频率惩罚<input id="fpv" type="number" min="0" max="1" step="0.05" value="0.15" style="width:48px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 3px"></label>
<label style="font-size:11px;color:#888">最小长度<input id="mnv" type="number" min="0" max="50" value="5" style="width:42px;background:#313244;border:1px solid #45475a;color:#cdd6f4;text-align:center;border-radius:4px;padding:2px 3px"></label></div>
<div style="display:flex;flex-direction:column;height:calc(100vh - 100px)">
<div id="chat-msgs"></div>
<div class="ci"><input id="ci2" placeholder="输入消息..." onkeydown="if(event.key==='Enter')send()">
<button class="btn btn-blue" onclick="send()">发送</button></div></div>
<div id="mo" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:20;align-items:center;justify-content:center" onclick="this.style.display='none'">
<div style="background:#1e1e2e;border-radius:12px;padding:14px;max-width:400px;width:90%;max-height:60vh;overflow-y:auto" onclick="event.stopPropagation()">
<h3 style="margin-bottom:8px">选择模型</h3><div id="ml" style="font-size:12px">加载中...</div>
<button class="btn btn-gray btn-sm" style="margin-top:8px" onclick="document.getElementById('mo').style.display='none'">关闭</button></div></div></div>
<div id="p-tok" class="panel">
<input placeholder="输入文本..." style="width:100%;background:#313244;border:1px solid #45475a;color:#cdd6f4;padding:8px;border-radius:8px;font-size:13px" id="ti" onkeydown="if(event.key==='Enter')tk()">
<button class="btn btn-blue btn-sm" style="margin-top:6px" onclick="tk()">分词</button><div id="tok-r"></div></div>
<div class="toast" id="toast"></div>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
const io_socket = io();
function sw(n){['train','chat','tok'].forEach((v,i)=>{document.querySelectorAll('.tab')[i].classList.toggle('active',v===n);document.getElementById('p-'+v).classList.toggle('active',v===n)})}
function nf(m,d=2000){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),d)}
function start(){io_socket.emit('start_train',{d_model:+c1.value,n_layers:+c2.value,n_heads:+c3.value,d_ff:+c4.value,dropout:+c5.value,seq_len:+c6.value,epochs:+c7.value,batch_size:+c8.value,lr:+c9.value,weight_decay:+ca.value,warmup_steps:+cb.value,n_experts:4,n_activated:2});bs.disabled=true;bp.disabled=false;ts.textContent='训练中...'}
function stop(){io_socket.emit('stop_train');bs.disabled=false;bp.disabled=true;ts.textContent='已停止'}
io_socket.on('train_progress',d=>{ss.textContent=d.step;sl.textContent=d.loss;sr.textContent=d.lr;sp.textContent=d.ppl||'-';ts.textContent='Epoch '+d.epoch+'/'+d.total_epochs+' | Step '+d.step;const p=Math.round(d.step/d.total_steps*100);pb.style.width=p+'%';pb.textContent=p+'%';bp.disabled=false;if(d.step_time>0)stept.textContent='步速: '+d.step_time.toFixed(1)+'s/步';if(d.eta_hours>0){var h=Math.floor(d.eta_hours);var m=Math.floor((d.eta_hours-h)*60);eta.textContent='预计剩余: '+h+'时'+(m<10?'0':'')+m+'分'}else{eta.textContent='预计剩余: --:--'}})
io_socket.on('train_log',d=>{const l=document.getElementById('log');l.textContent+=d.msg+'\\n';l.scrollTop=l.scrollHeight})
io_socket.on('train_done',d=>{bs.disabled=false;bp.disabled=true;ts.textContent='完成! 最佳PPL: '+d.best_ppl;nf('训练完成!')})
function fmtBytes(b){return b<1?b.toFixed(2)+'GB':b.toFixed(1)+'GB'}
io_socket.on('system_stats',function(d){if(d.cpu_percent!=null)scpu.textContent=d.cpu_percent.toFixed(0)+'%';if(d.ram_percent!=null)sram.textContent=d.ram_percent.toFixed(0)+'%';if(d.gpu_util!=null)sgpu.textContent=d.gpu_util+'%';else if(d.gpu_mem_percent!=null)sgpu.textContent='-';if(d.gpu_mem_percent!=null)svram.textContent=d.gpu_mem_percent.toFixed(0)+'%';if(d.gpu_temp!=null&&d.gpu_temp>0)stemp.textContent=d.gpu_temp+'\u00b0C';else stemp.textContent='-'})
setInterval(function(){fetch('/api/system_stats').then(function(r){return r.json()}).then(function(d){if(d.cpu_percent!=null)scpu.textContent=d.cpu_percent.toFixed(0)+'%';if(d.ram_percent!=null)sram.textContent=d.ram_percent.toFixed(0)+'%';if(d.gpu_util!=null)sgpu.textContent=d.gpu_util+'%';if(d.gpu_mem_percent!=null)svram.textContent=d.gpu_mem_percent.toFixed(0)+'%';if(d.gpu_temp!=null&&d.gpu_temp>0)stemp.textContent=d.gpu_temp+'\u00b0C';if(d.step_time>0)stept.textContent='步速: '+d.step_time.toFixed(1)+'s/步';if(d.eta_hours>0){var h=Math.floor(d.eta_hours);var m=Math.floor((d.eta_hours-h)*60);eta.textContent='预计剩余: '+h+'时'+(m<10?'0':'')+m+'分'}}).catch(function(){})},3000)
function lm(){document.getElementById('mo').style.display='flex';fetch('/api/models').then(r=>r.json()).then(a=>{ml.innerHTML=a.map(m=>'<div style="padding:7px;border-bottom:1px solid #313244;cursor:pointer;display:flex;justify-content:space-between" onclick="ld(\''+m.path+'\',\''+m.name+'\')"><span>'+m.name+'</span><span style="color:#888">'+m.size+'MB</span></div>').join('')||'<div style="color:#888">没有模型</div>'})}
function ld(p,n){mo.style.display='none';nf('加载中...');fetch('/api/model/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})}).then(r=>r.json()).then(d=>{if(d.ok){mi.textContent=n+' | Epoch:'+d.epoch+' | '+(d.params/1e6).toFixed(1)+'M | '+d.device;nf('已加载')}else{nf('失败: '+d.error,3000)}})}
var _chatDiv=null;
io_socket.on('chat_sync',function(d){var m=document.getElementById('chat-msgs');if(d.type==='user'){m.innerHTML+='<div class="chat-msg"><span class="cu">你:</span> <span class="ct">'+e(d.text)+'</span></div>';var b=document.createElement('div');b.className='chat-msg';b.innerHTML='<span class="cb">TGAI:</span> <span class="ct"></span>';m.appendChild(b);_chatDiv=b.querySelector('.ct')}else if(d.type==='chunk'&&_chatDiv){_chatDiv.textContent+=d.token;m.scrollTop=m.scrollHeight}else if(d.type==='done'){_chatDiv=null}else if(d.type==='error'){var b2=document.createElement('div');b2.className='chat-msg';b2.innerHTML='<span style="color:#f38ba8">[错误] '+e(d.error)+'</span>';m.appendChild(b2);_chatDiv=null}else if(d.type==='debug'&&d.lines&&_chatDiv){var de=document.createElement('div');de.className='cd';de.style.whiteSpace='pre-wrap';de.innerHTML=d.lines.map(function(l){return e(l)}).join('<br>');_chatDiv.parentElement.appendChild(de);m.scrollTop=m.scrollHeight}});
function send(){var t=ci2.value.trim();if(!t)return;ci2.value='';var tmp=parseFloat(document.querySelector('#p-chat input[type=range]').value);var rp=parseFloat(document.getElementById('rpn').value)||1.05;var mt=parseInt(document.getElementById('mtn').value)||128;var tk=parseInt(document.getElementById('tkv').value)||80;var tp=parseFloat(document.getElementById('tpv').value)||0.9;var fp=parseFloat(document.getElementById('fpv').value)||0.15;var mn=parseInt(document.getElementById('mnv').value)||5;io_socket.emit('send_message',{text:t,temperature:tmp,max_tokens:mt,top_k:tk,top_p:tp,frequency_penalty:fp,repetition_penalty:rp,min_new_tokens:mn,debug:dbg.checked})}
function e(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function tk(){const t=ti.value.trim();if(!t)return;io_socket.emit('tokenize',{text:t})}
io_socket.on('tokenize_result',d=>{let h='<div style="color:#888">词表:'+d.vocab_size+' | 编码:'+d.ids.join(' ')+'</div>';h+='<div style="color:#a6e3a1">解码:'+d.decoded+'</div><div style="margin-top:4px">';d.tokens.forEach(t=>{h+='<span style="color:'+(['<PAD>','<UNK>','<BOS>','<EOS>'].includes(t.token)?'#f38ba8':'#89b4fa')+';margin-right:3px">['+t.id+']'+t.token+'</span>'});h+='</div>';document.getElementById('tok-r').innerHTML=h})
document.addEventListener('DOMContentLoaded',()=>{fetch('/api/status').then(r=>r.json()).then(s=>{if(s.running){bs.disabled=true;bp.disabled=false;ts.textContent=s.message}})})
</script></body></html>'''

        @app_flask.route('/')
        def index():
            return render_template_string(html)

        @app_flask.route('/api/models')
        def api_models():
            ckpts = []
            if os.path.exists(ckpt_dir):
                for f in sorted(os.listdir(ckpt_dir), reverse=True):
                    if f.endswith('.pt') and not f.endswith('.tmp'):
                        fp = os.path.join(ckpt_dir, f)
                        ckpts.append({'name': f, 'size': round(os.path.getsize(fp) / 1e6, 1), 'path': fp.replace('\\', '/')})
            return jsonify(ckpts)

        @app_flask.route('/api/model/load', methods=['POST'])
        def api_model_load():
            data = request.get_json()
            path = data.get('path', '')
            if not path or not os.path.exists(path):
                return jsonify({'ok': False, 'error': '文件不存在'}), 400
            try:
                from model import create_model
                from tokenizer import ChineseTokenizer
                import torch
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                window.tokenizer = ChineseTokenizer.load(tok_path)
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                mc = ckpt.get('model_config', {})
                sd = ckpt['model_state_dict']
                window.model = create_model(
                    vocab_size=sd['token_embedding.weight'].shape[0],
                    d_model=mc.get('d_model', 256), n_layers=mc.get('n_layers', 8),
                    n_heads=mc.get('n_heads', 8), d_ff=mc.get('d_ff', 1024),
                    max_seq_len=mc.get('max_seq_len', 256), dropout=0.0,
                    n_experts=mc.get('n_experts', 4), n_activated=mc.get('n_activated', 2),
                )
                window.model.load_state_dict(_migrate_rope_buffers(sd))
                window.model.to(device_str)
                window.model.eval()
                window._model_device = device_str
                n_params = sum(p.numel() for p in window.model.parameters() if p.requires_grad)
                return jsonify({'ok': True, 'name': os.path.basename(path),
                                'epoch': ckpt.get('epoch', '?'), 'params': n_params, 'device': device_str})
            except Exception as ex:
                return jsonify({'ok': False, 'error': str(ex)}), 500

        @app_flask.route('/api/status')
        def api_status_view():
            return jsonify({
                'running': getattr(window, 'training_thread', None) is not None and
                           (window.training_thread.isRunning() if window.training_thread else False),
                'model_loaded': window.model is not None,
                'device': getattr(window, '_model_device', 'cpu'),
            })

        @app_flask.route('/api/system_stats')
        def api_system_stats():
            stats = window._collect_system_stats()
            stats['eta_hours'] = getattr(window, '_eta_hours', 0)
            stats['step_time'] = getattr(window, '_last_step_time', 0)
            return jsonify(stats)

        @socketio.on('connect')
        def on_connect():
            # 新客户端连接时，推送训练日志历史
            for msg in window._train_log_history:
                emit('train_log', {'msg': msg})
            # 推送当前训练状态
            is_training = (window.training_thread is not None and
                           window.training_thread.isRunning()) if window.training_thread else False
            if is_training:
                emit('train_progress', {
                    'epoch': getattr(window, '_last_epoch', 0),
                    'total_epochs': getattr(window, '_last_total_epochs', 100),
                    'step': 0, 'total_steps': 1,
                    'loss': getattr(window, '_last_loss', 0),
                    'lr': getattr(window, '_last_lr', 0),
                    'ppl': getattr(window, '_last_ppl', 0),
                })

        @socketio.on('start_train')
        def on_start(config):
            # 通过 GUI 的 _start_training 启动训练（复用现有逻辑）
            # 先设置参数到 GUI 控件
            window.cfg_d_model.setValue(config.get('d_model', 256))
            window.cfg_n_layers.setValue(config.get('n_layers', 8))
            window.cfg_n_heads.setValue(config.get('n_heads', 8))
            window.cfg_d_ff.setValue(config.get('d_ff', 1024))
            window.cfg_dropout.setValue(config.get('dropout', 0.1))
            window.cfg_seq_len.setValue(config.get('seq_len', 256))
            window.cfg_epochs.setValue(config.get('epochs', 30))
            window.cfg_batch.setValue(config.get('batch_size', 16))
            window.cfg_lr.setValue(config.get('lr', 0.0003))
            window.cfg_wd.setValue(config.get('weight_decay', 0.01))
            window.cfg_warmup.setValue(config.get('warmup_steps', 500))
            window.cfg_n_experts.setValue(config.get('n_experts', 4))
            window.cfg_n_activated.setValue(config.get('n_activated', 2))
            if not (window.training_thread and window.training_thread.isRunning()):
                window._start_training()
            emit('train_log', {'msg': '训练启动请求已发送'})

        @socketio.on('stop_train')
        def on_stop():
            window._stop_training()
            emit('train_log', {'msg': '停止请求已发送'})

        @socketio.on('send_message')
        def on_chat(data):
            prompt = data.get('text', '').strip()
            if not prompt:
                return
            temp = float(data.get('temperature', 0.8))
            max_tokens = int(data.get('max_tokens', 128))
            top_k = int(data.get('top_k', 80))
            top_p = float(data.get('top_p', 0.9))
            freq_pen = float(data.get('frequency_penalty', 0.15))
            min_new_tokens = int(data.get('min_new_tokens', 5))
            debug = bool(data.get('debug', False))
            rep_penalty = float(data.get('repetition_penalty', 1.05))
            # 通过信号桥接到 GUI 主线程，由 ChatThread 统一处理
            window._web_chat_request.emit(prompt, temp, max_tokens, top_k, top_p, freq_pen, rep_penalty, min_new_tokens, debug)

        @socketio.on('tokenize')
        def on_tokenize(data):
            from tokenizer import ChineseTokenizer as _CT
            text = data.get('text', '').strip()
            if not text:
                return
            tz = window.tokenizer
            if tz is None:
                tz = _CT.load(tok_path)
            ids = tz.encode(text, add_special=True)
            ids_ns = tz.encode(text, add_special=False)
            decoded = tz.decode(ids)
            tokens = []
            for tid in ids:
                tok = tz.id_to_token.get(tid, '<UNK>')
                tokens.append({'id': tid, 'token': tok})
            emit('tokenize_result', {'text': text, 'ids': ids, 'ids_no_special': ids_ns,
                                     'decoded': decoded, 'tokens': tokens,
                                     'vocab_size': tz.vocab_size_actual})

        def _run_server():
            socketio.run(app_flask, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

        self._web_thread = threading.Thread(target=_run_server, daemon=True)
        self._web_stop = False
        self._web_socketio = socketio  # 保存引用供训练进度广播
        self._web_thread.start()

        url = f'http://{ip}:5000'
        self.train_log.append(f'[WebUI] 服务已启动 → {url}')
        self.status_label.setText(f'WebUI: {url}')

    def _stop_web_server(self):
        self._web_stop = True
        # Flask-SocketIO 在 daemon 线程中，无法优雅停止，但勾掉复选框即可
        self.train_log.append('[WebUI] 服务已标记停止（请重启GUI完全关闭）')
        self.status_label.setText('就绪')

    def _collect_system_stats(self) -> dict:
        """收集系统性能指标 (CPU/内存/GPU)"""
        stats = {}
        try:
            import psutil
            stats['cpu_percent'] = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            stats['ram_used_gb'] = round(mem.used / 1e9, 1)
            stats['ram_total_gb'] = round(mem.total / 1e9, 1)
            stats['ram_percent'] = round(mem.percent, 1)
        except ImportError:
            stats['cpu_percent'] = 0
            stats['ram_used_gb'] = 0
            stats['ram_total_gb'] = 0
            stats['ram_percent'] = 0

        if torch.cuda.is_available():
            try:
                stats['gpu_mem_used'] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
                stats['gpu_mem_total'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                stats['gpu_mem_percent'] = round(stats['gpu_mem_used'] / max(stats['gpu_mem_total'], 0.01) * 100, 1)
                stats['gpu_name'] = torch.cuda.get_device_name(0)
                # 尝试 pynvml 获取 GPU 利用率和温度
                try:
                    import pynvml
                    if not hasattr(self, '_nvml_inited'):
                        pynvml.nvmlInit()
                        self._nvml_inited = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    stats['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats['gpu_util'] = util.gpu
                except Exception:
                    stats['gpu_temp'] = 0
                    stats['gpu_util'] = 0
            except Exception:
                pass

        return stats

    # ─── 训练操作 ────────────────────────────────────
    def _browse_checkpoint(self):
        """浏览选择自定义续训 checkpoint"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择续训 Checkpoint", "checkpoints",
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*.*)"
        )
        if path:
            self.cfg_custom_ckpt.setText(path)

    def _browse_teacher(self):
        """浏览选择教师模型 checkpoint"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择教师模型 Checkpoint", "checkpoints",
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*.*)"
        )
        if path:
            self.cfg_teacher_path.setText(path)

    def _browse_data_path(self):
        """浏览选择训练数据"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据", "data",
            "JSONL Files (*.jsonl);;JSON Files (*.json);;All Files (*.*)"
        )
        if path:
            self.cfg_data_path.setText(path)

    def _get_train_config(self) -> dict:
        return {
            'd_model': self.cfg_d_model.value(),
            'n_layers': self.cfg_n_layers.value(),
            'n_heads': self.cfg_n_heads.value(),
            'd_ff': self.cfg_d_ff.value(),
            'dropout': self.cfg_dropout.value(),
            'seq_len': self.cfg_seq_len.value(),
            'vocab_size': self.cfg_vocab.value(),
            'n_experts': self.cfg_n_experts.value(),
            'n_activated': self.cfg_n_activated.value(),
            'epochs': self.cfg_epochs.value(),
            'batch_size': self.cfg_batch.value(),
            'lr': self.cfg_lr.value(),
            'weight_decay': self.cfg_wd.value(),
            'warmup_steps': self.cfg_warmup.value(),
            'save_every': max(1, self.cfg_epochs.value() // 4),
            'save_every_steps': 500,
            'checkpoint_dir': 'checkpoints',
            'tokenizer_path': 'checkpoints/tokenizer.json',
            'data_path': self.cfg_data_path.text().strip() or 'data/train_all.jsonl',
            'stride': 64,
            'grad_accum_steps': 1,
            'distill_alpha': self.cfg_distill_alpha.value() if self.cfg_distill_enable.isChecked() else 0.0,
            'distill_temp': self.cfg_distill_temp.value(),
            'teacher_model_path': self.cfg_teacher_path.text().strip() if self.cfg_distill_enable.isChecked() else '',
            'teacher_device': 'cpu',
            'custom_checkpoint': self.cfg_custom_ckpt.text().strip(),
            'use_gpu': self.cfg_use_gpu.isChecked() and torch.cuda.is_available(),
        }

    def _start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "提示", "训练已在运行中")
            return

        config = self._get_train_config()

        # ── 断点续训检测 ──
        watchdog_path = os.path.join(config['checkpoint_dir'], 'last_step.pt')
        resume_from = None
        if os.path.exists(watchdog_path):
            try:
                import torch
                ckpt = torch.load(watchdog_path, map_location='cpu', weights_only=False)
                ep = ckpt.get('epoch', '?')
                gs = ckpt.get('global_step', '?')
                reply = QMessageBox.question(
                    self, "断点续训",
                    f"检测到上次训练中断:\n"
                    f"  Epoch: {ep}, Step: {gs}\n\n"
                    f"是否从断点继续训练？\n（选「否」将从头开始）",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    resume_from = watchdog_path
                    self.train_log.append(f"[断点续训] 从 {watchdog_path} 恢复 (epoch={ep}, step={gs})")
                else:
                    os.remove(watchdog_path)  # 用户不想续训，删掉看门狗
                    self.train_log.append("[断点续训] 用户选择不续训，从头开始")
            except Exception as e:
                self.train_log.append(f"[断点续训] 读取看门狗失败: {e}，将从头开始")
        # 如果用户指定了自定义 checkpoint，优先使用（覆盖看门狗）
        custom_ckpt = config.get('custom_checkpoint', '')
        if custom_ckpt and os.path.exists(custom_ckpt):
            resume_from = custom_ckpt
            self.train_log.append(f"[自定义续训] 从 {custom_ckpt} 恢复")

        config['resume_from'] = resume_from

        self.train_log.clear()
        self._train_log_history.clear()
        self.loss_history = []
        # 清空 Loss 曲线图
        if hasattr(self, 'loss_chart'):
            self.loss_chart.clear()

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备训练...")
        self.status_label.setText("训练中...")

        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self._on_train_log)
        self.training_thread.progress_signal.connect(self._on_train_progress)
        self.training_thread.batch_progress_signal.connect(self._on_batch_progress)
        self.training_thread.loss_signal.connect(self._on_train_loss)
        self.training_thread.finished_signal.connect(self._on_train_finished)
        self.training_thread.start()

    def _stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.train_log.append("[用户] 正在停止训练...")

    def _on_train_log(self, msg: str):
        self.train_log.append(msg)
        cursor = self.train_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.train_log.setTextCursor(cursor)
        # 缓冲日志供新WebUI连接推送
        self._train_log_history.append(msg)
        # Web 广播
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('train_log', {'msg': msg})
            except: pass

    def _on_train_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self._last_epoch = current
        self._last_total_epochs = total

    def _on_batch_progress(self, current: int, total: int):
        """每步更新进度条 + ETA + 系统状态"""
        now = time.time()

        # ETA 计算（滑动窗口平均步速）
        if not hasattr(self, '_step_times'):
            self._step_times = []
            self._last_progress_time = 0
            self._sys_stats_counter = 0

        if self._last_progress_time > 0:
            dt = now - self._last_progress_time
            if dt > 0 and dt < 60:  # 过滤异常值
                self._step_times.append(dt)
                if len(self._step_times) > 20:
                    self._step_times.pop(0)
        self._last_progress_time = now

        if self._step_times:
            avg_step = sum(self._step_times) / len(self._step_times)
            self._last_step_time = avg_step
            remaining = total - current
            eta_seconds = avg_step * remaining
            self._eta_hours = round(eta_seconds / 3600, 2)
        else:
            self._last_step_time = 0
            self._eta_hours = 0

        # GUI 进度条 + ETA
        eta_str = self._format_eta(self._eta_hours)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"%p% ({current}/{total} 步) ETA: {eta_str}")

        # Web 广播（进度 + 系统状态，节流系统状态每10步一次）
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try:
                ep = getattr(self, '_last_epoch', 0)
                ep_total = getattr(self, '_last_total_epochs', 100) or 100
                self._web_socketio.emit('train_progress', {
                    'epoch': ep, 'total_epochs': ep_total,
                    'step': current, 'total_steps': total,
                    'loss': getattr(self, '_last_loss', 0),
                    'lr': getattr(self, '_last_lr', 0),
                    'ppl': getattr(self, '_last_ppl', 0),
                    'eta_hours': self._eta_hours,
                    'step_time': round(self._last_step_time, 2),
                })
                # 每10步广播一次系统状态（避免过于频繁）
                self._sys_stats_counter += 1
                if self._sys_stats_counter >= 10:
                    self._sys_stats_counter = 0
                    stats = self._collect_system_stats()
                    self._web_socketio.emit('system_stats', stats)
            except: pass

    @staticmethod
    def _format_eta(eta_hours: float) -> str:
        """格式化 ETA 显示"""
        if eta_hours <= 0:
            return "--:--"
        if eta_hours < 1:
            mins = int(eta_hours * 60)
            return f"{mins}分钟"
        h = int(eta_hours)
        m = int((eta_hours - h) * 60)
        return f"{h}时{m:02d}分"

    def _on_train_loss(self, loss: float, ppl: float):
        self._last_loss = loss
        if ppl > 0:
            # epoch级更新：完整记录
            self.loss_history.append((loss, ppl))
            self._last_ppl = ppl
            if hasattr(self, 'loss_chart'):
                self.loss_chart.add_loss(loss, ppl)

    def _on_train_finished(self, success: bool, message: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)

        # 训练正常结束 → 删掉看门狗（不再提示续训）
        watchdog = os.path.join('checkpoints', 'last_step.pt')
        if success and os.path.exists(watchdog):
            os.remove(watchdog)

        if success:
            self.train_log.append(f"\n✓ {message}")
            self.train_log.append("提示: 请切换到「对话」标签页，系统会自动加载模型")
            self._auto_load_best_model()
        else:
            self.train_log.append(f"\n✗ 训练失败: {message}")
        # Web 广播完成
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('train_done', {'best_ppl': getattr(self, '_last_ppl', 0)})
            except: pass

    # ─── 模型加载 ────────────────────────────────────
    def _auto_load_best_model(self):
        ckpt_dir = 'checkpoints'
        best_path = os.path.join(ckpt_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self._load_model(best_path)
        else:
            # 找最新的 checkpoint (仅 epoch 级，排除 step 级)
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
                key=lambda x: int(x.split('_epoch')[-1].replace('.pt', '')),
            )
            if ckpts:
                self._load_model(os.path.join(ckpt_dir, ckpts[-1]))

    # ─── 快捷操作 ──────────────────────────────────
    def _clear_chat_memory(self):
        self.chat_history.clear()
        self.chat_memory.clear()

    def _start_api_server(self):
        """一键启动 API 服务器"""
        import subprocess, threading
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'scripts', 'api_server.py')
        if not os.path.exists(server_script):
            QMessageBox.critical(self, "错误", f"找不到 API 服务器:\n{server_script}")
            return

        self.status_label.setText("API 服务器启动中...")
        self.status_label.setStyleSheet("color: #61afef; padding: 4px;")

        def run_server():
            try:
                proc = subprocess.Popen(
                    [sys.executable, '-u', server_script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                self._api_proc = proc
                self.chat_history.append("<span style='color:#61afef;'>🌐 API 服务器已启动 (端口 6008)</span><br>")
                self.chat_history.append("<span style='color:#888;font-size:9pt;'>地址: http://127.0.0.1:6008</span><br>")
                self.status_label.setText("API 服务器运行中 — http://127.0.0.1:6008")
                self.status_label.setStyleSheet("color: #98c379; padding: 4px;")
            except Exception as e:
                self.chat_history.append(f"<span style='color:#e44;'>❌ 启动失败: {e}</span><br>")
                self.status_label.setText("API 服务器启动失败")
                self.status_label.setStyleSheet("color: #e44; padding: 4px;")

        threading.Thread(target=run_server, daemon=True).start()

    # ─── 设置面板 ──────────────────────────────────
    def _create_settings_panel(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget()
        sl = QVBoxLayout(content)
        sl.setSpacing(14)

        # Card: 生成参数
        gen_card = QGroupBox("🎲 生成参数")
        gen_card.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        gl = QGridLayout(gen_card)
        gl.setSpacing(8)

        params = [
            ("温度", "chat_temp", QDoubleSpinBox, (0.1, 2.0, 0.9, 0.1, 1),
             "越高越随机，越低越确定。建议 0.8~1.2"),
            ("Top-K", "chat_topk", QSpinBox, (1, 200, 80, 1, 0),
             "每步从 K 个最高概率词中采样"),
            ("Top-P", "chat_topp", QDoubleSpinBox, (0.0, 1.0, 0.90, 0.05, 2),
             "核采样累积概率阈值"),
            ("Max Tokens", "chat_max_tokens", QSpinBox, (16, 99999, 128, 16, 0),
             "最大生成 token 数"),
            ("重复惩罚", "rep_penalty_spin", QDoubleSpinBox, (0.1, 3.0, 1.05, 0.05, 2),
             ">1.0 抑制重复"),
            ("频率惩罚", "chat_freq_pen", QDoubleSpinBox, (0.0, 1.0, 0.15, 0.05, 2),
             "惩罚已出现 token"),
            ("最小长度", "chat_min_tokens", QSpinBox, (0, 99999, 5, 1, 0),
             "至少生成 N 个 token 后才允许结束"),
        ]

        for i, (label, attr, cls, args, tip) in enumerate(params):
            gl.addWidget(QLabel(label + ":"), i, 0)
            rng = args
            w = cls()
            w.setRange(rng[0], rng[1])
            w.setValue(rng[2])
            w.setSingleStep(rng[3])
            if hasattr(w, 'setDecimals') and rng[4] > 0:
                w.setDecimals(rng[4])
            w.setToolTip(tip)
            setattr(self, attr, w)
            gl.addWidget(w, i, 1)

        # 上下文扩展 (TGAI GO 引擎专用, NTK-aware RoPE)
        ext_row = len(params)
        gl.addWidget(QLabel("上下文扩展:"), ext_row, 0)
        self.tg_extend_combo = QComboBox()
        self.tg_extend_combo.addItems(["不扩展", "2048", "4096", "8192", "16384"])
        self.tg_extend_combo.setCurrentIndex(0)
        self.tg_extend_combo.setToolTip(
            "TGAI GO NTK-aware RoPE 上下文扩展\n"
            "超过模型原生 max_seq_len 时自动缩放 RoPE 频率\n"
            "更改后重新加载 .TG 模型生效")
        self.tg_extend_combo.currentTextChanged.connect(self._on_extend_change)
        gl.addWidget(self.tg_extend_combo, ext_row, 1)

        sl.addWidget(gen_card)

        # Card: 系统信息
        sys_card = QGroupBox("🖥 系统信息")
        sys_card.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        syl = QVBoxLayout(sys_card)
        self.sys_info_label = QLabel()
        self.sys_info_label.setFont(QFont("Consolas", 9))
        self.sys_info_label.setWordWrap(True)
        syl.addWidget(self.sys_info_label)
        self._update_sys_info()
        sl.addWidget(sys_card)

        sl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return tab

    def _update_sys_info(self):
        import platform, torch, psutil
        info = []
        info.append(f"Python: {platform.python_version()}")
        try:
            import torch; info.append(f"PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                info.append(f"CUDA: {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                info.append(f"VRAM: {mem:.1f} GB")
            else:
                info.append("CUDA: 不可用")
        except: pass
        info.append(f"CPU: {psutil.cpu_count(logical=False)}核/{psutil.cpu_count()}线程")
        info.append(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        try:
            dll = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TGAI GO', 'cpu', 'build', 'libtgai_engine.dll')
            if os.path.exists(dll):
                sz = os.path.getsize(dll) / 1024
                # Try to get version from format header
                info.append(f"TGAI GO: {sz:.1f} KB (v2: Llama/Mistral/Qwen2/GQA)")
        except: pass
        self.sys_info_label.setText("\n".join(info))

    def _load_model_dialog(self):
        """加载 .pt 或 .TG 模型，并注册到模型选择器"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "checkpoints",
            "模型文件 (*.pt *.TG);;PyTorch (*.pt);;TGAI Model (*.TG);;All (*.*)"
        )
        if not path:
            return

        if path.lower().endswith('.tg'):
            self._load_tg_model(path)
        else:
            self._load_model(path)

        # 注册到模型选择器
        name = os.path.basename(path)
        self._register_model(name, path)

    def _register_model(self, name: str, path: str):
        """将模型注册到选择器，支持多模型切换"""
        if not hasattr(self, '_loaded_models'):
            self._loaded_models = {}
        self._loaded_models[name] = {'name': name, 'path': path, 'is_tg': path.lower().endswith('.tg')}

        # 更新选择器
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        for key, info in self._loaded_models.items():
            tag = " [TG]" if info['is_tg'] else " [PT]"
            self.model_selector.addItem(info['name'] + tag, key)
        # 选最新加载的
        self.model_selector.setCurrentIndex(self.model_selector.count() - 1)
        self.model_selector.blockSignals(False)

    def _on_model_switch(self, idx):
        """切换已加载的模型"""
        model_id = self.model_selector.currentData()
        if not model_id or not hasattr(self, '_loaded_models'):
            return
        if model_id in self._loaded_models:
            info = self._loaded_models[model_id]
            if info['is_tg']:
                self._load_tg_model(info['path'])
            else:
                self._load_model(info['path'])
            self.chat_status.setText(f"已切换: {info['name']}")
            self.chat_status.setStyleSheet("color: #98c379; padding: 4px;")

    def _on_extend_change(self, text: str):
        """上下文扩展下拉框变化: 存到实例变量, 下次加载 .TG 时生效"""
        if text == "不扩展":
            self._tg_extend_ctx = 0
        else:
            try:
                self._tg_extend_ctx = int(text)
            except ValueError:
                self._tg_extend_ctx = 0

    def _load_tg_model_dialog(self):
        """加载 TGAI GO .TG 模型文件"""
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_dirs = [
            os.path.join(project_root, 'TGAI GO'),
            r'D:\TGAI\TGAI GO\test_run2',
            r'D:\TGAI\TGAI GO',
        ]
        default_dir = next((d for d in default_dirs if os.path.isdir(d)), '')
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 .TG 模型文件", default_dir, "TGAI Model (*.TG);;All (*.*)"
        )
        if path:
            self._load_tg_model(path)

    def _load_tg_model(self, tg_path: str):
        """用 TGAI GO 引擎加载 .TG 模型"""
        try:
            if self.tg_engine is None:
                self.tg_engine = TGEngine.instance()
            cfg = self.tg_engine.load_model(tg_path)
            self.use_go_engine = True

            # 自动设置最优线程数 (CPU 核心数, 上限 8)
            import os as _os
            cores = min(_os.cpu_count() or 4, 8)
            self.tg_engine.lib.tg_set_nthreads(cores)
            print(f"[TG] 线程数: {cores}")

            # NTK 上下文扩展: 从 ui 状态读取 (默认 0=不扩展)
            ext = getattr(self, '_tg_extend_ctx', 0) or 0
            if ext > 0:
                self.tg_engine.set_extend(ext)
                self.tg_engine.max_seq = ext  # 更新缓存大小
                print(f"[TG] 上下文: {ext}")

            # 释放 PyTorch 模型 (节省内存)，但保留分词器
            if self.model is not None:
                del self.model
                self.model = None
                import gc; gc.collect()
            # Tokenizer: TGAI原生模型用 checkpoints/tokenizer.json,
            # 外部模型(如 Qwen2)用 .TG 文件内置 tokenizer (来自 GGUF)
            is_external = (cfg.get('arch_type', 0) != 0 or cfg.get('vocab_size', 32768) != 32768)
            if is_external:
                self.tokenizer = None  # 用引擎内置 tokenizer
            elif self.tokenizer is None:
                from tokenizer import ChineseTokenizer
                self.tokenizer = ChineseTokenizer.load('checkpoints/tokenizer.json')
            self.chat_status.setText(
                f"状态: [GO 引擎] {os.path.basename(tg_path)} | "
                f"arch={cfg.get('arch_name','?')}{cfg.get('gqa_info','')} "
                f"L={cfg['n_layers']} d={cfg['d_model']} "
                f"vocab={cfg['vocab_size']} max_seq={self.tg_engine.max_seq} "
                f"线程={cores}"
            )
            self.chat_status.setStyleSheet("color: #4f4; padding: 4px;")
            arch_display = (f"架构: {cfg.get('arch_name','?')}"
                            f"{cfg.get('gqa_info','')} | "
                            if 'arch_name' in cfg else '')
            self.chat_history.append(
                f"<i style='color:#4f4'>[TGAI GO 引擎已加载] {os.path.basename(tg_path)}</i><br>"
                f"<i style='color:#888'>{arch_display}"
                f"layers={cfg['n_layers']} d_model={cfg['d_model']} "
                f"vocab={cfg['vocab_size']} max_seq={cfg['max_seq']} 线程={cores}</i>"
            )
            self._register_model(os.path.basename(tg_path), tg_path)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载 .TG 模型失败:\n{e}")
            self.use_go_engine = False

    def _convert_pt_to_tg_dialog(self):
        """将 PyTorch checkpoint 转换为 .TG 格式 (TGAI GO 引擎)，含量化选择"""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QComboBox, QPushButton, QFileDialog, QMessageBox,
                                     QGroupBox)
        import subprocess

        # ─── 弹出对话框 ────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("pt → TG 转换")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # ── 量化精度选择 ──
        grp = QGroupBox("量化精度")
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(6)

        self._convert_dtype_combo = QComboBox()
        dtype_options = [
            ("fp16", "FP16 半精度（推荐桌面/手机）"),
            ("q8_0", "Q8_0 分组 8-bit"),
            ("q4_0", "Q4_0 分组 4-bit（推荐手机端）"),
            ("int8", "INT8 全局（旧方案）"),
            ("fp32", "FP32 全精度"),
        ]
        for val, label in dtype_options:
            self._convert_dtype_combo.addItem(label, val)
        self._convert_dtype_combo.setCurrentIndex(0)
        grp_layout.addWidget(self._convert_dtype_combo)

        self._convert_dtype_desc = QLabel()
        self._convert_dtype_desc.setStyleSheet("color: #999; font-size: 10pt;")
        self._convert_dtype_desc.setWordWrap(True)
        grp_layout.addWidget(self._convert_dtype_desc)

        # 灰字说明随选择切换
        descs = {
            "fp16": "体积约为 FP32 的 50%。精度几乎无损失，手机/桌面均推荐。3B 模型约 6GB。",
            "q8_0": "每 32 个权重一组，独立 scale。精度接近 FP16，体积约 FP32 的 37%。3B 模型约 4.5GB。",
            "q4_0": "极致压缩。每 32 个权重一组 4-bit，体积约 FP32 的 18%。精度轻微损失，低端手机首选。3B 模型约 2.1GB。",
            "int8": "旧方案：一个全局 scale 管全部权重。精度不如 Q8_0，建议用 Q8_0 替代。3B 模型约 3GB。",
            "fp32": "体积最大，无任何精度损失。适合桌面端。3B 模型约 12GB。",
        }
        def on_dtype_changed(idx):
            val = self._convert_dtype_combo.currentData()
            self._convert_dtype_desc.setText(descs.get(val, ""))
        self._convert_dtype_combo.currentIndexChanged.connect(on_dtype_changed)
        on_dtype_changed(0)  # 初始显示
        layout.addWidget(grp)

        # ── 文件选择 ──
        file_grp = QGroupBox("文件路径")
        file_layout = QVBoxLayout(file_grp)
        file_layout.setSpacing(8)

        # 输入 .pt
        pt_row = QHBoxLayout()
        pt_row.addWidget(QLabel("源文件:"))
        self._convert_pt_label = QLabel("（未选择）")
        self._convert_pt_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._convert_pt_label.setWordWrap(True)
        pt_row.addWidget(self._convert_pt_label, 1)
        btn_pt = QPushButton("选择 .pt")
        btn_pt.clicked.connect(lambda: self._pick_convert_pt())
        pt_row.addWidget(btn_pt)
        file_layout.addLayout(pt_row)

        # 输出 .TG
        tg_row = QHBoxLayout()
        tg_row.addWidget(QLabel("输出:"))
        self._convert_tg_label = QLabel("（未选择）")
        self._convert_tg_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._convert_tg_label.setWordWrap(True)
        tg_row.addWidget(self._convert_tg_label, 1)
        btn_tg = QPushButton("保存为 .TG")
        btn_tg.clicked.connect(lambda: self._pick_convert_tg())
        tg_row.addWidget(btn_tg)
        file_layout.addLayout(tg_row)
        layout.addWidget(file_grp)

        # ── 按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)
        btn_ok = QPushButton("开始转换")
        btn_ok.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_ok.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        # ── 文件选择回调 ──
        self._convert_pt_path = ""
        self._convert_tg_path = ""

        def pick_pt():
            p, _ = QFileDialog.getOpenFileName(dlg, "选择 PyTorch checkpoint", "",
                                                "PyTorch Checkpoint (*.pt);;All (*.*)")
            if p:
                self._convert_pt_path = p
                self._convert_pt_label.setText(os.path.basename(p))
                self._convert_pt_label.setStyleSheet("color: #ccc; font-size: 9pt;")
                if not self._convert_tg_path:
                    default_tg = os.path.splitext(p)[0] + '.TG'
                    self._convert_tg_path = default_tg
                    self._convert_tg_label.setText(os.path.basename(default_tg))
                    self._convert_tg_label.setStyleSheet("color: #ccc; font-size: 9pt;")

        def pick_tg():
            p, _ = QFileDialog.getSaveFileName(dlg, "保存 .TG 文件", self._convert_tg_path or "",
                                                "TGAI Model (*.TG)")
            if p:
                self._convert_tg_path = p
                self._convert_tg_label.setText(os.path.basename(p))
                self._convert_tg_label.setStyleSheet("color: #ccc; font-size: 9pt;")

        self._pick_convert_pt = pick_pt
        self._pick_convert_tg = pick_tg

        # ── 验证 ──
        project_root = os.path.dirname(os.path.abspath(__file__))
        converter = os.path.join(project_root, 'TGAI GO', 'tools', 'tgai_convert.py')
        tokenizer = os.path.join(project_root, 'checkpoints', 'tokenizer.json')

        if not os.path.exists(converter):
            QMessageBox.critical(self, "错误", f"找不到 tgai_convert.py:\n{converter}")
            return
        if not os.path.exists(tokenizer):
            QMessageBox.critical(self, "错误", f"找不到 tokenizer.json:\n{tokenizer}")
            return

        # ── 运行 ──
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._convert_pt_path or not self._convert_tg_path:
            QMessageBox.warning(self, "提示", "请选择输入和输出文件。")
            return

        dtype_val = self._convert_dtype_combo.currentData()
        self.chat_status.setText(f"状态: 正在转换 .pt → .TG ({dtype_val.upper()}) ...")
        self.chat_status.setStyleSheet("color: #fa4; padding: 4px;")
        self.btn_pt_to_tg.setEnabled(False)

        class ConvertThread(QThread):
            log_signal = pyqtSignal(str)
            done_signal = pyqtSignal(bool, str)

            def __init__(self, converter, pt, tg, tok, dtype):
                super().__init__()
                self._converter = converter
                self._pt = pt
                self._tg = tg
                self._tok = tok
                self._dtype = dtype

            def run(self):
                try:
                    proc = subprocess.Popen(
                        [sys.executable, '-u', self._converter,
                         '--checkpoint', self._pt, '--output', self._tg,
                         '--tokenizer', self._tok, '--dtype', self._dtype],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True
                    )
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line:
                            self.log_signal.emit(line)
                    proc.wait()
                    if proc.returncode == 0 and os.path.exists(self._tg):
                        size_mb = os.path.getsize(self._tg) / 1024 / 1024
                        # 自动复制 tokenizer.json 到 .TG 文件同目录（手机端导入时需要）
                        tg_dir = os.path.dirname(self._tg)
                        dst_tok = os.path.join(tg_dir, 'tokenizer.json')
                        if not os.path.exists(dst_tok) and os.path.exists(self._tok):
                            try:
                                shutil.copy2(self._tok, dst_tok)
                            except Exception:
                                pass
                        self.done_signal.emit(True, f"完成! {size_mb:.1f} MB ({self._dtype.upper()})")
                    else:
                        self.done_signal.emit(False, f"转换失败 (exit={proc.returncode})")
                except Exception as e:
                    self.done_signal.emit(False, str(e))

        self._convert_thread = ConvertThread(converter, self._convert_pt_path,
                                             self._convert_tg_path, tokenizer, dtype_val)
        self._convert_thread.log_signal.connect(
            lambda s: self.chat_history.append(f"<span style='color:#888;font-size:10pt;'>{s}</span>")
        )
        self._convert_thread.done_signal.connect(self._on_convert_done)
        self._convert_thread.start()

    def _on_convert_done(self, success: bool, msg: str):
        if success:
            self.chat_history.append(f"<b style='color:#4f4;'>✅ {msg}</b><br>")
            self.chat_status.setText(f"状态: {msg}")
            self.chat_status.setStyleSheet("color: #4a9; padding: 4px;")
        else:
            self.chat_history.append(f"<b style='color:#e44;'>❌ {msg}</b><br>")
            self.chat_status.setText(f"状态: 转换失败")
            self.chat_status.setStyleSheet("color: #e44; padding: 4px;")
        self.btn_pt_to_tg.setEnabled(True)

    def _build_apk(self):
        """一键编译 TGAI CHAT APK — 弹出配置对话框支持自定义图标/签名/版本号"""
        import subprocess

        project_root = os.path.dirname(os.path.abspath(__file__))
        proj = os.path.join(project_root, 'tg_chat')
        flutter = os.path.join('D:', os.sep, 'flutter', 'flutter', 'bin', 'flutter.bat')
        java_home = os.path.join('D:', os.sep, 'jdk17', 'jdk-17.0.16+8')
        android_home = os.path.join('D:', os.sep, 'Android', 'Sdk')
        apk_out = os.path.join(proj, 'build', 'app', 'outputs', 'flutter-apk', 'app-release.apk')

        # ── 环境检查 ──
        if not os.path.exists(flutter):
            QMessageBox.critical(self, "缺少 Flutter",
                f"找不到 Flutter:\n{flutter}\n\n请将 Flutter 解压到 D:\\flutter")
            return
        if not os.path.exists(java_home):
            QMessageBox.critical(self, "缺少 JDK",
                f"找不到 JDK 17:\n{java_home}")
            return
        if not os.path.exists(android_home):
            QMessageBox.critical(self, "缺少 Android SDK",
                f"找不到 Android SDK:\n{android_home}")
            return
        if not os.path.exists(proj):
            QMessageBox.critical(self, "缺少项目", f"找不到 tg_chat 项目:\n{proj}")
            return

        # ── 读取当前配置作为默认值 ──
        # 图标：尝试多个已知路径
        current_icon = os.path.join(project_root, 'tg_chat', 'assets', 'icon', 'app_icon.png')
        if not os.path.exists(current_icon):
            alt_icon = r"D:\TG HELPER V0.1.5\TG HELPER\icon\TGAI.png"
            current_icon = alt_icon if os.path.exists(alt_icon) else ''

        # 签名配置
        key_props_path = os.path.join(proj, 'android', 'key.properties')
        ks_path_default = os.path.join(proj, 'android', 'app', 'tlstudio.keystore')
        existing_alias = 'tgchat'
        existing_store_pw = 'tlstudio2026'
        existing_key_pw = 'tlstudio2026'
        existing_ks = ''
        if os.path.exists(key_props_path):
            try:
                with open(key_props_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('keyAlias='):
                            existing_alias = line.split('=', 1)[1]
                        elif line.startswith('keyPassword='):
                            existing_key_pw = line.split('=', 1)[1]
                        elif line.startswith('storePassword='):
                            existing_store_pw = line.split('=', 1)[1]
                        elif line.startswith('storeFile='):
                            sf = line.split('=', 1)[1]
                            if not os.path.isabs(sf):
                                sf = os.path.join(proj, 'android', sf)
                            existing_ks = sf
            except Exception:
                pass
        if not existing_ks and os.path.exists(ks_path_default):
            existing_ks = ks_path_default

        # 版本号
        pubspec_path = os.path.join(proj, 'pubspec.yaml')
        current_version = '0.0.1'
        current_version_code = '1'
        try:
            with open(pubspec_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s.startswith('version:'):
                        v = s.split(':', 1)[1].strip()
                        if '+' in v:
                            current_version, current_version_code = v.split('+', 1)
                        else:
                            current_version = v
                        break
        except Exception:
            pass

        # ── 构建配置对话框 ──
        dlg = QDialog(self)
        dlg.setWindowTitle("编译 APK 配置")
        dlg.setMinimumWidth(560)
        layout = QVBoxLayout(dlg)

        # > 版本信息
        grp_ver = QGroupBox("版本信息")
        fl_ver = QFormLayout(grp_ver)
        txt_version_name = QLineEdit(current_version)
        txt_version_name.setPlaceholderText("如 0.0.1")
        fl_ver.addRow("版本号:", txt_version_name)
        txt_version_code = QLineEdit(current_version_code)
        txt_version_code.setPlaceholderText("如 1")
        fl_ver.addRow("版本代码:", txt_version_code)
        layout.addWidget(grp_ver)

        # > 应用图标
        grp_icon = QGroupBox("应用图标 (留空不修改)")
        fl_icon = QFormLayout(grp_icon)
        icon_row = QHBoxLayout()
        txt_icon = QLineEdit(current_icon)
        txt_icon.setPlaceholderText("选择 PNG 图标文件")
        btn_browse_icon = QPushButton("浏览...")
        btn_browse_icon.clicked.connect(lambda: self._browse_file_dlg(txt_icon, "PNG图片 (*.png)", dlg))
        icon_row.addWidget(txt_icon)
        icon_row.addWidget(btn_browse_icon)
        fl_icon.addRow("图标路径:", icon_row)
        layout.addWidget(grp_icon)

        # > 签名配置
        grp_sign = QGroupBox("签名配置 (使用现有配置可留空)")
        fl_sign = QFormLayout(grp_sign)
        ks_row = QHBoxLayout()
        txt_ks = QLineEdit(existing_ks)
        txt_ks.setPlaceholderText("选择 .keystore 或 .jks 文件")
        btn_browse_ks = QPushButton("浏览...")
        btn_browse_ks.clicked.connect(lambda: self._browse_file_dlg(txt_ks, "Keystore文件 (*.keystore *.jks);;所有文件 (*)", dlg))
        ks_row.addWidget(txt_ks)
        ks_row.addWidget(btn_browse_ks)
        fl_sign.addRow("签名文件:", ks_row)
        txt_alias = QLineEdit(existing_alias)
        fl_sign.addRow("别名:", txt_alias)
        txt_store_pw = QLineEdit(existing_store_pw)
        txt_store_pw.setEchoMode(QLineEdit.EchoMode.Password)
        fl_sign.addRow("Store密码:", txt_store_pw)
        txt_key_pw = QLineEdit(existing_key_pw)
        txt_key_pw.setEchoMode(QLineEdit.EchoMode.Password)
        fl_sign.addRow("Key密码:", txt_key_pw)
        layout.addWidget(grp_sign)

        # > 按钮
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ── 收集配置 ──
        config = {
            'icon_path': txt_icon.text().strip(),
            'ks_path': txt_ks.text().strip(),
            'ks_alias': txt_alias.text().strip(),
            'ks_store_pw': txt_store_pw.text().strip(),
            'ks_key_pw': txt_key_pw.text().strip(),
            'version_name': txt_version_name.text().strip(),
            'version_code': txt_version_code.text().strip(),
        }

        # ── 应用配置 ──
        try:
            # 1. 版本号 → pubspec.yaml
            if config['version_name']:
                with open(pubspec_path, 'r', encoding='utf-8') as f:
                    pub_lines = f.readlines()
                with open(pubspec_path, 'w', encoding='utf-8') as f:
                    for line in pub_lines:
                        s = line.strip()
                        if s.startswith('version:'):
                            new_v = f"{config['version_name']}+{config['version_code']}"
                            f.write(f"version: {new_v}\n")
                        else:
                            f.write(line)
                self.chat_history.append(f"<span style='color:#888;font-size:9pt;'>版本: {config['version_name']}+{config['version_code']}</span>")

            # 2. 图标 → mipmap 各密度目录
            if config['icon_path'] and os.path.exists(config['icon_path']):
                android_res = os.path.join(proj, 'android', 'app', 'src', 'main', 'res')
                for d in ('mipmap-mdpi', 'mipmap-hdpi', 'mipmap-xhdpi', 'mipmap-xxhdpi', 'mipmap-xxxhdpi'):
                    d_dir = os.path.join(android_res, d)
                    os.makedirs(d_dir, exist_ok=True)
                    shutil.copy2(config['icon_path'], os.path.join(d_dir, 'ic_launcher.png'))
                self.chat_history.append("<span style='color:#888;font-size:9pt;'>图标已更新</span>")

            # 3. 签名 → key.properties
            if config['ks_path'] and os.path.exists(config['ks_path']):
                ks_abs = os.path.abspath(config['ks_path'])
                ks_app_dir = os.path.join(proj, 'android', 'app')
                # 如果 keystore 在 android/app/ 下，用相对路径；否则用绝对路径
                try:
                    if os.path.commonpath([ks_abs, ks_app_dir]) == ks_app_dir.replace('/', os.sep):
                        ks_ref = os.path.basename(ks_abs)
                    else:
                        ks_ref = ks_abs.replace('\\', '\\\\')
                except ValueError:
                    ks_ref = ks_abs.replace('\\', '\\\\')
                with open(key_props_path, 'w', encoding='utf-8') as f:
                    f.write(f"storeFile={ks_ref}\n")
                    f.write(f"keyAlias={config['ks_alias']}\n")
                    f.write(f"keyPassword={config['ks_key_pw']}\n")
                    f.write(f"storePassword={config['ks_store_pw']}\n")
                self.chat_history.append("<span style='color:#888;font-size:9pt;'>签名配置已更新</span>")
            else:
                # fallback: 自动修复 key.properties
                ks_file = os.path.join(proj, 'android', 'app', 'tlstudio.keystore')
                if os.path.exists(key_props_path):
                    with open(key_props_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if 'storeFile=app/tlstudio.keystore' in content:
                        content = content.replace('storeFile=app/tlstudio.keystore', 'storeFile=tlstudio.keystore')
                        with open(key_props_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.chat_history.append("<span style='color:#888;font-size:9pt;'>已修复 key.properties</span>")
                elif os.path.exists(ks_file):
                    with open(key_props_path, 'w', encoding='utf-8') as f:
                        f.write("storeFile=tlstudio.keystore\nkeyAlias=tgchat\nkeyPassword=tlstudio2026\nstorePassword=tlstudio2026\n")
                    self.chat_history.append("<span style='color:#888;font-size:9pt;'>已生成 key.properties</span>")
        except Exception as e:
            QMessageBox.critical(self, "配置错误", f"应用配置失败:\n{e}")
            return

        # ── 开始编译 ──
        self.chat_status.setText("状态: 正在编译 APK (首次约需 3-5 分钟)...")
        self.chat_status.setStyleSheet("color: #fa4; padding: 4px;")
        self.chat_history.append("<br><b style='color:#61afef;'>📦 开始编译 APK ...</b><br>")

        class BuildAPKThread(QThread):
            log_signal = pyqtSignal(str)
            done_signal = pyqtSignal(bool, str)

            def run(self):
                env = os.environ.copy()
                env['JAVA_HOME'] = java_home
                env['ANDROID_HOME'] = android_home
                env['ANDROID_SDK_ROOT'] = android_home
                env['PATH'] = (os.path.join(java_home, 'bin') + os.pathsep +
                               os.path.dirname(flutter) + os.pathsep +
                               os.path.join(android_home, 'platform-tools') + os.pathsep +
                               env.get('PATH', ''))

                try:
                    # ── 预清理：防止工具进程锁冲突 ──
                    self.log_signal.emit("清理构建缓存...")

                    # 1. 停止 Gradle 守护进程 (残留的 Kotlin daemon 由 Gradle 托管)
                    gradlew = os.path.join(proj, 'android', 'gradlew.bat')
                    if os.path.exists(gradlew):
                        subprocess.run([gradlew, '--stop'],
                                       cwd=os.path.join(proj, 'android'), env=env,
                                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    time.sleep(0.5)

                    # 2. 杀死残留的 Kotlin 守护进程 + 清空其缓存目录
                    kotlin_daemon_dir = os.path.join(os.path.expanduser('~'),
                                                     'AppData', 'Local', 'kotlin', 'daemon')
                    subprocess.run(['taskkill', '/f', '/im', 'kotlin-daemon.exe'],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    time.sleep(0.3)
                    if os.path.isdir(kotlin_daemon_dir):
                        shutil.rmtree(kotlin_daemon_dir, ignore_errors=True)

                    # 3. 删除 Flutter SDK 锁文件
                    flutter_lock = os.path.join(os.path.dirname(flutter), 'cache', 'lockfile')
                    if os.path.exists(flutter_lock):
                        try:
                            os.remove(flutter_lock)
                        except Exception:
                            pass

                    # 4. 删除项目 Kotlin 编译错误缓存
                    kotlin_error_dir = os.path.join(proj, 'android', '.gradle', 'kotlin')
                    if os.path.isdir(kotlin_error_dir):
                        shutil.rmtree(kotlin_error_dir, ignore_errors=True)

                    # 5. flutter clean（清理项目构建缓存）
                    clean_proc = subprocess.run(
                        [flutter, 'clean'],
                        cwd=proj, env=env,
                        capture_output=True, text=True, encoding='utf-8', errors='replace'
                    )
                    for line in clean_proc.stdout.splitlines():
                        line = line.rstrip()
                        if line:
                            self.log_signal.emit(line)

                    self.log_signal.emit("开始编译...")

                    # ── 正式编译（最多重试一次）──
                    for attempt in range(2):
                        if attempt > 0:
                            self.log_signal.emit("首次失败，清理后重试...")
                            time.sleep(2)
                            # 重试前再次清理 Kotlin daemon
                            subprocess.run(['taskkill', '/f', '/im', 'kotlin-daemon.exe'],
                                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                            if os.path.isdir(kotlin_daemon_dir):
                                shutil.rmtree(kotlin_daemon_dir, ignore_errors=True)

                        proc = subprocess.Popen(
                            [flutter, 'build', 'apk', '--release'],
                            cwd=proj, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace'
                        )
                        for line in proc.stdout:
                            line = line.rstrip()
                            if line:
                                self.log_signal.emit(line)
                        proc.wait()
                        if proc.returncode == 0 and os.path.exists(apk_out):
                            size_mb = os.path.getsize(apk_out) / 1024 / 1024
                            self.done_signal.emit(True, f"编译成功! {size_mb:.1f} MB\n{apk_out}")
                            return
                        # 如果非 Kotlin daemon 错误，直接失败不重试
                        if proc.returncode != 0 and 'Kotlin compile daemon' not in str(proc.stdout):
                            break

                    self.done_signal.emit(False, f"编译失败 (exit={proc.returncode})")
                except Exception as e:
                    self.done_signal.emit(False, str(e))

        self._build_thread = BuildAPKThread()
        self._build_thread.log_signal.connect(
            lambda s: self.chat_history.append(f"<span style='color:#888;font-size:9pt;'>{s}</span>")
        )
        self._build_thread.done_signal.connect(self._on_build_apk_done)
        self._build_thread.start()

    def _browse_file_dlg(self, line_edit, filter_str, parent):
        """打开文件选择对话框并填入路径"""
        path, _ = QFileDialog.getOpenFileName(parent, "选择文件", "", filter_str)
        if path:
            line_edit.setText(path)

    def _on_build_apk_done(self, success: bool, msg: str):
        if success:
            self.chat_history.append(f"<b style='color:#98c379;'>✅ {msg}</b><br>")
            self.chat_status.setText("状态: APK 编译完成")
            self.chat_status.setStyleSheet("color: #98c379; padding: 4px;")
            # 自动打开输出目录
            apk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'tg_chat', 'build', 'app', 'outputs', 'flutter-apk')
            if os.path.isdir(apk_dir):
                os.startfile(apk_dir)
        else:
            self.chat_history.append(f"<b style='color:#e44;'>❌ {msg}</b><br>")
            self.chat_status.setText("状态: APK 编译失败")
            self.chat_status.setStyleSheet("color: #e44; padding: 4px;")

    def _load_model(self, path: str):
        try:
            import torch
            from tokenizer import ChineseTokenizer
            from model import create_model

            tok_path = 'checkpoints/tokenizer.json'
            if not os.path.exists(tok_path):
                QMessageBox.warning(self, "错误", f"分词器未找到: {tok_path}")
                return

            self.tokenizer = ChineseTokenizer.load(tok_path)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            mc = ckpt.get('model_config', {})
            state_dict = ckpt['model_state_dict']

            # 从 checkpoint 权重中获取真实的 vocab_size（最可靠）
            embed_weight = state_dict['token_embedding.weight']
            ckpt_vocab_size = embed_weight.shape[0]
            ckpt_d_model = embed_weight.shape[1]

            tokenizer_vocab = self.tokenizer.vocab_size_actual
            print(f"[加载] checkpoint词表: {ckpt_vocab_size}, 分词器词表: {tokenizer_vocab}")

            # 使用 checkpoint 的词表大小创建模型
            self.model = create_model(
                vocab_size=ckpt_vocab_size,
                d_model=mc.get('d_model', ckpt_d_model),
                n_layers=mc.get('n_layers', 8),
                n_heads=mc.get('n_heads', 8),
                d_ff=mc.get('d_ff', 1024),
                max_seq_len=mc.get('max_seq_len', 256),
                dropout=0.0,  # 推理时关闭 dropout
                n_experts=mc.get('n_experts', 4),
                n_activated=mc.get('n_activated', 2),
                self_extend=self.cfg_self_extend.isChecked() if hasattr(self, 'cfg_self_extend') else False,
                extend_max_seq_len=self.cfg_extend_len.value() if hasattr(self, 'cfg_extend_len') else 4096,
            )
            _migrate_rope_buffers(state_dict)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # 切换到 PyTorch 路径
            self.use_go_engine = False

            # GPU 加速
            use_gpu = self.cfg_use_gpu.isChecked() and self.cfg_use_gpu.isEnabled()
            self._model_device = 'cuda' if use_gpu else 'cpu'
            self.model.to(self._model_device)

            # 如果分词器词表比模型大，发出警告（那些 token 会被映射为 UNK）
            if tokenizer_vocab > ckpt_vocab_size:
                print(f"[加载] 警告: 分词器词表({tokenizer_vocab}) > 模型词表({ckpt_vocab_size})，超出部分将映射为 UNK")

            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            epoch = ckpt.get('epoch', '?')
            self.chat_status.setText(
                f"状态: ✓ 模型已加载 | Epoch: {epoch} | 参数: {n_params:,} | "
                f"词表: {ckpt_vocab_size} | 设备: {self._model_device} | {os.path.basename(path)}"
            )
            self.chat_status.setStyleSheet("color: #4a9; padding: 4px;")
            self.status_label.setText(f"模型已加载: {path}")

            QMessageBox.information(self, "成功",
                f"模型加载成功!\nEpoch: {epoch}\n参数: {n_params:,}\n"
                f"模型词表: {ckpt_vocab_size}\n分词器词表: {tokenizer_vocab}")

            self._register_model(os.path.basename(path), path)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            traceback.print_exc()

    # ─── 对话操作 ────────────────────────────────────
    def _stop_chat(self):
        """停止当前生成"""
        # 云端模式
        if self.use_cloud_api and hasattr(self, 'cloud_worker') and self.cloud_worker:
            self._cloud_abort = True
            if self.cloud_worker.isRunning():
                self.cloud_worker.quit()
                self.cloud_worker.wait(2000)
            self.cloud_worker = None
        elif self.chat_thread and self.chat_thread.isRunning():
            if hasattr(self.chat_thread, 'stop'):
                self.chat_thread.stop()
            self.chat_thread.wait(2000)  # 等待最多2秒
        self.btn_stop_chat.setEnabled(False)
        self.chat_input.setEnabled(True)
        self.btn_send.setEnabled(True)

    def _send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return

        # 云端模式 — 不需要本地模型
        if self.use_cloud_api:
            self._send_chat_cloud(text)
            return

        # 检查模型是否加载 (PyTorch 或 GO 引擎)
        if self.use_go_engine and self.tg_engine and self.tg_engine.model:
            pass  # GO 引擎已就绪
        elif self.model is not None and self.tokenizer is not None:
            pass  # PyTorch 模型已就绪
        else:
            QMessageBox.warning(self, "提示",
                "请先加载模型:\n"
                "  • 点击「加载 .TG 模型」使用 GO 引擎 (推荐, 更快)\n"
                "  • 或在「训练」标签页加载 .pt 模型"
            )
            return

        self.chat_input.clear()
        self.chat_input.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_stop_chat.setEnabled(True)

        # 用户消息
        self.chat_history.append(f"<b style='color:#4af'>你:</b> {text}")
        self.chat_history.append(f"<b style='color:#ff4'>TGAI:</b> ")
        self._stream_got_tokens = False

        # 构建带记忆的 prompt（与服务器默认格式一致）
        context = "系统:你是TGAI，一个由JXW独立开发的AI助手。\n"
        for q, a in self.chat_memory[-4:]:  # 最近 4 轮
            context += f"用户:{q}\nTGAI?{a}\n"
        full_prompt = f"{context}用户:{text}\nTGAI?"
        self._pending_prompt = text  # 保存当前问题，用于记忆存储

        # 广播用户消息到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'user', 'text': text})
            except: pass

        temp = self.chat_temp.value()
        rep_penalty = self.rep_penalty_spin.value()
        max_tokens = self.chat_max_tokens.value()
        top_k = self.chat_topk.value()
        top_p = self.chat_topp.value()
        freq_penalty = self.chat_freq_pen.value()
        min_tokens = self.chat_min_tokens.value()

        if self.use_go_engine and self.tg_engine and self.tg_engine.model:
            # ── GO 引擎路径 ──
            self.chat_thread = ChatThreadGO(
                self.tg_engine, full_prompt, temp, max_tokens,
                repetition_penalty=rep_penalty,
                top_k=top_k, top_p=top_p,
                freq_penalty=freq_penalty, min_tokens=min_tokens,
                tokenizer=self.tokenizer,
            )
            self.chat_thread.chunk_signal.connect(self._on_chat_chunk)
            self.chat_thread.response_signal.connect(self._on_chat_response)
            self.chat_thread.error_signal.connect(self._on_chat_error)
            self.chat_thread.finished_signal.connect(self._on_chat_finished)
            self.chat_thread.start()
        else:
            # ── PyTorch 路径 (原有逻辑) ──
            device = getattr(self, '_model_device', 'cpu')
            self.chat_thread = ChatThread(
                self.model, self.tokenizer, full_prompt, temp, max_tokens, device,
                debug=self.debug_check.isChecked(),
                repetition_penalty=rep_penalty,
                top_k=top_k, top_p=top_p, frequency_penalty=freq_penalty,
                min_new_tokens=min_tokens,
            )
            self.chat_thread.chunk_signal.connect(self._on_chat_chunk)
            self.chat_thread.response_signal.connect(self._on_chat_response)
            self.chat_thread.error_signal.connect(self._on_chat_error)
            self.chat_thread.finished_signal.connect(self._on_chat_finished)
            self.chat_thread.debug_signal.connect(self._on_chat_debug)
            self.chat_thread.start()

    def _on_chat_chunk(self, chunk: str):
        """流式: 逐 token 追加到聊天历史"""
        self._stream_got_tokens = True
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        # 自动滚动到底部
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # 广播到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'chunk', 'token': chunk})
            except: pass

    def _on_chat_response(self, response: str):
        """流式完成后的回调"""
        # 保存到对话记忆
        if hasattr(self, '_pending_prompt') and response and response != "[模型未生成回复]":
            self.chat_memory.append((self._pending_prompt, response))
            if len(self.chat_memory) > 20:
                self.chat_memory = self.chat_memory[-10:]
            self._pending_prompt = None

    def _on_chat_finished(self):
        """生成完成，恢复输入"""
        if not getattr(self, '_stream_got_tokens', True):
            # 模型没生成任何 token (EOS 在第一token)，追加提示
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("[模型未生成有效回复]")
        self.chat_history.append("")  # 空行分隔
        self.chat_input.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_stop_chat.setEnabled(False)
        self.chat_input.setFocus()
        # 广播完成到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'done'})
            except: pass

    def _on_chat_error(self, error: str):
        self.chat_history.append(f"<span style='color:red'>[错误] {error}</span>")
        self.chat_input.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_stop_chat.setEnabled(False)
        # 广播错误到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'error', 'error': error})
            except: pass

    def _toggle_debug(self, state: int):
        """切换调试模式 — 勾选后在对话中直接显示推理过程"""
        pass  # debug 开关本身不需要动作，_send_chat 会读取状态

    def _on_chat_debug(self, dbg_list: list):
        """全部生成完成后，一次性显示推理过程"""
        total_ms = sum(d['step_ms'] for d in dbg_list)
        total_tokens = len(dbg_list)

        # 汇总行
        temp_info = dbg_list[0].get('temp', '?') if dbg_list else '?'
        freq_info = dbg_list[0].get('freq_penalty', 0) if dbg_list else 0
        rep_info = dbg_list[0].get('repetition_penalty', 1.0) if dbg_list else 1.0
        lines = [
            f"<span style='color:#666;font-size:8pt;'>"
            f"━━ 推理过程: {total_tokens} tokens, {total_ms:.0f}ms "
            f"({total_ms/total_tokens:.0f}ms/token) "
            f"temp={temp_info} freq={freq_info:.2f} rep={rep_info:.2f} ━━"
            f"</span>"
        ]

        for d in dbg_list:
            step = d['step']
            text = d['text']
            prob = d['prob']
            ms = d['step_ms']

            candidates = d['top_sampled']
            cand_strs = []
            for c in candidates:
                cand_strs.append(f"{c[1]}({c[2]*100:.0f}%)")
            top3_text = "  ".join(cand_strs)

            lines.append(
                f"<span style='color:#888;font-size:8pt;'>"
                f"  #{step} 选「{text}」({prob*100:.0f}%) "
                f"候选: {top3_text}  {ms:.0f}ms"
                f"</span>"
            )

        self.chat_history.append("<br>".join(lines))

        # 广播调试信息到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            plain_lines = []
            t_info = dbg_list[0].get('temp', '?') if dbg_list else '?'
            f_info = dbg_list[0].get('freq_penalty', 0) if dbg_list else 0
            r_info = dbg_list[0].get('repetition_penalty', 1.0) if dbg_list else 1.0
            plain_lines.append(f"推理: {len(dbg_list)} tokens | temp={t_info} freq={f_info:.2f} rep={r_info:.2f}")
            for d in dbg_list:
                cs = d['top_sampled']
                cs_str = ' | '.join(f"{c[1]}({c[2]*100:.0f}%)" for c in cs)
                plain_lines.append(f"#{d['step']} 「{d['text']}」({d['prob']*100:.0f}%) [{cs_str}] {d['step_ms']:.0f}ms")
            try: self._web_socketio.emit('chat_sync', {'type': 'debug', 'lines': plain_lines})
            except: pass

    # ─── 云端 API 对话 ────────────────────────────
    def _send_chat_cloud(self, text: str):
        """通过云端 API 发送对话"""
        self.chat_input.clear()
        self.chat_input.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_stop_chat.setEnabled(True)

        # 用户消息
        self.chat_history.append(f"<b style='color:#4af'>你:</b> {text}")
        self.chat_history.append(f"<b style='color:#ff4'>TGAI:</b> ")
        self._stream_got_tokens = False

        # 构建带记忆的 prompt（云端 API 用 history 做多轮，这里直接用格式化文本）
        context = ""
        history_list = []
        for q, a in self.chat_memory[-4:]:
            context += f"用户:{q}\nTGAI?{a}\n"
            history_list.append({"role": "user", "content": q})
            history_list.append({"role": "assistant", "content": a})
        prompt = f"{context}用户:{text}\nTGAI?"
        self._pending_prompt = text

        # 读参数
        temp = self.chat_temp.value()
        max_tokens = self.chat_max_tokens.value()
        top_k = self.chat_topk.value()
        top_p = self.chat_topp.value()
        freq_pen = self.chat_freq_pen.value()
        rep_pen = self.rep_penalty_spin.value()
        min_tokens = self.chat_min_tokens.value()

        class CloudChatWorker(QThread):
            chunk_signal = pyqtSignal(str)
            error_signal = pyqtSignal(str)
            done_signal = pyqtSignal(str)

            def __init__(self, parent, cloud_fn, **kwargs):
                super().__init__(parent)
                self._fn = cloud_fn
                self._kwargs = kwargs

            def run(self):
                response = ""
                try:
                    for token in self._fn(**self._kwargs):
                        response += token
                        self.chunk_signal.emit(token)
                except Exception as e:
                    self.error_signal.emit(str(e))
                    return
                self.done_signal.emit(response)

        self.cloud_worker = CloudChatWorker(
            self, self._cloud_chat_stream,
            prompt=text, temp=temp, max_tokens=max_tokens,
            top_k=top_k, top_p=top_p, freq_penalty=freq_pen,
            rep_penalty=rep_pen, min_tokens=min_tokens,
            history=history_list,
        )
        self.cloud_worker.chunk_signal.connect(self._on_chat_chunk)
        self.cloud_worker.error_signal.connect(self._on_cloud_chat_error)
        self.cloud_worker.done_signal.connect(self._on_cloud_chat_done)
        self.cloud_worker.start()

    def _on_cloud_chat_done(self, response: str):
        """云端对话完成"""
        if not self._stream_got_tokens:
            self.chat_history.append("[模型未生成有效回复]")
        self.chat_memory.append((self._pending_prompt, response))
        if len(self.chat_memory) > 20:
            self.chat_memory = self.chat_memory[-10:]
        self._pending_prompt = ""
        self.chat_history.append("")
        self.btn_stop_chat.setEnabled(False)
        self.chat_input.setEnabled(True)
        self.btn_send.setEnabled(True)
        # 广播到 WebUI
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'assistant', 'text': response})
            except: pass

    def _on_cloud_chat_error(self, error: str):
        """云端对话错误"""
        self.chat_history.append(f"<span style='color:#e44;'>☁️ {error}</span>")
        self.chat_history.append("")
        self.btn_stop_chat.setEnabled(False)
        self.chat_input.setEnabled(True)
        self.btn_send.setEnabled(True)

    def _handle_web_chat(self, text: str, temp: float, max_tokens: int, top_k: int, top_p: float, freq_pen: float, rep_penalty: float, min_new_tokens: int, debug: bool):
        """处理来自WebUI的对话请求 — 在GUI主线程中运行，复用ChatThread"""
        if self.use_go_engine and self.tg_engine and self.tg_engine.model:
            pass  # GO 引擎已就绪
        elif self.model is not None and self.tokenizer is not None:
            pass  # PyTorch 模型已就绪
        else:
            if hasattr(self, '_web_socketio') and self._web_socketio:
                try: self._web_socketio.emit('chat_sync', {'type': 'error', 'error': '请先加载模型'})
                except: pass
            return

        # 在 GUI 聊天框中显示用户消息
        self.chat_history.append(f"<b style='color:#4af'>你:</b> {text}")
        self.chat_history.append(f"<b style='color:#ff4'>TGAI:</b> ")
        self._stream_got_tokens = False
        self.chat_input.setEnabled(False)
        self.btn_send.setEnabled(False)

        # 广播用户消息到所有 WebUI 客户端
        if hasattr(self, '_web_socketio') and self._web_socketio:
            try: self._web_socketio.emit('chat_sync', {'type': 'user', 'text': text})
            except: pass

        if self.use_go_engine and self.tg_engine and self.tg_engine.model:
            self.chat_thread = ChatThreadGO(
                self.tg_engine, text, temp, max_tokens,
                repetition_penalty=rep_penalty,
                top_k=top_k, top_p=top_p,
                freq_penalty=freq_penalty, min_tokens=min_tokens,
                tokenizer=self.tokenizer,
            )
        else:
            device = getattr(self, '_model_device', 'cpu')
            self.chat_thread = ChatThread(
                self.model, self.tokenizer, text, temp, max_tokens, device, debug=debug,
                repetition_penalty=rep_penalty,
                top_k=top_k, top_p=top_p, frequency_penalty=freq_pen,
                min_new_tokens=min_new_tokens,
            )
        self.chat_thread.chunk_signal.connect(self._on_chat_chunk)
        self.chat_thread.response_signal.connect(self._on_chat_response)
        self.chat_thread.error_signal.connect(self._on_chat_error)
        self.chat_thread.finished_signal.connect(self._on_chat_finished)
        if hasattr(self.chat_thread, 'debug_signal'):
            self.chat_thread.debug_signal.connect(self._on_chat_debug)
        self.chat_thread.start()

    # ─── 分词器操作 ──────────────────────────────────
    def _do_tokenize(self):
        text = self.tok_input.toPlainText().strip()
        if not text:
            return

        if self.tokenizer is None:
            # 尝试加载
            tok_path = 'checkpoints/tokenizer.json'
            if os.path.exists(tok_path):
                try:
                    from tokenizer import ChineseTokenizer
                    self.tokenizer = ChineseTokenizer.load(tok_path)
                except:
                    QMessageBox.warning(self, "提示", "请先在「训练」标签页训练模型以构建分词器")
                    return
            else:
                QMessageBox.warning(self, "提示", "请先在「训练」标签页训练模型以构建分词器")
                return

        ids = self.tokenizer.encode(text, add_special=True)
        ids_no_special = self.tokenizer.encode(text, add_special=False)
        decoded = self.tokenizer.decode(ids)

        result = f"输入: {text}\n"
        result += f"含特殊token: {ids}\n"
        result += f"不含特殊token: {ids_no_special}\n"
        result += f"解码: {decoded}\n\n"

        # 逐 token 展示
        result += "Token 明细:\n"
        for tid in ids:
            if tid in self.tokenizer.id_to_token:
                tok = self.tokenizer.id_to_token[tid]
                result += f"  [{tid:4d}] '{tok}'\n"
            else:
                result += f"  [{tid:4d}] <UNKNOWN>\n"

        self.tok_result.setText(result)

    def _show_vocab(self):
        if self.tokenizer is None:
            tok_path = 'checkpoints/tokenizer.json'
            if os.path.exists(tok_path):
                try:
                    from tokenizer import ChineseTokenizer
                    self.tokenizer = ChineseTokenizer.load(tok_path)
                except:
                    QMessageBox.warning(self, "提示", "请先训练模型")
                    return
            else:
                QMessageBox.warning(self, "提示", "请先训练模型")
                return

        vocab = self.tokenizer.id_to_token
        result = f"词表大小: {len(vocab)}\n\n"
        result += "ID → Token:\n"
        for i in sorted(vocab.keys()):
            tok = vocab[i]
            if tok in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                result += f"  [{i:4d}] {tok} (特殊)\n"
            else:
                result += f"  [{i:4d}] '{tok}'\n"

        self.tok_result.setText(result)

    # ─── 数据操作 ────────────────────────────────────
    def _load_demo_data(self):
        try:
            from train import _get_demo_texts
            texts = _get_demo_texts()
            self.data_editor.setPlainText('\n'.join(texts))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def _save_training_data(self):
        texts = self.data_editor.toPlainText().strip().split('\n')
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            QMessageBox.warning(self, "提示", "请输入至少一条数据")
            return

        os.makedirs('data', exist_ok=True)
        path = 'data/custom_train.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')

        QMessageBox.information(self, "已保存", f"已保存 {len(texts)} 条数据到 {path}\n下次训练时将自动使用此数据")

    def _load_data_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载训练数据", "data",
            "JSONL/JSON (*.jsonl *.json);;Text (*.txt);;All (*.*)"
        )
        if not path:
            return
        try:
            if path.endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8') as f:
                    texts = [json.loads(line)['text'] for line in f if line.strip()]
            elif path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts = data if isinstance(data, list) else data.get('texts', [])
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]

            self.data_editor.setPlainText('\n'.join(texts))
            self.status_label.setText(f"已加载 {path} ({len(texts)} 条)")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    # ─── 样式 ────────────────────────────────────────
    # ─── 主题系统 ──────────────────────────────────
    def _apply_theme(self):
        from PyQt6.QtGui import QPalette, QColor
        from PyQt6.QtWidgets import QStyleFactory

        # 强制使用 Fusion 样式（跨平台一致，Win11 兼容性最好）
        self.setStyle(QStyleFactory.create("Fusion"))

        if self._dark_mode:
            self._apply_dark_palette()
            self.btn_theme.setText("  🌙  深色模式")
        else:
            self._apply_light_palette()
            self.btn_theme.setText("  ☀️  浅色模式")

        # CSS 只用于自定义组件（侧边栏按钮、QGroupBox 圆角等 Qt Stylesheet 才能做到的）
        self._apply_extra_styles()

    def _apply_dark_palette(self):
        """基于 One Dark Pro 色系的 Fusion 暗色调色板"""
        from PyQt6.QtGui import QPalette, QColor

        p = self.palette()
        R = QPalette.ColorRole
        G = QPalette.ColorGroup
        bg      = QColor("#282c34")
        bg_alt  = QColor("#21252b")
        bg_inp  = QColor("#1e2127")
        fg      = QColor("#abb2bf")
        accent  = QColor("#61afef")
        accent2 = QColor("#98c379")
        border  = QColor("#3e4452")
        dis_fg  = QColor("#5c6370")

        p.setColor(R.Window,          bg)
        p.setColor(R.WindowText,      fg)
        p.setColor(R.Base,            bg_inp)
        p.setColor(R.AlternateBase,   bg_alt)
        p.setColor(R.ToolTipBase,     bg)
        p.setColor(R.ToolTipText,     fg)
        p.setColor(R.Text,            fg)
        p.setColor(R.Button,          bg)
        p.setColor(R.ButtonText,      fg)
        p.setColor(R.BrightText,      accent2)
        p.setColor(R.Link,            accent)
        p.setColor(R.LinkVisited,     QColor("#c678dd"))
        p.setColor(R.Highlight,       accent)
        p.setColor(R.HighlightedText, QColor("#282c34"))
        p.setColor(G.Disabled, R.WindowText, dis_fg)
        p.setColor(G.Disabled, R.Text,       dis_fg)
        p.setColor(G.Disabled, R.ButtonText, dis_fg)
        p.setColor(G.Disabled, R.Button,     bg_inp)
        p.setColor(R.Mid, border)
        p.setColor(R.Dark, QColor("#181a1f"))
        p.setColor(R.Shadow, QColor("#000000"))
        self.setPalette(p)

        self._dark_colors = {
            'bg': '#282c34', 'bg2': '#21252b', 'bg3': '#1e2127',
            'fg': '#abb2bf', 'accent': '#61afef', 'border': '#3e4452',
        }

    def _apply_light_palette(self):
        from PyQt6.QtGui import QPalette, QColor

        p = self.palette()
        R = QPalette.ColorRole
        G = QPalette.ColorGroup

        p.setColor(R.Window,          QColor("#fafafa"))
        p.setColor(R.WindowText,      QColor("#212121"))
        p.setColor(R.Base,            QColor("#ffffff"))
        p.setColor(R.AlternateBase,   QColor("#f5f5f5"))
        p.setColor(R.ToolTipBase,     QColor("#fafafa"))
        p.setColor(R.ToolTipText,     QColor("#212121"))
        p.setColor(R.Text,            QColor("#212121"))
        p.setColor(R.Button,          QColor("#f0f0f0"))
        p.setColor(R.ButtonText,      QColor("#212121"))
        p.setColor(R.Highlight,       QColor("#1e66f5"))
        p.setColor(R.HighlightedText, QColor("#ffffff"))
        p.setColor(G.Disabled, R.WindowText, QColor("#bdbdbd"))
        p.setColor(G.Disabled, R.Text,       QColor("#bdbdbd"))
        p.setColor(G.Disabled, R.ButtonText, QColor("#bdbdbd"))
        self.setPalette(p)

        self._dark_colors = {
            'bg': '#fafafa', 'bg2': '#f5f5f5', 'bg3': '#ffffff',
            'fg': '#212121', 'accent': '#1e66f5', 'border': '#c0c0c0',
        }

    def _apply_extra_styles(self):
        """仅 CSS 才能做到的自定义样式: 圆角、hover 效果、侧边栏等"""
        c = self._dark_colors
        self.setStyleSheet(f"""
            * {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }}
            #sidebar {{ background-color: {c['bg2']}; border-right: 1px solid {c['border']}; }}
            #sidebarTitle {{ color: {c['accent']}; font-size: 20px; font-weight: bold; padding: 8px; }}
            #sidebarSep {{ color: {c['border']}; }}
            #sidebarBtn {{ background: transparent; border: none; color: {c['fg']}; padding: 10px; text-align: left; font-size: 14px; border-radius: 8px; }}
            #sidebarBtn:hover {{ background: {c['border']}; }}
            #sidebarBtn:checked {{ background: {c['accent']}; color: #ffffff; font-weight: bold; }}
            #sidebarBtnSmall {{ background: transparent; border: none; color: {c['fg']}; padding: 6px; font-size: 11px; }}
            #sidebarBtnSmall:hover {{ color: {c['accent']}; }}
            #statusBar {{ color: {c['fg']}; padding: 6px; font-size: 12px; background: {c['bg2']}; }}
            #perfLabel {{ color: {c['border']}; font-size: 10px; padding: 4px; }}
            QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 8px; padding-top: 14px; font-weight: bold; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
            QProgressBar {{ border: 1px solid {c['border']}; border-radius: 6px; text-align: center; }}
            QProgressBar::chunk {{ background: {c['accent']}; border-radius: 4px; }}
            QSlider::groove:horizontal {{ height: 6px; background: {c['border']}; border-radius: 3px; }}
            QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -5px 0; background: {c['accent']}; border-radius: 8px; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; }}
            QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._apply_theme()

    def _update_perf_monitor(self):
        import psutil
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.perf_label.setText(f"CPU: {cpu}%  RAM: {ram}%")


# ─── 启动检查 ──────────────────────────────────────────
def _check_libraries():
    """检查依赖库，返回 (missing, warnings)"""
    missing = []
    warnings = []

    # 必须依赖
    for mod, name, pkg in [
        ("torch", "PyTorch", "torch"),
        ("tokenizers", "HuggingFace Tokenizers", "tokenizers"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append((name, pkg, mod))

    # 可选依赖（GUI）
    try:
        __import__("PyQt6")
    except ImportError:
        warnings.append("PyQt6 未安装，GUI 不可用 — pip install PyQt6")

    # 可选依赖
    for mod, name, pkg in [
        ("requests", "requests", "requests"),
        ("psutil", "psutil", "psutil"),
        ("flask", "Flask (WebUI)", "flask flask-socketio"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            warnings.append(f"{name} 未安装 — pip install {pkg}")

    # websocket-client 特殊处理
    try:
        import websocket
    except ImportError:
        warnings.append("websocket-client 未安装 — pip install websocket-client")

    return missing, warnings


def _install_torch(choice: str):
    """安装用户选择的 PyTorch 版本"""
    import subprocess
    if choice == "gpu":
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.0.0",
               "--extra-index-url", "https://download.pytorch.org/whl/cu124"]
        print("\n  🖥 安装 PyTorch GPU 版(CUDA 12.4)...")
    else:
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.0.0"]
        print("\n  💻 安装 PyTorch CPU 版...")

    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ PyTorch 安装成功")
        return True
    else:
        print(f"  ✗ 安装失败:\n{result.stderr[:500]}")
        return False


def _print_startup_info():
    """打印启动信息"""
    print("=" * 60)
    print("  TGAI NLP — 0.5B MoE Transformer 语言模型")
    print("=" * 60)

    # Python 版本
    import platform
    print(f"  Python: {platform.python_version()} | OS: {platform.system()}")

    # PyTorch 版本
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu} ({mem:.1f} GB)")
            print(f"  CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
        else:
            print("  GPU: 未检测到 CUDA（CPU 模式）")
    except:
        print("  PyTorch: 未安装")

    # 库检查
    missing, warnings = _check_libraries()
    if missing:
        print(f"\n  ❌ 缺少必须依赖: {len(missing)} 个")
        for name, pkg, mod in missing:
            print(f"     - {name}")
            # 如果是 PyTorch 缺失，询问安装版本
            if mod == "torch":
                print(f"\n  请选择 PyTorch 版本:")
                print(f"    [1] CPU 版 (兼容性好)")
                print(f"    [2] GPU 版 (CUDA 12.4, 需要 NVIDIA 显卡)")

                if sys.stdin.isatty():
                    choice = input("  输入 1 或 2 (默认 1): ").strip()
                else:
                    choice = "1"

                if choice == "2":
                    if not _install_torch("gpu"):
                        print("  GPU 版安装失败，尝试 CPU 版...")
                        _install_torch("cpu")
                else:
                    _install_torch("cpu")

                # 重新检查
                missing2, _ = _check_libraries()
                missing = [(n, p, m) for n, p, m in missing2 if m == "torch"]
                if not missing:
                    print("  ✓ PyTorch 已就绪\n")
                    break

        if missing:
            print(f"\n  请手动安装: pip install {' '.join(p for _, p, _ in missing)}")
            return False

    if warnings:
        print(f"\n  ⚠ 可选依赖: {len(warnings)} 个未安装")
        for w in warnings:
            print(f"     - {w}")

    print("\n  所有必须依赖 ✓")
    return True


# ─── 入口 ────────────────────────────────────────────────
def main():
    # 打印启动信息
    if not _print_startup_info():
        print("\n请安装缺失依赖后重试。")
        sys.exit(1)

    # 顶层异常捕获
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback as tb
        detail = ''.join(tb.format_exception(exc_type, exc_value, exc_tb))
        error_file = 'crash_log.txt'
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(detail)
        QMessageBox.critical(None, "程序崩溃",
            f"程序遇到未处理的错误:\n\n{exc_value}\n\n"
            f"详细日志已保存到: {error_file}")

    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setApplicationName("TGAI NLP")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    window = TGAIWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
