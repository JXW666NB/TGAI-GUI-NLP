#!/bin/bash
# =============================================
#  TGAI API 服务器 — AutoDL 部署脚本
#  项目路径: /root/TGAI/tgai_nlp/
#  模型路径: /root/autodl-tmp/TGAI_checkpoints_v3/
#  使用方法:
#    cd /root/TGAI/tgai_nlp
#    bash scripts/deploy/deploy_autodl.sh
# =============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

# ── 路径配置 ──
MODEL_DIR="/root/autodl-tmp/TGAI_checkpoints_v3"
CHECKPOINT="$MODEL_DIR/tgai_sft.pt"
TOKENIZER="$MODEL_DIR/tokenizer.json"

echo "========================================"
echo "  TGAI API Server — AutoDL 部署"
echo "========================================"
echo ""

# ── 1. 环境检查 ──
echo "[1/4] 检查环境..."

python3 --version
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo '检查失败，尝试 nvidia-smi')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo ""

# ── 2. 安装依赖 ──
echo ""
echo "[2/4] 安装 Python 依赖..."
pip install -r requirements_api.txt -q

# ── 3. 检查模型 ──
echo ""
echo "[3/4] 检查模型文件..."

if [ -f "$CHECKPOINT" ]; then
    echo "  [OK] $CHECKPOINT ($(du -h "$CHECKPOINT" | cut -f1))"
else
    echo "  [ERROR] 未找到 $CHECKPOINT"
    exit 1
fi

if [ -f "$TOKENIZER" ]; then
    echo "  [OK] $TOKENIZER"
else
    echo "  [ERROR] 未找到 $TOKENIZER"
    exit 1
fi

# ── 4. 启动服务 ──
echo ""
echo "[4/4] 启动 API 服务器..."
echo "========================================"
echo "  端口: 6008 (AutoDL 自定义服务)"
echo "  网页界面: http://你的实例地址:6008"
echo "  API文档: 见 docs/API_GUIDE.md"
echo "========================================"
echo ""

# 确保日志目录存在
mkdir -p logs

# 后台运行 + 日志
nohup python3 scripts/api_server.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --host 0.0.0.0 \
    --port 6008 \
    --extend 8192 \
    > logs/api_server.log 2>&1 &

echo "服务器已在后台启动 (PID: $!)"
echo ""
echo "查看日志: tail -f logs/api_server.log"
echo "停止服务: pkill -f api_server.py"
echo ""
echo "然后在 AutoDL 控制台 → 自定义服务 → 添加端口 6008"
