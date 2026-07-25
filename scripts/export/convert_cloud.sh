#!/bin/bash
# TGAI .pt → .TG 云端转换脚本
# 用法: bash convert_cloud.sh
# 自动将 models/ 目录下的 .pt 转换为 .TG 格式

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONVERTER="$PROJECT_DIR/TGAI GO/tgai_convert.py"
TOKENIZER="$PROJECT_DIR/checkpoints/tokenizer.json"
OUT_DIR="/root/autodl-tmp/TGAI_models"
MODEL_DIR="/root/autodl-tmp/TGAI_checkpoints_v3"

echo "════════════════════════════════════════"
echo "  TGAI 云端模型转换"
echo "════════════════════════════════════════"
echo ""
echo "转换器: $CONVERTER"
echo "分词器: $TOKENIZER"
echo "输出目录: $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# ─── V1 0.86B FP16 (桌面/性能优先) ───────────
echo "[1/4] TGAI NB V1 0.86B (FP16)"
python "$CONVERTER" \
  --checkpoint "$MODEL_DIR/tgai_sft.pt" \
  --output "$OUT_DIR/tgai_v1_fp16.TG" \
  --tokenizer "$TOKENIZER" \
  --dtype fp16
echo "  ✓ tgai_v1_fp16.TG"

# ─── V1 0.86B Q4_0 (低配手机) ───────────────
echo "[2/4] TGAI NB V1 0.86B (Q4_0)"
python "$CONVERTER" \
  --checkpoint "$MODEL_DIR/tgai_sft.pt" \
  --output "$OUT_DIR/tgai_v1_q4.TG" \
  --tokenizer "$TOKENIZER" \
  --dtype q4_0
echo "  ✓ tgai_v1_q4.TG"

# ─── V2 3B FP16 ──────────────────────────────
echo "[3/4] TGAI NB V2 3B (FP16)"
python "$CONVERTER" \
  --checkpoint "$MODEL_DIR/tgai_lora_1w.pt" \
  --output "$OUT_DIR/tgai_v2_3b_fp16.TG" \
  --tokenizer "$TOKENIZER" \
  --dtype fp16
echo "  ✓ tgai_v2_3b_fp16.TG"

# ─── V2 3B Q4_0 (极致压缩) ───────────────────
echo "[4/4] TGAI NB V2 3B (Q4_0)"
python "$CONVERTER" \
  --checkpoint "$MODEL_DIR/tgai_lora_1w.pt" \
  --output "$OUT_DIR/tgai_v2_3b_q4.TG" \
  --tokenizer "$TOKENIZER" \
  --dtype q4_0
echo "  ✓ tgai_v2_3b_q4.TG"

echo ""
echo "════════════════════════════════════════"
echo "  转换完成！"
echo "  输出目录: $OUT_DIR"
ls -lh "$OUT_DIR"/*.TG
echo "════════════════════════════════════════"
