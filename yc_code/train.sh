#!/usr/bin/env bash
# Minimal launcher for self-engine diffusion training
# - 固定常用超参，支持环境变量覆盖
# - 自动创建输出目录与日志
# - GPU/环境自检
# 用法：
#   chmod +x train.sh
#   ./train.sh
# 覆盖示例：
#   DATA_DIR=./data2025 IMAGE_SIZE=256 BATCH=32 MAX_STEPS=12000 ./train.sh

set -Eeuo pipefail

# ==== 路径与环境 ====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
cd "$PROJECT_ROOT"

# ==== 数据与输出 ====
DATA_DIR="${DATA_DIR:-./data_gray}"               # 你的数据根目录（已在 .gitignore 中忽略）
RUN_TAG="${RUN_TAG:-runs_vessel}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-.runs/${RUN_TAG}_${STAMP}}"
mkdir -p "$OUT_DIR" "$OUT_DIR/ckpts" "$OUT_DIR/samples"

# ==== 训练超参（可用环境变量覆盖）====
IMAGE_SIZE="${IMAGE_SIZE:-256}"                  # 必须被4整除（我们UNet下采样2次）
CHANNELS="${CHANNELS:-1}"                        # 1灰度 / 3彩色
BATCH="${BATCH:-32}"
MAX_STEPS="${MAX_STEPS:-12000}"
LR="${LR:-1e-4}"
TIMESTEPS="${TIMESTEPS:-1000}"
BASE="${BASE:-64}"
TIME_DIM="${TIME_DIM:-256}"
MID_ATTN="${MID_ATTN:-0}"                        # 0关闭 / 1开启
CENTER_CROP="${CENTER_CROP:-0}"                  # 0关闭 / 1开启
NO_AUG="${NO_AUG:-1}"                            # 1禁用轻度增强 / 0启用

# 训练中预览采样设置
PREVIEW_METHOD="${PREVIEW_METHOD:-ddim}"         # ddpm | ddim
DDIM_STEPS="${DDIM_STEPS:-50}"                   # 训练内预览的DDIM步数
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAMPLE_N="${SAMPLE_N:-16}"

# ==== 自检：GPU & PyTorch ====
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️  未检测到 nvidia-smi；若无GPU请确认已安装CUDA驱动（>=535）。"
else
  nvidia-smi || true
fi

python - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print(f"[check] torch.cuda.is_available() = {ok}")
sys.exit(0 if ok else 2)
PY

# ==== 组装参数 ====
ARGS=(
  --data_root "$DATA_DIR"
  --image_size "$IMAGE_SIZE"
  --channels "$CHANNELS"
  --batch_size "$BATCH"
  --max_steps "$MAX_STEPS"
  --lr "$LR"
  --timesteps "$TIMESTEPS"
  --base "$BASE"
  --time_dim "$TIME_DIM"
  --out_dir "$OUT_DIR"
  --log_every "$LOG_EVERY"
  --save_every "$SAVE_EVERY"
  --sample_n "$SAMPLE_N"
  --preview_method "$PREVIEW_METHOD"
  --ddim_steps "$DDIM_STEPS"
)

# flags
if [ "$MID_ATTN" = "1" ]; then
  ARGS+=( --mid_attn )
fi
if [ "$CENTER_CROP" = "1" ]; then
  ARGS+=( --center_crop )
fi
if [ "$NO_AUG" = "1" ]; then
  ARGS+=( --no_aug )
fi

# ==== 启动训练 ====
echo "[run] python train_diffusion.py ${ARGS[*]}"
python train_diffusion.py "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/train.log"

echo "[done] outputs -> $OUT_DIR"
