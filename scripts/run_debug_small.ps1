# scripts/run_debug_small.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# === 路径 ===
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Split-Path -Parent $SCRIPT_DIR
Set-Location $REPO_ROOT

# === 固定参数（无交互） ===
$DATA_ROOT   = Join-Path $REPO_ROOT "data\data_gray_sample"
$OUT_DIR     = Join-Path $REPO_ROOT "runs\runs_debug_small"
$IMAGE_SIZE  = 128
$BATCH_SIZE  = 8
$NUM_WORKERS = 0
$MAX_STEPS   = 400
$SAVE_EVERY  = 100
$LOG_EVERY   = 50
$DDIM_STEPS  = 25
$SAMPLE_N    = 4

# === 显示环境 ===
Write-Host "== Env ==" -ForegroundColor Cyan
@'
import torch, sys
print("python:", sys.version.split()[0])
print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("gpu   :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'@ | python -

# === 创建输出 ===
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

# === 跑训练（使用 diffusers 的 UNet2DModel） ===
Write-Host "== Train ==" -ForegroundColor Cyan
python yc_code/train_diffusion.py `
  --data_root "$DATA_ROOT" `
  --image_size $IMAGE_SIZE `
  --channels 1 `
  --out_dir "$OUT_DIR" `
  --batch_size $BATCH_SIZE `
  --num_workers $NUM_WORKERS `
  --max_steps $MAX_STEPS `
  --save_every $SAVE_EVERY `
  --log_every $LOG_EVERY `
  --preview_method ddim `
  --ddim_steps $DDIM_STEPS `
  --sample_n $SAMPLE_N `
  --use_diffusers
