# run_gray_cos_base128_lr1e4.ps1
# 在 2025_DATAGENERATION\scripts 目录下执行这个脚本

# 1. 项目根目录 = scripts 的上一级
$projRoot = Split-Path $PSScriptRoot -Parent

# 2. 关键路径
$trainPy  = Join-Path $projRoot "yc_code\train.py"

# ⚠️ 如果你的灰度数据在其他文件夹，比如 splits\gray_256\，
# 就把这一行改成对应的相对路径
$dataRoot = Join-Path $projRoot "data\data_gray"

$saveDir  = Join-Path $projRoot "runs"

# 3. 生成时间戳 + 实验名
$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
$exp = "gray_cos_base128_lr1e4_$ts"

# 4. 为这次实验建一个 log 目录（和 train.py 里的 save_root 对齐）
$expDir  = Join-Path $saveDir $exp
New-Item -ItemType Directory -Force -Path $expDir | Out-Null

# 5. 调用 train.py，并把终端输出同时写到文件里
python $trainPy `
  --data_root $dataRoot `
  --image_size 256 `
  --in_ch 1 `
  --base 128 `
  --mult "1,2,2,4" `
  --t_dim 256 `
  --batch_size 16 `
  --num_workers 8 `
  --val_ratio 0 `
  --epochs 120 `
  --timesteps 1000 `
  --schedule cosine `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --grad_clip 1.0 `
  --amp `
  --ema_decay 0.999 `
  --save_dir $saveDir `
  --exp_name $exp `
  --sample_every 2000 `
  --save_every 6000 `
  --log_every 50 `
  --sample_method ddpm `
  *>&1 | Tee-Object -FilePath (Join-Path $expDir "train_console.log")
