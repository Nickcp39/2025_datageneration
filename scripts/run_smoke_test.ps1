# run_smoke_test.ps1
# Execute this script from the '2025_DATAGENERATION\scripts' directory.

# 1. Define Project Root
$projRoot = Split-Path $PSScriptRoot -Parent

# 2. Define Key Paths
$trainPy = Join-Path $projRoot "yc_code\train.py"
$dataRoot = Join-Path $projRoot "data\data_gray"
$saveDir = Join-Path $projRoot "runs"

# 3. Generate Timestamp and Experiment Name
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$exp = "smoke_test_$ts"

# 4. Create Experiment Directory
$expDir = Join-Path $saveDir $exp
New-Item -ItemType Directory -Force -Path $expDir | Out-Null

Write-Host "--- 🚀 Starting Smoke Test: $exp ---"
Write-Host "Log Directory: $expDir"
Write-Host "--- ------------------------------------------------- ---"

# 5. Run train.py (Comments removed from command block to fix syntax error)
python $trainPy `
  --data_root $dataRoot `
  --image_size 256 `
  --in_ch 1 `
  --base 64 `
  --mult "1,2,2,4" `
  --t_dim 256 `
  --batch_size 4 `
  --num_workers 4 `
  --val_ratio 0 `
  --epochs 1 `
  --timesteps 1000 `
  --schedule cosine `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --grad_clip 1.0 `
  --amp `
  --ema_decay 0.999 `
  --save_dir $saveDir `
  --exp_name $exp `
  --sample_every 50 `
  --save_every 100 `
  --log_every 10 `
  --sample_method ddpm `
  *>&1 | Tee-Object -FilePath (Join-Path $expDir "train_console.log")

Write-Host "--- ✅ Smoke Test Complete. Please check 'train_console.log' in $expDir ---"