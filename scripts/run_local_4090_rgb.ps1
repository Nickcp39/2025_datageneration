# One-click training on RTX 4090 — RGB
$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_4090_rgb.ps1" -ForegroundColor Yellow

# (可选) 激活 venv
$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

# CUDA 自检
@"
import torch, sys
ok = torch.cuda.is_available()
print('cuda_available:', ok)
if ok: print('device:', torch.cuda.get_device_name(0))
sys.exit(0 if ok else 2)
"@ | python -
if ($LASTEXITCODE -ne 0) { throw "CUDA/GPU 未检测到，已终止运行。" }

# 环境变量
$env:CUDA_VISIBLE_DEVICES       = "0"
$env:PYTORCH_CUDA_ALLOC_CONF    = "expandable_segments:True"
$env:TORCH_ALLOW_TF32           = "1"
$env:NVIDIA_TF32_OVERRIDE       = "0"

# 目录与超参（按需修改）
$repo      = Resolve-Path (Join-Path $PSScriptRoot "..")
$data_root = Join-Path $repo "data_rgb"             # ← RGB 数据目录（换成你的）
$out_dir   = Join-Path $repo "runs_local_rgb_4090"

$image_size = 512
$channels   = 3      # RGB=3
$batch_size = 12     # RGB 显存更大，酌情调整
$num_workers= 8
$max_steps  = 20000
$save_every = 500
$log_every  = 50
$preview    = "ddim"
$ddim_steps = 25
$time_dim   = 256
$no_aug     = $true
$center_crop= $false

$py = "code\train_diffusion.py"
$logdir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$now = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logdir "train_rgb_4090_$now.txt"

$cmd = @(
  "python", (Join-Path $repo $py),
  "--data_root", $data_root,
  "--out_dir",   $out_dir,
  "--image_size", $image_size,
  "--channels",   $channels,
  "--batch_size", $batch_size,
  "--num_workers", $num_workers,
  "--max_steps",  $max_steps,
  "--save_every", $save_every,
  "--log_every",  $log_every,
  "--preview_method", $preview,
  "--ddim_steps", $ddim_steps,
  "--time_dim",   $time_dim,
  "--tensorboard"
)

if ($no_aug)      { $cmd += "--no_aug" }
if (-not $center_crop) { $cmd += "--no_center_crop" }

Write-Host ("Launching:`n" + ($cmd -join " ")) -ForegroundColor Cyan
& $cmd 2>&1 | Tee-Object -FilePath $log
Write-Host "Done. Outputs in: $out_dir" -ForegroundColor Green

# # 如需恢复训练请加：
# # $cmd += @("--resume", "<ckpt路径>")
