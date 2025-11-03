# One-click training on RTX 4090 — GRAYSCALE
# 用法：右键“用 PowerShell 运行”或在 pwsh 里：pwsh -File scripts/run_local_4090_gray.ps1
$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_4090_gray.ps1" -ForegroundColor Yellow

# === (可选) 激活本地 venv ===
$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

# === CUDA 自检（失败则退出）===
@"
import torch, sys
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
ok = torch.cuda.is_available()
print('cuda_available:', ok)
if ok: print('device:', torch.cuda.get_device_name(0))
sys.exit(0 if ok else 2)
"@ | python -
if ($LASTEXITCODE -ne 0) { throw "CUDA/GPU 未检测到，已终止运行。" }

# === 性能相关环境变量（可保留）===
$env:CUDA_VISIBLE_DEVICES       = "0"
$env:PYTORCH_CUDA_ALLOC_CONF    = "expandable_segments:True"
$env:TORCH_ALLOW_TF32           = "1"
$env:NVIDIA_TF32_OVERRIDE       = "0"

# === 目录与超参（按需修改）===
$repo      = Resolve-Path (Join-Path $PSScriptRoot "..")
$data_root = Join-Path $repo "data"                 # ← 灰度数据目录
$out_dir   = Join-Path $repo "runs_local_gray_4090" # 输出目录（不覆盖旧 runs）

$image_size = 512
$channels   = 1      # 灰度=1
$batch_size = 16     # 4090 建议 12~32 之间，OOM 就降
$num_workers= 8      # Windows 可用 0/4/8；WSL/Linux 8/12/16
$max_steps  = 20000
$save_every = 500
$log_every  = 50
$preview    = "ddim"
$ddim_steps = 25
$time_dim   = 256
$no_aug     = $true
$center_crop= $false

# === 组装命令 ===
$py = "code\train_diffusion.py"
$logdir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$now = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logdir "train_gray_4090_$now.txt"

# 注：train_diffusion.py 已经强制要求 cuda，可放心运行
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

# # 如需从某个 checkpoint 恢复，在上面 $cmd 最后追加：
# # "--resume", "<绝对或相对ckpt路径>"
