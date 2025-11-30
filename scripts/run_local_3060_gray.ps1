# scripts/run_local_3060_gray_bowen.ps1
# RTX 3060 · 256x256 灰度 · 配置尽量对齐 bowen（T=1000, linear beta, lr=1e-4, batch=8）

$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_3060_gray.ps1" -ForegroundColor Yellow

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

# === 性能相关环境变量 ===
$env:CUDA_VISIBLE_DEVICES    = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_ALLOW_TF32        = "1"
$env:NVIDIA_TF32_OVERRIDE    = "0"

# === 目录与超参（按需修改）===
$repo       = Resolve-Path (Join-Path $PSScriptRoot "..")
# 这里指向 8000 张灰度图所在目录（你之前说是 data\data_gray）
$data_root  = Join-Path $repo "data\data_gray"
# 3060 的输出单独放一个文件夹，避免和 4090 的混
$out_dir    = Join-Path $repo "runs_local"

# —— 关键超参：尽量对齐 bowen ——
$image_size = 256            # bowen: 256x256
$channels   = 1              # 灰度
$timesteps  = 1000           # T=1000，对齐 bowen
$batch_size = 8              # A100 用 8 也 OK，3060 8 一般没问题；OOM 就改成 4
$num_workers= 0              # Windows + 3060 建议 0，稳定
# bowen 约 190k steps；这里先给 200k，你可以根据时间改小
$max_steps  = 200000         
$save_every = 5000           # 每 5k step 存一次 ckpt + sample
$log_every  = 100
$preview    = "ddpm"         # 采样用 DDPM 更接近 bowen
$ddim_steps = 1000           # 这个参数在 ddpm 下不会用到，无所谓
$time_dim   = 256            # 和当前 UNetEps 设置一致
$lr         = 1e-4           # bowen: lr=1e-4
$no_aug     = $true          # bowen 没有 data augmentation
$center_crop= $false         # bowen 是直接 resize 到 256，不做 center crop

# === 组装命令 ===
# 注意这里路径按你现在工程来，如果你的 train_diffusion 在 yc_code 下，就改成 "yc_code\train_diffusion.py"
$py = "code\train_diffusion.py"

$logdir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$now = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logdir "train_gray_3060_$now.txt"

$cmd = @(
  "python", (Join-Path $repo $py),
  "--data_root", $data_root,
  "--out_dir",   $out_dir,
  "--image_size", $image_size,
  "--channels",   $channels,
  "--batch_size", $batch_size,
  "--num_workers", $num_workers,
  "--timesteps",  $timesteps,   # T=1000
  "--max_steps",  $max_steps,
  "--save_every", $save_every,
  "--log_every",  $log_every,
  "--preview_method", $preview,
  "--ddim_steps", $ddim_steps,
  "--time_dim",   $time_dim,
  "--lr",         $lr,
  "--tensorboard"
)

if ($no_aug)          { $cmd += "--no_aug" }
if (-not $center_crop){ $cmd += "--no_center_crop" }
# 如果你想开 AMP，再加一行：$cmd += "--amp"

Write-Host ("Launching:`n" + ($cmd -join " ")) -ForegroundColor Cyan
& $cmd 2>&1 | Tee-Object -FilePath $log
Write-Host "Done. Outputs in: $out_dir" -ForegroundColor Green

# 如需从某个 checkpoint 恢复，可在 $cmd 最后追加：
# "--resume", "<绝对或相对ckpt路径>"
