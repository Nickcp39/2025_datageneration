# One-click local training on 3060 (8GB), no YAML dependency. Save as UTF-8 (no BOM).
$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_3060.ps1" -ForegroundColor Yellow

# ---- 常量：基于脚本位置推导仓库根 ----
# $PSScriptRoot = 当前 .ps1 文件所在目录（scripts/）
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRoot = $repoRoot.Path   # 取字符串路径
$trainPy  = Join-Path $repoRoot "code\train_diffusion.py"

if (-not (Test-Path $trainPy)) {
  throw "Training script not found: $trainPy"
}

# ---- 训练参数（相对于仓库根，可写绝对路径）----
$data_root   = "data2025"     # 可以写 "D:\xxx\data2025"
$out_dir     = "runs_local"
$image_size  = 256
$channels    = 3             # 灰度=1；彩色=3
$batch_size  = 4
$num_workers = 0              # Windows 先 0 更稳
$max_steps   = 2000
$save_every  = 200
$log_every   = 20
$preview     = "ddim"
$ddim_steps  = 25
$time_dim    = 256
$seed        = ""             # 例：42；留空则不传
$use_tb      = $true
$use_amp     = $false         # 3060 先关

# ---- 环境变量 ----
$env:CUDA_VISIBLE_DEVICES    = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_ALLOW_TF32        = "1"
$env:NVIDIA_TF32_OVERRIDE    = "0"

# ---- CUDA 自检（没有就退出）----
@"
import torch, sys
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
ok = torch.cuda.is_available()
print('cuda_available:', ok)
if ok: print('device:', torch.cuda.get_device_name(0))
sys.exit(0 if ok else 2)
"@ | python -
if ($LASTEXITCODE -ne 0) { throw "❌ 未检测到 CUDA/GPU，已终止。" }

# ---- 规范化路径（支持相对/绝对）----
function _Abs([string]$p) {
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Resolve-Path (Join-Path $repoRoot $p)).Path
}
$absDataRoot = _Abs $data_root
$absOutDir   = _Abs $out_dir

# 确保输出目录存在
New-Item -ItemType Directory -Force -Path $absOutDir | Out-Null

# ---- 构造参数 ----
$argsList = @()
function Add-Arg($name, $value) {
  if ($null -ne $value -and $value -ne "") { $script:argsList += $name; $script:argsList += ([string]$value) }
}
Add-Arg "--data_root"      $absDataRoot
Add-Arg "--out_dir"        $absOutDir
Add-Arg "--image_size"     $image_size
Add-Arg "--channels"       $channels
Add-Arg "--batch_size"     $batch_size
Add-Arg "--num_workers"    $num_workers
Add-Arg "--max_steps"      $max_steps
Add-Arg "--save_every"     $save_every
Add-Arg "--log_every"      $log_every
Add-Arg "--preview_method" $preview
Add-Arg "--ddim_steps"     $ddim_steps
Add-Arg "--time_dim"       $time_dim
if ($seed -ne "") { Add-Arg "--seed" $seed }

# 布尔 flags
$argsList += "--no_aug"
$argsList += "--no_center_crop"
if ($use_tb)  { $argsList += "--tensorboard" }
if ($use_amp) { $argsList += "--amp" }

# ---- 启动 ----
Write-Host "Repo: $repoRoot" -ForegroundColor DarkGray
Write-Host "Data: $absDataRoot" -ForegroundColor DarkGray
Write-Host "Out : $absOutDir"   -ForegroundColor DarkGray

Write-Host "Launching:" -ForegroundColor Cyan
Write-Host "python `"$trainPy`" $($argsList -join ' ')"
& python $trainPy @argsList
Write-Host "Done. Outputs in: $absOutDir" -ForegroundColor Green
