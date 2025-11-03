# One-click local training on 3060 (8GB) - RGB with center crop to square
# 数据：python_code/data2025/  内含 1000x860 彩色图；先中心裁正方形，再缩放到 512
$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_rgb.ps1" -ForegroundColor Yellow

# --------- edit your params here ----------
$data_root   = "./data2025"      # ← RGB 数据文件夹
$out_dir     = "./runs_local_rgb"
$image_size  = 512
$channels    = 3                 # ← 彩色
$batch_size  = 4
$num_workers = 0
$max_steps   = 2000
$save_every  = 500
$log_every   = 50
$preview     = "ddim"
$ddim_steps  = 25
$time_dim    = 256
$seed        = ""                # 可填 20251102；留空则不传
$use_tb      = $true
$use_amp     = $false
# ------------------------------------------

# repo paths
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$trainPy  = Join-Path $repoRoot "code\train_diffusion.py"
if (-not (Test-Path $trainPy)) { throw "Training script not found: $trainPy" }

# env
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES = "0"

# make out dir
$outAbs = (Resolve-Path (Join-Path $repoRoot $out_dir) -ErrorAction SilentlyContinue)
if (-not $outAbs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot $out_dir) | Out-Null
    $outAbs = (Resolve-Path (Join-Path $repoRoot $out_dir))
}

# build args
$argsList = @()
function Add-Arg($name, $value) {
    if ($null -ne $value -and $value -ne "") { $script:argsList += $name; $script:argsList += ([string]$value) }
}

$absDataRoot = (Resolve-Path (Join-Path $repoRoot $data_root)).Path
$absOutDir   = $outAbs.Path

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

# flags
$argsList += "--no_aug"        # 如需翻转/微旋转，删掉这行
$argsList += "--center_crop"   # 关键：RGB 需中心裁成正方形
if ($use_tb)  { $argsList += "--tensorboard" }
if ($use_amp) { $argsList += "--amp" }

Write-Host "Launching:" -ForegroundColor Cyan
Write-Host "python `"$trainPy`" $($argsList -join ' ')"
& python $trainPy @argsList
Write-Host "Done. Outputs in: $absOutDir" -ForegroundColor Green
