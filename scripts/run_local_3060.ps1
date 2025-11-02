# One-click local training on 3060 (8GB), no YAML dependency. Save as UTF-8 (no BOM).
$ErrorActionPreference = "Stop"
Write-Host ">>> RUN: run_local_3060.ps1" -ForegroundColor Yellow

# --------- edit your params here ----------
# relative to repo root
$data_root   = "./data2025"
$out_dir     = "./runs_local"
$image_size  = 256          # ← 提升到 256
$channels    = 1
$batch_size  = 4
$num_workers = 0
$max_steps   = 2000
$save_every  = 200
$log_every   = 20
$preview     = "ddim"
$ddim_steps  = 25
$time_dim    = 256
$seed        = ""           # 如需固定，填 42 或 20251101；留空则不传参
$use_tb      = $true        # tensorboard flag
$use_amp     = $false       # 3060 建议先关
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

# boolean flags
$argsList += "--no_aug"           # 关闭增强，避免旋转黑角
$argsList += "--no_center_crop"   # 关闭中心正方裁剪
if ($use_tb)  { $argsList += "--tensorboard" }
if ($use_amp) { $argsList += "--amp" }

Write-Host "Launching:" -ForegroundColor Cyan
Write-Host "python `"$trainPy`" $($argsList -join ' ')"
& python $trainPy @argsList
Write-Host "Done. Outputs in: $absOutDir" -ForegroundColor Green
