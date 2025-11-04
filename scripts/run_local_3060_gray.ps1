# scripts/run_local_3060_gray.ps1 —— RTX3060 · 灰度稳跑（yc_code + EMA + mid-t 采样）
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# === 路径定位 ===
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Split-Path -Parent $SCRIPT_DIR
Set-Location $REPO_ROOT

# === 参数（按需改） ===
$DATA_ROOT   = Join-Path $REPO_ROOT "data\data_gray"
$OUT_DIR     = Join-Path $REPO_ROOT "runs_local_gray_3060"
$IMAGE_SIZE  = 256
$BATCH_SIZE  = 4
$NUM_WORKERS = 0
$MAX_STEPS   = 20000
$SAVE_EVERY  = 400
$LOG_EVERY   = 20
$DDIM_STEPS  = 50
$TIME_DIM    = 256
$NET_BASE    = 64
$T_TOTAL     = 1000
$LR          = 5e-5
$SEED        = 2025

# === 环境检查 ===
Write-Host "== Env check ==" -ForegroundColor Cyan
@'
import torch, sys
print("python:", sys.version.split()[0])
print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("is_cuda_available:", torch.cuda.is_available())
'@ | python -

# === 数据计数 ===
Write-Host "== Data check ==" -ForegroundColor Cyan
@"
import os, glob, sys
root = r"$DATA_ROOT"
exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
files=[]
for e in exts:
    files += glob.glob(os.path.join(root, "**", e), recursive=True)
print(f"[data] found {len(files)} images under: {root}")
if not files:
    sys.exit(2)
"@ | python -

# === 输出目录 ===
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OUT_DIR "samples") | Out-Null

# === 训练（内嵌 Python） ===
Write-Host "== Launch training (embedded) ==" -ForegroundColor Cyan
@"
import os, sys, math, time, copy, torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 让包可导入
sys.path.insert(0, os.getcwd())
from yc_code.dataset_gray import GrayImageFolder
from yc_code.diffusion.diffusion_engine import DiffusionEngine

# ---------- 超参 ----------
DATA_ROOT   = r"$DATA_ROOT"
OUT_DIR     = r"$OUT_DIR"
IMAGE_SIZE  = int($IMAGE_SIZE)
BATCH_SIZE  = int($BATCH_SIZE)
NUM_WORKERS = int($NUM_WORKERS)
MAX_STEPS   = int($MAX_STEPS)
SAVE_EVERY  = int($SAVE_EVERY)
LOG_EVERY   = int($LOG_EVERY)
TIME_DIM    = int($TIME_DIM)
NET_BASE    = int($NET_BASE)
T_TOTAL     = int($T_TOTAL)
LR          = float($LR)
SEED        = int($SEED)

# 固定种子
def set_seed(s):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------- 数据 ----------
ds = GrayImageFolder(
    root=DATA_ROOT, img_size=IMAGE_SIZE, channels=1,
    center_crop=False, aug=False, normalize="tanh",
    save_debug=True, debug_outdir=OUT_DIR,
)
print(f"[data] found {len(ds)} images under {DATA_ROOT}")
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
it = iter(dl)

# ---------- 模型 ----------
engine = DiffusionEngine(
    image_size=IMAGE_SIZE, channels=1, T=T_TOTAL,
    schedule="cosine", net_base=NET_BASE, time_dim=TIME_DIM,
    with_mid_attn=False,  # 先关注意力，更稳
).to(device)
opt = torch.optim.AdamW(engine.parameters(), lr=LR)

# ---------- EMA（用于采样/重建） ----------
ema = copy.deepcopy(engine).eval()
for p in ema.parameters(): p.requires_grad_(False)
EMA_DECAY = 0.9995

# ---------- 可选：p2 reweight（让高噪声时刻权重大一些） ----------
USE_P2 = True
def p2_weight(a_bar):
    w = (a_bar / (1.0 - a_bar + 1e-8)).clamp(0, 1e3).sqrt()  # p=0.5
    return w

# ---------- 工具：从真实图构造重建（用 net 反推） ----------
@torch.no_grad()
def reconstruct_from_real(net, x0_real, t_scalar=400):
    B = x0_real.size(0)
    t = torch.full((B,), t_scalar, device=x0_real.device, dtype=torch.long)
    a_bar = net._extract(net.alphas_cumprod, t, x0_real.shape)
    noise = torch.randn_like(x0_real)
    x_t = a_bar.sqrt()*x0_real + (1.0 - a_bar).sqrt()*noise
    for cur in range(t_scalar, -1, -1):
        tt = torch.full((B,), cur, device=x0_real.device, dtype=torch.long)
        x_t = net.p_sample(x_t, tt, log_every=False)
    return x_t

# ---------- 训练 ----------
step = 0
engine.train()
while step < MAX_STEPS:
    try:
        x0 = next(it)
    except StopIteration:
        it = iter(dl); x0 = next(it)
    x0 = x0.to(device, non_blocking=True)
    t  = torch.randint(0, engine.T, (x0.size(0),), dtype=torch.long, device=device)

    out  = engine(x0, t)              # out['loss'] 为 MSE(epŝ, eps)
    loss = out["loss"]
    if USE_P2:
        a_bar = engine._extract(engine.alphas_cumprod, t, x0.shape)  # [B,1,1,1]
        loss  = (loss * p2_weight(a_bar)).mean()
    else:
        loss = loss.mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
    opt.step()

    # EMA 更新
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), engine.parameters()):
            p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0-EMA_DECAY)

    step += 1
    if step % LOG_EVERY == 0:
        print(f"[train] step={step:06d} | loss={loss.item():.4f}", flush=True)

    if step % SAVE_EVERY == 0:
        # 用 EMA 做采样与重建
        net = ema.eval()
        with torch.no_grad():
            # ① 纯生成：从中等噪声 t_start 起步（更稳）
            B = 4; t_start = 400
            xg = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
            for cur in range(t_start, -1, -1):
                tt = torch.full((B,), cur, device=device, dtype=torch.long)
                xg = net.p_sample(xg, tt, log_every=False)
            xg = (xg + 1.0)/2.0
            xg = xg.clamp(0,1)
            sp = os.path.join(OUT_DIR, "samples", f"sample_{step:06d}.png")
            save_image(xg, sp, nrow=2)
            print(f"[sample] saved -> {sp}", flush=True)

            # ② 重建可视化：真图 → 加噪到 t=400 → 反推到 0
            real = x0[:4]
            xr   = reconstruct_from_real(net, real, t_scalar=400)
            vis  = torch.cat([ (real+1)/2.0, (xr+1)/2.0 ], dim=0).clamp(0,1)
            rp = os.path.join(OUT_DIR, "samples", f"recon_{step:06d}.png")
            save_image(vis, rp, nrow=4)
            print(f"[recon]  saved -> {rp}", flush=True)

print("[done] training finished.", flush=True)
"@ | python -
