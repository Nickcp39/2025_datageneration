# scripts/run_local_3060_gray.ps1 — RTX3060 · 256x256 灰度 · t* 采样 + EMA 重建
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ==== 路径 ====
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Split-Path -Parent $SCRIPT_DIR
Set-Location $REPO_ROOT

# ==== 数据 & 输出 ====
$DATA_ROOT   = Join-Path $REPO_ROOT "data\data_gray"         # ← 这里指向你的 8000 张图
$OUT_DIR     = Join-Path $REPO_ROOT "runs\runs_local_gray_3060"

# ==== 训练超参（按需改） ====
$IMAGE_SIZE  = 256
$BATCH_SIZE  = 8            # 3060Ti (8GB) 通常 8~16；不够就降到 6/4
$NUM_WORKERS = 0            # Windows 建议 0，最稳
$MAX_STEPS   = 20000
$SAVE_EVERY  = 500
$LOG_EVERY   = 50
$T_TOTAL     = 400          # 先用 400，离纯噪声没那么远，更容易出结构
$LR          = 1e-4
$EMA_DECAY   = 0.9995
$SEED        = 2025

# ==== 环境信息 ====
Write-Host "== Env ==" -ForegroundColor Cyan
@'
import torch, sys
print("python:", sys.version.split()[0])
print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("gpu   :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'@ | python -

# ==== 数据计数 ====
Write-Host "== Data check ==" -ForegroundColor Cyan
@"
import os, glob, sys
root = r"$DATA_ROOT"
exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
files=[]
for e in exts:
    files += glob.glob(os.path.join(root, "**", e), recursive=True)
print(f"[data] found {len(files)} images under: {root}")
if len(files) == 0: sys.exit(2)
"@ | python -

# ==== 开始训练（内嵌 Python） ====
Write-Host "== Train ==" -ForegroundColor Cyan
@"
import os, sys, math, time, copy, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
T_TOTAL     = int($T_TOTAL)
LR          = float($LR)
EMA_DECAY   = float($EMA_DECAY)
SEED        = int($SEED)

# ---------- 稳定性设置 ----------
def set_seed(s):
    import numpy as np, torch, random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------- 数据 ----------
ds = GrayImageFolder(
    root=DATA_ROOT, img_size=IMAGE_SIZE, channels=1,
    center_crop=False, aug=True, normalize="tanh",    # [-1,1]
    save_debug=True, debug_outdir=OUT_DIR,
)
print(f"[data] found {len(ds)} images under {DATA_ROOT}")
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
it = iter(dl)

# ---------- 模型与优化 ----------
engine = DiffusionEngine(
    image_size=IMAGE_SIZE, channels=1, T=T_TOTAL,
    schedule="cosine", net_base=64, time_dim=256,
    with_mid_attn=False, eps_guard=1e-8
).to(device)
opt = torch.optim.AdamW(engine.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=0.0)

# ---------- EMA ----------
ema = copy.deepcopy(engine).eval()
for p in ema.parameters(): p.requires_grad_(False)

# ---------- p2 reweight（可提高高噪声阶段权重） ----------
USE_P2 = True
def p2_weight(a_bar):
    # p=0.5：sqrt(a_bar/(1-a_bar))
    return ((a_bar / (1.0 - a_bar + 1e-8)).clamp(0,1e6)).sqrt()

# ---------- 可达 t* 探测 ----------
@torch.no_grad()
def find_reachable_t(engine, ema_net, real_batch, probe=(300,250,200,150,120,100,80,60)):
    device = real_batch.device
    ema_net.eval()
    for t_scalar in probe:
        B = real_batch.size(0)
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)
        a_bar = engine._extract(engine.alphas_cumprod, t, real_batch.shape)
        noise = torch.randn_like(real_batch)
        x_t = a_bar.sqrt()*real_batch + (1.0 - a_bar).sqrt()*noise
        x = x_t.clone()
        for cur in range(t_scalar, -1, -1):
            tt = torch.full((B,), cur, device=device, dtype=torch.long)
            x = engine.p_sample(x, tt, log_every=False)
        l1 = (x - real_batch).abs().mean().item()
        print(f"[probe] t={t_scalar:4d} | L1={l1:.4f}")
        if l1 < 0.35:      # 经验阈值，可按数据调
            return t_scalar
    return 0

# ---------- 训练主循环 ----------
step = 0
engine.train()
os.makedirs(os.path.join(OUT_DIR, "ckpts"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)

while step < MAX_STEPS:
    try:
        x0 = next(it)
    except StopIteration:
        it = iter(dl); x0 = next(it)
    x0 = x0.to(device, non_blocking=True)           # [-1,1]
    t  = torch.randint(0, engine.T, (x0.size(0),), device=device, dtype=torch.long)

    out  = engine(x0, t)                            # {'loss', 'pred_noise', 'x_noisy'}
    loss = out["loss"]                              # [B,...] 或标量，按你的实现而定
    if loss.ndim > 0:                               # 做成标量
        if USE_P2:
            a_bar = engine._extract(engine.alphas_cumprod, t, x0.shape)
            loss = (loss * p2_weight(a_bar)).mean()
        else:
            loss = loss.mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
    opt.step()

    # EMA
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), engine.parameters()):
            p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0-EMA_DECAY)

    step += 1
    if step % LOG_EVERY == 0:
        print(f"[{step:06d}/{MAX_STEPS:06d}] loss={loss.item():.4f}", flush=True)

    if step % SAVE_EVERY == 0 or step == MAX_STEPS:
        net = ema.eval()

        # 1) 先探测可达 t*
        real = x0[:4]
        t_star = find_reachable_t(engine, net, real, probe=(300,250,200,150,120,100,80,60))
        t_star = max(60, int(t_star))
        print(f"[sample] start from t*={t_star}")

        # 2) 生成：从 t* 起步
        B = 8
        xg = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        for cur in range(t_star, -1, -1):
            tt = torch.full((B,), cur, device=device, dtype=torch.long)
            xg = engine.p_sample(xg, tt, log_every=False)
        xg = ((xg+1)/2).clamp(0,1)
        sp = os.path.join(OUT_DIR, "samples", f"sample_{step:06d}.png")
        save_image(xg, sp, nrow=4); print(f"[sample] -> {sp}")

        # 3) 重建：真图→噪到 t*→反推
        @torch.no_grad()
        def reconstruct_from_real(net, x0, t_scalar):
            B = x0.size(0)
            t = torch.full((B,), t_scalar, device=device, dtype=torch.long)
            a_bar = engine._extract(engine.alphas_cumprod, t, x0.shape)
            noise = torch.randn_like(x0)
            x_t = a_bar.sqrt()*x0 + (1.0 - a_bar).sqrt()*noise
            x = x_t
            for cur in range(t_scalar, -1, -1):
                tt = torch.full((B,), cur, device=device, dtype=torch.long)
                x = engine.p_sample(x, tt, log_every=False)
            return x

        xr = reconstruct_from_real(net, real, t_star)
        vis = torch.cat([ (real+1)/2, (xr+1)/2 ], dim=0).clamp(0,1)
        rp = os.path.join(OUT_DIR, "samples", f"recon_{step:06d}.png")
        save_image(vis, rp, nrow=4); print(f"[recon ] -> {rp}")

        # 4) 存 ckpt
        torch.save({"ema": net.state_dict(), "cfg":{
            "image_size": IMAGE_SIZE, "timesteps": T_TOTAL
        }}, os.path.join(OUT_DIR, "ckpts", f"ckpt_{step:06d}.pt"))
"@ | python -
