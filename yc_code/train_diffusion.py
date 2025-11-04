# train_diffusion.py
# Minimal training loop for DDPM/DDIM with EMA and periodic sampling.
# - Robust tqdm progress bar: percentage + ETA (PowerShell/VSCode 终端均可)
# - Data snapshot to samples/data_debug.png
# - DataLoader uses eff_batch & drop_last consistent with dataset length

import os
import sys
import time
import argparse
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm.auto import tqdm  # ✅ auto 版更适配不同终端

from dataset_gray import GrayImageFolder

def save_image_grid(x, path, nrow=8):
    """统一到 CPU/float32 再保存，避免半精度/CUDA 导致的黑图。"""
    x = x.detach().to('cpu', dtype=torch.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if x_min < 0.0 or x_max > 1.0:
        x = (x.clamp(-1, 1) + 1) / 2
    else:
        x = x.clamp(0, 1)
    vutils.save_image(x, path, nrow=nrow, padding=2)


def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--channels', type=int, default=1, choices=[1, 3])
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--center_crop',    dest='center_crop', action='store_true',  default=False)
    ap.add_argument('--no_center_crop', dest='center_crop', action='store_false')
    ap.add_argument('--no_aug', action='store_true', help='disable light augmentation')

    # training
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_steps', type=int, default=12000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--timesteps', type=int, default=1000)  # diffusion T

    # model
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--time_dim', type=int, default=256)
    ap.add_argument('--mid_attn', action='store_true', default=True)

    # logging / io
    ap.add_argument('--out_dir', type=str, default='.runs/runs_vessel')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--sample_n', type=int, default=16)
    ap.add_argument('--preview_method', type=str, default='ddim', choices=['ddpm', 'ddim'])
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--tensorboard', action='store_true')

    # switch: use diffusers-like UNet2D or your own UNetEps + engine
    ap.add_argument('--use_diffusers', action='store_true', default=True)

    return ap.parse_args()


def main():
    args = parse_args()

    # ===== 强制非缓冲输出，提升 tqdm 刷新稳定性（PowerShell/VSCode 里很关键） =====
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):  # Python 3.7+
        sys.stdout.reconfigure(line_buffering=True)

    # ===== 必须使用 GPU：没有就直接退出 =====
    if not torch.cuda.is_available():
        print("❌ CUDA/GPU 未检测到：请安装 GPU 版 PyTorch 或检查显卡/驱动。训练已终止。")
        sys.exit(2)

    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------- IO & dirs -------
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'ckpts')
    samp_dir = os.path.join(args.out_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samp_dir, exist_ok=True)

    # ------- Dataset & Loader -------
    ds = GrayImageFolder(
        root=args.data_root,
        img_size=args.image_size,
        channels=args.channels,
        center_crop=args.center_crop,
        aug=not args.no_aug,
        normalize="none",           # 输出 [0,1]，训练时再转 [-1,1]
    )
    print(f"[data] found {len(ds)} images under {args.data_root}")
    if len(ds) == 0:
        raise RuntimeError("No images found. Check data_root and extensions.")

    # 自适应 batch_size 与 drop_last（和实际一致，便于 ETA）
    eff_batch = min(args.batch_size, len(ds))
    drop_last = len(ds) >= args.batch_size

    dl = DataLoader(
        ds,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    # 快照一张 batch，确认不是黑的
    dbg = next(iter(dl))[:min(16, eff_batch)]   # [B,C,H,W] in [0,1]
    print("[data] batch:", dbg.shape, "min/max:", float(dbg.min()), float(dbg.max()))
    save_image_grid(dbg, os.path.join(samp_dir, 'data_debug.png'),
                    nrow=max(1, int(len(dbg) ** 0.5)))

    dl_iter = cycle(dl)

    # ------- Model & Engine -------
    if args.use_diffusers:
        # 走 HuggingFace 风格 UNet + 轻量引擎（需你已有 models.hf_unet2d 和 diffusion.hf_engine）
        from models.hf_unet2d import build_unet2d
        from diffusion.hf_engine import HFEngine
        net = build_unet2d(
            sample_size=args.image_size,
            in_ch=args.channels,
            out_ch=args.channels
        ).to(device)
        engine = HFEngine(
            net,
            timesteps=args.timesteps,
            schedule='ddpm'  # or 'cosine' if你的 HFEngine 支持
        ).to(device)
    else:
        # 走你自研 UNetEps + 自研 DiffusionEngine
        from models.unet_eps import UNetEps
        from diffusion.diffusion_engine import DiffusionEngine
        net = UNetEps(
            in_ch=args.channels,
            base=args.base,
            time_dim=args.time_dim,
            with_mid_attn=args.mid_attn
        ).to(device)
        engine = DiffusionEngine(
            net,
            img_size=args.image_size,
            channels=args.channels,
            timesteps=args.timesteps,
            device=str(device)
        ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    # EMA（与引擎无关，包模型权重）
    class EMA:
        def __init__(self, model, beta=0.9999):
            self.model = model
            self.beta = beta
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        @torch.no_grad()
        def update(self, model=None):
            m = self.model if model is None else model
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_((self.beta)).add_(v, alpha=(1 - self.beta))
        def copy_to(self, model):
            model.load_state_dict(self.shadow, strict=False)

    ema = EMA(net, beta=0.9999)

    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.out_dir)

    # ------- Training loop -------
    net.train()

    BAR_FMT = (
        "{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    pbar = tqdm(
        total=args.max_steps,
        desc=f"train bs={eff_batch} drop_last={drop_last}",
        bar_format=BAR_FMT,
        dynamic_ncols=True,
        ascii=True,          # PowerShell 下建议保留
        smoothing=0.1,
        miniters=1,
        mininterval=0.1,
        leave=True,
        disable=False,
    )

    t0 = time.time()
    try:
        for step in range(1, args.max_steps + 1):
            x = next(dl_iter).to(device, non_blocking=True)  # [B,C,H,W], in [0,1]
            x = x * 2 - 1                                    # -> [-1,1]

            t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()

            opt.zero_grad(set_to_none=True)
            out  = engine.p_losses(x, t)
            # 兼容：自研引擎返回 dict；HFEngine 可能直接返回 Tensor
            loss = out["loss"] if isinstance(out, dict) else out
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            ema.update(net)

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                iters   = max(step, 1)
                sec_per_it = elapsed / iters
                eta_sec = (args.max_steps - step) * sec_per_it
                pct = 100.0 * step / args.max_steps
                print(f"[{step:06d}/{args.max_steps:06d}] {pct:5.1f}% | "
                      f"elapsed {elapsed/60:.1f}m < ETA {eta_sec/60:.1f}m | "
                      f"{1.0/sec_per_it:.1f} it/s | loss={loss.item():.4f}",
                      flush=True)
                pbar.set_postfix_str(f" loss={loss.item():.4f}", refresh=False)
                if tb:
                    tb.add_scalar('train/loss', loss.item(), step)

            pbar.update(1)

            # —— 定期保存 + 预览采样（EMA + DDPM + FP32）——
            if step % args.save_every == 0 or step == args.max_steps:
                # 1) 用 EMA 权重做采样（别覆盖训练态）
                _train_state = net.state_dict()
                net.load_state_dict(ema.shadow, strict=False)

                # 2) 采样只用 FP32（不要 autocast/half）
                # === 采样（更容易出结构）：从中等噪声起步 ===
                net.eval()
                with torch.no_grad():
                    B, H = args.sample_n, args.image_size
                    x = torch.randn(B, args.channels, H, H, device=device)
                    t_start = 400  # 200~600 都行，400 常用
                    for cur in range(t_start, -1, -1):
                        tt = torch.full((B,), cur, device=device, dtype=torch.long)
                        x = engine.p_sample(x, tt)   # 用 DDPM 单步
                x = x.detach().to('cpu', dtype=torch.float32).clamp(-1, 1)
                save_image_grid((x+1)/2, os.path.join(samp_dir, f'sample_{step:06d}.png'),
                                nrow=max(1, int(args.sample_n ** 0.5)))
                net.train()

                # 3) 还原原来的训练权重
                net.load_state_dict(_train_state, strict=False)

                # 4) 存 ckpt + 预览
                ckpt_path = os.path.join(ckpt_dir, f'ckpt_{step:06d}.pt')
                torch.save({
                    'model': _train_state,       # 保存的是训练权重
                    'ema': ema.shadow,
                    'meta': {
                        'image_size': args.image_size,
                        'channels': args.channels,
                        'timesteps': args.timesteps,
                        'base': args.base,
                        'time_dim': args.time_dim,
                    }
                }, ckpt_path)
                pbar.write(f"saved: {ckpt_path}")

                x_sample = x_sample.detach().to('cpu', dtype=torch.float32)
                grid_path = os.path.join(samp_dir, f'sample_{step:06d}.png')
                save_image_grid(x_sample, grid_path, nrow=max(1, int(args.sample_n ** 0.5)))
                pbar.write(f"preview saved: {grid_path}")

            # ------- 兜底：若 tqdm 被禁（非 TTY），手动打印 ETA/百分比 -------
            if pbar.disable and (step % args.log_every == 0):
                elapsed = time.time() - t0
                done = step
                total = args.max_steps
                rate = done / max(elapsed, 1e-9)
                remain = (total - done) / max(rate, 1e-9)
                print(f"[{done:>6}/{total}] {done/total*100:5.1f}% "
                      f"elapsed={elapsed:6.1f}s ETA={remain:6.1f}s rate={rate:6.2f} it/s "
                      f"loss={loss.item():.4f}")
                sys.stdout.flush()

    finally:
        pbar.close()
        if tb:
            tb.close()


if __name__ == '__main__':
    main()
