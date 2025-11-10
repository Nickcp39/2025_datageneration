# train_diffusion.py
# Minimal training loop for our own DiffusionEngine (ε-pred), with EMA, AMP, and periodic sampling.
# 目录约定：yc_code 与 data 同级；运行时在 yc_code/ 下，--data_root ../data/data_gray

import os
import sys
import time
import argparse
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm.auto import tqdm


# -------------------- utils --------------------
def save_image_grid(x, path, nrow=8):
    """
    统一到 CPU/float32 再保存。
    允许输入在 [-1,1] 或 [0,1]，自动映射为 [0,1]。
    """
    x = x.detach().to('cpu', dtype=torch.float32)
    if x.min() < 0.0 or x.max() > 1.0:
        x = (x.clamp(-1, 1) + 1) * 0.5
    else:
        x = x.clamp(0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(x, path, nrow=nrow, padding=2)


# -------------------- args --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train diffusion with our single-source engine (ε-pred).")
    # data
    ap.add_argument('--data_root', type=str, required=True, help="e.g., ../data/data_gray")
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
    ap.add_argument('--amp', action='store_true', help='use torch.cuda.amp autocast')

    # model
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--time_dim', type=int, default=256)
    ap.add_argument('--mid_attn', action='store_true', default=True)

    # logging / io
    ap.add_argument('--out_dir', type=str, default='./runs/runs_gray')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--sample_n', type=int, default=16)
    ap.add_argument('--preview_method', type=str, default='ddim', choices=['ddpm', 'ddim'])
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--tensorboard', action='store_true')

    # 兼容旧开关（强制不用 HF）
    ap.add_argument('--use_diffusers', action='store_true', default=False)

    return ap.parse_args()


# -------------------- main --------------------
def main():
    args = parse_args()

    # 更稳定的 tqdm 刷新
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # 必须用 GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA/GPU，退出。")
        sys.exit(2)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # IO
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'ckpts')
    samp_dir = os.path.join(args.out_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samp_dir, exist_ok=True)

    # dataset
    try:
        # 如果你有封装在 yc_code/ 里的实现
        from yc_code.data.gray_image_folder import GrayImageFolder
    except Exception:
        from dataset_gray import GrayImageFolder

    ds = GrayImageFolder(
        root=args.data_root,
        img_size=args.image_size,
        channels=args.channels,
        center_crop=args.center_crop,
        train=True,
        aug=not args.no_aug,
        normalize="none",  # 数据集输出 [0,1]
    )
    print(f"[data] found {len(ds)} images under {args.data_root}")
    if len(ds) == 0:
        raise RuntimeError("No images found. Check data_root and extensions.")

    eff_batch = min(args.batch_size, len(ds))
    drop_last = len(ds) >= args.batch_size
    dl = DataLoader(
        ds, batch_size=eff_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=drop_last
    )
    dl_iter = cycle(dl)

    # quick snapshot
    batch0 = next(iter(dl))[:min(16, eff_batch)]
    print("[data] batch:", batch0.shape, "min/max:", float(batch0.min()), float(batch0.max()))
    save_image_grid(batch0, os.path.join(samp_dir, 'data_debug.png'),
                    nrow=max(1, int(len(batch0) ** 0.5)))

    # 模型与引擎（只用我们的引擎）
    if args.use_diffusers:
        raise RuntimeError("HF engine 已弃用。本项目只用自研 DiffusionEngine。")

    from models.unet_eps import UNetEps
    from diffusion.diffusion_engine import DiffusionEngine

    net = UNetEps(
        in_ch=args.channels,
        base=args.base,
        time_dim=args.time_dim,
        with_mid_attn=args.mid_attn
    ).to(device)

    if hasattr(net, "set_max_timesteps"):
        net.set_max_timesteps(args.timesteps)

    engine = DiffusionEngine(
        image_size=args.image_size,
        channels=args.channels,
        T=args.timesteps,   # 注意：参数名是 T
        # schedule="cosine",
    ).to(device)            # 引擎实现了 to()，会把内部表移动到 GPU

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # EMA（shadow dict）
    class EMA:
        def __init__(self, model, beta=0.9999):
            self.beta = beta
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def update(self, model):
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.beta).add_(v, alpha=(1 - self.beta))

        def copy_to(self, model):
            model.load_state_dict(self.shadow, strict=False)

    ema = EMA(net, beta=0.9999)

    # TensorBoard
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.out_dir)

    # 进度条
    BAR_FMT = "{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    pbar = tqdm(
        total=args.max_steps,
        desc=f"train bs={eff_batch} drop_last={drop_last}",
        bar_format=BAR_FMT,
        dynamic_ncols=True,
        ascii=True,
        smoothing=0.1,
        miniters=1,
        mininterval=0.1,
        leave=True,
    )

    # -------------------- train loop --------------------
    net.train()
    t0 = time.time()
    try:
        for step in range(1, args.max_steps + 1):
            x0 = next(dl_iter).to(device, non_blocking=True)  # [B,C,H,W] in [0,1]
            x0 = x0 * 2.0 - 1.0                               # -> [-1,1]
            t  = torch.randint(0, args.timesteps, (x0.size(0),), device=device).long()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out  = engine.p_losses(model=net, x_start=x0, t=t)  # ✅ 传 model
                loss = out["loss"] if isinstance(out, dict) else out

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            ema.update(net)

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                it_s = step / max(elapsed, 1e-9)
                eta  = (args.max_steps - step) / max(it_s, 1e-9)
                pbar.set_postfix_str(f"loss={loss.item():.4f}", refresh=False)
                print(f"[{step:06d}/{args.max_steps:06d}] {100*step/args.max_steps:5.1f}% | "
                      f"{it_s:5.1f} it/s | ETA {eta/60:5.1f}m | loss={loss.item():.4f}",
                      flush=True)
                if tb:
                    tb.add_scalar('train/loss', loss.item(), step)

            pbar.update(1)

            # ---- save & preview ----
            if step % args.save_every == 0 or step == args.max_steps:
                # 1) 备份训练权重，用 EMA 做采样
                train_state = net.state_dict()
                net.load_state_dict(ema.shadow, strict=False)

                # 2) 采样（FP32，统一接口）
                net.eval()
                with torch.no_grad():
                    preview = engine.sample(
                        model=net,
                        batch_size=args.sample_n,
                        method=('ddim' if args.preview_method == 'ddim' else 'ddpm'),
                        ddim_steps=args.ddim_steps,
                        device=device
                    )  # [-1,1]

                # 3) 存样图
                grid_path = os.path.join(samp_dir, f'sample_{step:06d}.png')
                save_image_grid(preview, grid_path, nrow=max(1, int(args.sample_n ** 0.5)))
                pbar.write(f"preview saved: {grid_path}")

                # 4) 还原训练权重
                net.load_state_dict(train_state, strict=False)
                net.train()

                # 5) 存 ckpt（训练态 + EMA）
                ckpt_path = os.path.join(ckpt_dir, f'ckpt_{step:06d}.pt')
                torch.save({
                    'model': train_state,
                    'ema': ema.shadow,
                    'opt': opt.state_dict(),
                    'meta': {
                        'image_size': args.image_size,
                        'channels': args.channels,
                        'timesteps': args.timesteps,
                        'base': args.base,
                        'time_dim': args.time_dim,
                        'mid_attn': args.mid_attn,
                        'preview_method': args.preview_method,
                        'ddim_steps': args.ddim_steps,
                    }
                }, ckpt_path)
                pbar.write(f"saved: {ckpt_path}")

    finally:
        pbar.close()
        if tb:
            tb.close()


if __name__ == '__main__':
    main()
