# train.py
import os
import math
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from tqdm.auto import tqdm

from dataset import build_dataloaders, save_debug_batch
from unet import UNetEps
from engine import DiffusionEngine

# -----------------------
# 小工具：存图（不依赖 torchvision）
# -----------------------
from PIL import Image
import numpy as np

def save_grid(x: torch.Tensor, out_path: str, nrow: int = 4):
    """
    x: [B,1,H,W] in [-1,1]
    """
    x = x.detach().float().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    b, c, h, w = x.shape
    assert c == 1, "expected grayscale [B,1,H,W]"
    cols = nrow
    rows = (b + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    k = 0
    for r in range(rows):
        for cidx in range(cols):
            if k >= b: break
            tile = (x[k, 0].cpu().numpy() * 255.0).astype(np.uint8)
            canvas[r*h:(r+1)*h, cidx*w:(cidx+1)*w] = tile
            k += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)

# -----------------------
# EMA（直接内置，无需额外文件）
# -----------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            p.data.copy_(self.shadow[name].data)

# -----------------------
# 训练主逻辑
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--val_ratio', type=float, default=0.05)
    ap.add_argument('--augment_train', action='store_true', help='启用轻量增强')
    # model/engine
    ap.add_argument('--in_ch', type=int, default=1)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--mult', type=str, default="1,2,2,4", help='通道倍率, 逗号分隔')
    ap.add_argument('--t_dim', type=int, default=256)
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'])
    ap.add_argument('--beta_start', type=float, default=1e-4)
    ap.add_argument('--beta_end', type=float, default=2e-2)
    # optim
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--amp', action='store_true', help='开启混合精度（torch.amp）')
    ap.add_argument('--ema_decay', type=float, default=0.9999)
    # logging / save
    ap.add_argument('--save_dir', type=str, default='./runs')
    ap.add_argument('--exp_name', type=str, default='exp')
    ap.add_argument('--sample_every', type=int, default=1000)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--resume', type=str, default='')
    # sampling preview
    ap.add_argument('--sample_method', type=str, default='ddim', choices=['ddim','ddpm'])
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--ddim_eta', type=float, default=0.0)
    return ap.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mult = tuple(int(x) for x in args.mult.split(','))

    # dirs
    save_root = Path(args.save_dir) / args.exp_name
    (save_root / 'ckpt').mkdir(parents=True, exist_ok=True)
    (save_root / 'samples').mkdir(parents=True, exist_ok=True)

    # dataloaders
    train_dl, val_dl = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        augment_train=args.augment_train,
    )

    # quick sanity visualization
    save_debug_batch(train_dl, str(save_root / 'samples' / 'data_debug.png'), n=16)

    # model / engine
    net = UNetEps(in_ch=args.in_ch, base=args.base, mult=mult, t_dim=args.t_dim).to(device)
    eng = DiffusionEngine(
        model=net, image_size=args.image_size, T=args.timesteps,
        schedule=args.schedule, beta_start=args.beta_start, beta_end=args.beta_end,
        device=device
    )

    # optim & EMA
    opt = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    scaler = amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(net, decay=args.ema_decay) if args.ema_decay > 0 else None

    # resume
    start_ep, global_step = 0, 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state(b := ckpt['model'])
        if 'engine_buffers' in ckpt:
            eng.load_state_dict(ckpt['engine_buffers'], strict=False)
        opt.load_state_dict(ckpt['opt'])
        if scaler.is_enabled() and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if ema and 'ema' in ckpt:
            ema.shadow = ckpt['ema']
        start_ep = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"Resumed from {args.resume} (epoch={start_ep}, step={global_step})")

    # training loop
    print("Start training.")
    for ep in range(start_ep, args.epochs):
        net.train()
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {ep:03d}", ncols=100, leave=False)
        for x0 in pbar:
            x0 = x0.to(device, non_blocking=True)
            b = x0.size(0)
            t = torch.randint(0, eng.T, (b,), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with amp.autocast('cuda'):
                    loss = eng.p_losses(x0, t)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss = eng.p_losses(x0, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                opt.step()

            if ema:
                ema.update(net)

            # logging on progress bar
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    xt, eps_true = eng.q_sample(x0, t)
                    eps_pred = net(xt, t)
                    eps_corr = torch.nn.functional.cosine_similarity(
                        eps_pred.flatten(1), eps_true.flatten(1)
                    ).mean().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", eps_corr=f"{eps_corr:+.3f}")

            # periodic sample
            if global_step % args.sample_every == 0 and global_step > 0:
                net.eval()
                with torch.no_grad():
                    if ema: ema.copy_to(net)
                    samples = eng.sample(
                        batch_size=min(8, args.batch_size),
                        shape=(args.in_ch, args.image_size, args.image_size),
                        method=args.sample_method,
                        steps=args.ddim_steps,
                        eta=args.ddim_eta,
                    )
                save_grid(samples, str(save_root / 'samples' / f'step_{global_step:06d}.png'), nrow=4)
                pbar.write(f"[sample] step {global_step} → saved preview")
                net.train()

            # periodic checkpoint
            if global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = save_root / 'ckpt' / f'ckpt_step_{global_step:06d}.pt'
                to_save = {
                    'model': net.state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': ep,
                    'global_step': global_step,
                    'args': vars(args),
                }
                try:
                    to_save['engine_buffers'] = eng.state_dict()
                except Exception:
                    pass
                if scaler.is_enabled():
                    to_save['scaler'] = scaler.state_dict()
                if ema:
                    to_save['ema'] = {k: v.cpu() for k, v in ema.shadow.items()}
                torch.save(to_save, ckpt_path)
                pbar.write(f"[ckpt ] saved {ckpt_path}")

            global_step += 1

        # epoch-end sample
        net.eval()
        with torch.no_grad():
            if ema: ema.copy_to(net)
            samples = eng.sample(
                batch_size=min(8, args.batch_size),
                shape=(args.in_ch, args.image_size, args.image_size),
                method=args.sample_method,
                steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
        save_grid(samples, str(save_root / 'samples' / f'ep_{ep:03d}.png'), nrow=4)

        # epoch-end ckpt
        ckpt_path = save_root / 'ckpt' / f'ckpt_ep_{ep:03d}.pt'
        to_save = {
            'model': net.state_dict(),
            'opt': opt.state_dict(),
            'epoch': ep + 1,
            'global_step': global_step,
            'args': vars(args),
        }
        if scaler.is_enabled():
            to_save['scaler'] = scaler.state_dict()
        if ema:
            to_save['ema'] = {k: v.cpu() for k, v in ema.shadow.items()}
        torch.save(to_save, ckpt_path)

        dt = time.time() - t0
        tqdm.write(f"[ep {ep:03d}] done in {dt:.1f}s → saved epoch ckpt & samples.")

if __name__ == "__main__":
    main()
