# train_diffusion.py
# Minimal training loop for DDPM/DDIM with EMA and periodic sampling.

import os
import argparse
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from dataset_gray import GrayImageFolder
from models.unet_eps import UNetEps
from diffusion.diffusion_engine import DiffusionEngine, EMA
from tqdm import tqdm


def save_image_grid(x, path, nrow=8):
    # x: [-1,1] -> [0,1]
    x = (x.clamp(-1, 1) + 1) / 2
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
    ap.add_argument('--out_dir', type=str, default='./runs_vessel')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--sample_n', type=int, default=16)
    ap.add_argument('--preview_method', type=str, default='ddim', choices=['ddpm','ddim'])
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--tensorboard', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_iter = cycle(dl)

    # ------- Model & Engine -------
    net = UNetEps(in_ch=args.channels, base=args.base, time_dim=args.time_dim, with_mid_attn=args.mid_attn).to(device)
    engine = DiffusionEngine(net, img_size=args.image_size, channels=args.channels,
                             timesteps=args.timesteps, device=device).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    ema = EMA(net, beta=0.9999)

    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.out_dir)

    # ------- Training loop -------
    net.train()
    pbar = tqdm(range(1, args.max_steps + 1), ncols=100, desc="train", smoothing=0.1)

    for step in pbar:
        x = next(dl_iter).to(device)   # [B,C,H,W], in [0,1]
        x = x * 2 - 1                  # -> [-1,1]

        t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
        loss = engine.p_losses(x, t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        ema.update(net)

        # 实时在进度条上显示 loss
        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if tb:
                tb.add_scalar('train/loss', loss.item(), step)

        # 定期保存 + 预览采样
        if step % args.save_every == 0 or step == args.max_steps:
            ema.copy_to(net)
            ckpt_path = os.path.join(ckpt_dir, f'ckpt_{step:06d}.pt')
            torch.save({
                'model': net.state_dict(),
                'ema': ema.shadow,
                'meta': {
                    'image_size': args.image_size,
                    'channels': args.channels,
                    'timesteps': args.timesteps,
                    'base': args.base,
                    'time_dim': args.time_dim,
                }
            }, ckpt_path)
            # 用 pbar.write() 避免打乱进度条
            pbar.write(f"saved: {ckpt_path}")

            with torch.no_grad():
                x_sample = engine.sample(
                    n=args.sample_n,
                    method=args.preview_method,
                    ddim_steps=args.ddim_steps
                )
            grid_path = os.path.join(samp_dir, f'sample_{step:06d}.png')
            save_image_grid(x_sample, grid_path, nrow=int(args.sample_n ** 0.5))
            pbar.write(f"preview saved: {grid_path}")

    if tb:
        tb.close()
  


if __name__ == '__main__':
    main()
