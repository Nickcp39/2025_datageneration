# main.py
import os
import argparse
import math
import random
from typing import Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import wandb
import json
from dataset import build_dataloaders
from network import UNetModel, GaussianDiffusion

from copy import deepcopy

def update_ema(ema_model, model, decay: float = 0.999):
    """
    简单的 EMA 更新： ema = decay * ema + (1-decay) * model
    只对参数做更新，不参与反向传播
    """
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if k in msd:
                v.copy_(decay * v + (1.0 - decay) * msd[k])


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default=None,help="Path to Json config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="dataset root, 包含 train/ 和 val/ 子目录")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--project", type=str, default="PA_image_generation")
    parser.add_argument("--run_name", type=str, default="P1")
    parser.add_argument("--save_dir", type=str, default="./Results/results_image/P1_test_results")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="保存模型的 epoch 间隔")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config,'r') as f:
            config_json = json.load(f)
        for k,v in config_json.items():
            if hasattr(args,k):
                setattr(args,k,v)
            else:
                print(f"[Warning] Unknown key in JOSN: {k}")

    return args


def build_model_and_diffusion(args) -> GaussianDiffusion:
    model = UNetModel(
        in_ch=1,
        base_ch=64,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
    )
    model = model.to(args.device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        device=args.device,
    )
    diffusion = diffusion.to(args.device)

    return diffusion


def validate(diffusion: GaussianDiffusion,
             val_loader: DataLoader,
             device: str) -> Dict[str, float]:
    """
    在验证集上估计 loss
    """
    diffusion.eval()
    val_losses = []

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            b = x.size(0)
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
            loss = diffusion.p_losses(x_start=x, t=t)
            val_losses.append(loss.item())

    diffusion.train()
    val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
    return {"val_loss": val_loss}


def sample_and_log_images(diffusion: GaussianDiffusion,
                          global_step: int,
                          num_samples: int = 16):
    """
    采样一批图像，并上传到 wandb
    """
    diffusion.eval()
    with torch.no_grad():
        samples = diffusion.sample(batch_size=num_samples)  # [-1,1]
    diffusion.train()

    # 做一个 grid
    nrow = int(math.sqrt(num_samples))
    grid = vutils.make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
    wandb.log({"val/samples": wandb.Image(grid)}, step=global_step)


def main():
    args = parse_args()
    set_seed(args.seed)
    wandb.login(key="3c81fd27191085f58004d8d98cd450171018a724")
    wandb.init(project=args.project, name=args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== Data ==========
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ========== Model & Diffusion ==========
    diffusion = build_model_and_diffusion(args)
    model = diffusion.model  # 方便直接访问
    device = args.device
    # ========== EMA Model ==========
    ema_decay = 0.999  # 可以先用 0.999，后面再调
    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)  # EMA 不参与训练


    # ========== Optimizer ==========
    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=1e-4,
    )

    # ========== wandb ==========
    config = vars(args)

    global_step = 0

    # 预先创建一个固定 noise，用于每次 epoch 采样对比（如果你想可复现）
    fixed_noise = torch.randn(16, 1, args.image_size, args.image_size, device=device)

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        for i, x in enumerate(train_loader):
            x = x.to(device)  # [B,1,H,W]
            b = x.size(0)

            # 均匀采样 t
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()

            loss = diffusion.p_losses(x_start=x, t=t)

            optimizer.zero_grad()
            update_ema(ema_model, model, decay=ema_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            if global_step % 50 == 0:
                wandb.log({"train/loss": loss.item(),
                           "train/epoch": epoch},
                          step=global_step)

        # ===== 每个 epoch 结束做一次验证 =====
        metrics = validate(diffusion, val_loader, device=device)
        wandb.log(metrics, step=global_step)
        print(f"[Epoch {epoch}] val_loss={metrics['val_loss']:.6f}")

        # ===== 采样 + 上传到 wandb =====
        # 使用 fixed_noise 的版本，也可以直接调用 diffusion.sample()
        diffusion.eval()
        with torch.no_grad():
            # 这里示范一种：把 fixed_noise 当做 x_T，然后只改 p_sample_loop 稍微支持自定义初始值的话
            # 目前 p_sample_loop 内部自己生成 x_T，因此这里直接用 diffusion.sample()
            orig_model = diffusion.model
            diffusion.model = ema_model
            samples = diffusion.sample(batch_size=16)
            diffusion.model = orig_model
        diffusion.train()

        grid = vutils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        wandb.log({"val/samples": wandb.Image(grid),
                   "val/epoch": epoch},
                  step=global_step)

        # ===== 周期性保存模型 =====
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "diffusion_state": diffusion.state_dict(),
                "ema_model_state": ema_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"[Epoch {epoch}] checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
