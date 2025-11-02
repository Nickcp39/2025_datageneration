# sample_diffusion.py
# Load checkpoint (prefers EMA), run DDPM/DDIM sampling, save grid and singles.

import os
import argparse
import torch
import torchvision.utils as vutils

from models.unet_eps import UNetEps
from diffusion.diffusion_engine import DiffusionEngine


def save_grid_and_singles(x, out_dir, nrow=8, prefix="gen"):
    os.makedirs(out_dir, exist_ok=True)
    # x: [-1,1] -> [0,1]
    x_vis = (x.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(x_vis, nrow=nrow, padding=2)
    vutils.save_image(grid, os.path.join(out_dir, f"{prefix}_grid.png"))

    for i in range(x.size(0)):
        vutils.save_image(x_vis[i], os.path.join(out_dir, f"{prefix}_{i:04d}.png"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint (.pt)")
    ap.add_argument("--out_dir", type=str, default="./samples_vessel")
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--ddim_steps", type=int, default=50)

    # fallback overrides (if ckpt lacks meta or你想强制指定)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--time_dim", type=int, default=256)

    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load ckpt -----
    ckpt = torch.load(args.ckpt, map_load_location="cpu") if hasattr(torch, "compile") else torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})

    image_size = meta.get("image_size", args.image_size)
    channels = meta.get("channels", args.channels)
    timesteps = meta.get("timesteps", args.timesteps)
    base = meta.get("base", args.base)
    time_dim = meta.get("time_dim", args.time_dim)

    # ----- build model -----
    net = UNetEps(in_ch=channels, base=base, time_dim=time_dim, with_mid_attn=True).to(device)
    state = None
    # 优先使用 EMA（若存在）
    if "ema" in ckpt and isinstance(ckpt["ema"], dict):
        state = ckpt["ema"]
        print("[INFO] Loaded EMA weights.")
    elif "model" in ckpt:
        state = ckpt["model"]
        print("[INFO] Loaded model weights (no EMA found).")
    else:
        # 兼容其它保存格式
        state = ckpt.get("unet", ckpt)
        print("[INFO] Loaded raw state dict.")

    net.load_state_dict(state, strict=False)
    net.eval()

    # ----- diffusion engine -----
    engine = DiffusionEngine(net, img_size=image_size, channels=channels,
                             timesteps=timesteps, device=device).to(device)

    # ----- sampling -----
    with torch.no_grad():
        x = engine.sample(n=args.num, method=args.method, ddim_steps=args.ddim_steps)

    # ----- save -----
    os.makedirs(args.out_dir, exist_ok=True)
    save_grid_and_singles(x, args.out_dir, nrow=args.nrow, prefix=f"{args.method}_n{args.num}")

    print(f"[DONE] Saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
