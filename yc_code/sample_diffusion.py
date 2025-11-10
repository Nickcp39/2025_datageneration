# sample_diffusion.py
# Load checkpoint (prefers EMA), run true DDPM/DDIM sampling, save grid and singles.
# 约定对齐：
# - 只用自研 DiffusionEngine；engine 不是 nn.Module，**不要** .to(device)
# - UNetEps in_ch == out_ch == channels；with_mid_attn 从 ckpt meta 读取（默认 False）
# - 采样方法 --method ddpm|ddim 确实走不同代码路径
# - 优先使用 EMA 权重；采样输出为 [-1,1]，保存前映射到 [0,1]
# - t_start 按比例给定（0.4*T）

import os
import argparse
import torch
import torchvision.utils as vutils

from models.unet_eps import UNetEps
from diffusion.diffusion_engine import DiffusionEngine


def save_grid_and_singles(x, out_dir, nrow=8, prefix="gen"):
    os.makedirs(out_dir, exist_ok=True)
    # x: [-1,1] -> [0,1]
    x_vis = (x.detach().to(torch.float32).clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(x_vis, nrow=nrow, padding=2)
    vutils.save_image(grid, os.path.join(out_dir, f"{prefix}_grid.png"))
    for i in range(x_vis.size(0)):
        vutils.save_image(x_vis[i], os.path.join(out_dir, f"{prefix}_{i:04d}.png"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint (.pt)")
    ap.add_argument("--out_dir", type=str, default="./samples_vessel")
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--t_ratio", type=float, default=0.4, help="start ratio of T for partial sampling, e.g., 0.4")
    # fallback overrides (if ckpt lacks meta or你想强制指定)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--time_dim", type=int, default=256)
    ap.add_argument("--with_mid_attn", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load ckpt -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})

    image_size = meta.get("image_size", args.image_size)
    channels   = meta.get("channels", args.channels)
    timesteps  = meta.get("timesteps", args.timesteps)
    base       = meta.get("base", args.base)
    time_dim   = meta.get("time_dim", args.time_dim)
    mid_attn   = meta.get("mid_attn", args.with_mid_attn)

    # ----- build model -----
    net = UNetEps(in_ch=channels, base=base, time_dim=time_dim, with_mid_attn=mid_attn).to(device)
    if hasattr(net, "set_max_timesteps"):
        net.set_max_timesteps(timesteps)

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
    # 注意：engine 不是 nn.Module，**不要** .to(device)
    engine = DiffusionEngine(
            image_size=args.image_size,
            channels=args.channels,
            T=args.timesteps,          # 参数名是 T
            # schedule="cosine",       # 需要 linear 再改
            )
    engine.to(device) 

    # ----- sampling -----
    with torch.no_grad():
        T = timesteps
        t_start = max(0, min(T - 1, int(args.t_ratio * T)))
        B, H = args.num, image_size

        if args.method == "ddim":
            print(f"[SAMPLER] method=DDIM | steps={args.steps} | t_start={t_start}/{T-1}")
            # 优先调用引擎提供的 DDIM 接口；否则退化到 DDPM
            if hasattr(engine, "ddim_partial"):
                x = engine.ddim_partial(batch=B, channels=channels, size=H,
                                        t_start=t_start, steps=args.steps)  # 期望返回 [-1,1]
            elif hasattr(engine, "ddim_sample"):
                x = engine.ddim_sample(n=B, steps=args.steps)  # 若实现的是完整 DDIM
            else:
                print("[WARN] Engine has no DDIM method; falling back to DDPM loop.")
                x = torch.randn(B, channels, H, H, device=device)
                for cur in range(t_start, -1, -1):
                    tt = torch.full((B,), cur, device=device, dtype=torch.long)
                    x = engine.p_sample(x, tt)   # DDPM 单步
        else:  # DDPM
            print(f"[SAMPLER] method=DDPM | t_start={t_start}/{T-1}")
            x = torch.randn(B, channels, H, H, device=device)
            for cur in range(t_start, -1, -1):
                tt = torch.full((B,), cur, device=device, dtype=torch.long)
                x = engine.p_sample(x, tt)       # DDPM 单步

    # ----- save -----
    os.makedirs(args.out_dir, exist_ok=True)
    save_grid_and_singles(x, args.out_dir, nrow=args.nrow, prefix=f"{args.method}_n{args.num}")
    print(f"[DONE] Saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
