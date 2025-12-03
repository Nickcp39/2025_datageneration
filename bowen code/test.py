# test.py
import os
import argparse
import math

import torch
import torchvision.utils as vutils
import wandb

from network import UNetModel, GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的 checkpoint 路径")
    parser.add_argument("--out_dir", type=str, default="./samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--project", type=str, default="pa_diffusion")
    parser.add_argument("--run_name", type=str, default="ddpm_test")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device

    # ===== 构建网络和 diffusion，注意参数要和训练时保持一致 =====
    model = UNetModel(
        in_ch=1,
        base_ch=64,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
    )
    model = model.to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        beta_1=args.beta_start,
        beta_T=args.beta_end,
        device=device,
    )
    diffusion = diffusion.to(device)

    # ===== 加载 checkpoint =====
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    diffusion.load_state_dict(ckpt["diffusion_state"], strict=False)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'NA')}")

    # ===== wandb (可选，用于记录测试样本) =====
    wandb.init(project=args.project, name=args.run_name)

    diffusion.eval()
    with torch.no_grad():
        samples = diffusion.sample(batch_size=args.num_samples)  # [-1,1]

    # 保存单张 & grid
    grid = vutils.make_grid(samples, nrow=int(math.sqrt(args.num_samples)),
                            normalize=True, value_range=(-1, 1))
    grid_path = os.path.join(args.out_dir, "samples_grid.png")
    vutils.save_image(grid, grid_path)
    print(f"Saved grid image to {grid_path}")

    # 单张保存
    for i in range(args.num_samples):
        img = samples[i]
        img_path = os.path.join(args.out_dir, f"sample_{i:03d}.png")
        # [-1,1] -> [0,1]
        vutils.save_image((img + 1) / 2, img_path)

    # 上传到 wandb
    wandb.log({"test/samples": wandb.Image(grid)})
    print("Samples logged to wandb.")


if __name__ == "__main__":
    main()
