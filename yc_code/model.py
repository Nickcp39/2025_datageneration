"""
VesselDiffusionMinimal: 从零实现的轻量级 DDPM，用于 2D 灰度（或伪 RGB）血管图生成。
- 纯 PyTorch，无预训练、无外部库依赖（可选 TensorBoard）。
- 支持 1/3 通道输入；默认灰度 1ch，更贴近你的数据。
- 采用 cosine beta schedule；UNet 带 SiLU、GroupNorm、ResBlock、Attention（中层可开关）。
- 训练时实时写入 TensorBoard（可选），并保存 EMA 权重与周期性样图。
- 采样支持 DDPM (ancestral) 和 DDIM（更快）两种推断方式。

使用：
1) 安装依赖：
   pip install torch torchvision tensorboard

2) 组织数据：
   data_root/ 目录下放你的 8000 张图（可含子目录）。

3) 训练：
   python vessel_diffusion_minimal.py train \
       --data_root /path/to/images \
       --image_size 256 --channels 1 \
       --out_dir ./runs_vessel \
       --batch_size 32 --steps 12000

4) 采样（用最后一次保存的权重）：
   python vessel_diffusion_minimal.py sample \
       --ckpt ./runs_vessel/ckpt_final.pt \
       --out_dir ./runs_vessel/samples --num 64 --ddim 1 --ddim_steps 50

5) 启动 TensorBoard（可选）：
   tensorboard --logdir ./runs_vessel

作者：ChatGPT（为课堂作业设计“可解释的自定义模型”，可在报告中说明改动点与取舍）
"""

import os, math, argparse, glob, random, time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from PIL import Image

# ----------------------------
# Utility
# ----------------------------

def exists(x): return x is not None

def default(val, d): return val if exists(val) else d

def cycle(dl):
    while True:
        for x in dl:
            yield x

# ----------------------------
# Dataset
# ----------------------------
class ImgFolder(Dataset):
    def __init__(self, root, size=256, channels=1, center_crop=True):
        self.files = []
        for e in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            self.files += glob.glob(os.path.join(root, "**", e), recursive=True)
        if len(self.files) == 0:
            raise RuntimeError(f"No images under: {root}")
        self.size = size
        self.channels = channels
        tfm = [transforms.Resize(size, interpolation=Image.BICUBIC)]
        if center_crop:
            tfm.insert(0, transforms.Lambda(lambda img: _center_square(img)))
        tfm += [
            transforms.ToTensor(),        # [0,1]
        ]
        self.tf = transforms.Compose(tfm)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)                 # [3,H,W]
        if self.channels == 1:
            x = x.mean(0, keepdim=True)  # 灰度化：取三通道平均
        # 轻量数据增强（类似 ADA 的保守版）：
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # 水平翻转
        if random.random() < 0.1:
            # 轻度旋转（-5~5 度），防止过拟合到固定角度
            angle = random.uniform(-5, 5)
            x = transforms.functional.rotate(x, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        return x


def _center_square(img: Image.Image):
    s = min(img.width, img.height)
    l = (img.width - s)//2; t = (img.height - s)//2
    return img.crop((l,t,l+s,t+s))

# ----------------------------
# Diffusion schedule (cosine)
# ----------------------------
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    s: float = 0.008  # from Nichol & Dhariwal for cosine schedule

class CosineSchedule:
    def __init__(self, cfg: DiffusionConfig):
        self.N = cfg.timesteps
        self.s = cfg.s
        t = torch.linspace(0, self.N, self.N+1, dtype=torch.float32)
        f = lambda x: torch.cos(((x/self.N)+self.s)/(1+self.s) * math.pi/2) ** 2
        alphas_cumprod = f(t) / f(torch.tensor(0.0))
        self.alphas_cumprod = alphas_cumprod.clamp(min=1e-8)
        self.alphas = self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
        self.betas = (1 - self.alphas).clamp(min=1e-8, max=0.999)

# ----------------------------
# Positional / timestep embedding
# ----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), value=0)
        return emb

# ----------------------------
# UNet building blocks
# ----------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act  = nn.SiLU()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, t=None):
        h = self.block1(x)
        if exists(self.mlp) and exists(t):
            h = h + self.mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, inner*3, 1, bias=False)
        self.to_out = nn.Conv2d(inner, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h*w), qkv)
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class UNet(nn.Module):
    def __init__(self, channels=1, dim=128, dim_mults=(1,2,2,4), with_attn=(False, True, False, False), time_dim=512):
        super().__init__()
        self.channels = channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        # Down
        dims = [channels, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.skips = []
        for i, (di, do) in enumerate(in_out):
            attn = with_attn[i]
            self.downs.append(nn.ModuleList([
                ResBlock(di, do, time_dim),
                ResBlock(do, do, time_dim),
                Attention(do) if attn else nn.Identity(),
                nn.Conv2d(do, do, 3, 2, 1),  # downsample
            ]))
        # Mid
        mid_dim = dims[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_dim, mid_dim, time_dim),
            Attention(mid_dim),
            ResBlock(mid_dim, mid_dim, time_dim),
        ])
        # Up
        self.ups = nn.ModuleList([])
        for i, (di, do) in enumerate(reversed(in_out)):
            attn = with_attn[len(in_out)-1-i]
            self.ups.append(nn.ModuleList([
                ResBlock(do*2, di, time_dim),
                ResBlock(di, di, time_dim),
                Attention(di) if attn else nn.Identity(),
                nn.ConvTranspose2d(di, di, 4, 2, 1),  # upsample
            ]))
        self.final = nn.Sequential(
            Block(dim, dim), nn.Conv2d(dim, channels, 1)
        )
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        hs = []
        for res1, res2, attn, down in self.downs:
            x = res1(x, t); x = res2(x, t); x = attn(x)
            hs.append(x)
            x = down(x)
        for midlayer in self.mid:
            x = midlayer(x, t) if isinstance(midlayer, ResBlock) else midlayer(x)
        for res1, res2, attn, up in self.ups:
            x = torch.cat([x, hs.pop()], dim=1)
            x = res1(x, t); x = res2(x, t); x = attn(x)
            x = up(x)
        return self.final(x)

# ----------------------------
# Exponential Moving Average
# ----------------------------
class EMA:
    def __init__(self, model, beta=0.9999):
        self.model = model; self.beta = beta
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.beta).add_(v, alpha=1-self.beta)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# ----------------------------
# Diffusion helper
# ----------------------------
class Diffusion(nn.Module):
    def __init__(self, unet: UNet, img_size=256, channels=1, timesteps=1000, device='cuda'):
        super().__init__()
        self.unet = unet
        self.img_size = img_size
        self.channels = channels
        self.device = device
        self.N = timesteps
        self.sch = CosineSchedule(DiffusionConfig(timesteps=timesteps))
        # register buffers
        self.register_buffer('betas', self.sch.betas)
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - self.betas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:]))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.unet(xt, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n=16, ddim=False, ddim_steps=50):
        self.eval()
        x = torch.randn(n, self.channels, self.img_size, self.img_size, device=self.device)
        if not ddim:
            # ancestral sampling
            for i in reversed(range(self.N)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                eps = self.unet(x, t)
                a_t = self.alphas[i]; ac_t = self.alphas_cumprod[i]
                sqrt_one_minus_ac = torch.sqrt(1-ac_t)
                mean = (1.0/torch.sqrt(a_t))*(x - (1-a_t)/sqrt_one_minus_ac * eps)
                if i > 0:
                    var = self.betas[i]
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean
        else:
            # DDIM sampling (fast)
            ts = torch.linspace(self.N-1, 0, steps=ddim_steps, device=self.device).long()
            for i, ti in enumerate(ts):
                t = torch.full((n,), ti, device=self.device, dtype=torch.long)
                eps = self.unet(x, t)
                ac_t = self.alphas_cumprod[ti]
                ac_prev = self.alphas_cumprod[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=self.device)
                x0_pred = (x - torch.sqrt(1-ac_t)*eps) / torch.sqrt(ac_t)
                dir_xt = torch.sqrt(1-ac_prev) * eps
                x = torch.sqrt(ac_prev)*x0_pred + dir_xt
        return x.clamp(-1, 1)

# ----------------------------
# Train / Sample entrypoints
# ----------------------------

def save_image_grid(x, path, nrow=8):
    # x in [-1,1]
    x = (x + 1)/2
    x = x.clamp(0,1)
    vutils.save_image(x, path, nrow=nrow)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = ImgFolder(args.data_root, size=args.image_size, channels=args.channels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dl_iter = cycle(dl)

    unet = UNet(channels=args.channels, dim=args.model_dim,
                dim_mults=tuple(args.dim_mults), with_attn=(False, True, False, False))
    unet.to(device)
    diffusion = Diffusion(unet, img_size=args.image_size, channels=args.channels, timesteps=args.timesteps, device=device).to(device)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    ema = EMA(unet, beta=0.9999)

    os.makedirs(args.out_dir, exist_ok=True)
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.out_dir)

    steps = args.steps
    log_every = args.log_every
    save_every = args.save_every

    unet.train()
    for step in range(1, steps+1):
        x = next(dl_iter).to(device)
        # 归一化到 [-1,1]
        x = x * 2 - 1
        t = torch.randint(0, diffusion.N, (x.size(0),), device=device).long()
        loss = diffusion.p_losses(x, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        ema.update()

        if step % log_every == 0:
            print(f"[Step {step}/{steps}] loss={loss.item():.6f}")
            if tb is not None:
                tb.add_scalar("train/loss", loss.item(), step)

        if step % save_every == 0 or step == steps:
            # 采样 & 保存权重
            ema.copy_to(unet)
            with torch.no_grad():
                samples = diffusion.sample(n=args.sample_n, ddim=args.ddim, ddim_steps=args.ddim_steps)
            save_image_grid(samples, os.path.join(args.out_dir, f"sample_{step:06d}.png"), nrow=int(args.sample_n**0.5))
            torch.save({
                'unet': unet.state_dict(),
                'ema': ema.shadow,
                'cfg': vars(args)
            }, os.path.join(args.out_dir, f"ckpt_{step:06d}.pt"))
    # final save
    torch.save({
        'unet': unet.state_dict(), 'ema': ema.shadow, 'cfg': vars(args)
    }, os.path.join(args.out_dir, f"ckpt_final.pt"))


def sample(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    channels = cfg.get('channels', args.channels)
    image_size = cfg.get('image_size', args.image_size)

    unet = UNet(channels=channels, dim=args.model_dim,
                dim_mults=tuple(args.dim_mults), with_attn=(False, True, False, False))
    unet.load_state_dict(ckpt['unet'], strict=False)
    unet.to(device).eval()

    diffusion = Diffusion(unet, img_size=image_size, channels=channels, timesteps=args.timesteps, device=device).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        x = diffusion.sample(n=args.num, ddim=args.ddim, ddim_steps=args.ddim_steps)
    save_image_grid(x, os.path.join(args.out_dir, f"grid_{args.num}.png"), nrow=int(args.num**0.5))
    # 逐张输出
    for i in range(args.num):
        save_image_grid(x[i:i+1], os.path.join(args.out_dir, f"img_{i:04d}.png"), nrow=1)

# ----------------------------
# CLI
# ----------------------------

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    pt = sub.add_parser('train')
    pt.add_argument('--data_root', type=str, required=True)
    pt.add_argument('--out_dir', type=str, default='./runs_vessel')
    pt.add_argument('--image_size', type=int, default=256)
    pt.add_argument('--channels', type=int, default=1, choices=[1,3])
    pt.add_argument('--batch_size', type=int, default=32)
    pt.add_argument('--steps', type=int, default=12000)
    pt.add_argument('--timesteps', type=int, default=1000)
    pt.add_argument('--lr', type=float, default=1e-4)
    pt.add_argument('--model_dim', type=int, default=128)
    pt.add_argument('--dim_mults', type=int, nargs='+', default=[1,2,2,4])
    pt.add_argument('--tensorboard', action='store_true')
    pt.add_argument('--log_every', type=int, default=100)
    pt.add_argument('--save_every', type=int, default=1000)
    pt.add_argument('--sample_n', type=int, default=16)
    pt.add_argument('--ddim', type=int, default=1)
    pt.add_argument('--ddim_steps', type=int, default=50)

    # sample
    ps = sub.add_parser('sample')
    ps.add_argument('--ckpt', type=str, required=True)
    ps.add_argument('--out_dir', type=str, default='./samples_vessel_min')
    ps.add_argument('--num', type=int, default=16)
    ps.add_argument('--image_size', type=int, default=256)
    ps.add_argument('--channels', type=int, default=1)
    ps.add_argument('--timesteps', type=int, default=1000)
    ps.add_argument('--model_dim', type=int, default=128)
    ps.add_argument('--dim_mults', type=int, nargs='+', default=[1,2,2,4])
    ps.add_argument('--ddim', type=int, default=1)
    ps.add_argument('--ddim_steps', type=int, default=50)

    return p

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    else:
        sample(args)
