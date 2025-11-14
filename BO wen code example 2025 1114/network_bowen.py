# network.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 一些小工具函数 =====

def default(val, d):
    return val if val is not None else d


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    从长度为 T 的向量 a 中根据 t (B,) 取出对应元素，并 reshape 成 [B,1,1,1] 方便 broadcast
    """
    # a: [T], t: [B]
    b = t.shape[0]
    out = a.gather(-1, t)  # [B]
    return out.view(b, *((1,) * (len(x_shape) - 1)))


# ===== 时间 embedding =====

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] 的时间步整数，先转 float 再做 sin/cos 编码
        输出: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)  # [half]
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)            # [B,half]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)   # [B,2*half]
        return emb


# ===== 基本 Block: ResBlock + 时间注入 =====

class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 time_emb_dim: int,
                 groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        t_emb: [B, time_emb_dim]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 将时间 embedding 注入到通道上
        t = self.time_mlp(t_emb)              # [B,out_ch]
        t = t[:, :, None, None]               # [B,out_ch,1,1]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


# ===== UNet 主体 =====

class UNetModel(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 base_ch: int = 64,
                 ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
                 num_res_blocks: int = 2,
                 time_emb_dim: int = 256):
        super().__init__()

        self.in_ch = in_ch
        self.base_ch = base_ch

        # 时间 embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 输入卷积
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Down path
        in_chs = [base_ch]
        downs = []
        ch_in = base_ch
        self.num_levels = len(ch_mult)
        for i, mult in enumerate(ch_mult):
            ch_out = base_ch * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock(ch_in, ch_out, time_emb_dim))
                ch_in = ch_out
                in_chs.append(ch_in)
            if i != len(ch_mult) - 1:
                downs.append(Downsample(ch_in))
                in_chs.append(ch_in)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.mid_block1 = ResBlock(ch_in, ch_in, time_emb_dim)
        self.mid_block2 = ResBlock(ch_in, ch_in, time_emb_dim)

        # Up path
        ups = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            ch_out = base_ch * mult
            for _ in range(num_res_blocks + 1):
                # 每次上采样会 concat skip, 通道变多
                ups.append(ResBlock(ch_in + in_chs.pop(), ch_out, time_emb_dim))
                ch_in = ch_out
            if i != 0:
                ups.append(Upsample(ch_in))
        self.ups = nn.ModuleList(ups)

        # 输出卷积
        self.out_norm = nn.GroupNorm(8, ch_in)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch_in, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W], t: [B] int64
        返回预测的 noise ε
        """
        # 时间 embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # 下采样
        h = self.in_conv(x)
        hs = [h]
        for module in self.downs:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # 上采样
        for module in self.ups:
            if isinstance(module, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        # 输出
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h


# ===== Gaussian Diffusion 封装 =====

class GaussianDiffusion(nn.Module):
    """
    标准 DDPM 封装：负责 q_sample, p_losses, sample 等
    """

    def __init__(self,
                 model: nn.Module,
                 image_size: int = 256,
                 timesteps: int = 1000,
                 beta_1: float = 1e-4,
                 beta_T: float = 2e-2,
                 device: str = "cuda"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.device = device

        # 线性 beta schedule
        betas = torch.linspace(beta_1, beta_T, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ----------------- 前向扩散 q(x_t | x_0) -----------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    # ----------------- 训练损失 -----------------

    def p_losses(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        标准 epsilon-prediction MSE 损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_pred = self.model(x_noisy, t)

        loss = F.mse_loss(eps_pred, noise)
        return loss

    # ----------------- 反向采样 p(x_{t-1} | x_t) -----------------

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        使用公式：
        x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta )
                  + sigma_t * z
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        eps_theta = self.model(x, t)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps_theta / sqrt_one_minus_alpha_bar_t)

        # t == 0 时不加噪声
        posterior_var_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape) -> torch.Tensor:
        """
        反向迭代 T 步
        """
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)

        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16) -> torch.Tensor:
        """
        从纯噪声采样生成图像
        输出: [B,1,H,W] in [-1,1]
        """
        shape = (batch_size, 1, self.image_size, self.image_size)
        return self.p_sample_loop(shape)
