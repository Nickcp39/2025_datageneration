# models/unet_eps.py
# UNet epsilon predictor for diffusion models (DDPM/DDIM)
# - Sinusoidal timestep embedding
# - ResBlocks with GroupNorm + SiLU
# - Optional self-attention at bottleneck
# - Input:  x_t  in [-1, 1], shape [B, C, H, W]
# - Timestep: t as float/long tensor shape [B]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Timestep embedding ----------
class SinusoidalPosEmb(nn.Module):
    """Classic sinusoidal embedding used for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (float or long)
        if t.dtype in (torch.long, torch.int32, torch.int64):
            t = t.float()
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb  # [B, dim]


# ---------- Building blocks ----------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=None, groups=8):
        super().__init__()
        self.block1 = ConvGNAct(in_ch, out_ch, groups)
        self.block2 = ConvGNAct(out_ch, out_ch, groups)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if t_emb_dim is not None:
            self.to_t = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        else:
            self.to_t = None

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if self.to_t is not None and t_emb is not None:
            h = h + self.to_t(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Simple spatial self-attention block."""
    def __init__(self, channels, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_qkv = nn.Conv2d(channels, inner * 3, 1, bias=False)
        self.proj = nn.Conv2d(inner, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.heads, -1, h * w) for t in qkv]
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)          # [B, H, HW, HW]
        attn = attn.softmax(dim=-1)
        out = attn @ v                           # [B, H, HW, DH]
        out = out.reshape(b, -1, h, w)
        return self.proj(out)


# ---------- UNet ε-predictor ----------
class UNetEps(nn.Module):
    """
    A lightweight UNet used to predict noise epsilon for diffusion models.

    Args:
        in_ch:    input channels (1 for grayscale, 3 for pseudo-RGB)
        base:     base channel width
        time_dim: dimension of timestep embedding MLP
        with_mid_attn: whether to enable self-attention at bottleneck
    """
    def __init__(self, in_ch=1, base=64, time_dim=256, with_mid_attn=True):
        super().__init__()
        self.in_ch = in_ch
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),   # ✅ 输出维度 = time_dim
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )


        # encoder
        self.inc = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.down1 = ResBlock(base, base * 2, t_emb_dim=time_dim)
        self.pool1 = nn.Conv2d(base * 2, base * 2, 3, 2, 1)  # 1/2

        self.down2 = ResBlock(base * 2, base * 4, t_emb_dim=time_dim)
        self.pool2 = nn.Conv2d(base * 4, base * 4, 3, 2, 1)  # 1/4

        # bottleneck
        self.mid1 = ResBlock(base * 4, base * 4, t_emb_dim=time_dim)
        self.mid_attn = SelfAttention2d(base * 4) if with_mid_attn else nn.Identity()
        self.mid2 = ResBlock(base * 4, base * 4, t_emb_dim=time_dim)

        # decoder
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResBlock(base * 4 + base * 4, base * 2, t_emb_dim=time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ResBlock(base * 2 + base * 2, base, t_emb_dim=time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 1),
        )

    def forward(self, x, t):
        """
        x: [-1,1], shape [B, C, H, W]
        t: [B] (float or long)
        return: predicted epsilon (same shape as x)
        """
        t_emb = self.time_mlp(t)

        # encoder
        x0 = self.inc(x)                               # [B, base, H, W]
        e1 = self.down1(x0, t_emb)                     # [B, 2b, H, W]
        p1 = self.pool1(e1)                            # [B, 2b, H/2, W/2]

        e2 = self.down2(p1, t_emb)                     # [B, 4b, H/2, W/2]
        p2 = self.pool2(e2)                            # [B, 4b, H/4, W/4]

        # bottleneck
        m = self.mid1(p2, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        # decoder
        u2 = self.up2(m)                               # [B, 4b, H/2, W/2]
        d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb)  # -> [B, 2b, H/2, W/2]

        u1 = self.up1(d2)                              # [B, 2b, H, W]
        d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb)  # -> [B, b, H, W]

        out = self.out(d1)                             # [B, C, H, W]
        return out


# quick sanity check
if __name__ == "__main__":
    net = UNetEps(in_ch=1, base=64, time_dim=256, with_mid_attn=True)
    x = torch.randn(2, 1, 256, 256)
    t = torch.randint(0, 1000, (2,))
    y = net(x, t)
    print("out:", y.shape)  # 期望: torch.Size([2, 1, 256, 256])

