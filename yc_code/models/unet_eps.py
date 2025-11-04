# models/unet_eps.py
# UNet epsilon predictor for diffusion models (DDPM/DDIM)
# - Stable sinusoidal timestep embedding (log-spaced, * 2π)
# - ResBlocks with GroupNorm + FiLM(time): scale-shift on normalized activations
# - Optional mid self-attention
# - I/O:
#     x_t in [-1, 1], shape [B, C, H, W]
#     t   as float/long tensor shape [B]
# - Guarantee: output channels == input channels (in_ch)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Stable Timestep Embedding ----------
class SinusoidalPosEmb(nn.Module):
    """
    Stable sinusoidal embedding for diffusion timesteps.
    log-uniform frequencies in [1, 10000], multiplied by 2π for better coverage.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim > 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (float or long)
        if t.dtype in (torch.int32, torch.int64, torch.long):
            t = t.float()
        b = t.shape[0]
        device = t.device
        half = self.dim // 2

        # log-spaced freqs and 2π scaling (improves numerical behavior)
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
        args = t[:, None] * freqs[None, :] * (2.0 * math.pi)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)                      # pad to odd dim
        return emb  # [B, dim]


# ---------- Building blocks ----------
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


class ResBlockFiLM(nn.Module):
    """
    ResBlock with GroupNorm and FiLM time modulation:
      - apply FiLM (gamma, beta) *after* first GroupNorm and *before* activation.
      - to_t maps time embedding -> 2*out_ch (gamma, beta).
    """
    def __init__(self, in_ch, out_ch, t_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        g1 = min(groups, out_ch)
        g2 = min(groups, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.gn1   = nn.GroupNorm(num_groups=g1, num_channels=out_ch)
        self.act1  = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.gn2   = nn.GroupNorm(num_groups=g2, num_channels=out_ch)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if t_emb_dim is not None:
            self.to_t = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_ch * 2)  # -> gamma, beta
            )
        else:
            self.to_t = None

    def forward(self, x, t_emb=None):
        # first conv + norm
        h = self.conv1(x)
        h = self.gn1(h)

        # FiLM on normalized activations
        if self.to_t is not None and t_emb is not None:
            tb = self.to_t(t_emb)                        # [B, 2*out_ch]
            gamma, beta = tb.chunk(2, dim=1)            # [B, out_ch], [B, out_ch]
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        h = self.act1(h)

        # second conv
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act2(h)
        h = self.drop2(h)

        return h + self.skip(x)


# ---------- UNet ε-predictor ----------
class UNetEps(nn.Module):
    """
    A lightweight UNet to predict noise epsilon for diffusion models.

    Args:
        in_ch:         input channels (1 for grayscale, 3 for RGB)
        base:          base channel width
        time_dim:      dimension of timestep embedding MLP
        with_mid_attn: whether to enable self-attention at bottleneck
        groups:        GroupNorm groups
        dropout:       dropout inside ResBlocks (usually 0 for med images)
    """
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        time_dim: int = 256,
        with_mid_attn: bool = True,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch = in_ch

        # time embedding MLP (kept within time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # encoder
        self.inc   = nn.Conv2d(in_ch, base, 3, 1, 1)

        self.down1 = ResBlockFiLM(base, base * 2, t_emb_dim=time_dim, groups=groups, dropout=dropout)
        self.pool1 = nn.Conv2d(base * 2, base * 2, 3, 2, 1)  # stride2 downsample (H/2, W/2)

        self.down2 = ResBlockFiLM(base * 2, base * 4, t_emb_dim=time_dim, groups=groups, dropout=dropout)
        self.pool2 = nn.Conv2d(base * 4, base * 4, 3, 2, 1)  # (H/4, W/4)

        # bottleneck
        self.mid1      = ResBlockFiLM(base * 4, base * 4, t_emb_dim=time_dim, groups=groups, dropout=dropout)
        self.mid_attn  = SelfAttention2d(base * 4) if with_mid_attn else nn.Identity()
        self.mid2      = ResBlockFiLM(base * 4, base * 4, t_emb_dim=time_dim, groups=groups, dropout=dropout)

        # decoder
        self.up2  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResBlockFiLM(base * 4 + base * 4, base * 2, t_emb_dim=time_dim, groups=groups, dropout=dropout)

        self.up1  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ResBlockFiLM(base * 2 + base * 2, base, t_emb_dim=time_dim, groups=groups, dropout=dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(groups, base), num_channels=base),
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
        x0 = self.inc(x)                                    # [B, b,  H,   W]
        e1 = self.down1(x0, t_emb)                          # [B, 2b, H,   W]
        p1 = self.pool1(e1)                                 # [B, 2b, H/2, W/2]

        e2 = self.down2(p1, t_emb)                          # [B, 4b, H/2, W/2]
        p2 = self.pool2(e2)                                 # [B, 4b, H/4, W/4]

        # bottleneck
        m = self.mid1(p2, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        # decoder
        u2 = self.up2(m)                                    # [B, 4b, H/2, W/2]
        d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb)   # -> [B, 2b, H/2, W/2]

        u1 = self.up1(d2)                                   # [B, 2b, H,   W]
        d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb)   # -> [B, b,  H,  W]

        out = self.out(d1)                                  # [B, C=in_ch, H, W]
        return out


# quick sanity check
if __name__ == "__main__":
    for c in (1, 3):
        net = UNetEps(in_ch=c, base=64, time_dim=256, with_mid_attn=True)
        x = torch.randn(2, c, 256, 256)
        t = torch.randint(0, 1000, (2,))
        y = net(x, t)
        print(f"in_ch={c} -> out shape:", y.shape)  # expect: [2, c, 256, 256]
        assert y.shape[1] == c
