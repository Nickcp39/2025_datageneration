# unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Sinusoidal time embedding ----------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] (int/long/float) -> [B, dim]
    """
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / max(half, 1)))
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# --------- Basic Blocks ----------
class ConvGNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=32):
        super().__init__()
        self.gn  = nn.GroupNorm(num_groups=min(groups, c_in), num_channels=c_in)
        self.act = nn.SiLU()
        self.conv= nn.Conv2d(c_in, c_out, k, s, p)

    def forward(self, x):
        return self.conv(self.act(self.gn(x)))

class ResBlock(nn.Module):
    """
    ResBlock with time embedding injection.
    in: c_in -> out: c_out
    """
    def __init__(self, c_in, c_out, t_dim, groups=32, dropout=0.0):
        super().__init__()
        self.conv1 = ConvGNAct(c_in, c_out, 3, 1, 1, groups=groups)
        self.temb  = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, c_out))
        self.conv2 = ConvGNAct(c_out, c_out, 3, 1, 1, groups=groups)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.temb(t_emb).unsqueeze(-1).unsqueeze(-2)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, 2)
    def forward(self, x):
        return self.up(x)

# --------- UNet ε-prediction ----------
class UNetEps(nn.Module):
    """
    Minimal UNet for epsilon prediction.
    Args:
      in_ch:  输入通道（灰度=1）
      base:   最底层通道基数
      mult:   每层通道倍率，如 (1,2,2,4)
      t_dim:  时间嵌入维度
      num_res:每层 ResBlock 个数
    """
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        mult=(1, 2, 2, 4),
        t_dim: int = 256,
        num_res: int = 2,
        groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch  = in_ch
        self.base   = base
        self.mult   = mult
        self.t_dim  = t_dim
        self.num_res= num_res

        chs = [base * m for m in mult]

        # time embedding MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # input stem
        self.in_conv = nn.Conv2d(in_ch, chs[0], kernel_size=3, stride=1, padding=1)

        # ---------------- down path ----------------
        downs = []
        in_c = chs[0]
        self.down_skips = []
        for li, out_c in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(num_res):
                blocks.append(ResBlock(in_c, out_c, t_dim, groups=groups, dropout=dropout))
                in_c = out_c
            downs.append(blocks)
            self.down_skips.append(out_c)
            if li < len(chs) - 1:
                downs.append(Downsample(in_c, chs[li + 1]))
                in_c = chs[li + 1]
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock(in_c, in_c, t_dim, groups=groups, dropout=dropout)
        self.mid2 = ResBlock(in_c, in_c, t_dim, groups=groups, dropout=dropout)

        # ---------------- up path（修正：首块 concat，后续不再 concat） ----------------
        ups = []
        up_in = in_c
        for li in reversed(range(len(chs))):
            out_c = chs[li]
            # 第一个 up 层用上采样，其最高层用 Identity
            ups.append(Upsample(up_in, out_c) if li < len(chs) - 1 else nn.Identity())

            blocks = nn.ModuleList()
            for k in range(num_res):
                in_channels = (out_c + out_c) if k == 0 else out_c  # 只有第一个 block 看到 concat 后的通道数
                blocks.append(ResBlock(in_channels, out_c, t_dim, groups=groups, dropout=dropout))
            ups.append(blocks)

            up_in = out_c
        self.ups = nn.ModuleList(ups)

        # output head
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(groups, chs[0]), num_channels=chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_ch, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_requires_grad(self, flag: bool):
        for p in self.parameters():
            p.requires_grad = flag

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C=in_ch, H, W] in [-1,1]
        t: [B] int/long/float in [0, T-1]
        return: eps_hat same shape as x
        """
        # time embedding
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))

        # down path
        feats = []
        h = self.in_conv(x)
        i = 0
        while i < len(self.downs):
            blocks: nn.ModuleList = self.downs[i]
            for block in blocks:
                h = block(h, t_emb)
            feats.append(h)  # skip
            i += 1
            if i < len(self.downs) and isinstance(self.downs[i], Downsample):
                h = self.downs[i](h)
                i += 1

        # mid
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # up path —— 每层消费两项：[Upsample/Identity] + [blocks]；且只 concat 一次
        j = 0
        skip_idx = len(feats) - 1
        while j < len(self.ups):
            up_op = self.ups[j]; j += 1
            if isinstance(up_op, Upsample):
                h = up_op(h)

            blocks: nn.ModuleList = self.ups[j]; j += 1
            skip = feats[skip_idx]; skip_idx -= 1

            if skip.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")

            # 仅在进入第一个 block 前 concat 一次
            h = torch.cat([h, skip], dim=1)

            for bi, block in enumerate(blocks):
                h = block(h, t_emb)
                # 第一个 block 输出通道为 out_c，后续 block 的输入正是 out_c，无需再次 concat

        eps = self.out(h)
        return eps
