# diffusion/diffusion_engine.py
# Cosine schedule + DiffusionEngine (q_sample, loss, DDPM/DDIM sampling) + EMA

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# EMA (Exponential Moving Average)
# ----------------------------
class EMA:
    """Keep an exponential moving average of model parameters to improve sampling quality."""
    def __init__(self, model: nn.Module, beta: float = 0.9999):
        self.beta = beta
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            self.shadow[k].mul_(self.beta).add_(v, alpha=1 - self.beta)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# ----------------------------
# Cosine schedule (Nichol & Dhariwal, s=0.008)
# ----------------------------
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    s: float = 0.008  # small offset to avoid alpha_bar = 0


class CosineSchedule:
    """Generates betas via cosine alpha_bar schedule."""
    def __init__(self, cfg: DiffusionConfig):
        N, s = cfg.timesteps, cfg.s
        # t grid: 0..N
        t = torch.linspace(0, N, N + 1, dtype=torch.float32)
        # cosine squared schedule (normalized by value at t=0)
        def f(x):
            return torch.cos(((x / N) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f(t) / f(torch.tensor(0.0))
        alpha_bar = alpha_bar.clamp(min=1e-8, max=1.0)

        # betas from alpha_bar
        # alpha_t = alpha_bar[t] / alpha_bar[t-1]
        alphas = alpha_bar[1:] / alpha_bar[:-1]
        betas = (1.0 - alphas).clamp(min=1e-8, max=0.999)
        self.betas = betas  # [N]


# ----------------------------
# Helper: gather by t as [B,1,1,1]
# ----------------------------
def _gather_coef(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x: [T], t: [B] -> [B,1,1,1]
    return x.gather(0, t).view(-1, 1, 1, 1)


# ----------------------------
# Diffusion Engine
# ----------------------------
class DiffusionEngine(nn.Module):
    """
    Wraps a noise predictor (εθ) and provides:
      - q_sample: forward noising
      - p_losses: training loss (MSE on noise)
      - sample: DDPM or DDIM sampling

    Expectation:
      - model(x_t, t) predicts noise with same shape as x_t
      - Inputs to model should be in [-1, 1]
      - Training script将 [0,1] 转为 [-1,1]：x = x*2 - 1
      - 采样输出会在外部再反归一化为 [0,1]
    """
    def __init__(self, model: nn.Module, img_size: int, channels: int,
                 timesteps: int = 1000, device: Optional[str] = None):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.channels = channels
        self.T = timesteps

        # device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # schedule buffers
        betas = CosineSchedule(DiffusionConfig(timesteps=timesteps)).betas.to(self.device)  # [T]
        alphas = 1.0 - betas                                                                 # [T]
        alpha_bar = torch.cumprod(alphas, dim=0)                                             # [T]

        # register as buffers for safe .to() etc.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))

        # posterior variance for DDPM sampling
        # Var[q(x_{t-1} | x_t, x_0)] = beta_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer('posterior_var', posterior_var)  # [T]

    # ------------- forward diffusion -------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Sample x_t ~ q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        x0: [-1,1], [B,C,H,W]
        t:  [B] long
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _gather_coef(self.sqrt_alpha_bar, t)
        sqrt_om = _gather_coef(self.sqrt_one_minus_alpha_bar, t)
        return sqrt_ab * x0 + sqrt_om * noise, noise

    # ------------- training loss -------------
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Predict noise and compute MSE.
        """
        xt, noise = self.q_sample(x0, t)
        pred = self.model(xt, t.float())
        return F.mse_loss(pred, noise)

    # ------------- sampling (DDPM / DDIM) -------------
    @torch.no_grad()
    def sample(self, n: int = 16, method: str = 'ddim', ddim_steps: int = 50) -> torch.Tensor:
        """
        Return x in [-1,1], shape [n, C, H, W].
        method: 'ddpm' or 'ddim'
        """
        shape = (n, self.channels, self.img_size, self.img_size)
        x = torch.randn(shape, device=self.device)

        if method.lower() == 'ddpm':
            # ancestral sampling
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                eps = self.model(x, t.float())
                a_t = self.alphas[i]
                ab_t = self.alpha_bar[i]
                sqrt_recip_a = torch.sqrt(1.0 / a_t)
                sqrt_one_minus_ab = torch.sqrt(1.0 - ab_t)

                # mean of posterior q(x_{t-1} | x_t, x_0) with predicted noise
                x = sqrt_recip_a * (x - (1 - a_t) / (sqrt_one_minus_ab + 1e-8) * eps)

                if i > 0:
                    var = self.posterior_var[i].clamp(min=1e-20)
                    x = x + torch.sqrt(var) * torch.randn_like(x)

            return x.clamp(-1, 1)

        # DDIM (deterministic fast sampling)
        steps = int(ddim_steps)
        ts = torch.linspace(self.T - 1, 0, steps=steps, device=self.device).long()
        for si, ti in enumerate(ts):
            t = torch.full((n,), ti.item(), device=self.device, dtype=torch.long)
            eps = self.model(x, t.float())
            ab_t = self.alpha_bar[ti]
            ab_prev = self.alpha_bar[ts[si + 1]] if si < steps - 1 else torch.tensor(1.0, device=self.device)

            x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps) / (torch.sqrt(ab_t) + 1e-8)
            dir_xt = torch.sqrt(1.0 - ab_prev) * eps
            x = torch.sqrt(ab_prev) * x0_pred + dir_xt

        return x.clamp(-1, 1)
