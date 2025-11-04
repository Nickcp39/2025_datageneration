# code/diffusion/diffusion_engine.py
# Diffusion engine (training + sampling) with numeric guards for stability.
# - Cosine schedule (default) or linear
# - Epsilon prediction objective: MSE(pred_eps, true_eps)
# - FP32 sampling (AMP disabled) to avoid "black images"
# - DDPM / DDIM sampling consistent with eps-prediction
# - Progress logging: x.min/max and eps_pred.min/max

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal, Dict

# 你自己的 UNet
from yc_code.models.unet_eps import UNetEps


def _cosine_schedule(T: int, s: float = 0.008, eps: float = 1e-8):
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = torch.clamp(alphas_cumprod, min=eps, max=1.0)  # 数值保护
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=eps, max=0.999)
    return betas


def _linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2, eps: float = 1e-12):
    betas = torch.linspace(beta_start, beta_end, T)
    betas = torch.clamp(betas, min=eps, max=0.999)
    return betas


class DiffusionEngine(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 1,
        T: int = 1000,
        schedule: Literal["cosine", "linear"] = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        net_base: int = 64,
        time_dim: int = 256,
        with_mid_attn: bool = True,
        eps_guard: float = 1e-8,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.T = T
        self.eps_guard = float(eps_guard)

        # ------- noise schedule -------
        if schedule == "cosine":
            betas = _cosine_schedule(T, s=0.008, eps=eps_guard)
        elif schedule == "linear":
            betas = _linear_schedule(T, beta_start, beta_end, eps=eps_guard)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # 关键 buffers（全部加 eps 防护，避免 sqrt(负数) 或除 0）
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", torch.clamp(alphas, min=eps_guard, max=1.0).float())
        self.register_buffer("alphas_cumprod", torch.clamp(alphas_cumprod, min=eps_guard, max=1.0).float())
        self.register_buffer("alphas_cumprod_prev", torch.clamp(alphas_cumprod_prev, min=eps_guard, max=1.0).float())

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(torch.clamp(self.alphas_cumprod, min=eps_guard)).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - self.alphas_cumprod, min=eps_guard)).float()
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(torch.clamp(1.0 / self.alphas, min=eps_guard)).float())
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(torch.clamp(1.0 / self.alphas - 1.0, min=eps_guard)).float()
        )
        self.register_buffer(
            "posterior_variance",
            torch.clamp(
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + eps_guard),
                min=eps_guard
            ).float()
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=eps_guard)).float()
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (torch.sqrt(torch.clamp(alphas_cumprod_prev, min=eps_guard)) * betas / torch.clamp(1.0 - alphas_cumprod, min=eps_guard)).float()
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (torch.sqrt(torch.clamp(alphas, min=eps_guard)) * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=eps_guard)).float()
        )

        # ------- network -------
        self.net = UNetEps(in_ch=channels, base=net_base, time_dim=time_dim, with_mid_attn=with_mid_attn)

    # -------------------- Forward diffusion helpers --------------------
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_om * noise

    def _extract(self, a, t, x_shape):
        # a: [T] buffer on device/dtype
        # t: [B] (long/float ok)
        if t.dtype != torch.long:
            t = t.long()
        t = t.clamp(0, self.T - 1).to(a.device)        # 非原地
        vals = a.index_select(0, t)                    # [B]
        while vals.dim() < len(x_shape):               # -> [B,1,1,1]
            vals = vals.unsqueeze(-1)
        return vals.expand(x_shape)                    # broadcast 到 x_shape


    # -------------------- Training objective: eps-prediction --------------------
    def p_losses(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        # 预测的是 epsilon
        pred_noise = self.net(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "pred_noise": pred_noise.detach(), "x_noisy": x_noisy.detach()}

    # -------------------- DDPM sampling step (eps-prediction) --------------------
    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, log_every: bool = False) -> Tensor:
        """
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
        sigma_t^2 = posterior_variance_t
        """
        eps = self.net(x_t, t)
        if log_every:
            # 打印统计，易于发现“全零/饱和”
            print(f"[DDPM] t={t[0].item():4d} | x.min={x_t.min().item():+.4f}, x.max={x_t.max().item():+.4f} | "
                  f"eps.min={eps.min().item():+.4f}, eps.max={eps.max().item():+.4f}")

        b = x_t.shape[0]
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 预测 x0（可用于可视化或指导），但主公式使用 eps 形式
        # x0_pred = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)  # 这里不直接用，防止数值不稳

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / torch.clamp(sqrt_one_minus_ab_t, min=self.eps_guard)) * eps)

        # 当 t > 0 才加噪声；t=0 直接返回均值
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().view(b, *([1] * (x_t.dim() - 1)))
        var = self._extract(self.posterior_variance, t, x_t.shape)
        nonzero_term = mask * torch.sqrt(torch.clamp(var, min=self.eps_guard)) * noise
        return mean + nonzero_term

    # -------------------- DDIM sampling (eps-prediction, eta controls stochasticity) --------------------
    @torch.no_grad()
    def ddim_step(self, x_t: Tensor, t: Tensor, t_prev: Tensor, eta: float = 0.0, log_every: bool = False) -> Tensor:
        """
        DDIM update (ε-pred). When eta=0, deterministic.
        """
        eps = self.net(x_t, t)
        if log_every:
            print(f"[DDIM] t={t[0].item():4d}->{t_prev[0].item():4d} | x.min={x_t.min().item():+.4f}, "
                  f"x.max={x_t.max().item():+.4f} | eps.min={eps.min().item():+.4f}, eps.max={eps.max().item():+.4f}")

        a_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        a_bar_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        # 预测 x0
        x0_pred = torch.clamp((x_t - torch.sqrt(torch.clamp(1 - a_bar_t, min=self.eps_guard)) * eps) /
                              torch.sqrt(torch.clamp(a_bar_t, min=self.eps_guard)), min=-1.0, max=1.0)

        # 确定性项
        dir_xt = torch.sqrt(torch.clamp(a_bar_prev, min=self.eps_guard)) * x0_pred
        # 残差项
        sigma_t = eta * torch.sqrt(
            torch.clamp((1 - a_bar_prev) / (1 - a_bar_t), min=self.eps_guard) * (1 - a_bar_t / a_bar_prev)
        )
        noise = torch.randn_like(x_t) if (eta > 0) else 0.0
        x_prev = dir_xt + torch.sqrt(torch.clamp(1 - a_bar_prev, min=self.eps_guard)) * eps + sigma_t * noise
        return x_prev

    # -------------------- Public sampling API --------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        method: Literal["ddpm", "ddim"] = "ddpm",
        ddim_steps: int = 50,
        eta: float = 0.0,
        log_every_n: int = 50,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Returns: x_0 in [-1,1], shape [B,C,H,W]
        Note: sampling is forced to FP32 (no autocast) for stability.
        """
        if device is None:
            device = next(self.parameters()).device

        # 关闭 AMP：全程 FP32
        torch.set_grad_enabled(False)
        prev_autocast = torch.is_autocast_enabled()
        if prev_autocast:
            torch.set_autocast_enabled(False)

        try:
            x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device, dtype=torch.float32)

            if method == "ddpm":
                for i in reversed(range(self.T)):
                    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                    log_now = (i % max(1, log_every_n) == 0)
                    x = self.p_sample(x, t, log_every=log_now)
                x0 = torch.clamp(x, -1.0, 1.0)
                return x0

            elif method == "ddim":
                # 均匀子序列
                ts = torch.linspace(self.T - 1, 0, steps=ddim_steps, device=device).long()
                for si in range(len(ts)):
                    t = ts[si].expand(batch_size)
                    t_prev = ts[si + 1].expand(batch_size) if si + 1 < len(ts) else torch.zeros_like(t)
                    log_now = (si % max(1, log_every_n) == 0)
                    x = self.ddim_step(x, t, t_prev, eta=eta, log_every=log_now)
                x0 = torch.clamp(x, -1.0, 1.0)
                return x0
            else:
                raise ValueError(f"Unknown sample method: {method}")
        finally:
            if prev_autocast:
                torch.set_autocast_enabled(True)

    # -------------------- Utility --------------------
    def forward(self, x_start: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """For training: returns dict with 'loss'."""
        return self.p_losses(x_start, t)
