# engine.py
from __future__ import annotations
import math
from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================
# 调度器（β / ᾱ）
# =====================
def linear_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    return betas.clamp_(eps, 1.0 - eps)


def cosine_beta_schedule(
    T: int,
    s: float = 0.008,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    # from "Improved Denoising Diffusion Probabilistic Models"
    steps = T + 1
    t = torch.linspace(0, T, steps, device=device) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp_(eps, 0.999)


# =====================
# 小工具
# =====================
def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    从长度 T 的 1D 张量 a 中，按 batch 索引 t 取值，并 reshape 成 [B,1,1,1] 扩展到 x_shape
    用 index_select，避免 gather + 原地 clamp_ 带来的重叠写入问题。
    """
    # 保证 t 是 1D long，且不原地修改
    t = t.to(dtype=torch.long).view(-1)
    t = t.clamp(0, a.size(0) - 1)
    out = a.index_select(0, t)          # [B]
    return out.view(-1, 1, 1, 1).expand(x_shape)

    


@torch.no_grad()
def randn_like_x0(x0: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x0)


# =====================
# Diffusion 引擎（ε-pred）
# =====================
class DiffusionEngine(nn.Module):
    """
    纯自研扩散引擎：
      - 目标：ε-prediction（训练=ε-MSE；采样严格用 ε 公式）
      - 提供 DDPM 与 DDIM 两种采样
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 512,
        T: int = 1000,
        schedule: Literal["linear", "cosine"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.T = int(T)
        self.device = device if device is not None else next(model.parameters()).device

        if schedule == "linear":
            betas = linear_beta_schedule(T=self.T, beta_start=beta_start, beta_end=beta_end, device=self.device)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T=self.T, device=self.device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}, ᾱ_-1 = 1

        # 注册为 buffer，参与 device/ckpt 同步
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 预计算常用项
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-12))

    # ------------- 前向加噪（训练用） -------------
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε
        返回 (x_t, ε_true)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ac * x0 + sqrt_om * noise
        return x_t, noise

    # ------------- 训练损失（ε-MSE） -------------
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
        """
        E_{x0,t,ε} || ε - ε̂(x_t, t) ||^2
        """
        x_t, noise = self.q_sample(x0, t)
        eps_pred = self.model(x_t, t)
        loss = (noise - eps_pred).pow(2)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    # ------------- DDPM 单步 -------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_{t-1} = 1/sqrt(α_t) [ x_t - β_t / sqrt(1-ᾱ_t) * ε̂ ] + σ_t * z,  t>0
                = 1/sqrt(α_t) [ x_t - β_t / sqrt(1-ᾱ_t) * ε̂ ],          t=0
        σ_t 可取 posterior_variance 的 sqrt（更稳定）
        """
        betas_t = _extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_ac_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        posterior_var_t = _extract(self.posterior_variance, t, x_t.shape)

        eps_pred = self.model(x_t, t)
        # 预测均值 μ_t
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps_pred / (sqrt_one_minus_ac_t + 1e-12))

        if t.ndim == 0:
            nonzero_mask = torch.tensor(0.0, device=x_t.device) if t.item() == 0 else torch.tensor(1.0, device=x_t.device)
            nonzero_mask = nonzero_mask.view(1, 1, 1, 1)
        else:
            nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t)
        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return x_prev.clamp_(-1.0, 1.0)

    # ------------- DDPM 全程采样 -------------
    @torch.no_grad()
    def sample_ddpm(self, batch_size: int, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        从 x_T ~ N(0,I) 开始，逐步推到 x_0
        shape: (C, H, W)
        """
        C, H, W = shape
        x = torch.randn((batch_size, C, H, W), device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp_(-1.0, 1.0)

    # ------------- DDIM 单步 -------------
    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM 确定性/半随机单步（η=0 为确定性）
        公式（ε-pred 版）：
          x0_hat = (x_t - sqrt(1-ᾱ_t)*ε̂) / sqrt(ᾱ_t)
          σ = eta * sqrt( (1-ᾱ_{t_prev})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t_prev}) )
          dir = sqrt(1-ᾱ_{t_prev}) * ε̂
          x_{t_prev} = sqrt(ᾱ_{t_prev})*x0_hat + dir + σ*z
        """
        ac_t = _extract(self.alphas_cumprod, t, x_t.shape)
        ac_prev = _extract(self.alphas_cumprod, t_prev, x_t.shape)
        eps_pred = self.model(x_t, t)

        x0_hat = (x_t - torch.sqrt(1.0 - ac_t) * eps_pred) / torch.sqrt(ac_t + 1e-12)

        # sigma per DDIM
        sigma = (
            eta
            * torch.sqrt((1.0 - ac_prev) / (1.0 - ac_t + 1e-12))
            * torch.sqrt(1.0 - ac_t / (ac_prev + 1e-12))
        )

        dir_xt = torch.sqrt(torch.clamp(1.0 - ac_prev, min=0.0)) * eps_pred
        noise = torch.randn_like(x_t) if (eta > 0.0) else torch.zeros_like(x_t)

        x_prev = torch.sqrt(torch.clamp(ac_prev, min=0.0)) * x0_hat + dir_xt + sigma * noise
        return x_prev.clamp_(-1.0, 1.0)

    # ------------- DDIM 全程采样 -------------
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        shape: Tuple[int, int, int],
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        以 steps 等分 T 的时间网格做 DDIM 采样（eta=0 确定性、速度快）
        """
        C, H, W = shape
        x = torch.randn((batch_size, C, H, W), device=self.device)

        # 构造等分时间表（含 t=0）
        times = torch.linspace(0, self.T - 1, steps + 1, device=self.device).long()
        # 从大到小迭代：t_k -> t_{k-1}
        for i in reversed(range(1, len(times))):
            t = times[i].expand(batch_size)
            t_prev = times[i - 1].expand(batch_size)
            x = self.ddim_step(x, t, t_prev, eta=eta)
        return x.clamp_(-1.0, 1.0)

    # ------------- 便捷接口 -------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        shape: Tuple[int, int, int],
        method: Literal["ddpm", "ddim"] = "ddim",
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if method == "ddpm":
            return self.sample_ddpm(batch_size, shape)
        elif method == "ddim":
            return self.sample_ddim(batch_size, shape, steps=steps, eta=eta)
        else:
            raise ValueError(f"Unknown sample method: {method}")

    @torch.no_grad()
    def reconstruct_one(self, x0: torch.Tensor, t_star: int = 700, method: Literal["ddpm", "ddim"] = "ddim", steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重建自测：对真图 x0 加噪到 t_star，再往回还原，看结构是否能恢复。
        返回 (x_rec, x_t)
        """
        b = x0.shape[0]
        t = torch.full((b,), int(max(0, min(self.T - 1, t_star))), device=x0.device, dtype=torch.long)
        x_t, _ = self.q_sample(x0, t)

        if method == "ddpm":
            x = x_t.clone()
            for i in reversed(range(t_star)):
                ti = torch.full((b,), i, device=x0.device, dtype=torch.long)
                x = self.p_sample(x, ti)
            return x.clamp_(-1, 1), x_t
        else:
            # 把 [0..t_star] 等分为 steps 步
            times = torch.linspace(0, t_star, steps + 1, device=x0.device).long()
            x = x_t.clone()
            for i in reversed(range(1, len(times))):
                tt = times[i].expand(b)
                tt_prev = times[i - 1].expand(b)
                x = self.ddim_step(x, tt, tt_prev, eta=0.0)
            return x.clamp_(-1, 1), x_t
