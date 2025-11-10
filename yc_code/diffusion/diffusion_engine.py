# yc_code/diffusion/diffusion_engine.py
# ---------------------------------------------------------------------
# DiffusionEngine: 单一调度器 + 训练/采样公式（ε-pred）
#  - 不继承 nn.Module，不绑定 UNet；UNet 在外部脚本创建并传入
#  - 统一 β/ᾱ 表（cosine 默认，可选 linear），所有公式共用同一真源
#  - 训练目标：MSE(pred_eps, true_eps)
#  - 采样：DDPM / DDIM（eta=0 默认确定性），严格 ε-pred 一致
#  - 数值护栏：对 sqrt/除法/对数等加入 clamp(eps)
#  - 采样强制 FP32（不启用 AMP），避免黑图/漂移
#  - 轻量日志：可选打印 x/eps min/max 便于诊断“全零/饱和”
# ---------------------------------------------------------------------

import math
from typing import Optional, Literal, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
# ------- Model & Engine -------
from models.unet_eps import UNetEps


# ------------------------- schedules -------------------------

def _cosine_schedule(T: int, s: float = 0.008, eps: float = 1e-8) -> Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021), aka squaredcos_cap_v2。
    返回: betas, shape [T]
    """
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = torch.clamp(alphas_cumprod, min=eps, max=1.0)
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=eps, max=0.999)
    return betas.float()


def _linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2, eps: float = 1e-12) -> Tensor:
    """
    线性 beta 表，返回 betas, shape [T]
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    betas = torch.clamp(betas, min=eps, max=0.999)
    return betas.float()


# ------------------------- engine -------------------------

class DiffusionEngine:
    """
    单一扩散引擎（不继承 nn.Module，不持有 UNet）。
    外部脚本需自行:
      model = UNetEps(...)
      out = engine.p_losses(model, x_start, t)
      x0  = engine.sample(model, batch_size, method="ddim", ddim_steps=50)
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        channels: int = 1,
        T: int = 1000,
        schedule: Literal["cosine", "linear"] = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        eps_guard: float = 1e-8,
    ):
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.T = int(T)
        self.eps_guard = float(eps_guard)

        # --------- schedule (唯一真源) ---------
        if schedule == "cosine":
            betas = _cosine_schedule(self.T, s=0.008, eps=self.eps_guard)
        elif schedule == "linear":
            betas = _linear_schedule(self.T, beta_start=beta_start, beta_end=beta_end, eps=self.eps_guard)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]])

        # --------- 持久张量（作为 engine 状态）---------
        eg = self.eps_guard
        self.betas = betas.float()  # [T]
        self.alphas = torch.clamp(alphas, min=eg, max=1.0).float()  # [T]
        self.alphas_cumprod = torch.clamp(alphas_cumprod, min=eg, max=1.0).float()  # [T]
        self.alphas_cumprod_prev = torch.clamp(alphas_cumprod_prev, min=eg, max=1.0).float()  # [T]

        self.sqrt_alphas_cumprod = torch.sqrt(torch.clamp(self.alphas_cumprod, min=eg)).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - self.alphas_cumprod, min=eg)).float()
        self.sqrt_recip_alphas = torch.sqrt(torch.clamp(1.0 / self.alphas, min=eg)).float()
        self.sqrt_recipm1_alphas = torch.sqrt(torch.clamp(1.0 / self.alphas - 1.0, min=eg)).float()

        # 后验参数（供 DDPM 用）
        self.posterior_variance = torch.clamp(
            self.betas * (1.0 - self.alphas_cumprod_prev) / torch.clamp(1.0 - self.alphas_cumprod, min=eg),
            min=eg
        ).float()
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=eg)).float()

        # 文档目的打印（可由外部日志接管）
        self._meta = dict(
            timesteps=self.T,
            beta_schedule=schedule,
            prediction_type="epsilon",
        )

    # -------------------- device/ dtype helpers --------------------

    def to(self, device: torch.device) -> "DiffusionEngine":
        """
        把内部状态张量移动到 device。注意：Engine 不是 nn.Module。
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def _extract(self, a: Tensor, t: Tensor, x_shape) -> Tensor:
        """
        从长度为 T 的 1D 张量 a 中按索引 t（[B]）取值，并 broadcast 到 x_shape。
        """
        if t.dtype != torch.long:
            t = t.long()
        t = t.clamp(0, self.T - 1)
        vals = a.index_select(0, t)  # [B]
        while len(vals.shape) < len(x_shape):
            vals = vals.unsqueeze(-1)
        return vals.expand(x_shape)

    # -------------------- Forward diffusion --------------------

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
        约定：x_* 均在 [-1, 1]。
        """
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return (sqrt_ab * x_start + sqrt_om * noise).float()

    # -------------------- Training objective: ε-pred --------------------

    def p_losses(
        self,
        model: torch.nn.Module,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        训练损失：MSE(pred_eps, true_eps)
        - x_start: [-1, 1]
        """
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
        x_noisy = self.q_sample(x_start, t, noise)
        # 模型预测的就是 ε̂
        pred_noise = model(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": loss,
            "pred_noise": pred_noise.detach(),
            "x_noisy": x_noisy.detach(),
        }

    # -------------------- DDPM sampling step (ε-pred) --------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: Tensor,
        t: Tensor,
        *,
        log_every: bool = False
    ) -> Tensor:
        """
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-a_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
        其中 sigma_t^2 = posterior_variance_t
        """
        # 预测 ε̂
        eps = model(x_t, t).float()
        if log_every:
            print(f"[DDPM] t={int(t[0]):4d}"
                  f" | x.min={x_t.min().item():+.4f}, x.max={x_t.max().item():+.4f}"
                  f" | eps.min={eps.min().item():+.4f}, eps.max={eps.max().item():+.4f}")

        b = x_t.shape[0]
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / torch.clamp(sqrt_one_minus_ab_t, min=self.eps_guard)) * eps)

        # t > 0 才加噪声
        var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t, dtype=torch.float32)
        mask = (t > 0).float().view(b, *([1] * (x_t.dim() - 1)))
        nonzero_term = mask * torch.sqrt(torch.clamp(var, min=self.eps_guard)) * noise
        return (mean + nonzero_term).float()

    # -------------------- DDIM sampling step (ε-pred) --------------------

    @torch.no_grad()
    def ddim_step(
        self,
        model: torch.nn.Module,
        x_t: Tensor,
        t: Tensor,
        t_prev: Tensor,
        *,
        eta: float = 0.0,
        log_every: bool = False
    ) -> Tensor:
        """
        DDIM（ε-pred）单步：
          x0_pred = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)
          sigma_t^2 = eta^2 * (1 - a_bar_prev)/(1 - a_bar_t) * (1 - a_bar_t/a_bar_prev)
          x_{t-1} = sqrt(a_bar_prev) * x0_pred
                    + sqrt(1 - a_bar_prev - sigma_t^2) * eps
                    + sigma_t * z
        eta=0 -> 确定性 DDIM
        """
        eps = model(x_t, t).float()
        if log_every:
            print(f"[DDIM] t={int(t[0]):4d}->{int(t_prev[0]):4d}"
                  f" | x.min={x_t.min().item():+.4f}, x.max={x_t.max().item():+.4f}"
                  f" | eps.min={eps.min().item():+.4f}, eps.max={eps.max().item():+.4f}")

        eg = self.eps_guard
        a_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        a_bar_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        # 预测 x0，并裁剪到训练域 [-1, 1]
        x0_pred = (x_t - torch.sqrt(torch.clamp(1.0 - a_bar_t, min=eg)) * eps) / torch.sqrt(
            torch.clamp(a_bar_t, min=eg)
        )
        x0_pred = torch.clamp(x0_pred, min=-1.0, max=1.0)

        # dir_xt 确定性主项
        dir_xt = torch.sqrt(torch.clamp(a_bar_prev, min=eg)) * x0_pred

        # 正确的 DDIM 噪声项
        sigma_t_sq = (eta ** 2) * torch.clamp((1.0 - a_bar_prev) / torch.clamp(1.0 - a_bar_t, min=eg), min=eg) * \
                     torch.clamp(1.0 - (a_bar_t / torch.clamp(a_bar_prev, min=eg)), min=0.0)
        sigma_t_sq = torch.clamp(sigma_t_sq, min=0.0)
        sigma_t = torch.sqrt(sigma_t_sq)

        # 残差 eps 前的系数应是 sqrt(1 - a_bar_prev - sigma_t^2)
        coeff_eps = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t_sq, min=0.0))

        if eta > 0:
            noise = torch.randn_like(x_t, dtype=torch.float32)
        else:
            noise = 0.0

        x_prev = dir_xt + coeff_eps * eps + sigma_t * noise
        return x_prev.float()

    # -------------------- Public sampling API --------------------

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        batch_size: int,
        *,
        method: Literal["ddpm", "ddim"] = "ddpm",
        ddim_steps: int = 50,
        eta: float = 0.0,
        log_every_n: int = 50,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        采样返回 x_0 ∈ [-1, 1]，shape [B, C, H, W]
        - 采样强制使用 FP32（不使用 AMP），避免数值发散/黑片
        - method: "ddpm" 或 "ddim"
        """
        # 设备推断：以模型参数为准
        if device is None:
            device = next(model.parameters()).device

        # 初始化 x_T
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size,
                        device=device, dtype=torch.float32)

        if method == "ddpm":
            for i in reversed(range(self.T)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                log_now = (i % max(1, log_every_n) == 0)
                x = self.p_sample(model, x, t, log_every=log_now)
            x0 = torch.clamp(x, -1.0, 1.0)
            return x0.float()

        elif method == "ddim":
            # 均匀子序列
            ts = torch.linspace(self.T - 1, 0, steps=int(ddim_steps), device=device).long()
            for si in range(len(ts)):
                t = ts[si].expand(batch_size)
                t_prev = ts[si + 1].expand(batch_size) if si + 1 < len(ts) else torch.zeros_like(t)
                log_now = (si % max(1, log_every_n) == 0)
                x = self.ddim_step(model, x, t, t_prev, eta=eta, log_every=log_now)
            x0 = torch.clamp(x, -1.0, 1.0)
            return x0.float()

        else:
            raise ValueError(f"Unknown sample method: {method}")

    # -------------------- meta/info --------------------

    @property
    def meta(self) -> Dict[str, str]:
        """
        返回关键信息，便于训练/推理日志记录与复现。
        """
        return dict(self._meta)
