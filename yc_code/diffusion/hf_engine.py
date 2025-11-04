import torch, torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

class HFEngine(nn.Module):
    def __init__(self, unet, timesteps=1000, schedule='ddpm'):
        super().__init__()
        self.unet = unet
        self.ddpm = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")
        self.ddim = DDIMScheduler(num_train_timesteps=timesteps)  # 推理可选
        self.timesteps = timesteps
        self.schedule = schedule

    # 训练：用调度器加噪
    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.ddpm.add_noise(x0, noise, t)
        pred = self.unet(xt, t).sample   # 预测 ε
        return torch.nn.functional.mse_loss(pred, noise)

    # 采样：统一交给调度器 step
    @torch.no_grad()
    def sample(self, n=16, method='ddpm', ddim_steps=50, device=None, channels=1, image_size=256):
        device = device or next(self.parameters()).device
        latents = torch.randn(n, channels, image_size, image_size, device=device)
        if method == 'ddim':
            sch = self.ddim; sch.set_timesteps(ddim_steps, device=device)
        else:
            sch = self.ddpm; sch.set_timesteps(self.timesteps, device=device)

        for t in sch.timesteps:
            noise_pred = self.unet(latents, t).sample
            latents = sch.step(noise_pred, t, latents).prev_sample
        return latents.clamp(-1, 1)
