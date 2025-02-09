import torch
import torch.nn as nn

from typing import Union


class DiffusionModel(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.generator = (
            generator  # your noise predictor, e.g. a UNet that takes (x, t)
        )
        self.timesteps = timesteps
        self.register_buffer("beta", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        # t is a tensor of shape [batch_size] with timestep indices
        sqrt_alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_t)

        sqrt_alpha = self.alpha[t].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus_alpha = (1 - self.alpha[t]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha * x_t + sqrt_one_minus_alpha * noise

    def forward(
        self, x0: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # model predicts noise given x_t and t
        pred_noise = self.generator(x_t, t)
        return pred_noise, noise

    def inference(
        self,
        noise: torch.Tensor,
        cond: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        for i in range(self.timesteps - 1, -1, -1):
            pred_noise = self.generator(noise, i)
            noise = self.q_sample(noise, i, pred_noise)
        return noise
