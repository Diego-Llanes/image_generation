import torch
from torch import nn

from typing import Union


class GAN(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        noise: Union[torch.Tensor, None] = None,
        real_imgs: Union[torch.Tensor, None] = None,
        condition: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        noise: a random tensor to generate fake images that is of shape [N, noise_dim]
        real_image: a real image tensor that is of shape [M, height, width]

        returns: fake_preds, real_preds

        Note: you will need to keep track of which predictions are which externally. You can also just only specify one of the inputs
        and the other as None.
        If noise is not specified, fake_preds will be None. If real_image is not specified, real_preds will be None.
        """
        assert (
            noise is not None or real_imgs is not None
        ), "Either noise or real_image must be provided"

        gen_preds, real_preds = None, None

        if noise is not None:
            fake_images = self.generator(noise, condition)
            gen_preds = self.discriminator(fake_images, condition)

        if real_imgs is not None:
            real_preds = self.discriminator(real_imgs, condition)

        return gen_preds, real_preds

    def inferencer(
        self, noise: torch.Tensor, condition: Union[None, torch.Tensor]
    ) -> torch.Tensor:
        """
        noise: a random tensor to generate fake images that is of shape [N, noise_dim]

        returns: fake_images
        """
        return self.generator(noise, condition)
