import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class DownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        skip = x
        if self.pool:
            x = self.pool(x)
        return x, skip


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad x if needed to match skip dimensions
        if x.size() != skip.size():
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditional_channels: int = 0,
        depth: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        # Down path
        self.down_path = nn.ModuleList()
        prev_ch = in_channels + conditional_channels
        ch = base_channels
        for _ in range(depth):
            self.down_path.append(DownConvBlock(prev_ch, ch, use_pool=True))
            prev_ch = ch
            ch *= 2

        # Bottom block
        self.bottom = nn.Sequential(
            nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # Up path
        self.up_path = nn.ModuleList()
        for _ in range(depth):
            self.up_path.append(UpConvBlock(ch, ch // 2))
            ch //= 2

        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, cond: Union[None, torch.Tenrsor] = None
    ) -> torch.Tensor:
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        if cond is not None:
            cond = cond.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = torch.cat([x, cond], dim=1)

        skips = []
        for down in self.down_path:
            x, skip = down(x)
            skips.append(skip)
        x = self.bottom(x)
        for up in self.up_path:
            skip = skips.pop()
            x = up(x, skip)
        return self.final_conv(x)
