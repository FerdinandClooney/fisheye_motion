from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, mid_ch: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(gate_ch, mid_ch, 1, bias=False)
        self.skip = nn.Conv2d(skip_ch, mid_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(mid_ch, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.gate(g) + self.skip(x))
        return x * a


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, max(out_ch // 2, 8))
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class FisheyeMotionNet(nn.Module):
    def __init__(self, in_channels: int = 15, base: int = 32) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.u4 = UpBlock(base * 16, base * 8, base * 8)
        self.u3 = UpBlock(base * 8, base * 4, base * 4)
        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.u4(b, e4)
        x = self.u3(x, e3)
        x = self.u2(x, e2)
        x = self.u1(x, e1)
        return self.head(x)
