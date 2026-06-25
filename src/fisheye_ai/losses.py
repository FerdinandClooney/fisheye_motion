from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = torch.sum(prob * target, dims)
        union = torch.sum(prob + target, dims)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=prob.dtype, device=prob.device).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        edge_p = torch.sqrt(F.conv2d(prob, kx, padding=1).pow(2) + F.conv2d(prob, ky, padding=1).pow(2) + 1e-6)
        edge_t = torch.sqrt(F.conv2d(target, kx, padding=1).pow(2) + F.conv2d(target, ky, padding=1).pow(2) + 1e-6)
        return F.l1_loss(edge_p, edge_t)


class MotionLoss(nn.Module):
    def __init__(self, bce: float = 1.0, dice: float = 1.0, boundary: float = 0.25, edge_weight: float = 2.0) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.weights = (bce, dice, boundary)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wb, wd, wbd = self.weights
        dilated = F.max_pool2d(target, 5, stride=1, padding=2)
        eroded = 1.0 - F.max_pool2d(1.0 - target, 5, stride=1, padding=2)
        edge = (dilated - eroded).clamp(0, 1)
        weights = 1.0 + self.edge_weight * edge
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
        return wb * bce + wd * self.dice(logits, target) + wbd * self.boundary(logits, target)
