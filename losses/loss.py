# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mu, sigma, x_hat) -> torch.Tensor:
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1 + L2) / batch_size


class HierarchicalVAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mu1, sigma1, x_hat, mu2, sigma2, z_hat) -> torch.Tensor:
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = -torch.sum(1 + torch.log(sigma2**2) - mu2**2 - sigma2**2)
        L3 = -torch.sum(1 + torch.log(sigma1**2) - (mu1 - z_hat) ** 2 - sigma1**2)
        return (L1 + L2 + L3) / batch_size
