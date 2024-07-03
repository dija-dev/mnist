# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def _pos_encoding(time_idx: int, output_dim: int) -> torch.Tensor:
    t, D = time_idx, output_dim
    v = torch.zeros(D)

    i = torch.arange(0, D)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def pos_encoding(timesteps: torch.Tensor, output_dim: int) -> torch.Tensor:
    batch_size = len(timesteps)
    v = torch.zeros(batch_size, output_dim)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, time_embed_dim: int = 100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(256 + 128, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        v = pos_encoding(timesteps, self.time_embed_dim).detach()
        if label is not None:
            v += self.label_emb(label)

        x1 = self.down1(x, v)

        x = self.maxpool(x1)
        x2 = self.down2(x, v)

        x = self.maxpool(x2)
        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)

        x = self.out(x)
        return x


class Diffuser:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)  # , device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple:
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(
        self,
        model: UNet,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor = None,
        gamma: float = None,
    ) -> torch.Tensor:

        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, label)
            if gamma is not None:
                eps_uncond = model(x, t)
                eps = eps_uncond + gamma * (eps - eps_uncond)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x: torch.Tensor) -> Image:
        x = x * 255
        x = x.clamp(0, 255).to(torch.uint8).cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(
        self,
        model: UNet,
        x_shape: tuple = (20, 1, 28, 28),
        label: torch.Tensor = None,
        gamma: float = None,
    ) -> list:
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        if label is None:
            label = torch.randint(0, 10, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, dtype=torch.long, device=self.device)
            x = self.denoise(model, x, t, label, gamma)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, label
