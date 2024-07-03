# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, use_sigmoid: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        if self.use_sigmoid:
            h = F.sigmoid(h)
        return h


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 100, latent_dim: int = 20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True)

    def forward(self, x: torch.Tensor) -> tuple:
        # x --> z
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)

        # z --> x
        x_hat = self.decoder(z)

        return mu, sigma, x_hat


class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 100, latent_dim: int = 20):
        super().__init__()
        self.encoder1 = Encoder(input_dim, hidden_dim, latent_dim)
        self.encoder2 = Encoder(latent_dim, hidden_dim, latent_dim)
        self.decoder1 = Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True)
        self.decoder2 = Decoder(latent_dim, hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        # x --> z1
        mu1, sigma1 = self.encoder1(x)
        z1 = reparameterize(mu1, sigma1)

        # z1 --> x
        x_hat = self.decoder1(z1)

        # z1 --> z2
        mu2, sigma2 = self.encoder2(z1)
        z2 = reparameterize(mu2, sigma2)

        # z2 --> z1
        z_hat = self.decoder2(z2)

        return mu1, sigma1, x_hat, mu2, sigma2, z_hat
