# -*- coding: utf-8 -*-
import torch
import lightning as L
import numpy as np
import torch.nn as nn
from models.classifier import Classifier
from models.vae import VAE, HierarchicalVAE
from models.diffusion import UNet, Diffuser
from losses.loss import VAELoss, HierarchicalVAELoss


class MyLightningModule(L.LightningModule):
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "classifier":
            self.model = Classifier()
            self.loss = nn.CrossEntropyLoss()
        elif model_name == "vae":
            self.model = VAE()
            self.loss = VAELoss()
        elif model_name == "hierarchical-vae":
            self.model = HierarchicalVAE()
            self.loss = HierarchicalVAELoss()
        elif model_name in ["diffusion", "conditional-diffusion", "classifier-free-guidance-diffusion"]:
            num_labels = None if model_name == "diffusion" else 10
            self.model, self.diffuser = UNet(num_labels=num_labels), Diffuser()
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"<< unknown model_name: {model_name} >>")

    def forward(self, batch) -> torch.Tensor:
        x, label = batch

        if self.hparams.model_name == "classifier":
            output = self.model(x)
            loss = self.loss(output, label)

        elif self.hparams.model_name in ["vae", "hierarchical-vae"]:
            output = self.model(x)
            loss = self.loss(x, *output)

        elif self.hparams.model_name in ["diffusion", "conditional-diffusion", "classifier-free-guidance-diffusion"]:
            t = torch.randint(1, self.diffuser.num_timesteps + 1, (len(x),))
            x_noisy, noise = self.diffuser.add_noise(x, t)

            if self.hparams.model_name == "diffusion":
                label = None
            elif self.hparams.model_name == "conditional-diffusion":
                pass
            elif self.hparams.model_name == "classifier-free-guidance-diffusion":
                if np.random.random() < 0.1:
                    label = None

            noise_pred = self.model(x_noisy, t, label)
            loss = self.loss(noise, noise_pred)

        else:
            raise ValueError(f"<< unknown model_name: {self.hparams.model_name} >>")

        return loss

    def training_step(self, batch, batch_nb) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("train_accuracy", accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.forward(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("val_accuracy", accuracy, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
