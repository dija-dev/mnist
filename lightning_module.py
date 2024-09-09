# -*- coding: utf-8 -*-
import torch
import lightning as L
import numpy as np
import os
import pandas as pd
import torch.nn as nn
from typing import List

# 自作モジュール
from models.classifier import Classifier
from models.vae import VAE, HierarchicalVAE
from models.diffusion import UNet, Diffuser
from losses.loss import VAELoss, HierarchicalVAELoss


class MyLightningModule(L.LightningModule):
    def __init__(self, model_name: str, lr: float, **kwargs) -> None:
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

        self.td = kwargs.get("td") and model_name == "classifier"
        return None

    def forward(self, x, label) -> torch.Tensor:

        if self.hparams.model_name == "classifier":
            output = self.model(x)
            loss = self.loss(output, label)

        elif self.hparams.model_name in ["vae", "hierarchical-vae"]:
            output = self.model(x)
            loss = self.loss(x, *output)

        elif "diffusion" in self.hparams.model_name:
            t = torch.randint(1, self.diffuser.num_timesteps + 1, (len(x),))
            x_noisy, noise = self.diffuser.add_noise(x, t)

            if self.hparams.model_name == "diffusion":
                label = None
            elif self.hparams.model_name == "conditional-diffusion":
                pass
            elif self.hparams.model_name == "classifier-free-guidance-diffusion":
                if np.random.random() < 0.1:
                    label = None

            output = self.model(x_noisy, t, label)  # prediction of noise
            loss = self.loss(noise, output)

        else:
            raise ValueError(f"<< unknown model_name: {self.hparams.model_name} >>")

        return loss, output

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, label, identifier = batch
        loss, output = self.forward(x, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("train_accuracy", accuracy, on_step=True, on_epoch=True)

        if self.td:
            self.train_ids[batch_idx] = identifier.detach().cpu().numpy()
            self.train_logits[batch_idx] = output.detach().cpu().numpy()
            self.train_golds[batch_idx] = label.detach().cpu().numpy()

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, label, identifier = batch
        loss, output = self.forward(x, label)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("val_accuracy", accuracy, on_step=True, on_epoch=True)
        return None

    def test_step(self, batch, batch_idx) -> None:
        x, label, identifier = batch
        loss, output = self.forward(x, label)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # pred = output.argmax(-1)
        # n_correct = (pred == label).sum().item()
        # accuracy = n_correct / pred.size(0)
        # self.log("accuracy", accuracy)
        return None

    def on_train_epoch_start(self) -> None:
        if self.td:
            total_batches = len(self.trainer.train_dataloader)
            batch_size = self.trainer.train_dataloader.batch_size
            self.train_ids = np.zeros((total_batches, batch_size), dtype=np.int32)
            self.train_logits = np.zeros((total_batches, batch_size, self.model.output_dim), dtype=np.float32)
            self.train_golds = np.zeros((total_batches, batch_size), dtype=np.int32)
        return None

    def on_train_epoch_end(self) -> None:
        """Save training dynamics (logits) from given epoch as records of a `.jsonl` file."""
        if self.td:
            td_dir = os.path.join(self.trainer.logger.log_dir, f"training_dynamics")
            if not os.path.exists(td_dir):
                os.makedirs(td_dir)

            epoch = self.trainer.current_epoch
            epoch_file_path = os.path.join(td_dir, f"dynamics_epoch_{epoch}.jsonl")

            td_df = pd.DataFrame(
                {
                    "guid": self.train_ids.reshape(-1),
                    f"logits_epoch_{epoch}": self.train_logits.reshape(-1, self.model.output_dim).tolist(),
                    "gold": self.train_golds.reshape(-1),
                }
            )
            td_df.to_json(epoch_file_path, lines=True, orient="records")
        return None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
