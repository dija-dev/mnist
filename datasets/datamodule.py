# -*- coding: utf-8 -*-
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 自作モジュール
from datasets.dataset import MNISTWithIDs


class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, transform: transforms.Compose):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    # 必ず呼び出される関数
    def prepare_data(self):
        # download MNIST
        MNISTWithIDs(self.data_dir, train=True, download=True)
        MNISTWithIDs(self.data_dir, train=False, download=True)

    # 必ず呼び出される関数
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full = MNISTWithIDs(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(
                dataset=full,
                lengths=(0.9, 0.1),
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = MNISTWithIDs(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = MNISTWithIDs(self.data_dir, train=False, transform=self.transform)

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    # Trainer.test() 時に呼び出される
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    # Trainer.predict() 時に呼び出される
    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)
