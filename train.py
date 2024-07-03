# -*- coding: utf-8 -*-
import argparse
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

# 自作モジュール
from datasets.datamodule import MyDataModule
from lightning_module import MyLightningModule


def main():
    # 引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        choices=[
            "classifier",
            "vae",
            "hierarchical-vae",
            "diffusion",
            "conditional-diffusion",
            "classifier-free-guidance-diffusion",
        ],
        required=True,
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    model_name = args.model_name
    lr = args.lr
    batch_size = args.batch_size

    # 学習時にモデルの重みを保存する条件を指定
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        dirpath=f"saved_model/{model_name}",
    )

    # Early stopping の条件を指定
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    )

    # 学習の方法を指定
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=10,
        callbacks=[checkpoint, early_stopping],
        # fast_dev_run=7,  # runs 7 train, val, test batches and program ends
        # limit_train_batches=10, # use 10 batches of train
        # limit_val_batches=5, # use 5 batches of val
    )

    # LightningModule をロード
    model = MyLightningModule(model_name=model_name, lr=lr)

    # transform を定義
    transform = transforms.Compose([transforms.ToTensor()])
    if "vae" in model_name:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])

    # LightningDataModule をロード
    datamodule = MyDataModule(data_dir="./data", batch_size=batch_size, transform=transform)

    # 学習を実行
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
