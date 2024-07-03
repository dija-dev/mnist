# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
        )
        self.full_layer1 = nn.Sequential(
            nn.Linear(in_features=32 * 5 * 5, out_features=120),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.full_layer2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.full_layer1(x)
        x = self.full_layer2(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = Classifier()

    N, C, W, H = 20, 1, 28, 28
    summary(model=model, input_size=(N, C, W, H))
