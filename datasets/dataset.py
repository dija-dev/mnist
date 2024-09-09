# -*- coding: utf-8 -*-
from torchvision.datasets import MNIST


class MNISTWithIDs(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ids = list(range(len(self.data)))

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        identifier = self.ids[index]
        return image, label, identifier
