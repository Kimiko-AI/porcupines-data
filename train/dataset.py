"""
CIFAR-10 dataset with I2E event-stream encoding.

Each ``__getitem__`` returns ``(events, label)`` where ``events`` is a
``(T=8, 2, H, W)`` float32 tensor of ON/OFF spikes.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

import sys
import os

# Ensure snn_encoder is importable from the parent project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snn_encoder.encoders.i2e import I2EEncoder


class CIFAR10Events(Dataset):
    """CIFAR-10 with I2E event-stream encoding.

    Parameters
    ----------
    root : str
        Directory for CIFAR-10 download/cache.
    train : bool
        Train or test split.
    s_th0 : float
        I2E sensitivity threshold (0.07 recommended for CIFAR).
    download : bool
        Download the dataset if not present.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        s_th0: float = 0.07,
        download: bool = True,
    ):
        # Standard CIFAR-10 augmentations (applied BEFORE encoding)
        if train:
            self.pre_transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # → (3, 32, 32) float [0, 1]
            ])
        else:
            self.pre_transform = T.Compose([
                T.ToTensor(),
            ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download,
            transform=self.pre_transform,
        )

        self.s_th0 = s_th0
        self.encoder = I2EEncoder.get_module(s_th0=s_th0, device="cpu")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, label = self.dataset[idx]
        # img_tensor: (3, 32, 32) float [0, 1]

        # I2E expects (B, C, H, W), so unsqueeze batch dim
        with torch.no_grad():
            events = self.encoder(img_tensor.unsqueeze(0))  # (T=8, 1, 2, H, W)
            events = events[:, 0]  # (T=8, 2, H, W)

        return events.float(), label


def get_dataloaders(
    root: str = "./data",
    batch_size: int = 128,
    s_th0: float = 0.07,
    num_workers: int = 4,
    download: bool = True,
):
    """Create train and test DataLoaders.

    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    train_ds = CIFAR10Events(root=root, train=True, s_th0=s_th0,
                              download=download)
    test_ds = CIFAR10Events(root=root, train=False, s_th0=s_th0,
                             download=download)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
