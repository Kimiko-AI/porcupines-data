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


# ---------------------------------------------------------------------------
# CIFAR-10 DVS (neuromorphic, for transfer evaluation)
# ---------------------------------------------------------------------------

class CIFAR10DVSEval(Dataset):
    """CIFAR-10 DVS wrapper that resizes to 32×32 for transfer eval.

    Uses spikingjelly's ``CIFAR10DVS`` with ``data_type='frame'``
    and a fixed number of frames matching the I2E timestep count.

    Parameters
    ----------
    root : str
        Directory where CIFAR-10 DVS is stored.
    train : bool
        Train or test split.
    frames_number : int
        Number of frames to split events into (default 8 to match I2E).
    target_size : tuple[int, int]
        Spatial size to resize frames to (default ``(32, 32)``).
    """

    def __init__(
        self,
        root: str,
        train: bool = False,
        frames_number: int = 8,
        target_size: tuple[int, int] = (32, 32),
    ):
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

        self.dataset = CIFAR10DVS(
            root=root,
            train=train,
            data_type="frame",
            frames_number=frames_number,
            split_by="number",
        )
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frame, label = self.dataset[idx]
        # frame: (T, 2, 128, 128) ndarray or tensor
        frame = torch.as_tensor(frame, dtype=torch.float32)

        # Resize spatially: (T, 2, 128, 128) → (T, 2, 32, 32)
        T, C, H, W = frame.shape
        tgt_h, tgt_w = self.target_size
        if H != tgt_h or W != tgt_w:
            # Reshape to (T*C, 1, H, W) for F.interpolate, then back
            frame = torch.nn.functional.interpolate(
                frame.view(T * C, 1, H, W),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).view(T, C, tgt_h, tgt_w)

        # Normalise to [0, 1] if not already
        fmax = frame.max()
        if fmax > 0:
            frame = frame / fmax

        return frame, label


def get_dvs_dataloader(
    root: str,
    batch_size: int = 128,
    frames_number: int = 8,
    num_workers: int = 4,
    train: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for CIFAR-10 DVS evaluation.

    Parameters
    ----------
    root : str
        Path to CIFAR-10 DVS data.
    batch_size : int
        Batch size.
    frames_number : int
        Number of event frames (default 8).
    num_workers : int
        DataLoader workers.
    train : bool
        If True, load training split; otherwise test split.
    """
    ds = CIFAR10DVSEval(
        root=root, train=train,
        frames_number=frames_number,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

