"""
Train a Spiking ResNet-18 on CIFAR-10 with I2E event-stream encoding.

Usage:
    python train.py                         # defaults: 200 epochs, batch 128
    python train.py --epochs 50 --lr 0.05   # quick run
    python train.py --device cuda            # GPU training

Logs are written to ``runs/`` (TensorBoard).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from model import spiking_resnet18_cifar
from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer,
                    global_step):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (events, labels) in enumerate(loader):
        # events: (B, T=8, 2, H, W)
        events = events.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        T = events.shape[1]

        # Multi-step forward: accumulate output over T timesteps
        out_sum = 0.0
        for t in range(T):
            frame = events[:, t]  # (B, 2, H, W)
            out_sum = out_sum + model(frame)

        # Mean firing rate as logit
        out_mean = out_sum / T

        loss = criterion(out_mean, labels)
        loss.backward()
        optimizer.step()

        # Reset neuron states for next sample
        functional.reset_net(model)

        # Metrics
        total_loss += loss.item() * labels.size(0)
        _, predicted = out_mean.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        global_step += 1

        if batch_idx % 100 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {100. * correct / total:.1f}%"
            )

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", accuracy, epoch)

    return avg_loss, accuracy, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for events, labels in loader:
        events = events.to(device)
        labels = labels.to(device)

        T = events.shape[1]
        out_sum = 0.0
        for t in range(T):
            out_sum = out_sum + model(events[:, t])
        out_mean = out_sum / T

        loss = criterion(out_mean, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = out_mean.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        functional.reset_net(model)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", accuracy, epoch)

    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNN ResNet-18 CIFAR-10 Training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--s-th0", type=float, default=0.07,
                        help="I2E sensitivity threshold (0.07 for CIFAR)")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="LIF neuron time constant")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print("Loading CIFAR-10 with I2E encoding...")
    train_loader, test_loader = get_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        s_th0=args.s_th0,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    model = spiking_resnet18_cifar(
        num_classes=10,
        in_channels=2,
        tau=args.tau,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Logging
    writer = SummaryWriter(log_dir=args.log_dir)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training Spiking ResNet-18 on CIFAR-10 (I2E encoded)")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"I2E s_th0: {args.s_th0}  |  LIF tau: {args.tau}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step,
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, epoch, writer,
        )

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"Train: {train_loss:.4f} / {train_acc:.1f}%  "
            f"Val: {val_loss:.4f} / {val_acc:.1f}%  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}  "
            f"({elapsed:.1f}s)"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
            }, ckpt_path)
            print(f"  ✓ New best: {val_acc:.1f}% → saved {ckpt_path}")

        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
