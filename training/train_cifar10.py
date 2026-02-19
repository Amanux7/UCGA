"""
train_cifar10.py — Train UCGA on CIFAR-10 image classification.

Uses the real CIFAR-10 dataset (60,000 images, 10 classes) via
torchvision to validate the image encoder + cognitive loop pipeline
on a widely-used benchmark.

Usage:
    python training/train_cifar10.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import ImageEncoder
from utils.logger import get_logger

logger = get_logger("train_cifar10")

# ======================================================================
# Hyperparameters
# ======================================================================
IMAGE_SIZE = 32
IN_CHANNELS = 3
STATE_DIM = 128
NUM_CLASSES = 10
MEMORY_SLOTS = 64
COGNITIVE_STEPS = 2
REASONING_STEPS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# Data
# ======================================================================
def get_dataloaders():
    """Download and prepare CIFAR-10 train/test dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    return train_loader, test_loader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, encoder, test_loader):
    """Evaluate on the test set and return accuracy."""
    model.eval()
    encoder.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        encoded = encoder(images)
        logits = model(encoded)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA × CIFAR-10 Training")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Downloading CIFAR-10 to {DATA_DIR}...")

    train_loader, test_loader = get_dataloaders()
    logger.info(f"Train samples: {len(train_loader.dataset):,}")
    logger.info(f"Test  samples: {len(test_loader.dataset):,}")

    # ---- Model ----
    encoder = ImageEncoder(in_channels=IN_CHANNELS, output_dim=STATE_DIM).to(DEVICE)
    model = UCGAModel(
        input_dim=STATE_DIM,
        state_dim=STATE_DIM,
        output_dim=NUM_CLASSES,
        memory_slots=MEMORY_SLOTS,
        cognitive_steps=COGNITIVE_STEPS,
        reasoning_steps=REASONING_STEPS,
    ).to(DEVICE)

    total_params = model.count_parameters() + sum(
        p.numel() for p in encoder.parameters()
    )
    logger.info(f"Total parameters: {total_params:,}")

    # ---- Optimiser ----
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            encoded = encoder(images)
            logits = model(encoded)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{correct / total:.3f}",
            )

        scheduler.step()

        train_acc = correct / total
        avg_loss = epoch_loss / len(train_loader)
        test_acc = evaluate(model, encoder, test_loader)

        tag = ""
        if test_acc > best_acc:
            best_acc = test_acc
            tag = " ★"

        logger.info(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Train Acc: {train_acc:.4f}  |  "
            f"Test Acc: {test_acc:.4f}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}{tag}"
        )

    logger.info("")
    logger.info(f"Training complete.  Best test accuracy: {best_acc:.4f}")
    logger.info(f"Classes: {CIFAR10_CLASSES}")


if __name__ == "__main__":
    main()
