"""
train_visual.py — Train UCGA on an image classification task.

Task: Classify synthetic random images (noise-based patterns) into
categories, validating the image encoder + cognitive loop pipeline.

Usage:
    python training/train_visual.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import ImageEncoder
from utils.logger import get_logger

logger = get_logger("train_visual")

# ======================================================================
# Hyperparameters
# ======================================================================
IMAGE_SIZE = 32
IN_CHANNELS = 3
STATE_DIM = 128
NUM_CLASSES = 5
MEMORY_SLOTS = 64
COGNITIVE_STEPS = 2
REASONING_STEPS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# Synthetic image data
# ======================================================================
def generate_image_batch(batch_size: int):
    """
    Generate synthetic images with class-dependent patterns.
    Each class has a distinct frequency pattern added to noise.
    """
    images = torch.randn(batch_size, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 0.3
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)

    for i in range(batch_size):
        freq = (labels[i].item() + 1) * 2
        x_coords = torch.linspace(0, freq * 3.14159, IMAGE_SIZE, device=DEVICE)
        pattern = torch.sin(x_coords).unsqueeze(0).expand(IMAGE_SIZE, -1)
        images[i, 0] += pattern * 0.5

    return images, labels

# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=== UCGA Visual Reasoning Training ===")
    logger.info(f"Device: {DEVICE}")

    # Image encoder: (B, C, H, W) → (B, STATE_DIM)
    encoder = ImageEncoder(in_channels=IN_CHANNELS, output_dim=STATE_DIM).to(DEVICE)

    # UCGA model
    model = UCGAModel(
        input_dim=STATE_DIM,
        state_dim=STATE_DIM,
        output_dim=NUM_CLASSES,
        memory_slots=MEMORY_SLOTS,
        cognitive_steps=COGNITIVE_STEPS,
        reasoning_steps=REASONING_STEPS,
    ).to(DEVICE)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 50

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for _ in pbar:
            images, labels = generate_image_batch(BATCH_SIZE)

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

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        accuracy = correct / total

        if accuracy > best_acc:
            best_acc = accuracy
            tag = " ★"
        else:
            tag = ""

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  Loss: {avg_loss:.4f}  |  "
            f"Accuracy: {accuracy:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}{tag}"
        )

    logger.info(f"\nTraining complete.  Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
