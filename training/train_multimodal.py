"""
train_multimodal.py — Train UCGA on a synthetic image-text matching task.

Demonstrates the multimodal pipeline:
    1. MultimodalEncoder fuses random images + token sequences
    2. UCGA cognitive loop processes the fused representation
    3. Model classifies whether image-text pairs are matched or mismatched

This is a self-contained proof-of-concept — no external datasets needed.

Usage:
    python training/train_multimodal.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import MultimodalEncoder
from utils.logger import get_logger

logger = get_logger("train_multimodal")

# ======================================================================
# Hyperparameters
# ======================================================================
VOCAB_SIZE = 500
MAX_SEQ_LEN = 16
EMBED_DIM = 64
STATE_DIM = 64
NUM_CLASSES = 2      # Matched vs Mismatched
MEMORY_SLOTS = 16
COGNITIVE_STEPS = 1
NUM_SAMPLES = 4000
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Synthetic data generation
# ======================================================================
def generate_multimodal_data(n_samples: int, n_classes: int = 5):
    """
    Generate synthetic image-text pairs with match/mismatch labels.

    Each class has a distinct visual pattern and a distinct token pattern.
    Matched pairs = same class for image and text.
    Mismatched pairs = different classes.
    """
    images = torch.zeros(n_samples, 3, 32, 32)
    tokens = torch.zeros(n_samples, MAX_SEQ_LEN, dtype=torch.long)
    labels = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        img_class = torch.randint(0, n_classes, (1,)).item()

        # Decide matched or mismatched
        is_matched = torch.rand(1).item() > 0.5
        if is_matched:
            txt_class = img_class
            labels[i] = 1  # matched
        else:
            txt_class = (img_class + torch.randint(1, n_classes, (1,)).item()) % n_classes
            labels[i] = 0  # mismatched

        # Generate image with class-dependent pattern
        base_color = torch.zeros(3)
        base_color[img_class % 3] = 0.8
        images[i] = base_color.view(3, 1, 1).expand(3, 32, 32)
        # Add class-specific spatial pattern
        start = img_class * 6
        images[i, :, start:start+6, start:start+6] += 0.5

        # Generate text with class-dependent token pattern
        base_token = txt_class * (VOCAB_SIZE // n_classes) + 1
        seq_len = torch.randint(4, MAX_SEQ_LEN, (1,)).item()
        tokens[i, :seq_len] = torch.randint(
            base_token, base_token + VOCAB_SIZE // n_classes,
            (seq_len,),
        )

    # Add noise
    images += torch.randn_like(images) * 0.1

    return images, tokens, labels


# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA × Multimodal Image-Text Matching")
    logger.info("  Encoder: MultimodalEncoder (Phase 3)")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    # Generate data
    logger.info("Generating synthetic multimodal data...")
    images, tokens, labels = generate_multimodal_data(NUM_SAMPLES)

    split = int(0.8 * NUM_SAMPLES)
    train_loader = DataLoader(
        TensorDataset(images[:split], tokens[:split], labels[:split]),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(images[split:], tokens[split:], labels[split:]),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    logger.info(f"Train: {split} samples | Val: {NUM_SAMPLES - split} samples")

    # ---- Models ----
    encoder = MultimodalEncoder(
        image_channels=3,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        output_dim=STATE_DIM,
        num_heads=4,
        num_fusion_layers=2,
        text_encoder_type="transformer",
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    model = UCGAModel(
        input_dim=STATE_DIM,
        state_dim=STATE_DIM,
        output_dim=NUM_CLASSES,
        memory_slots=MEMORY_SLOTS,
        cognitive_steps=COGNITIVE_STEPS,
    ).to(DEVICE)

    total_params = model.count_parameters() + sum(
        p.numel() for p in encoder.parameters()
    )
    logger.info(f"Total parameters: {total_params:,}")

    # ---- Optimiser ----
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
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
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False,
        )
        for imgs, toks, lbls in pbar:
            imgs, toks, lbls = imgs.to(DEVICE), toks.to(DEVICE), lbls.to(DEVICE)

            encoded = encoder(imgs, toks)
            logits = model(encoded)
            loss = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

        # ---- Validation ----
        model.eval()
        encoder.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, toks, lbls in val_loader:
                imgs, toks, lbls = imgs.to(DEVICE), toks.to(DEVICE), lbls.to(DEVICE)
                encoded = encoder(imgs, toks)
                logits = model(encoded)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_loss = epoch_loss / len(train_loader)
        tag = " ★" if val_acc > best_acc else ""
        best_acc = max(best_acc, val_acc)

        logger.info(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Train Acc: {train_acc:.4f}  |  "
            f"Val Acc: {val_acc:.4f}{tag}"
        )

    logger.info("")
    logger.info(f"Training complete.  Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
