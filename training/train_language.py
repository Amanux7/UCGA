"""
train_language.py — Train UCGA on a basic language reasoning task.

Task: Classify short synthetic "sentences" (random token sequences)
into categories, validating the text encoder + cognitive loop pipeline.

Usage:
    python training/train_language.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import TextEncoder
from utils.logger import get_logger

logger = get_logger("train_language")

# ======================================================================
# Hyperparameters
# ======================================================================
VOCAB_SIZE = 1000
EMBED_DIM = 64
STATE_DIM = 128
NUM_CLASSES = 10
SEQ_LEN = 16
MEMORY_SLOTS = 64
COGNITIVE_STEPS = 2
REASONING_STEPS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# Synthetic language data
# ======================================================================
def generate_language_batch(batch_size: int):
    """
    Generate random token sequences and assign a label based on
    a deterministic rule (sum of first 3 tokens mod NUM_CLASSES).
    """
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
    labels = (tokens[:, :3].sum(dim=1) % NUM_CLASSES).long()
    return tokens, labels

# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=== UCGA Language Reasoning Training ===")
    logger.info(f"Device: {DEVICE}")

    # Text encoder: token IDs → cognitive vector
    encoder = TextEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        output_dim=STATE_DIM,
        max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    # UCGA model with classification head
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
        num_batches = 100

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for _ in pbar:
            tokens, labels = generate_language_batch(BATCH_SIZE)

            encoded = encoder(tokens)
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
