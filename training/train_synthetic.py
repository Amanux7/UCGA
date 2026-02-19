"""
train_synthetic.py — Train UCGA on a synthetic vector reasoning task.

Task: Given two random vectors a and b, predict a learned non-linear
combination  f(a, b) = tanh(W · [a; b] + c) .  This validates that
the cognitive loop can learn input–output mappings through its
graph of nodes.

Usage:
    python training/train_synthetic.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("train_synthetic")

# ======================================================================
# Hyperparameters
# ======================================================================
INPUT_DIM = 32           # Raw input dimensionality
STATE_DIM = 128          # Cognitive state dimensionality
OUTPUT_DIM = 32          # Task output dimensionality
MEMORY_SLOTS = 64        # Persistent memory slots
COGNITIVE_STEPS = 2      # Outer cognitive-loop iterations
REASONING_STEPS = 2      # Inner reasoning refinement steps
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# Synthetic data generator
# ======================================================================
def generate_batch(batch_size: int, dim: int):
    """Generate a synthetic reasoning task: y = tanh(W_true · [a; b])."""
    # Fixed (but random) ground-truth transform
    if not hasattr(generate_batch, "W_true"):
        generate_batch.W_true = torch.randn(dim, dim * 2, device=DEVICE) * 0.5
        generate_batch.bias = torch.randn(dim, device=DEVICE) * 0.1

    a = torch.randn(batch_size, dim, device=DEVICE)
    b = torch.randn(batch_size, dim, device=DEVICE)
    x = torch.cat([a, b], dim=-1)                       # (B, 2*dim)
    y = torch.tanh(x @ generate_batch.W_true.T + generate_batch.bias)  # (B, dim)
    return torch.cat([a, b], dim=-1), y  # input, target

# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=== UCGA Synthetic Training ===")
    logger.info(f"Device: {DEVICE}")

    # Encoder: maps raw 2*INPUT_DIM → STATE_DIM
    encoder = VectorEncoder(input_dim=INPUT_DIM * 2, output_dim=STATE_DIM).to(DEVICE)

    # UCGA model
    model = UCGAModel(
        input_dim=STATE_DIM,
        state_dim=STATE_DIM,
        output_dim=OUTPUT_DIM,
        memory_slots=MEMORY_SLOTS,
        cognitive_steps=COGNITIVE_STEPS,
        reasoning_steps=REASONING_STEPS,
    ).to(DEVICE)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Optimizer & loss
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.MSELoss()

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        num_batches = 100  # synthetic — unlimited data

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for _ in pbar:
            raw_input, target = generate_batch(BATCH_SIZE, INPUT_DIM)

            encoded = encoder(raw_input)
            output, meta = model(encoded, return_meta=True)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            tag = " ★"
        else:
            tag = ""

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  Loss: {avg_loss:.6f}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}  |  "
            f"Confidence: {meta['confidences'][-1]:.3f}  |  "
            f"Corrections: {meta['corrections']}{tag}"
        )

    logger.info(f"\nTraining complete.  Best loss: {best_loss:.6f}")
    logger.info("Model ready for evaluation.")


if __name__ == "__main__":
    main()
