"""
experiment_memory.py — Benchmark persistent memory retrieval accuracy.

Tests whether the UCGA can store information in memory and retrieve
it accurately after several intervening episodes.

Usage:
    python experiments/experiment_memory.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("experiment_memory")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    logger.info("=== UCGA Memory Experiment ===")

    dim = 32
    state_dim = 64
    encoder = VectorEncoder(input_dim=dim, output_dim=state_dim).to(DEVICE)
    model = UCGAModel(
        input_dim=state_dim, state_dim=state_dim, output_dim=dim,
        memory_slots=32, cognitive_steps=2, reasoning_steps=2,
    ).to(DEVICE)

    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    criterion = nn.MSELoss()

    # Phase 1: Store patterns in memory via training
    logger.info("\n--- Phase 1: Memorization ---")
    patterns = torch.randn(8, dim, device=DEVICE)

    for epoch in range(1, 51):
        model.train(); encoder.train()
        idx = torch.randint(0, 8, (16,))
        x = patterns[idx]
        target = x.clone()  # identity recall

        encoded = encoder(x)
        pred = model(encoded)
        loss = criterion(pred, target)

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}  |  Loss: {loss.item():.6f}")

    # Phase 2: Recall — feed same patterns and measure output similarity
    logger.info("\n--- Phase 2: Recall Test ---")
    model.eval(); encoder.eval()
    with torch.no_grad():
        for i in range(8):
            x = patterns[i].unsqueeze(0)
            pred = model(encoder(x))
            sim = nn.functional.cosine_similarity(pred, x, dim=-1).item()
            logger.info(f"  Pattern {i}  |  Cosine similarity: {sim:.4f}")

    # Phase 3: Distractor robustness
    logger.info("\n--- Phase 3: Distractor Robustness ---")
    with torch.no_grad():
        noise = torch.randn(4, dim, device=DEVICE)
        for i in range(4):
            x = noise[i].unsqueeze(0)
            pred = model(encoder(x))
            sim = nn.functional.cosine_similarity(pred, x, dim=-1).item()
            logger.info(f"  Noise {i}   |  Cosine similarity: {sim:.4f}")

    logger.info("\n✓ Memory experiment complete.")


if __name__ == "__main__":
    main()
