"""
experiment_reasoning.py — Benchmark reasoning capability on synthetic tasks.

Tests how well the cognitive loop can learn and generalise simple
logical / arithmetic reasoning patterns.

Usage:
    python experiments/experiment_reasoning.py

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

logger = get_logger("experiment_reasoning")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Reasoning tasks
# ======================================================================
def make_addition_task(batch_size: int, dim: int = 16):
    """y = a + b (element-wise)."""
    a = torch.randn(batch_size, dim, device=DEVICE)
    b = torch.randn(batch_size, dim, device=DEVICE)
    return torch.cat([a, b], dim=-1), a + b


def make_comparison_task(batch_size: int, dim: int = 16):
    """y_i = 1 if a_i > b_i else 0."""
    a = torch.randn(batch_size, dim, device=DEVICE)
    b = torch.randn(batch_size, dim, device=DEVICE)
    return torch.cat([a, b], dim=-1), (a > b).float()


# ======================================================================
# Experiment runner
# ======================================================================
def run_experiment(task_name: str, task_fn, epochs: int = 30):
    logger.info(f"\n{'='*50}")
    logger.info(f"Experiment: {task_name}")
    logger.info(f"{'='*50}")

    dim = 16
    state_dim = 64
    encoder = VectorEncoder(input_dim=dim * 2, output_dim=state_dim).to(DEVICE)
    model = UCGAModel(
        input_dim=state_dim, state_dim=state_dim, output_dim=dim,
        memory_slots=32, cognitive_steps=2, reasoning_steps=2,
    ).to(DEVICE)

    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(1, epochs + 1):
        model.train(); encoder.train()
        x, y = task_fn(256, dim)
        pred = model(encoder(x))
        loss = criterion(pred, y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(loss.item())
        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:3d}  |  Loss: {loss.item():.6f}")

    logger.info(f"  Final loss: {losses[-1]:.6f}")
    return losses


def main():
    logger.info("=== UCGA Reasoning Experiments ===")
    run_experiment("Element-wise Addition", make_addition_task)
    run_experiment("Element-wise Comparison", make_comparison_task)
    logger.info("\n✓ All reasoning experiments complete.")


if __name__ == "__main__":
    main()
