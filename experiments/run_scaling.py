"""
run_scaling.py — Scaling behavior analysis for UCGA.

Produces:
  1. Parameter count breakdown (per-node)
  2. Accuracy vs. state_dim curves (64 → 256) on CIFAR-10
  3. Accuracy vs. cognitive steps T curves (1 → 5) on CIFAR-10
  4. Wall-clock and estimated FLOPs at each configuration
  5. Projected compute estimates for full-scale runs

Usage:
    python experiments/run_scaling.py [--seeds 3] [--epochs 15]

Author: Dr. Elena Voss / Aman Singh
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

from ucga.ucga_model import UCGAModel
from ucga.encoders import ImageEncoder
from utils.compute_metrics import (
    count_parameters,
    count_parameters_by_module,
    estimate_flops_from_forward,
    measure_wall_clock,
    format_params,
    format_flops,
)
from utils.logger import get_logger

logger = get_logger("run_scaling")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "scaling")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Data
# ======================================================================
def get_cifar10_loaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                   pin_memory=True, drop_last=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                   pin_memory=True),
    )


# ======================================================================
# Training helper
# ======================================================================
@torch.no_grad()
def evaluate(model, encoder, loader, device):
    model.eval()
    encoder.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(encoder(images))
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_one(encoder, model, train_loader, test_loader, device, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = encoder.to(device)
    model = model.to(device)
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        encoder.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(encoder(images))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        acc = evaluate(model, encoder, test_loader, device)
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ======================================================================
# 1. Parameter breakdown
# ======================================================================
def report_parameter_breakdown(state_dim=128):
    logger.info(f"\n{'='*60}")
    logger.info(f"  PARAMETER BREAKDOWN (state_dim={state_dim})")
    logger.info(f"{'='*60}")

    encoder = ImageEncoder(in_channels=3, output_dim=state_dim)
    model = UCGAModel(
        input_dim=state_dim, state_dim=state_dim, output_dim=10,
        memory_slots=64, cognitive_steps=3, reasoning_steps=2,
    )

    enc_params = count_parameters(encoder)
    model_breakdown = count_parameters_by_module(model)
    total = count_parameters(model) + enc_params

    logger.info(f"  {'Component':<25} {'Parameters':>12}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  {'ImageEncoder':<25} {format_params(enc_params):>12}")
    for name, count in model_breakdown.items():
        logger.info(f"  {name:<25} {format_params(count):>12}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  {'TOTAL':<25} {format_params(total):>12}")

    return {"encoder": enc_params, **model_breakdown, "total": total}


# ======================================================================
# 2. Accuracy vs. state_dim
# ======================================================================
def sweep_state_dim(train_loader, test_loader, device, args):
    dims = [64, 128, 192, 256]
    logger.info(f"\n{'='*60}")
    logger.info("  ACCURACY vs. STATE_DIM (CIFAR-10)")
    logger.info(f"{'='*60}")

    results = {}
    for dim in dims:
        logger.info(f"\n  state_dim={dim}:")
        accs = []
        for seed in range(args.seeds):
            encoder = ImageEncoder(in_channels=3, output_dim=dim)
            model = UCGAModel(
                input_dim=dim, state_dim=dim, output_dim=10,
                memory_slots=64, cognitive_steps=3, reasoning_steps=2,
            )
            acc = train_one(encoder, model, train_loader, test_loader, device, args.epochs, 42 + seed)
            accs.append(acc)
            logger.info(f"    Seed {seed+1}: {acc:.4f}")
            del encoder, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        n_params = count_parameters(ImageEncoder(in_channels=3, output_dim=dim)) + \
                   count_parameters(UCGAModel(input_dim=dim, state_dim=dim, output_dim=10, memory_slots=64))

        results[str(dim)] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "params": n_params,
        }
        logger.info(f"    → {np.mean(accs):.4f} ± {np.std(accs):.4f}  ({format_params(n_params)})")

    return results


# ======================================================================
# 3. Accuracy vs. T (cognitive steps)
# ======================================================================
def sweep_cognitive_steps(train_loader, test_loader, device, args):
    T_values = [1, 2, 3, 5]
    logger.info(f"\n{'='*60}")
    logger.info("  ACCURACY vs. COGNITIVE STEPS T (CIFAR-10)")
    logger.info(f"{'='*60}")

    results = {}
    for T in T_values:
        logger.info(f"\n  T={T}:")
        accs = []
        for seed in range(args.seeds):
            encoder = ImageEncoder(in_channels=3, output_dim=128)
            model = UCGAModel(
                input_dim=128, state_dim=128, output_dim=10,
                memory_slots=64, cognitive_steps=T, reasoning_steps=2,
            )
            acc = train_one(encoder, model, train_loader, test_loader, device, args.epochs, 42 + seed)
            accs.append(acc)
            logger.info(f"    Seed {seed+1}: {acc:.4f}")
            del encoder, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # FLOPs and timing for this T
        enc_ref = ImageEncoder(in_channels=3, output_dim=128).to(device)
        mdl_ref = UCGAModel(
            input_dim=128, state_dim=128, output_dim=10,
            memory_slots=64, cognitive_steps=T, reasoning_steps=2,
        ).to(device)
        sample = torch.randn(1, 3, 32, 32, device=device)
        sample_enc = enc_ref(sample)
        flops = estimate_flops_from_forward(mdl_ref, sample_enc)
        fwd_mean, fwd_std = measure_wall_clock(mdl_ref, sample_enc)
        del enc_ref, mdl_ref

        results[str(T)] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "flops": flops,
            "fwd_ms": f"{fwd_mean:.1f}±{fwd_std:.1f}",
        }
        logger.info(f"    → {np.mean(accs):.4f} ± {np.std(accs):.4f}  FLOPs={format_flops(flops)}")

    return results


# ======================================================================
# 4. Compute projection
# ======================================================================
def compute_projection():
    logger.info(f"\n{'='*60}")
    logger.info("  COMPUTE PROJECTION — Full-Scale Estimates")
    logger.info(f"{'='*60}")

    configs = [
        {"label": "Current (128d, 9 nodes)", "state_dim": 128, "nodes": 9, "scale": 1},
        {"label": "Medium (512d, 9 nodes)", "state_dim": 512, "nodes": 9, "scale": 16},
        {"label": "Large  (1024d, 18 nodes)", "state_dim": 1024, "nodes": 18, "scale": 128},
        {"label": "XL     (2048d, 36 nodes)", "state_dim": 2048, "nodes": 36, "scale": 1024},
    ]

    logger.info(f"  {'Config':<30} {'Est. Params':>12} {'Est. FLOPs/sample':>18} {'GPU Projection':>25}")
    logger.info(f"  {'-'*85}")

    for cfg in configs:
        # Rough param estimate: O(nodes * state_dim^2)
        est_params = int(cfg["nodes"] * cfg["state_dim"]**2 * 3)  # W, bias, gate per node
        est_flops = est_params * 6  # forward + backward ≈ 3× forward, 2 FLOPs/MAC
        if cfg["scale"] <= 16:
            gpu = "GTX 1650 (4GB) — hours"
        elif cfg["scale"] <= 128:
            gpu = "1×A100 (80GB) — days"
        else:
            gpu = "4-8×A100 — weeks"

        logger.info(
            f"  {cfg['label']:<30} {format_params(est_params):>12} {format_flops(est_flops):>18} {gpu:>25}"
        )

    logger.info(f"\n  Roadmap: Full-scale (10-50M params) projected to require 4-8×A100")
    logger.info(f"  weeks for ImageNet-scale; plan LoRA adapters + distributed graph")
    logger.info(f"  partitioning for efficient scaling.")

    return configs


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="UCGA Scaling Analysis")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Scaling Analysis | Seeds: {args.seeds} | Epochs: {args.epochs} | Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)

    # 1. Parameter breakdown
    param_breakdown = report_parameter_breakdown()

    # 2. Accuracy vs. state_dim
    dim_results = sweep_state_dim(train_loader, test_loader, device, args)

    # 3. Accuracy vs. T
    T_results = sweep_cognitive_steps(train_loader, test_loader, device, args)

    # 4. Compute projection
    projections = compute_projection()

    # Save all results
    all_results = {
        "param_breakdown": param_breakdown,
        "dim_scaling": dim_results,
        "T_scaling": T_results,
    }
    out_path = os.path.join(RESULTS_DIR, "scaling_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
