"""
run_baselines.py — Train and compare all baselines + UCGA on CIFAR-10.

Reports accuracy, parameter count, estimated FLOPs, and wall-clock time.
Supports multi-seed runs for mean ± std reporting.

Usage:
    python experiments/run_baselines.py [--seeds 5] [--epochs 20] [--batch_size 64]

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
from tqdm import tqdm

from experiments.baselines import MLPBaseline, CNNBaseline, ViTBaseline, UCGABaseline
from utils.compute_metrics import (
    count_parameters,
    estimate_flops_from_forward,
    measure_wall_clock,
    format_params,
    format_flops,
)
from utils.logger import get_logger

logger = get_logger("run_baselines")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Data
# ======================================================================
def get_cifar10_loaders(batch_size: int = 64):
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
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


# ======================================================================
# Training / Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_one_seed(model, train_loader, test_loader, device, args, seed):
    """Train model for one seed. Returns (best_test_acc, epochs_to_90pct, epoch_times)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_acc = 0.0
    final_acc_90 = None
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        test_acc = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc

        # Track epochs to reach 90% of eventual best
        if final_acc_90 is None and test_acc >= best_acc * 0.9:
            final_acc_90 = epoch

    return best_acc, final_acc_90 or args.epochs, np.mean(epoch_times)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline comparison on CIFAR-10")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--amp", action="store_true", help="Mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)

    models = {
        "MLP-3L": lambda: MLPBaseline(num_classes=10),
        "CNN-Light": lambda: CNNBaseline(num_classes=10),
        "ViT-Tiny": lambda: ViTBaseline(num_classes=10),
        "UCGA": lambda: UCGABaseline(num_classes=10, state_dim=128),
    }

    results = {}
    for name, model_fn in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Model: {name}")
        logger.info(f"{'='*60}")

        # Create a reference instance for metrics
        ref_model = model_fn().to(device)
        n_params = count_parameters(ref_model)
        sample = torch.randn(1, 3, 32, 32, device=device)
        flops = estimate_flops_from_forward(ref_model, sample)
        fwd_mean, fwd_std = measure_wall_clock(ref_model, sample)
        del ref_model

        logger.info(f"  Params: {format_params(n_params)} | FLOPs: {format_flops(flops)} | Fwd: {fwd_mean:.1f}±{fwd_std:.1f} ms")

        accs, convergence_epochs, epoch_times_list = [], [], []

        for seed in range(args.seeds):
            logger.info(f"  Seed {seed+1}/{args.seeds}...")
            model = model_fn()
            acc, conv_ep, avg_time = train_one_seed(
                model, train_loader, test_loader, device, args, seed=42 + seed
            )
            accs.append(acc)
            convergence_epochs.append(conv_ep)
            epoch_times_list.append(avg_time)
            logger.info(f"    Acc: {acc:.4f} | Conv@90%: epoch {conv_ep} | {avg_time:.1f}s/epoch")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[name] = {
            "params": n_params,
            "params_fmt": format_params(n_params),
            "flops": flops,
            "flops_fmt": format_flops(flops),
            "fwd_ms": f"{fwd_mean:.1f}",
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "conv_mean": float(np.mean(convergence_epochs)),
            "conv_std": float(np.std(convergence_epochs)),
            "time_per_epoch": float(np.mean(epoch_times_list)),
        }

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("  BASELINE COMPARISON RESULTS (CIFAR-10)")
    logger.info(f"{'='*80}")
    header = f"{'Model':<12} {'Params':>8} {'FLOPs':>8} {'Fwd(ms)':>8} {'Accuracy':>14} {'Conv(ep)':>10} {'s/ep':>6}"
    logger.info(header)
    logger.info("-" * 80)
    for name, r in results.items():
        logger.info(
            f"{name:<12} {r['params_fmt']:>8} {r['flops_fmt']:>8} {r['fwd_ms']:>8} "
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f} "
            f"{r['conv_mean']:>5.1f}±{r['conv_std']:.1f} "
            f"{r['time_per_epoch']:>5.1f}"
        )

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
