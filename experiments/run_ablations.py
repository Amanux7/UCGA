"""
run_ablations.py — Comprehensive ablation study for UCGA.

Runs all UCGA ablation variants on CIFAR-10 and AG News with multiple seeds.
Reports mean ± std accuracy and convergence speed (Δ columns).

Usage:
    python experiments/run_ablations.py [--seeds 5] [--epochs 20] [--dataset cifar10]

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
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from experiments.ablation_models import create_ablation_variants
from utils.logger import get_logger

logger = get_logger("run_ablations")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "ablations")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Data loaders
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


def get_agnews_loaders(batch_size=128, vocab_size=8000):
    from collections import Counter
    from datasets import load_dataset

    logger.info("Loading AG News...")
    ds = load_dataset("ag_news", trust_remote_code=True)

    train_texts = ds["train"]["text"]
    train_labels = torch.tensor(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = torch.tensor(ds["test"]["label"])

    # Build vocabulary
    counter = Counter()
    for text in train_texts:
        counter.update(text.lower().split())
    word2idx = {w: i for i, (w, _) in enumerate(counter.most_common(vocab_size))}

    def texts_to_bow(texts):
        dim = len(word2idx)
        vecs = torch.zeros(len(texts), dim)
        for i, text in enumerate(texts):
            for w in text.lower().split():
                if w in word2idx:
                    vecs[i, word2idx[w]] += 1
            n = vecs[i].sum()
            if n > 0:
                vecs[i] /= n
        return vecs

    train_X = texts_to_bow(train_texts)
    test_X = texts_to_bow(test_texts)

    return (
        DataLoader(TensorDataset(train_X, train_labels), batch_size=batch_size,
                   shuffle=True, drop_last=True),
        DataLoader(TensorDataset(test_X, test_labels), batch_size=batch_size,
                   shuffle=False),
    )


# ======================================================================
# Training
# ======================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_one_seed(model, train_loader, test_loader, device, epochs, seed, use_amp=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    best_acc = 0.0
    epoch_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        epoch_accs.append(acc)
        if acc > best_acc:
            best_acc = acc

    # Convergence speed: first epoch to reach 90% of best
    target = best_acc * 0.9
    conv_epoch = epochs
    for i, a in enumerate(epoch_accs):
        if a >= target:
            conv_epoch = i + 1
            break

    return best_acc, conv_epoch


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="UCGA Ablation Study")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "agnews"])
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Ablation Study | Dataset: {args.dataset} | Seeds: {args.seeds} | Epochs: {args.epochs}")

    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    else:
        train_loader, test_loader = get_agnews_loaders(args.batch_size)

    variants = create_ablation_variants(mode=args.dataset)

    results = {}
    full_ucga_acc = None

    for name, factory_fn in variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Variant: {name}")
        logger.info(f"{'='*60}")

        accs, convs = [], []
        for seed in range(args.seeds):
            logger.info(f"  Seed {seed+1}/{args.seeds}...")
            model = factory_fn()
            acc, conv = train_one_seed(
                model, train_loader, test_loader, device,
                args.epochs, seed=42 + seed, use_amp=args.amp,
            )
            accs.append(acc)
            convs.append(conv)
            logger.info(f"    Acc: {acc:.4f} | Conv@90%: epoch {conv}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_conv = float(np.mean(convs))
        std_conv = float(np.std(convs))

        if name == "Full UCGA":
            full_ucga_acc = mean_acc

        results[name] = {
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "conv_mean": mean_conv,
            "conv_std": std_conv,
            "all_accs": accs,
            "all_convs": convs,
        }

    # Print results table with Δ columns
    logger.info(f"\n{'='*90}")
    logger.info(f"  ABLATION RESULTS ({args.dataset.upper()})")
    logger.info(f"{'='*90}")
    header = f"{'Variant':<18} {'Accuracy':>14} {'Conv(ep)':>12} {'Δ Acc':>10} {'Δ Conv':>10}"
    logger.info(header)
    logger.info("-" * 90)

    for name, r in results.items():
        delta_acc = r["acc_mean"] - full_ucga_acc if full_ucga_acc else 0.0
        delta_conv = r["conv_mean"] - results.get("Full UCGA", r)["conv_mean"]
        logger.info(
            f"{name:<18} "
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f} "
            f"{r['conv_mean']:>6.1f}±{r['conv_std']:.1f} "
            f"{delta_acc:>+8.4f} "
            f"{delta_conv:>+8.1f}"
        )

    # Save
    out_path = os.path.join(RESULTS_DIR, f"ablation_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
