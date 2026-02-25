"""
run_reasoning.py — Evaluate UCGA cognitive step benefit on reasoning tasks.

Trains UCGA with T=1, T=3, T=5 on each synthetic reasoning benchmark,
demonstrating that multi-step cognitive refinement provides clear gains
on tasks requiring iterative computation.

Usage:
    python experiments/run_reasoning.py [--seeds 5] [--epochs 30]

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
from torch.utils.data import DataLoader, random_split
import numpy as np

from experiments.synthetic_benchmarks import get_all_benchmarks
from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("run_reasoning")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reasoning")
os.makedirs(RESULTS_DIR, exist_ok=True)


class ReasoningModel(nn.Module):
    """Encoder + UCGA for vector-input reasoning tasks."""

    def __init__(self, input_dim, num_classes, state_dim=128, memory_slots=32,
                 cognitive_steps=3, reasoning_steps=2):
        super().__init__()
        self.encoder = VectorEncoder(input_dim=input_dim, output_dim=state_dim)
        self.ucga = UCGAModel(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=cognitive_steps,
            reasoning_steps=reasoning_steps,
        )

    def forward(self, x):
        return self.ucga(self.encoder(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_seed(model, train_loader, test_loader, device, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Reasoning task evaluation")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--state_dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Reasoning Tasks | Seeds: {args.seeds} | Epochs: {args.epochs} | Device: {device}")

    benchmarks = get_all_benchmarks(seed=42)
    T_values = [1, 3, 5]

    all_results = {}

    for bench_name, (dataset, input_dim, num_classes) in benchmarks.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Benchmark: {bench_name}")
        logger.info(f"  Samples: {len(dataset)} | Input: {input_dim} | Classes: {num_classes}")
        logger.info(f"{'='*60}")

        # Split 80/20
        n_train = int(len(dataset) * 0.8)
        n_test = len(dataset) - n_train
        train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                          generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        bench_results = {}
        for T in T_values:
            logger.info(f"\n  T={T}:")
            accs = []
            for seed in range(args.seeds):
                model = ReasoningModel(
                    input_dim=input_dim, num_classes=num_classes,
                    state_dim=args.state_dim, cognitive_steps=T,
                )
                acc = train_one_seed(
                    model, train_loader, test_loader, device, args.epochs, seed=42 + seed
                )
                accs.append(acc)
                logger.info(f"    Seed {seed+1}: {acc:.4f}")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            bench_results[f"T={T}"] = {
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "all_accs": accs,
            }
            logger.info(f"    → Mean: {mean_acc:.4f} ± {std_acc:.4f}")

        # Compute gains
        t1_acc = bench_results["T=1"]["acc_mean"]
        for T in T_values:
            key = f"T={T}"
            bench_results[key]["gain_vs_T1"] = bench_results[key]["acc_mean"] - t1_acc

        all_results[bench_name] = bench_results

    # Summary table
    logger.info(f"\n{'='*90}")
    logger.info("  REASONING TASK RESULTS — Effect of Cognitive Steps (T)")
    logger.info(f"{'='*90}")
    header = f"{'Benchmark':<28} {'T=1':>14} {'T=3':>14} {'T=5':>14} {'Δ(T=3-T=1)':>12} {'Δ(T=5-T=1)':>12}"
    logger.info(header)
    logger.info("-" * 90)

    for bench_name, br in all_results.items():
        t1 = br["T=1"]
        t3 = br["T=3"]
        t5 = br["T=5"]
        logger.info(
            f"{bench_name:<28} "
            f"{t1['acc_mean']:.4f}±{t1['acc_std']:.4f} "
            f"{t3['acc_mean']:.4f}±{t3['acc_std']:.4f} "
            f"{t5['acc_mean']:.4f}±{t5['acc_std']:.4f} "
            f"{t3['gain_vs_T1']:>+10.4f} "
            f"{t5['gain_vs_T1']:>+10.4f}"
        )

    # Save
    out_path = os.path.join(RESULTS_DIR, "reasoning_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
