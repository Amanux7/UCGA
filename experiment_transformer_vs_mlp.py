"""
UCGA Phase 3 Experiment — MLP vs Transformer ReasoningNode
Compares 4 variants across Sorting and Parity tasks:
  - mlp   / T=1
  - mlp   / T=3
  - transformer / T=1
  - transformer / T=3

Tracks: Test performance, per-step state norms, gradient norms, training time.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from ucga.ucga_model import UCGAModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_grad_norm(model: nn.Module) -> float:
    """Return L2 norm of all gradients concatenated."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def build_model(reasoning_type: str, state_dim: int = 64, output_dim: int = 1):
    model = UCGAModel(input_dim=state_dim, state_dim=state_dim,
                      output_dim=output_dim, reasoning_type=reasoning_type)
    return model


# ---------------------------------------------------------------------------
# Sorting experiment (MSE regression)
# ---------------------------------------------------------------------------

def run_sorting(T_values, reasoning_types, seeds=5, epochs=50, batch_size=256, seq_len=8):
    state_dim = 64

    N = 10000
    x = torch.rand(N, seq_len) * 10.0
    y, _ = torch.sort(x, dim=1)

    train_x = x[:8000]; train_y = y[:8000]
    test_x  = x[8000:]; test_y  = y[8000:]

    train_ds  = TensorDataset(train_x, train_y)
    criterion = nn.MSELoss()

    results = {}

    for rtype in reasoning_types:
        for T in T_values:
            key = f"{rtype}/T={T}"
            results[key] = {"mse": [], "time": [], "state_norms": [], "grad_norms": []}
            print(f"\n=== Sorting | {key} ===")

            for seed in range(seeds):
                torch.manual_seed(seed); np.random.seed(seed)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model     = build_model(rtype, state_dim=state_dim, output_dim=seq_len).to(device)
                input_proj = nn.Linear(seq_len, state_dim).to(device)
                params     = list(model.parameters()) + list(input_proj.parameters())
                optimizer  = torch.optim.Adam(params, lr=1e-3)
                loader     = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                step_state_norms = []
                step_grad_norms  = []

                t0 = time.time()
                for epoch in range(epochs):
                    model.train()
                    e_loss = 0.0
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out, meta = model(input_proj(bx), cognitive_steps=T, return_meta=True)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        e_loss += loss.item() * bx.size(0)
                        if epoch == epochs - 1:  # Log on last epoch
                            step_state_norms.append(np.mean(meta.get("state_norms", [0])))
                            step_grad_norms.append(compute_grad_norm(model))

                    if (epoch + 1) % 10 == 0 and seed == 0:
                        print(f"  Epoch {epoch+1:02d}/{epochs} | Train MSE: {e_loss/8000:.4f}")

                elapsed = time.time() - t0

                model.eval()
                with torch.no_grad():
                    preds = model(input_proj(test_x.to(device)), cognitive_steps=T)
                    mse = criterion(preds, test_y.to(device)).item()

                results[key]["mse"].append(mse)
                results[key]["time"].append(elapsed)
                results[key]["state_norms"].append(np.mean(step_state_norms) if step_state_norms else 0)
                results[key]["grad_norms"].append(np.mean(step_grad_norms) if step_grad_norms else 0)

                print(f"  [*] Seed {seed+1}/{seeds} | MSE: {mse:.4f} | Time: {elapsed:.1f}s | "
                      f"StateNorm: {results[key]['state_norms'][-1]:.3f} | "
                      f"GradNorm: {results[key]['grad_norms'][-1]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Parity experiment (binary classification)
# ---------------------------------------------------------------------------

def run_parity(T_values, reasoning_types, seeds=5, epochs=50, batch_size=256, seq_len=16):
    state_dim = 64

    N = 10000
    x = torch.randint(0, 2, (N, seq_len), dtype=torch.float32)
    y = (x.sum(dim=1) % 2).long()

    train_x = x[:8000]; train_y = y[:8000]
    test_x  = x[8000:]; test_y  = y[8000:]

    train_ds  = TensorDataset(train_x, train_y)
    criterion = nn.CrossEntropyLoss()

    results = {}

    for rtype in reasoning_types:
        for T in T_values:
            key = f"{rtype}/T={T}"
            results[key] = {"acc": [], "time": [], "state_norms": [], "grad_norms": []}
            print(f"\n=== Parity | {key} ===")

            for seed in range(seeds):
                torch.manual_seed(seed); np.random.seed(seed)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model      = build_model(rtype, state_dim=state_dim, output_dim=2).to(device)
                input_proj = nn.Linear(seq_len, state_dim).to(device)
                params     = list(model.parameters()) + list(input_proj.parameters())
                optimizer  = torch.optim.Adam(params, lr=1e-3)
                loader     = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                step_state_norms = []
                step_grad_norms  = []

                t0 = time.time()
                for epoch in range(epochs):
                    model.train()
                    correct = total = 0
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out, meta = model(input_proj(bx), cognitive_steps=T, return_meta=True)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        preds = out.argmax(dim=1)
                        correct += (preds == by).sum().item()
                        total   += by.size(0)
                        if epoch == epochs - 1:
                            step_state_norms.append(np.mean(meta.get("state_norms", [0])))
                            step_grad_norms.append(compute_grad_norm(model))

                    if (epoch + 1) % 10 == 0 and seed == 0:
                        print(f"  Epoch {epoch+1:02d}/{epochs} | Train Acc: {correct/total:.4f}")

                elapsed = time.time() - t0

                model.eval()
                with torch.no_grad():
                    out = model(input_proj(test_x.to(device)), cognitive_steps=T)
                    preds = out.argmax(dim=1)
                    acc = (preds == test_y.to(device)).sum().item() / test_y.size(0)

                results[key]["acc"].append(acc)
                results[key]["time"].append(elapsed)
                results[key]["state_norms"].append(np.mean(step_state_norms) if step_state_norms else 0)
                results[key]["grad_norms"].append(np.mean(step_grad_norms) if step_grad_norms else 0)

                print(f"  [*] Seed {seed+1}/{seeds} | Acc: {acc:.4f} | Time: {elapsed:.1f}s | "
                      f"StateNorm: {results[key]['state_norms'][-1]:.3f} | "
                      f"GradNorm: {results[key]['grad_norms'][-1]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_table(sorting_results, parity_results):
    T_values       = [1, 3]
    reasoning_types = ["mlp", "transformer"]

    header = f"{'Variant':<28} | {'Sort MSE':>10} | {'Sort Std':>10} | {'Parity Acc':>10} | {'Par Std':>8} | {'Time(s)':>9} | {'StateNorm':>10} | {'GradNorm':>10}"
    divider = "-" * len(header)
    print("\n" + divider)
    print("FINAL COMPARISON TABLE")
    print(divider)
    print(header)
    print(divider)

    for rtype in reasoning_types:
        for T in T_values:
            key = f"{rtype}/T={T}"

            s_mse  = np.mean(sorting_results[key]["mse"])
            s_std  = np.std(sorting_results[key]["mse"])
            p_acc  = np.mean(parity_results[key]["acc"])
            p_std  = np.std(parity_results[key]["acc"])
            t_mean = np.mean(sorting_results[key]["time"])
            sn     = np.mean(sorting_results[key]["state_norms"])
            gn     = np.mean(sorting_results[key]["grad_norms"])

            label = f"{rtype.upper():>11}  T={T}"
            print(f"{label:<28} | {s_mse:>10.4f} | {s_std:>10.4f} | {p_acc:>10.4f} | {p_std:>8.4f} | {t_mean:>9.1f} | {sn:>10.4f} | {gn:>10.4f}")

    print(divider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    T_values        = [1, 3]
    reasoning_types = ["mlp", "transformer"]
    SEEDS           = 5
    EPOCHS          = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running 4 variants x {SEEDS} seeds x {EPOCHS} epochs each on Sorting + Parity\n")

    sorting_results = run_sorting(
        T_values=T_values,
        reasoning_types=reasoning_types,
        seeds=SEEDS,
        epochs=EPOCHS,
    )

    parity_results = run_parity(
        T_values=T_values,
        reasoning_types=reasoning_types,
        seeds=SEEDS,
        epochs=EPOCHS,
    )

    print_table(sorting_results, parity_results)
