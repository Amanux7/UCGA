import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from ucga.ucga_model import UCGAModel

def run_experiment(T_values=[1, 3], seeds=5, epochs=50, batch_size=256):
    print("--- UCGA Regression Experiment: Recursion Utility ---")
    print(f"Testing T values: {T_values} over {seeds} seeds")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1) Data Generate (Synthetic polynomial)
    N = 8000
    x = torch.randn(N, 1)
    a = torch.randn(N, 1)
    b = torch.randn(N, 1)
    c = torch.randn(N, 1)
    y = a * x**2 + b * x + c
    dataset = torch.cat([x, a, b, c], dim=1)

    # Split
    train_x = dataset[:6000].to(device)
    train_y = y[:6000].to(device)
    test_x = dataset[6000:].to(device)
    test_y = y[6000:].to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    results = {T: [] for T in T_values}

    for T in T_values:
        print("="*60)
        print(f"--- Training for T={T} ---")
        print("="*60)
        
        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 2) Model (regression version)
            # input_dim=64 to match state_dim=64 so that residual x = balanced + x_original works.
            # We pre-project the 4D input to 64D.
            model = UCGAModel(input_dim=64, state_dim=64, output_dim=1).to(device)
            input_proj = nn.Linear(4, 64).to(device)
            
            p_count = model.count_parameters() + sum(p.numel() for p in input_proj.parameters())
            if seed == 0:
                print(f"[Seed {seed}] Model initialized with {p_count:,} parameters.")
            
            optimizer = torch.optim.Adam(list(model.parameters()) + list(input_proj.parameters()), lr=5e-4)
            criterion = nn.MSELoss()

            # 3) Train loop
            start_time = time.time()
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    x_proj = input_proj(bx)
                    out = model(x_proj, cognitive_steps=T)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * bx.size(0)
                
                if (epoch + 1) % 10 == 0 and seed == 0:
                    print(f"  Epoch {epoch+1:02d}/{epochs} | Train MSE: {epoch_loss/6000:.4f}")
            
            run_time = time.time() - start_time
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_x_proj = input_proj(test_x)
                preds = model(test_x_proj, cognitive_steps=T)
                mse = criterion(preds.squeeze(), test_y.squeeze())
                
            print(f"[*] Seed {seed+1}/{seeds} | Test MSE: {mse.item():.4f} | Time: {run_time:.1f}s")
            results[T].append(mse.item())

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS: T-LOOP RECURSION UTILITY")
    print("="*60)
    for T in T_values:
        arr = np.array(results[T])
        mean_mse = arr.mean()
        std_mse = arr.std()
        print(f"T={T}: {mean_mse:.4f} \u00b1 {std_mse:.4f}")
        
    print("\nInterpretation:")
    print("If T=3 MSE is significantly lower than T=1 MSE, then iterative")
    print("refinement allows the model to approximate the polynomial function better.")

if __name__ == "__main__":
    run_experiment(T_values=[1, 3], seeds=5, epochs=50)
