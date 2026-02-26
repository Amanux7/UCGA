import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from ucga.ucga_model import UCGAModel

def run_experiment(T_values=[1, 3], seeds=5, epochs=50, batch_size=256, seq_len=8):
    print("--- UCGA Phase 2: Sorting Experiment ---")
    print(f"Testing T values: {T_values} over {seeds} seeds")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Seq Len: {seq_len}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1) Data Generate (Regression: Sorting 8 numbers)
    N = 10000
    # Random floats between 0 and 10
    x = torch.rand(N, seq_len, dtype=torch.float32) * 10.0
    y, _ = torch.sort(x, dim=1) # Target is the sorted values

    # Split
    train_x = x[:8000].to(device)
    train_y = y[:8000].to(device)
    test_x = x[8000:].to(device)
    test_y = y[8000:].to(device)
    
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
            
            # 2) Model
            # pre-project seq_len to state_dim=64
            # predict seq_len outputs continuously
            model = UCGAModel(input_dim=64, state_dim=64, output_dim=seq_len).to(device)
            input_proj = nn.Linear(seq_len, 64).to(device)
            
            p_count = model.count_parameters() + sum(p.numel() for p in input_proj.parameters())
            if seed == 0:
                print(f"[Seed {seed}] Model initialized with {p_count:,} parameters.")
            
            optimizer = torch.optim.Adam(list(model.parameters()) + list(input_proj.parameters()), lr=1e-3)
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
                    print(f"  Epoch {epoch+1:02d}/{epochs} | Train MSE: {epoch_loss/8000:.4f}")
            
            run_time = time.time() - start_time
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_x_proj = input_proj(test_x)
                preds = model(test_x_proj, cognitive_steps=T)
                mse = criterion(preds, test_y).item()
                
            print(f"[*] Seed {seed+1}/{seeds} | Test MSE: {mse:.4f} | Time: {run_time:.1f}s")
            results[T].append(mse)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS: SORTING")
    print("="*60)
    for T in T_values:
        arr = np.array(results[T])
        mean_mse = arr.mean()
        std_mse = arr.std()
        print(f"T={T}: {mean_mse:.4f} \u00b1 {std_mse:.4f}")
        
    print("\nInterpretation:")
    print("If T=3 MSE is lower than T=1 MSE, iterating helps structural re-ordering.")

if __name__ == "__main__":
    run_experiment(T_values=[1, 3], seeds=5, epochs=50)
