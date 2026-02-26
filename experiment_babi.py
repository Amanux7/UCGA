import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from ucga.ucga_model import UCGAModel

def run_experiment(T_values=[1, 3], seeds=5, epochs=60, batch_size=256):
    print("--- UCGA Phase 2: Multi-Hop Reasoning Experiment ---")
    print(f"Testing T values: {T_values} over {seeds} seeds")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1) Data Generate (Synthetic Multi-Hop)
    # Task: Fact 1 (E1 -> E2), Fact 2 (E2 -> E3), Question: Where is E1? (Ans: E3)
    # Vocabulary size = 20 entities
    N = 10000
    vocab_size = 20
    embed_dim = 8
    
    # Generate random entities
    E1 = torch.randint(2, vocab_size, (N,)) # Reserve 0 and 1 for Fact/Question tokens
    E2 = torch.randint(2, vocab_size, (N,))
    E3 = torch.randint(2, vocab_size, (N,))
    
    # Build raw dataset (B, 3 sentences, 3 tokens)
    # Sentence 1: [0, E1, E2] (Fact)
    # Sentence 2: [0, E2, E3] (Fact)
    # Sentence 3: [1, E1, 0]  (Question: Where is E1?)
    X_raw = torch.zeros(N, 3, 3, dtype=torch.long)
    X_raw[:, 0, 0] = 0; X_raw[:, 0, 1] = E1; X_raw[:, 0, 2] = E2
    X_raw[:, 1, 0] = 0; X_raw[:, 1, 1] = E2; X_raw[:, 1, 2] = E3
    X_raw[:, 2, 0] = 1; X_raw[:, 2, 1] = E1; X_raw[:, 2, 2] = 0
    
    Y = E3.clone().long() # Target is E3

    # Split
    train_x_raw = X_raw[:8000].to(device)
    train_y = Y[:8000].to(device)
    test_x_raw = X_raw[8000:].to(device)
    test_y = Y[8000:].to(device)
    
    train_dataset = TensorDataset(train_x_raw, train_y)
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
            # T-loop architecture needs to predict V=20 classes.
            # We embed tokens, flatten them, and project to state_dim.
            # Flattened embedding size: 3 sentences * 3 tokens * embed_dim = 9 * embed_dim = 72
            model = UCGAModel(input_dim=64, state_dim=64, output_dim=vocab_size).to(device)
            embedding = nn.Embedding(vocab_size, embed_dim).to(device)
            input_proj = nn.Linear(9 * embed_dim, 64).to(device)
            
            p_count = model.count_parameters() + sum(p.numel() for p in input_proj.parameters()) + sum(p.numel() for p in embedding.parameters())
            if seed == 0:
                print(f"[Seed {seed}] Model initialized with {p_count:,} parameters.")
            
            optimizer = torch.optim.Adam(list(model.parameters()) + list(input_proj.parameters()) + list(embedding.parameters()), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # 3) Train loop
            start_time = time.time()
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    # bx: (B, 3, 3) -> embed -> (B, 3, 3, 8) -> flatten -> (B, 72)
                    emb = embedding(bx).view(bx.size(0), -1)
                    x_proj = input_proj(emb)
                    
                    out = model(x_proj, cognitive_steps=T)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * bx.size(0)
                    preds = out.argmax(dim=1)
                    correct += (preds == by).sum().item()
                    total += by.size(0)
                
                if (epoch + 1) % 10 == 0 and seed == 0:
                    print(f"  Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_loss/8000:.4f}, Acc: {correct/total:.4f}")
            
            run_time = time.time() - start_time
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                emb = embedding(test_x_raw).view(test_x_raw.size(0), -1)
                test_x_proj = input_proj(emb)
                out = model(test_x_proj, cognitive_steps=T)
                preds = out.argmax(dim=1)
                acc = (preds == test_y).sum().item() / test_y.size(0)
                
            print(f"[*] Seed {seed+1}/{seeds} | Test Acc: {acc:.4f} | Time: {run_time:.1f}s")
            results[T].append(acc)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS: MULTI-HOP REASONING")
    print("="*60)
    for T in T_values:
        arr = np.array(results[T])
        mean_acc = arr.mean()
        std_acc = arr.std()
        print(f"T={T}: {mean_acc:.4f} \u00b1 {std_acc:.4f}")
        
    print("\nInterpretation:")
    print("If T=3 accuracy is significantly higher than T=1 accuracy, then iterative")
    print("refinement allows the model to map Fact 1 -> Fact 2 -> Answer.")

if __name__ == "__main__":
    run_experiment(T_values=[1, 3], seeds=5, epochs=60)
