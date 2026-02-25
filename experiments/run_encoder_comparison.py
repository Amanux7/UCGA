"""
run_encoder_comparison.py — Side-by-side text encoder comparison on AG News.

Compares:
  1. BOW + VectorEncoder (current approach)
  2. Learned 1D-CNN (TextEncoder)
  3. Frozen MiniLM-L6-v2 + projection
  4. Frozen DistilBERT + projection (Colab only, --distilbert flag)

Reports accuracy, trainable params, total params, and wall-clock per epoch.

Usage:
    python experiments/run_encoder_comparison.py [--seeds 5] [--epochs 10]
    python experiments/run_encoder_comparison.py --distilbert   # Colab only

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
import numpy as np
from collections import Counter

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder, TextEncoder
from utils.compute_metrics import count_parameters, format_params
from utils.logger import get_logger

logger = get_logger("run_encoder_comparison")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "encoders")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Data loading
# ======================================================================
VOCAB_SIZE = 8_000
STATE_DIM = 128
NUM_CLASSES = 4
MAX_SEQ_LEN = 128


def load_agnews_raw():
    """Return raw texts and labels."""
    from datasets import load_dataset
    ds = load_dataset("ag_news", trust_remote_code=True)
    return (
        ds["train"]["text"], ds["train"]["label"],
        ds["test"]["text"], ds["test"]["label"],
    )


def build_vocab(texts, vocab_size=8000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    return {w: i for i, (w, _) in enumerate(counter.most_common(vocab_size))}


def texts_to_bow(texts, word2idx):
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


def texts_to_token_ids(texts, word2idx, max_len=128):
    """Convert texts to token ID sequences for CNN encoder."""
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        for j, w in enumerate(tokens):
            if w in word2idx:
                ids[i, j] = word2idx[w] + 1  # +1 for padding=0
    return ids


# ======================================================================
# Model wrappers
# ======================================================================
class BOWModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VectorEncoder(input_dim=VOCAB_SIZE, output_dim=STATE_DIM)
        self.ucga = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM, output_dim=NUM_CLASSES,
            memory_slots=32, cognitive_steps=2, reasoning_steps=2,
        )

    def forward(self, x):
        return self.ucga(self.encoder(x))


class CNNTextModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size=vocab_size + 1,  # +1 for padding
            embed_dim=64, output_dim=STATE_DIM, max_seq_len=MAX_SEQ_LEN,
        )
        self.ucga = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM, output_dim=NUM_CLASSES,
            memory_slots=32, cognitive_steps=2, reasoning_steps=2,
        )

    def forward(self, x):
        return self.ucga(self.encoder(x))


class PretrainedModel(nn.Module):
    """Model using frozen pretrained embeddings + UCGA."""
    def __init__(self, pretrained_dim, model_name):
        super().__init__()
        # Projection from pretrained dim → state_dim
        self.projection = nn.Sequential(
            nn.Linear(pretrained_dim, STATE_DIM),
            nn.LayerNorm(STATE_DIM),
            nn.GELU(),
            nn.Linear(STATE_DIM, STATE_DIM),
            nn.LayerNorm(STATE_DIM),
        )
        self.ucga = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM, output_dim=NUM_CLASSES,
            memory_slots=32, cognitive_steps=2, reasoning_steps=2,
        )
        self.model_name = model_name

    def forward(self, x):
        return self.ucga(self.projection(x))


def precompute_embeddings(texts, model_name, device, batch_size=256):
    """Pre-compute frozen embeddings from a pretrained model."""
    logger.info(f"  Pre-computing embeddings with {model_name}...")
    if "MiniLM" in model_name or "minilm" in model_name.lower():
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            st_model = st_model.half()
        with torch.no_grad():
            embeddings = st_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=True,
                device=device, batch_size=batch_size,
            )
        if embeddings.dtype == torch.float16:
            embeddings = embeddings.float()
        pretrained_dim = embeddings.shape[1]
        del st_model
    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModel.from_pretrained(model_name).to(device)
        if device == "cuda":
            hf_model = hf_model.half()
        hf_model.eval()

        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = hf_model(**encoded)
            emb = outputs.last_hidden_state.mean(dim=1).float()
            embeddings_list.append(emb.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        pretrained_dim = embeddings.shape[1]
        del hf_model

    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"  Embeddings: shape={embeddings.shape}, dim={pretrained_dim}")
    return embeddings, pretrained_dim


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


def train_one_seed(model, train_loader, test_loader, device, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
        scheduler.step()
        epoch_times.append(time.time() - t0)

        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc

    return best_acc, np.mean(epoch_times)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Encoder comparison on AG News")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--distilbert", action="store_true", help="Include DistilBERT (Colab only)")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Encoder Comparison | Seeds: {args.seeds} | Epochs: {args.epochs}")

    # Load data
    train_texts, train_labels_list, test_texts, test_labels_list = load_agnews_raw()
    train_labels = torch.tensor(train_labels_list)
    test_labels = torch.tensor(test_labels_list)
    word2idx = build_vocab(train_texts, VOCAB_SIZE)

    # Prepare data variants
    logger.info("Preparing BOW features...")
    train_bow = texts_to_bow(train_texts, word2idx)
    test_bow = texts_to_bow(test_texts, word2idx)

    logger.info("Preparing token ID sequences...")
    train_ids = texts_to_token_ids(train_texts, word2idx, MAX_SEQ_LEN)
    test_ids = texts_to_token_ids(test_texts, word2idx, MAX_SEQ_LEN)

    # Encoder configurations
    encoders = {
        "BOW + VectorEncoder": {
            "train_data": train_bow,
            "test_data": test_bow,
            "model_fn": lambda: BOWModel(),
        },
        "Learned 1D-CNN": {
            "train_data": train_ids,
            "test_data": test_ids,
            "model_fn": lambda: CNNTextModel(VOCAB_SIZE),
        },
    }

    # MiniLM
    logger.info("\nPre-computing MiniLM embeddings...")
    train_emb_mini, dim_mini = precompute_embeddings(
        train_texts, "all-MiniLM-L6-v2", device_str
    )
    test_emb_mini, _ = precompute_embeddings(
        test_texts, "all-MiniLM-L6-v2", device_str
    )
    encoders["Frozen MiniLM-L6-v2"] = {
        "train_data": train_emb_mini,
        "test_data": test_emb_mini,
        "model_fn": lambda: PretrainedModel(dim_mini, "all-MiniLM-L6-v2"),
    }

    # DistilBERT (optional)
    if args.distilbert:
        logger.info("\nPre-computing DistilBERT embeddings...")
        train_emb_db, dim_db = precompute_embeddings(
            train_texts, "distilbert-base-uncased", device_str
        )
        test_emb_db, _ = precompute_embeddings(
            test_texts, "distilbert-base-uncased", device_str
        )
        encoders["Frozen DistilBERT"] = {
            "train_data": train_emb_db,
            "test_data": test_emb_db,
            "model_fn": lambda: PretrainedModel(dim_db, "distilbert-base-uncased"),
        }

    # Run comparisons
    results = {}
    for enc_name, cfg in encoders.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Encoder: {enc_name}")
        logger.info(f"{'='*60}")

        train_loader = DataLoader(
            TensorDataset(cfg["train_data"], train_labels),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            TensorDataset(cfg["test_data"], test_labels),
            batch_size=args.batch_size, shuffle=False,
        )

        ref_model = cfg["model_fn"]()
        trainable_p = count_parameters(ref_model, trainable_only=True)
        total_p = count_parameters(ref_model, trainable_only=False)
        logger.info(f"  Trainable: {format_params(trainable_p)} | Total: {format_params(total_p)}")
        del ref_model

        accs, times = [], []
        for seed in range(args.seeds):
            model = cfg["model_fn"]()
            acc, avg_time = train_one_seed(
                model, train_loader, test_loader, device,
                args.epochs, seed=42 + seed,
            )
            accs.append(acc)
            times.append(avg_time)
            logger.info(f"  Seed {seed+1}: {acc:.4f} ({avg_time:.1f}s/epoch)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[enc_name] = {
            "trainable_params": trainable_p,
            "total_params": total_p,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "time_per_epoch": float(np.mean(times)),
        }

    # Summary table
    logger.info(f"\n{'='*85}")
    logger.info("  ENCODER COMPARISON RESULTS (AG News)")
    logger.info(f"{'='*85}")
    header = f"{'Encoder':<25} {'Train Params':>12} {'Total Params':>12} {'Accuracy':>14} {'s/epoch':>8}"
    logger.info(header)
    logger.info("-" * 85)
    for name, r in results.items():
        logger.info(
            f"{name:<25} "
            f"{format_params(r['trainable_params']):>12} "
            f"{format_params(r['total_params']):>12} "
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f} "
            f"{r['time_per_epoch']:>7.1f}"
        )

    # Save
    out_path = os.path.join(RESULTS_DIR, "encoder_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
