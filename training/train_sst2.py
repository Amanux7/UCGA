"""
train_sst2.py — Train UCGA on SST-2 sentiment classification.

SST-2 (Stanford Sentiment Treebank v2) is a binary sentiment analysis
benchmark with ~67k training and ~872 validation sentences.

Uses the new TransformerTextEncoder + UCGA cognitive loop to classify
movie reviews as positive or negative.

Requires:  pip install datasets

Usage:
    python training/train_sst2.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import TransformerTextEncoder
from utils.logger import get_logger

logger = get_logger("train_sst2")

# ======================================================================
# Hyperparameters
# ======================================================================
VOCAB_SIZE = 15_000         # Token vocabulary size
MAX_SEQ_LEN = 64            # Max tokens per sentence
EMBED_DIM = 128             # Transformer embed dim
STATE_DIM = 128             # Cognitive state dim
NUM_CLASSES = 2             # Positive / Negative
MEMORY_SLOTS = 32
COGNITIVE_STEPS = 1
REASONING_STEPS = 1
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SST2_LABELS = ["Negative", "Positive"]

# ======================================================================
# Tokenizer
# ======================================================================
def build_vocab(texts, vocab_size=15_000):
    """Build word → index mapping.  Index 0 is reserved for padding."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    top_words = [w for w, _ in counter.most_common(vocab_size - 1)]
    word2idx = {w: i + 1 for i, w in enumerate(top_words)}  # 0 = pad
    return word2idx


def tokenize_batch(texts, word2idx, max_len=64):
    """Convert texts to padded token-ID tensors."""
    batch = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        for j, w in enumerate(tokens):
            batch[i, j] = word2idx.get(w, 0)
    return batch


# ======================================================================
# Data loading
# ======================================================================
def load_sst2():
    """Download SST-2 and prepare dataloaders."""
    from datasets import load_dataset

    logger.info("Downloading SST-2 dataset...")
    ds = load_dataset("glue", "sst2", trust_remote_code=True)

    train_texts = ds["train"]["sentence"]
    train_labels = torch.tensor(ds["train"]["label"])
    val_texts = ds["validation"]["sentence"]
    val_labels = torch.tensor(ds["validation"]["label"])

    # Build vocabulary from training data
    logger.info("Building vocabulary...")
    word2idx = build_vocab(train_texts, VOCAB_SIZE)
    logger.info(f"Vocabulary size: {len(word2idx):,}")

    # Tokenize
    logger.info("Tokenizing training data...")
    train_ids = tokenize_batch(train_texts, word2idx, MAX_SEQ_LEN)
    logger.info("Tokenizing validation data...")
    val_ids = tokenize_batch(val_texts, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(
        TensorDataset(train_ids, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_ids, val_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    return train_loader, val_loader


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, encoder, val_loader):
    """Evaluate on the validation set and return accuracy."""
    model.eval()
    encoder.eval()
    correct = 0
    total = 0

    for token_ids, labels in val_loader:
        token_ids, labels = token_ids.to(DEVICE), labels.to(DEVICE)
        encoded = encoder(token_ids)
        logits = model(encoded)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA × SST-2 Sentiment Classification")
    logger.info("  Encoder: TransformerTextEncoder (Phase 2)")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    train_loader, val_loader = load_sst2()
    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val   batches: {len(val_loader):,}")

    # ---- Model ----
    encoder = TransformerTextEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        output_dim=STATE_DIM,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(DEVICE)

    model = UCGAModel(
        input_dim=STATE_DIM,
        state_dim=STATE_DIM,
        output_dim=NUM_CLASSES,
        memory_slots=MEMORY_SLOTS,
        cognitive_steps=COGNITIVE_STEPS,
        reasoning_steps=REASONING_STEPS,
        correction_threshold=0.3,
    ).to(DEVICE)

    total_params = model.count_parameters() + sum(
        p.numel() for p in encoder.parameters()
    )
    logger.info(f"Total parameters: {total_params:,}")

    # ---- Optimiser ----
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False
        )
        for token_ids, labels in pbar:
            token_ids, labels = token_ids.to(DEVICE), labels.to(DEVICE)

            encoded = encoder(token_ids)
            logits = model(encoded)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{correct / total:.3f}",
            )

        scheduler.step()

        train_acc = correct / total
        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, encoder, val_loader)

        tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            tag = " ★"

        logger.info(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Train Acc: {train_acc:.4f}  |  "
            f"Val Acc: {val_acc:.4f}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}{tag}"
        )

    logger.info("")
    logger.info(f"Training complete.  Best validation accuracy: {best_acc:.4f}")
    logger.info(f"Classes: {SST2_LABELS}")


if __name__ == "__main__":
    main()
