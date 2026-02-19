"""
train_agnews.py — Train UCGA on the AG News text classification dataset.

AG News has 4 classes:  World · Sports · Business · Sci/Tech
120,000 train /  7,600 test samples.

Uses a bag-of-words (BOW) representation fed through a VectorEncoder
into the UCGA cognitive loop.  This approach avoids the gradient-
vanishing problem that plagues small learned-embedding encoders on
classification tasks and achieves strong results with simple features.

Uses the HuggingFace ``datasets`` library for download.

Usage:
    python training/train_agnews.py

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
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("train_agnews")

# ======================================================================
# Hyperparameters
# ======================================================================
VOCAB_SIZE = 8_000          # BOW vocabulary size
STATE_DIM = 128
NUM_CLASSES = 4
MEMORY_SLOTS = 32
COGNITIVE_STEPS = 1
REASONING_STEPS = 1
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# ======================================================================
# BOW featurizer
# ======================================================================
def build_vocab(texts, vocab_size=8_000):
    """Build word → index mapping from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    top_words = [w for w, _ in counter.most_common(vocab_size)]
    word2idx = {w: i for i, w in enumerate(top_words)}
    return word2idx


def texts_to_bow(texts, word2idx):
    """Convert a list of texts to normalised bag-of-words tensors."""
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


# ======================================================================
# Data loading
# ======================================================================
def load_agnews():
    """Download AG News and prepare BOW dataloaders."""
    from datasets import load_dataset

    logger.info("Downloading AG News dataset...")
    ds = load_dataset("ag_news", trust_remote_code=True)

    train_texts = ds["train"]["text"]
    train_labels = torch.tensor(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = torch.tensor(ds["test"]["label"])

    # Build vocabulary
    logger.info("Building BOW vocabulary...")
    word2idx = build_vocab(train_texts, VOCAB_SIZE)
    logger.info(f"Vocabulary size: {len(word2idx):,}")

    # Convert to BOW
    logger.info("Featurizing training data...")
    train_X = texts_to_bow(train_texts, word2idx)
    logger.info("Featurizing test data...")
    test_X = texts_to_bow(test_texts, word2idx)

    train_loader = DataLoader(
        TensorDataset(train_X, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    return train_loader, test_loader


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, encoder, test_loader):
    """Evaluate on the test set and return accuracy."""
    model.eval()
    encoder.eval()
    correct = 0
    total = 0

    for bow_vecs, labels in test_loader:
        bow_vecs, labels = bow_vecs.to(DEVICE), labels.to(DEVICE)
        encoded = encoder(bow_vecs)
        logits = model(encoded)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ======================================================================
# Training
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA × AG News Training  (BOW features)")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    train_loader, test_loader = load_agnews()
    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Test  batches: {len(test_loader):,}")

    # ---- Model ----
    encoder = VectorEncoder(
        input_dim=VOCAB_SIZE, output_dim=STATE_DIM
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
        for bow_vecs, labels in pbar:
            bow_vecs, labels = bow_vecs.to(DEVICE), labels.to(DEVICE)

            encoded = encoder(bow_vecs)
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
        test_acc = evaluate(model, encoder, test_loader)

        tag = ""
        if test_acc > best_acc:
            best_acc = test_acc
            tag = " ★"

        logger.info(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Train Acc: {train_acc:.4f}  |  "
            f"Test Acc: {test_acc:.4f}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}{tag}"
        )

    logger.info("")
    logger.info(f"Training complete.  Best test accuracy: {best_acc:.4f}")
    logger.info(f"Classes: {AG_NEWS_CLASSES}")


if __name__ == "__main__":
    main()
