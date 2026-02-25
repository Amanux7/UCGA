"""
train_optimized.py — Unified, optimized UCGA training pipeline.

Implements all 10 training optimizations for scaling UCGA intelligence
from ~0.20 → 0.60+ WITHOUT modifying any architecture code.

Optimizations
─────────────
 1. Increased model capacity (state_dim=512, memory_slots=256, etc.)
 2. Training stability (AdamW, warmup, cosine schedule, grad clipping)
 3. Extended training (300 epochs, checkpoint every 10)
 4. Large batch size (128 + gradient accumulation)
 5. Mixed precision (torch.cuda.amp)
 6. Multi-phase dataset diversity (Synthetic → CIFAR-10 → AG News → Multimodal)
 7. Xavier initialization for all Linear layers
 8. Per-epoch intelligence monitoring with plots
 9. GPU optimizations (pin_memory, num_workers)
10. Architecture integrity verification

Usage:
    python training/train_optimized.py
    python training/train_optimized.py --epochs 5    # quick smoke test

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import hashlib
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder, ImageEncoder, MultimodalEncoder
from utils.logger import get_logger
from training.config import TrainingConfig

logger = get_logger("train_optimized")


# ======================================================================
# Step 10: Architecture Integrity Verification
# ======================================================================
ARCHITECTURE_FILES = [
    "ucga/ucga_model.py",
    "ucga/nodes/perception_node.py",
    "ucga/nodes/memory_node.py",
    "ucga/nodes/reasoning_node.py",
    "ucga/nodes/planning_node.py",
    "ucga/nodes/evaluation_node.py",
    "ucga/nodes/correction_node.py",
    "ucga/nodes/balancer_node.py",
    "ucga/nodes/output_node.py",
    "ucga/nodes/cognitive_node.py",
    "ucga/memory/persistent_memory.py",
]


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        return "NOT_FOUND"
    return h.hexdigest()


def verify_architecture_integrity(root_dir: str) -> dict:
    """
    Compute hashes of all architecture files.
    Returns a dict of {relative_path: sha256_hash}.
    """
    hashes = {}
    for rel in ARCHITECTURE_FILES:
        full = os.path.join(root_dir, rel)
        hashes[rel] = compute_file_hash(full)
    return hashes


def assert_no_modification(before: dict, after: dict):
    """Raise if any architecture file was modified during training."""
    for path, h in before.items():
        if after.get(path) != h:
            raise RuntimeError(
                f"ARCHITECTURE INTEGRITY VIOLATION: {path} was modified!"
            )
    logger.info("✓ Architecture integrity verified — no files modified.")


# ======================================================================
# Step 7: Xavier Initialization
# ======================================================================
def xavier_init(module: nn.Module):
    """Apply Xavier uniform initialization to all Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ======================================================================
# Step 2: Warmup + Cosine Scheduler
# ======================================================================
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        lr_scale = self._get_scale()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale

    def _get_scale(self) -> float:
        if self._step <= self.warmup_steps:
            return self._step / max(1, self.warmup_steps)
        progress = (self._step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self):
        scale = self._get_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ======================================================================
# Step 8: Lightweight Intelligence Monitor
# ======================================================================
class IntelligenceMonitor:
    """
    Lightweight per-epoch intelligence evaluation.

    Computes approximate ROS, LOS, and GIB scores without running
    full heavy-weight validation suites.
    """

    def __init__(self, model: UCGAModel, state_dim: int, device: str):
        self.model = model
        self.state_dim = state_dim
        self.device = device
        self.history = {
            "epoch": [],
            "intelligence_score": [],
            "ros_score": [],
            "los_score": [],
            "gib_score": [],
            "convergence_rate": [],
        }

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict:
        """Run lightweight intelligence evaluation."""
        self.model.eval()

        # --- ROS: Reasoning convergence ---
        x = torch.randn(4, self.state_dim, device=self.device)
        self.model._reset_all(4)
        x_loop = x
        x_original = x
        memory_bank = self.model.persistent_memory.get_memory_bank(4)
        prev_state = None
        errors = []

        for t in range(self.model.cognitive_steps):
            percept = self.model.perception([x_loop])
            mem_state = self.model.memory_node([percept], memory_bank=memory_bank)
            reason_state = self.model.reasoning([percept, mem_state])
            plan_state = self.model.planning([reason_state])
            eval_state = self.model.evaluation([plan_state, reason_state])
            confidence = self.model.evaluation.get_confidence()

            if confidence.mean().item() < self.model.correction_threshold:
                corrected = self.model.correction([plan_state, eval_state])
            else:
                corrected = plan_state

            balanced = self.model.balancer([reason_state, corrected, mem_state])
            x_loop = balanced + x_original

            if prev_state is not None:
                err = (reason_state - prev_state).norm(dim=-1).mean().item()
            else:
                err = reason_state.norm(dim=-1).mean().item()
            errors.append(err)
            prev_state = reason_state.clone()

        if errors[0] > 1e-8:
            ros_score = max(0.0, (errors[0] - errors[-1]) / errors[0])
        else:
            ros_score = 0.0

        if len(errors) > 1 and errors[0] > 1e-8:
            convergence_rate = 1.0 - (errors[-1] / errors[0]) ** (
                1.0 / (len(errors) - 1)
            )
        else:
            convergence_rate = 0.0

        # --- LOS: Memory contribution ---
        x_test = torch.randn(4, self.state_dim, device=self.device)
        out_with, meta_with = self.model(x_test, return_meta=True)

        saved_mem = self.model.persistent_memory.memory.clone()
        saved_use = self.model.persistent_memory.usage.clone()
        self.model.persistent_memory.memory.zero_()
        self.model.persistent_memory.usage.zero_()
        out_without, meta_without = self.model(x_test, return_meta=True)
        self.model.persistent_memory.memory.copy_(saved_mem)
        self.model.persistent_memory.usage.copy_(saved_use)

        conf_with = np.mean(meta_with["confidences"])
        conf_without = np.mean(meta_without["confidences"])
        los_score = conf_with - conf_without

        # --- GIB: Balancer adaptivity ---
        gib_weights = []
        for scale in [0.1, 1.0, 5.0]:
            x_gib = torch.randn(4, self.state_dim, device=self.device) * scale
            self.model._reset_all(4)
            x_loop = x_gib
            mem_bank = self.model.persistent_memory.get_memory_bank(4)

            for t in range(min(2, self.model.cognitive_steps)):
                p = self.model.perception([x_loop])
                m = self.model.memory_node([p], memory_bank=mem_bank)
                r = self.model.reasoning([p, m])
                pl = self.model.planning([r])
                ev = self.model.evaluation([pl, r])
                co = self.model.evaluation.get_confidence()
                cr = pl if co.mean().item() >= self.model.correction_threshold else self.model.correction([pl, ev])

                inputs = [r, cr, m]
                stacked = torch.stack(inputs, dim=1)
                B, S, D = stacked.shape
                concat = stacked.reshape(B, S * D)
                w = self.model.balancer.stream_gate(concat)
                gib_weights.append(w.mean(dim=0).cpu().numpy())

                balanced = self.model.balancer(inputs)
                x_loop = balanced + x_gib

        if gib_weights:
            wa = np.array(gib_weights)
            gib_score = float(wa.var(axis=0).mean())
        else:
            gib_score = 0.0

        # Composite
        intelligence = 0.4 * ros_score + 0.35 * los_score + 0.25 * gib_score

        self.history["epoch"].append(epoch)
        self.history["intelligence_score"].append(intelligence)
        self.history["ros_score"].append(ros_score)
        self.history["los_score"].append(los_score)
        self.history["gib_score"].append(gib_score)
        self.history["convergence_rate"].append(convergence_rate)

        self.model.train()
        return {
            "intelligence_score": intelligence,
            "ros_score": ros_score,
            "los_score": los_score,
            "gib_score": gib_score,
            "convergence_rate": convergence_rate,
        }

    def save_plots(self, save_dir: str):
        """Save intelligence progress plots."""
        os.makedirs(save_dir, exist_ok=True)
        epochs = self.history["epoch"]
        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("UCGA Intelligence Training Progress", fontsize=16, fontweight="bold")

        metrics = [
            ("intelligence_score", "Intelligence Score", "#264653"),
            ("ros_score", "ROS Score", "#e63946"),
            ("los_score", "LOS Score", "#2a9d8f"),
            ("gib_score", "GIB Score", "#457b9d"),
            ("convergence_rate", "Convergence Rate", "#e76f51"),
        ]

        for ax, (key, title, color) in zip(axes.flat, metrics):
            vals = self.history[key]
            ax.plot(epochs, vals, "-o", color=color, linewidth=2, markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.fill_between(epochs, vals, alpha=0.1, color=color)

        # Summary text in last subplot
        ax_last = axes.flat[-1]
        ax_last.axis("off")
        latest = {k: self.history[k][-1] for k in [
            "intelligence_score", "ros_score", "los_score",
            "gib_score", "convergence_rate",
        ]}
        summary = "\n".join(f"{k}: {v:.4f}" for k, v in latest.items())
        ax_last.text(0.5, 0.5, f"Latest Metrics\n\n{summary}",
                     transform=ax_last.transAxes, fontsize=12,
                     ha="center", va="center", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0"))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "intelligence_progress.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


# ======================================================================
# Step 6: Dataset Loaders
# ======================================================================
def make_synthetic_data(cfg: TrainingConfig):
    """Generate synthetic reasoning dataset."""
    dim = cfg.synthetic_input_dim
    device = cfg.device

    if not hasattr(make_synthetic_data, "W_true") or make_synthetic_data._dim != dim:
        make_synthetic_data.W_true = torch.randn(dim, dim * 2, device=device) * 0.5
        make_synthetic_data.bias = torch.randn(dim, device=device) * 0.1
        make_synthetic_data._dim = dim

    def gen(batch_size):
        a = torch.randn(batch_size, dim, device=device)
        b = torch.randn(batch_size, dim, device=device)
        x = torch.cat([a, b], dim=-1)
        y = torch.tanh(x @ make_synthetic_data.W_true.T + make_synthetic_data.bias)
        return x, y

    return gen


def make_cifar10_loaders(cfg: TrainingConfig):
    """Create CIFAR-10 train/test DataLoaders."""
    import torchvision
    import torchvision.transforms as T

    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=transform_test,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    return train_loader, test_loader


def make_agnews_loaders(cfg: TrainingConfig):
    """Create AG News train/test DataLoaders using BOW features."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset

    ds = load_dataset("ag_news", trust_remote_code=True)

    train_texts = ds["train"]["text"]
    train_labels = torch.tensor(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = torch.tensor(ds["test"]["label"])

    # Build vocabulary
    counter = Counter()
    for t in train_texts:
        counter.update(t.lower().split())
    top_words = [w for w, _ in counter.most_common(cfg.agnews_vocab_size)]
    word2idx = {w: i for i, w in enumerate(top_words)}

    def to_bow(texts):
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

    logger.info("Featurizing AG News data...")
    train_X = to_bow(train_texts)
    test_X = to_bow(test_texts)

    train_loader = DataLoader(
        TensorDataset(train_X, train_labels),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_labels),
        batch_size=cfg.batch_size, shuffle=False,
    )
    return train_loader, test_loader, len(word2idx)


def make_multimodal_data(cfg: TrainingConfig):
    """Generate synthetic multimodal image-text matching data."""
    from torch.utils.data import DataLoader, TensorDataset

    n = cfg.multimodal_num_samples
    n_classes = 5
    images = torch.zeros(n, 3, 32, 32)
    tokens = torch.zeros(n, cfg.multimodal_max_seq_len, dtype=torch.long)
    labels = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        ic = torch.randint(0, n_classes, (1,)).item()
        matched = torch.rand(1).item() > 0.5
        tc = ic if matched else (ic + torch.randint(1, n_classes, (1,)).item()) % n_classes
        labels[i] = 1 if matched else 0

        base = torch.zeros(3)
        base[ic % 3] = 0.8
        images[i] = base.view(3, 1, 1).expand(3, 32, 32)
        s = ic * 6
        images[i, :, s:s+6, s:s+6] += 0.5

        base_tok = tc * (cfg.multimodal_vocab_size // n_classes) + 1
        slen = torch.randint(4, cfg.multimodal_max_seq_len, (1,)).item()
        tokens[i, :slen] = torch.randint(
            base_tok, base_tok + cfg.multimodal_vocab_size // n_classes, (slen,),
        )

    images += torch.randn_like(images) * 0.1
    split = int(0.8 * n)

    train_loader = DataLoader(
        TensorDataset(images[:split], tokens[:split], labels[:split]),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(images[split:], tokens[split:], labels[split:]),
        batch_size=cfg.batch_size, shuffle=False,
    )
    return train_loader, val_loader


# ======================================================================
# Phase Training Loops
# ======================================================================
def train_phase_synthetic(
    model, encoder, optimizer, scheduler, scaler, cfg, monitor,
    start_epoch, end_epoch, best_score,
):
    """Phase A: Synthetic vector reasoning."""
    logger.info("=" * 60)
    logger.info("  Phase A: Synthetic Vector Reasoning")
    logger.info("=" * 60)

    gen = make_synthetic_data(cfg)
    criterion = nn.MSELoss()
    params = list(model.parameters()) + list(encoder.parameters())

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0

        pbar = tqdm(range(cfg.batches_per_epoch_synthetic),
                     desc=f"Epoch {epoch}/{end_epoch} [Synth]", leave=False)

        optimizer.zero_grad()
        for step, _ in enumerate(pbar):
            raw_x, target = gen(cfg.batch_size)

            # AMP: encode in float16, model forward in float32 (persistent memory needs float32)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                encoded = encoder(raw_x)
            encoded = encoded.float()
            output = model(encoded)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                loss = criterion(output, target)
                loss = loss / cfg.gradient_accumulation_steps

            if cfg.use_amp and cfg.device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.use_amp and cfg.device == "cuda":
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            pbar.set_postfix(loss=f"{loss.item() * cfg.gradient_accumulation_steps:.4f}")

        avg_loss = epoch_loss / cfg.batches_per_epoch_synthetic

        # Intelligence monitoring
        scores = monitor.evaluate(epoch)
        tag = ""
        if scores["intelligence_score"] > best_score:
            best_score = scores["intelligence_score"]
            save_checkpoint(model, encoder, optimizer, epoch, best_score, cfg, tag="best")
            tag = " ★"

        logger.info(
            f"Epoch {epoch:3d}  |  Loss: {avg_loss:.6f}  |  "
            f"IQ: {scores['intelligence_score']:.4f}  |  "
            f"ROS: {scores['ros_score']:.4f}  |  LOS: {scores['los_score']:.4f}  |  "
            f"GIB: {scores['gib_score']:.6f}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}{tag}"
        )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, encoder, optimizer, epoch, best_score, cfg)
            monitor.save_plots(cfg.plot_dir)

    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return best_score


def train_phase_cifar10(
    model, optimizer, scheduler, scaler, cfg, monitor,
    start_epoch, end_epoch, best_score,
):
    """Phase B: CIFAR-10 image classification."""
    logger.info("=" * 60)
    logger.info("  Phase B: CIFAR-10 Image Classification")
    logger.info("=" * 60)

    train_loader, test_loader = make_cifar10_loaders(cfg)
    encoder = ImageEncoder(in_channels=cfg.cifar_in_channels, output_dim=cfg.state_dim).to(cfg.device)
    encoder.apply(xavier_init)

    params = list(model.parameters()) + list(encoder.parameters())
    # Create a new optimizer that includes encoder params
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} [CIFAR]", leave=False)
        opt.zero_grad()

        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(cfg.device), labels.to(cfg.device)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                encoded = encoder(images)
            encoded = encoded.float()
            logits = model(encoded)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                loss = criterion(logits, labels)
                loss = loss / cfg.gradient_accumulation_steps

            if cfg.use_amp and cfg.device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.use_amp and cfg.device == "cuda":
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    opt.step()
                opt.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item() * cfg.gradient_accumulation_steps:.3f}",
                             acc=f"{correct/total:.3f}")

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # Test accuracy
        model.eval()
        encoder.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                logits = model(encoder(images))
                test_correct += (logits.argmax(-1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total

        scores = monitor.evaluate(epoch)
        tag = ""
        if scores["intelligence_score"] > best_score:
            best_score = scores["intelligence_score"]
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg, tag="best")
            tag = " ★"

        logger.info(
            f"Epoch {epoch:3d}  |  Loss: {avg_loss:.4f}  |  "
            f"Train: {train_acc:.4f}  |  Test: {test_acc:.4f}  |  "
            f"IQ: {scores['intelligence_score']:.4f}  |  "
            f"ROS: {scores['ros_score']:.4f}{tag}"
        )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg)
            monitor.save_plots(cfg.plot_dir)

    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return best_score


def train_phase_agnews(
    model, optimizer, scheduler, scaler, cfg, monitor,
    start_epoch, end_epoch, best_score,
):
    """Phase C: AG News text classification."""
    logger.info("=" * 60)
    logger.info("  Phase C: AG News Text Classification")
    logger.info("=" * 60)

    train_loader, test_loader, actual_vocab = make_agnews_loaders(cfg)
    encoder = VectorEncoder(input_dim=actual_vocab, output_dim=cfg.state_dim).to(cfg.device)
    encoder.apply(xavier_init)

    params = list(model.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} [AG]", leave=False)
        opt.zero_grad()

        for step, (bow, labels) in enumerate(pbar):
            bow, labels = bow.to(cfg.device), labels.to(cfg.device)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                encoded = encoder(bow)
            encoded = encoded.float()
            logits = model(encoded)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                loss = criterion(logits, labels)
                loss = loss / cfg.gradient_accumulation_steps

            if cfg.use_amp and cfg.device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.use_amp and cfg.device == "cuda":
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    opt.step()
                opt.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item() * cfg.gradient_accumulation_steps:.3f}",
                             acc=f"{correct/total:.3f}")

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # Test
        model.eval()
        encoder.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for bow, labels in test_loader:
                bow, labels = bow.to(cfg.device), labels.to(cfg.device)
                logits = model(encoder(bow))
                tc += (logits.argmax(-1) == labels).sum().item()
                tt += labels.size(0)
        test_acc = tc / tt

        scores = monitor.evaluate(epoch)
        tag = ""
        if scores["intelligence_score"] > best_score:
            best_score = scores["intelligence_score"]
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg, tag="best")
            tag = " ★"

        logger.info(
            f"Epoch {epoch:3d}  |  Loss: {avg_loss:.4f}  |  "
            f"Train: {train_acc:.4f}  |  Test: {test_acc:.4f}  |  "
            f"IQ: {scores['intelligence_score']:.4f}{tag}"
        )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg)
            monitor.save_plots(cfg.plot_dir)

    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return best_score


def train_phase_multimodal(
    model, optimizer, scheduler, scaler, cfg, monitor,
    start_epoch, end_epoch, best_score,
):
    """Phase D: Multimodal image-text matching."""
    logger.info("=" * 60)
    logger.info("  Phase D: Multimodal Image-Text Matching")
    logger.info("=" * 60)

    train_loader, val_loader = make_multimodal_data(cfg)
    encoder = MultimodalEncoder(
        image_channels=3,
        vocab_size=cfg.multimodal_vocab_size,
        embed_dim=cfg.multimodal_embed_dim,
        output_dim=cfg.state_dim,
        num_heads=4,
        num_fusion_layers=2,
        text_encoder_type="transformer",
        max_seq_len=cfg.multimodal_max_seq_len,
    ).to(cfg.device)
    encoder.apply(xavier_init)

    params = list(model.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} [MM]", leave=False)
        opt.zero_grad()

        for step, (imgs, toks, lbls) in enumerate(pbar):
            imgs = imgs.to(cfg.device)
            toks = toks.to(cfg.device)
            lbls = lbls.to(cfg.device)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                encoded = encoder(imgs, toks)
            encoded = encoded.float()
            logits = model(encoded)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                loss = criterion(logits, lbls)
                loss = loss / cfg.gradient_accumulation_steps

            if cfg.use_amp and cfg.device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.use_amp and cfg.device == "cuda":
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip_max_norm)
                    opt.step()
                opt.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            preds = logits.argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            pbar.set_postfix(loss=f"{loss.item() * cfg.gradient_accumulation_steps:.3f}",
                             acc=f"{correct/total:.3f}")

        # Validation
        model.eval()
        encoder.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, toks, lbls in val_loader:
                imgs = imgs.to(cfg.device)
                toks = toks.to(cfg.device)
                lbls = lbls.to(cfg.device)
                logits = model(encoder(imgs, toks))
                vc += (logits.argmax(-1) == lbls).sum().item()
                vt += lbls.size(0)
        val_acc = vc / vt if vt > 0 else 0.0

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        scores = monitor.evaluate(epoch)
        tag = ""
        if scores["intelligence_score"] > best_score:
            best_score = scores["intelligence_score"]
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg, tag="best")
            tag = " ★"

        logger.info(
            f"Epoch {epoch:3d}  |  Loss: {avg_loss:.4f}  |  "
            f"Train: {train_acc:.4f}  |  Val: {val_acc:.4f}  |  "
            f"IQ: {scores['intelligence_score']:.4f}{tag}"
        )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, encoder, opt, epoch, best_score, cfg)
            monitor.save_plots(cfg.plot_dir)

    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return best_score


# ======================================================================
# Checkpoint
# ======================================================================
def save_checkpoint(model, encoder, optimizer, epoch, best_score, cfg, tag=""):
    """Save a training checkpoint."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    suffix = f"_epoch{epoch}" if not tag else f"_{tag}"
    path = os.path.join(cfg.checkpoint_dir, f"ucga{suffix}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_intelligence_score": best_score,
    }, path)
    logger.info(f"  → Checkpoint saved: {path}")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="UCGA Optimized Training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs (for smoke tests)")
    parser.add_argument("--skip-phases", type=str, default="",
                        help="Phases to skip, e.g. 'BCD' to run only synthetic")
    args = parser.parse_args()

    cfg = TrainingConfig()

    if args.epochs:
        # Distribute epochs proportionally
        total = args.epochs
        ratios = [e / sum(cfg.phase_epochs) for e in cfg.phase_epochs]
        cfg.phase_epochs = [max(1, int(total * r)) for r in ratios]
        cfg.num_epochs = sum(cfg.phase_epochs)

    logger.info("=" * 60)
    logger.info("  UCGA Optimized Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Device:           {cfg.device}")
    logger.info(f"  State dim:        {cfg.state_dim}")
    logger.info(f"  Memory slots:     {cfg.memory_slots}")
    logger.info(f"  Cognitive steps:  {cfg.cognitive_steps}")
    logger.info(f"  Reasoning steps:  {cfg.reasoning_steps}")
    logger.info(f"  Batch size:       {cfg.batch_size}")
    logger.info(f"  Grad accum:       {cfg.gradient_accumulation_steps}")
    logger.info(f"  Mixed precision:  {cfg.use_amp}")
    logger.info(f"  Phase epochs:     {cfg.phase_epochs}")
    logger.info(f"  Total epochs:     {sum(cfg.phase_epochs)}")
    logger.info("=" * 60)

    root_dir = os.path.join(os.path.dirname(__file__), "..")

    # Step 10: Record architecture hashes
    hashes_before = verify_architecture_integrity(root_dir)
    logger.info("✓ Architecture file hashes recorded.")

    # Step 1: Create model with increased capacity
    encoder_synth = VectorEncoder(
        input_dim=cfg.synthetic_input_dim * 2,
        output_dim=cfg.state_dim,
    ).to(cfg.device)

    model = UCGAModel(
        input_dim=cfg.state_dim,
        state_dim=cfg.state_dim,
        output_dim=cfg.synthetic_output_dim,
        memory_slots=cfg.memory_slots,
        cognitive_steps=cfg.cognitive_steps,
        reasoning_steps=cfg.reasoning_steps,
        correction_threshold=cfg.correction_threshold,
    ).to(cfg.device)

    logger.info(f"  UCGA Parameters:    {model.count_parameters():,}")
    logger.info(f"  Encoder Parameters: {sum(p.numel() for p in encoder_synth.parameters()):,}")

    # Step 7: Xavier initialization
    model.apply(xavier_init)
    encoder_synth.apply(xavier_init)
    logger.info("✓ Xavier initialization applied.")

    # Step 2: Optimizer + scheduler
    params = list(model.parameters()) + list(encoder_synth.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = sum(cfg.phase_epochs) * cfg.batches_per_epoch_synthetic
    warmup_steps = int(total_steps * cfg.warmup_pct)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Step 5: Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and cfg.device == "cuda")

    # Step 8: Intelligence monitor
    monitor = IntelligenceMonitor(model, cfg.state_dim, cfg.device)

    best_score = -float("inf")
    epoch_offset = 1

    # ── Phase A: Synthetic ──
    if "A" not in args.skip_phases.upper():
        end_epoch = epoch_offset + cfg.phase_epochs[0] - 1
        best_score = train_phase_synthetic(
            model, encoder_synth, optimizer, scheduler, scaler, cfg, monitor,
            epoch_offset, end_epoch, best_score,
        )
        epoch_offset = end_epoch + 1

    # For subsequent phases, we rebuild the output head.
    # Phase B: CIFAR-10 — output_dim = 10
    if "B" not in args.skip_phases.upper() and len(cfg.phase_epochs) > 1:
        model_cifar = UCGAModel(
            input_dim=cfg.state_dim,
            state_dim=cfg.state_dim,
            output_dim=cfg.cifar_num_classes,
            memory_slots=cfg.memory_slots,
            cognitive_steps=cfg.cognitive_steps,
            reasoning_steps=cfg.reasoning_steps,
        ).to(cfg.device)

        # Transfer cognitive weights from synthetic phase
        _transfer_weights(model, model_cifar)
        model_cifar.apply(xavier_init)  # re-init only the output head
        _restore_cognitive_weights(model, model_cifar)

        end_epoch = epoch_offset + cfg.phase_epochs[1] - 1
        best_score = train_phase_cifar10(
            model_cifar, optimizer, scheduler, scaler, cfg, monitor,
            epoch_offset, end_epoch, best_score,
        )
        model = model_cifar
        epoch_offset = end_epoch + 1

    # Phase C: AG News — output_dim = 4
    if "C" not in args.skip_phases.upper() and len(cfg.phase_epochs) > 2:
        model_ag = UCGAModel(
            input_dim=cfg.state_dim,
            state_dim=cfg.state_dim,
            output_dim=cfg.agnews_num_classes,
            memory_slots=cfg.memory_slots,
            cognitive_steps=cfg.cognitive_steps,
            reasoning_steps=cfg.reasoning_steps,
        ).to(cfg.device)
        _transfer_weights(model, model_ag)
        model_ag.apply(xavier_init)
        _restore_cognitive_weights(model, model_ag)

        end_epoch = epoch_offset + cfg.phase_epochs[2] - 1
        best_score = train_phase_agnews(
            model_ag, optimizer, scheduler, scaler, cfg, monitor,
            epoch_offset, end_epoch, best_score,
        )
        model = model_ag
        epoch_offset = end_epoch + 1

    # Phase D: Multimodal — output_dim = 2
    if "D" not in args.skip_phases.upper() and len(cfg.phase_epochs) > 3:
        model_mm = UCGAModel(
            input_dim=cfg.state_dim,
            state_dim=cfg.state_dim,
            output_dim=cfg.multimodal_num_classes,
            memory_slots=cfg.memory_slots,
            cognitive_steps=cfg.cognitive_steps,
            reasoning_steps=cfg.reasoning_steps,
        ).to(cfg.device)
        _transfer_weights(model, model_mm)
        model_mm.apply(xavier_init)
        _restore_cognitive_weights(model, model_mm)

        end_epoch = epoch_offset + cfg.phase_epochs[3] - 1
        best_score = train_phase_multimodal(
            model_mm, optimizer, scheduler, scaler, cfg, monitor,
            epoch_offset, end_epoch, best_score,
        )
        model = model_mm
        epoch_offset = end_epoch + 1

    # ── Final intelligence evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("  Final Intelligence Evaluation")
    logger.info("=" * 60)
    final_scores = monitor.evaluate(epoch_offset)
    monitor.save_plots(cfg.plot_dir)

    logger.info(f"  Intelligence Score: {final_scores['intelligence_score']:.4f}")
    logger.info(f"  ROS Score:          {final_scores['ros_score']:.4f}")
    logger.info(f"  LOS Score:          {final_scores['los_score']:.4f}")
    logger.info(f"  GIB Score:          {final_scores['gib_score']:.6f}")
    logger.info(f"  Convergence Rate:   {final_scores['convergence_rate']:.4f}")
    logger.info(f"  Best IQ:            {best_score:.4f}")

    # Step 10: Verify integrity
    hashes_after = verify_architecture_integrity(root_dir)
    assert_no_modification(hashes_before, hashes_after)

    logger.info("\n✓ UCGA Optimized Training Complete.")
    logger.info(f"  Checkpoints: {cfg.checkpoint_dir}")
    logger.info(f"  Plots:       {cfg.plot_dir}")

    return final_scores


def _transfer_weights(src: UCGAModel, dst: UCGAModel):
    """
    Transfer cognitive node weights from src to dst model.
    Only transfers matching keys (skips output_node which may differ).
    """
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    transfer = {k: v for k, v in src_dict.items()
                if k in dst_dict and v.shape == dst_dict[k].shape
                and "output_node" not in k}
    dst_dict.update(transfer)
    dst.load_state_dict(dst_dict)


def _restore_cognitive_weights(src: UCGAModel, dst: UCGAModel):
    """Restore cognitive weights after xavier_init overwrote them."""
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    restore = {k: v for k, v in src_dict.items()
               if k in dst_dict and v.shape == dst_dict[k].shape
               and "output_node" not in k}
    dst_dict.update(restore)
    dst.load_state_dict(dst_dict)


if __name__ == "__main__":
    main()
