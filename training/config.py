"""
config.py — Centralized training configuration for UCGA optimization.

All tunable hyperparameters in one place.  Only training parameters
are defined here — no model architecture is modified.

Author: Aman Singh
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class TrainingConfig:
    """Optimized training configuration for UCGA intelligence scaling."""

    # ── Model capacity (passed to UCGAModel constructor) ──
    state_dim: int = 512
    memory_slots: int = 256
    cognitive_steps: int = 8
    reasoning_steps: int = 6
    correction_threshold: float = 0.5

    # ── Optimizer ──
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 1.0
    warmup_pct: float = 0.05          # first 5% of total steps

    # ── Training duration ──
    num_epochs: int = 50
    checkpoint_every: int = 10
    batches_per_epoch_synthetic: int = 200   # synthetic phases

    # ── Batch size / accumulation (adjusted for 4GB VRAM) ──
    batch_size: int = 1
    gradient_accumulation_steps: int = 128     # effective = 128

    # ── Mixed precision ──
    use_amp: bool = True

    # ── Data loading ──
    num_workers: int = 4
    pin_memory: bool = True

    # ── Multi-phase training (epoch budgets) ──
    phase_epochs: List[int] = field(default_factory=lambda: [10, 20, 10, 10])
    phase_names: List[str] = field(
        default_factory=lambda: [
            "Synthetic",
            "CIFAR-10",
            "AG News",
            "Multimodal",
        ]
    )

    # ── Synthetic dataset ──
    synthetic_input_dim: int = 32
    synthetic_output_dim: int = 32

    # ── CIFAR-10 ──
    cifar_num_classes: int = 10
    cifar_image_size: int = 32
    cifar_in_channels: int = 3

    # ── AG News ──
    agnews_vocab_size: int = 8_000
    agnews_num_classes: int = 4

    # ── Multimodal ──
    multimodal_vocab_size: int = 500
    multimodal_max_seq_len: int = 16
    multimodal_embed_dim: int = 64
    multimodal_num_classes: int = 2
    multimodal_num_samples: int = 6_000

    # ── Paths ──
    data_dir: str = "data"
    checkpoint_dir: str = "outputs/checkpoints"
    plot_dir: str = "outputs/training_plots"
    dashboard_dir: str = "outputs/dashboard"

    # ── Device ──
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
