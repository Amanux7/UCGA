"""
baselines.py — Reference baseline models for fair comparison with UCGA.

Implements:
  1. MLPBaseline  — 3-layer MLP with GELU + LayerNorm
  2. CNNBaseline  — Lightweight CNN (MobileNetV3-Small flavour, from scratch)
  3. ViTBaseline  — Tiny Vision Transformer via timm (ViT-Tiny/16 adapted to 32×32)

All baselines expose a unified interface:
  - forward(images) → logits
  - count_parameters() → int

Author: Dr. Elena Voss / Aman Singh
"""

import torch
import torch.nn as nn
from typing import Optional


# ======================================================================
#  1.  3-Layer MLP Baseline
# ======================================================================
class MLPBaseline(nn.Module):
    """
    3-layer MLP baseline matching roughly the UCGA parameter budget.

    Flattens the image, then applies three FC layers with GELU + LayerNorm.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
#  2.  Lightweight CNN Baseline (MobileNetV3-Small flavour)
# ======================================================================
class CNNBaseline(nn.Module):
    """
    Lightweight CNN baseline inspired by MobileNetV3-Small.

    Uses depthwise-separable convolutions for parameter efficiency, trained
    from scratch (no pretrained weights) for a fair compute-budget comparison.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: standard conv
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            # Block 2: depthwise separable
            self._depthwise_sep(16, 24, stride=2),
            # Block 3
            self._depthwise_sep(24, 40, stride=2),
            # Block 4
            self._depthwise_sep(40, 48, stride=1),
            # Block 5
            self._depthwise_sep(48, 96, stride=2),
            # Global pool
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _depthwise_sep(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
#  3.  ViT-Tiny Baseline
# ======================================================================
class ViTBaseline(nn.Module):
    """
    Tiny Vision Transformer baseline via timm.

    Uses ViT-Tiny with patch size 16, adapted for 32×32 images (2×2 patches).
    Trained from scratch (pretrained=False) for fair comparison.
    Falls back to a manual implementation if timm is not installed.
    """

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        in_channels: int = 3,
    ):
        super().__init__()
        self.image_size = image_size

        try:
            import timm
            # ViT-Tiny: embed_dim=192, depth=12, num_heads=3
            # Use patch_size=4 for 32×32 → 8×8 = 64 patches (reasonable)
            self.model = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                img_size=image_size,
                patch_size=4,  # 32/4 = 8 patches per side
                in_chans=in_channels,
                num_classes=num_classes,
            )
        except ImportError:
            # Fallback: simple manual ViT
            self.model = _ManualViTTiny(
                num_classes=num_classes,
                image_size=image_size,
                in_channels=in_channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ManualViTTiny(nn.Module):
    """Minimal ViT-Tiny fallback when timm is not available."""

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)  # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ======================================================================
#  UCGA wrapper for baseline comparison
# ======================================================================
class UCGABaseline(nn.Module):
    """
    Thin wrapper that bundles encoder + UCGAModel for apples-to-apples
    comparison with the baseline models.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        state_dim: int = 128,
        memory_slots: int = 64,
        cognitive_steps: int = 2,
        reasoning_steps: int = 2,
    ):
        super().__init__()
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from ucga.ucga_model import UCGAModel
        from ucga.encoders import ImageEncoder

        self.encoder = ImageEncoder(in_channels=in_channels, output_dim=state_dim)
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=num_classes,
            memory_slots=memory_slots,
            cognitive_steps=cognitive_steps,
            reasoning_steps=reasoning_steps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.encoder(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
