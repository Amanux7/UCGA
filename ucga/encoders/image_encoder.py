"""
ImageEncoder — Simple CNN encoder for image inputs.

Produces a fixed-size cognitive vector from an image tensor.

Author: Aman Singh
"""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    Lightweight CNN image encoder for the UCGA architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB).
    output_dim : int
        Output cognitive vector dimensionality.
    """

    def __init__(self, in_channels: int = 3, output_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, in_channels, H, W) → (B, 32, H/2, W/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Block 2: → (B, 64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Block 3: → (B, 128, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Global average pool
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch into cognitive vectors.

        Parameters
        ----------
        images : torch.Tensor
            Image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Cognitive vector of shape ``(B, output_dim)``.
        """
        feats = self.features(images)
        return self.projection(feats)
