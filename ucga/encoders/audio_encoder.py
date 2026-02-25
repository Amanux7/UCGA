"""
AudioEncoder — Mel-spectrogram CNN encoder for audio inputs.

Converts raw mel-spectrogram features into a fixed-size cognitive vector
that can be passed into the UCGA cognitive loop, enabling audio as a
first-class modality alongside vision and language.

Author: Aman Singh
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    CNN-based audio encoder for the UCGA architecture.

    Processes mel-spectrogram inputs (2D time-frequency representations)
    using a stack of 2D convolutions with batch normalisation and GELU
    activation, followed by adaptive pooling and projection to produce
    a fixed-size cognitive vector.

    Parameters
    ----------
    n_mels : int
        Number of mel frequency bins (height of spectrogram).
    output_dim : int
        Output cognitive vector dimensionality.
    in_channels : int
        Number of input channels (1 for mono, 2 for stereo).
    """

    def __init__(
        self,
        n_mels: int = 128,
        output_dim: int = 256,
        in_channels: int = 1,
    ):
        super().__init__()
        self.n_mels = n_mels

        # ---- CNN Feature Extractor ----
        # Input: (B, in_channels, n_mels, T) where T = time frames
        self.features = nn.Sequential(
            # Block 1: (B, C, n_mels, T) → (B, 32, n_mels/2, T/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Block 2: → (B, 64, n_mels/4, T/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 3: → (B, 128, n_mels/8, T/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Block 4: → (B, 256, n_mels/16, T/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Global average pool → (B, 256, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # ---- Projection to Cognitive Space ----
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode a mel-spectrogram batch into cognitive vectors.

        Parameters
        ----------
        mel_spec : torch.Tensor
            Mel-spectrogram of shape ``(B, C, n_mels, T)`` where ``C``
            is the number of channels and ``T`` is the number of time frames.
            Also accepts ``(B, n_mels, T)`` (auto-adds channel dim).

        Returns
        -------
        torch.Tensor
            Cognitive vector of shape ``(B, output_dim)``.
        """
        # Auto-add channel dimension if 3D input
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        feats = self.features(mel_spec)
        return self.projection(feats)
