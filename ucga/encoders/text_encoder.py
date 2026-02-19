"""
TextEncoder — Embedding + 1D-CNN feature extraction + projection.

Encodes a sequence of token IDs into a fixed-size cognitive vector.
Uses multi-scale 1D convolutions for n-gram feature extraction,
giving much richer representations than simple mean pooling.

Author: Aman Singh
"""

import math
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    CNN-based text encoder for the UCGA architecture.

    Uses multiple 1D convolution filters (unigram, bigram, trigram)
    followed by global max-pooling to produce a fixed-size vector
    suitable for the UCGA cognitive loop.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimensionality.
    output_dim : int
        Output cognitive vector dimensionality.
    max_seq_len : int
        Maximum sequence length supported.
    num_filters : int
        Number of filters per convolution kernel size.
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embed_dim: int = 128,
        output_dim: int = 256,
        max_seq_len: int = 512,
        num_filters: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)

        # Multi-scale 1D convolutions (n-gram feature extraction)
        kernel_sizes = [1, 2, 3, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)

        # Projection to cognitive space
        self.projection = nn.Sequential(
            nn.Linear(total_filters, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs into a single cognitive vector.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Cognitive vector of shape ``(B, output_dim)``.
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)

        # Embed tokens + positions
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Transpose for Conv1d: (B, embed_dim, L)
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions + global max pool
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)                    # (B, num_filters, L')
            c = c.max(dim=2).values        # (B, num_filters)  — global max pool
            conv_outputs.append(c)

        # Concatenate all n-gram features
        features = torch.cat(conv_outputs, dim=1)   # (B, total_filters)

        return self.projection(features)
