"""
TransformerTextEncoder — Self-attention-based text encoder for the UCGA architecture.

Uses a small Transformer encoder (multi-head self-attention + feed-forward)
with [CLS]-token pooling to produce a fixed-size cognitive vector from a
sequence of token IDs.

This is the Phase-2 upgrade over the CNN-based TextEncoder, enabling richer
contextual representations and better handling of long-range dependencies.

Author: Aman Singh
"""

import math
import torch
import torch.nn as nn


class TransformerTextEncoder(nn.Module):
    """
    Transformer-based text encoder for the UCGA architecture.

    Uses multiple self-attention layers with learned positional embeddings
    and [CLS]-token pooling to produce a fixed-size cognitive vector.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.  Index 0 is reserved for padding.
    embed_dim : int
        Embedding / hidden dimensionality.
    output_dim : int
        Output cognitive vector dimensionality.
    max_seq_len : int
        Maximum sequence length supported.
    num_layers : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads per layer.
    ff_dim : int, optional
        Feed-forward inner dimensionality (default: ``4 * embed_dim``).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 30_000,
        embed_dim: int = 128,
        output_dim: int = 256,
        max_seq_len: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        ff_dim = ff_dim or embed_dim * 4

        # ---- Embeddings ----
        # +1 for the prepended [CLS] token
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # ---- Transformer Encoder Layers ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ---- Projection to cognitive space ----
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode token IDs into a single cognitive vector.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(B, L)``.
        attention_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, L)`` where ``True`` means ignore
            (i.e. padding positions).  If ``None``, derived from padding
            index 0.

        Returns
        -------
        torch.Tensor
            Cognitive vector of shape ``(B, output_dim)``.
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Derive padding mask if not provided
        if attention_mask is None:
            # True = pad = ignore
            attention_mask = (token_ids == 0)

        # Token embeddings
        tok_emb = self.token_embedding(token_ids)           # (B, L, D)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        tok_emb = torch.cat([cls_tokens, tok_emb], dim=1)   # (B, 1+L, D)

        # Extend attention mask for [CLS] (never masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 1+L)

        # Position embeddings
        seq_len = tok_emb.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        # Combine
        x = self.embed_norm(tok_emb + pos_emb)
        x = self.embed_dropout(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attention_mask)  # (B, 1+L, D)

        # [CLS] pooling — take the first token
        cls_output = x[:, 0, :]  # (B, D)

        # Project to cognitive space
        return self.projection(cls_output)
