"""
CrossModalAttention — Bidirectional cross-attention fusion for multimodal signals.

Allows two modalities (e.g. vision and language) to attend to each other,
producing fused representations that capture cross-modal interactions.

Author: Aman Singh
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention fusion module.

    Given two sets of features (modality A and modality B), produces
    fused representations where each modality attends to the other.

    Parameters
    ----------
    dim : int
        Feature dimensionality (both modalities must share this).
    num_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # A → B cross attention (A queries, B provides keys/values)
        self.q_a = nn.Linear(dim, dim)
        self.k_b = nn.Linear(dim, dim)
        self.v_b = nn.Linear(dim, dim)

        # B → A cross attention (B queries, A provides keys/values)
        self.q_b = nn.Linear(dim, dim)
        self.k_a = nn.Linear(dim, dim)
        self.v_a = nn.Linear(dim, dim)

        # Output projections
        self.out_proj_a = nn.Linear(dim, dim)
        self.out_proj_b = nn.Linear(dim, dim)

        # Layer norms and dropout
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward refinement after fusion
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ff_b = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ff_norm_a = nn.LayerNorm(dim)
        self.ff_norm_b = nn.LayerNorm(dim)

        self.scale = self.head_dim ** -0.5

    def _multi_head_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaled dot-product multi-head attention."""
        B, L_q, D = Q.shape
        L_k = K.size(1)
        H = self.num_heads
        d = self.head_dim

        Q = Q.view(B, L_q, H, d).transpose(1, 2)  # (B, H, L_q, d)
        K = K.view(B, L_k, H, d).transpose(1, 2)  # (B, H, L_k, d)
        V = V.view(B, L_k, H, d).transpose(1, 2)  # (B, H, L_k, d)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L_q, L_k)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = torch.matmul(weights, V)               # (B, H, L_q, d)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # (B, L_q, D)
        return out

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-modal attention.

        Parameters
        ----------
        feat_a : torch.Tensor
            Features from modality A, shape ``(B, D)`` or ``(B, L_a, D)``.
        feat_b : torch.Tensor
            Features from modality B, shape ``(B, D)`` or ``(B, L_b, D)``.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Fused features for A and B, same shapes as inputs.
        """
        # Handle 2D inputs by adding sequence dimension
        squeeze_a = feat_a.dim() == 2
        squeeze_b = feat_b.dim() == 2
        if squeeze_a:
            feat_a = feat_a.unsqueeze(1)
        if squeeze_b:
            feat_b = feat_b.unsqueeze(1)

        # A attends to B
        Q_a = self.q_a(feat_a)
        K_b = self.k_b(feat_b)
        V_b = self.v_b(feat_b)
        cross_a = self.out_proj_a(self._multi_head_attention(Q_a, K_b, V_b))
        feat_a = self.norm_a(feat_a + cross_a)  # residual + norm

        # B attends to A
        Q_b = self.q_b(feat_b)
        K_a = self.k_a(feat_a)
        V_a = self.v_a(feat_a)
        cross_b = self.out_proj_b(self._multi_head_attention(Q_b, K_a, V_a))
        feat_b = self.norm_b(feat_b + cross_b)  # residual + norm

        # Feed-forward refinement
        feat_a = self.ff_norm_a(feat_a + self.ff_a(feat_a))
        feat_b = self.ff_norm_b(feat_b + self.ff_b(feat_b))

        # Restore original shape if input was 2D
        if squeeze_a:
            feat_a = feat_a.squeeze(1)
        if squeeze_b:
            feat_b = feat_b.squeeze(1)

        return feat_a, feat_b
