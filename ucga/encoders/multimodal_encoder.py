"""
MultimodalEncoder — Vision-Language fusion encoder using cross-modal attention.

Combines an ImageEncoder and a TextEncoder (CNN or Transformer) through
cross-modal attention to produce a single unified cognitive vector that
captures joint visual and linguistic information.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Optional

from .image_encoder import ImageEncoder
from .cross_modal_attention import CrossModalAttention


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder that fuses vision and language features.

    Uses separate unimodal encoders to produce initial representations,
    then applies cross-modal attention for bidirectional feature fusion,
    and finally projects the combined representation into cognitive space.

    Parameters
    ----------
    image_channels : int
        Number of input image channels (3 for RGB).
    vocab_size : int
        Token vocabulary size for the text encoder.
    embed_dim : int
        Internal feature dimensionality shared by both modalities.
    output_dim : int
        Output cognitive vector dimensionality.
    num_heads : int
        Number of attention heads for cross-modal fusion.
    num_fusion_layers : int
        Number of cross-modal attention layers.
    text_encoder_type : str
        ``"cnn"`` for TextEncoder or ``"transformer"`` for TransformerTextEncoder.
    max_seq_len : int
        Maximum text sequence length.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        image_channels: int = 3,
        vocab_size: int = 10_000,
        embed_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        text_encoder_type: str = "transformer",
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ---- Unimodal Encoders ----
        self.image_encoder = ImageEncoder(
            in_channels=image_channels, output_dim=embed_dim
        )

        if text_encoder_type == "transformer":
            from .transformer_text_encoder import TransformerTextEncoder
            self.text_encoder = TransformerTextEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                output_dim=embed_dim,
                max_seq_len=max_seq_len,
                num_layers=2,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            from .text_encoder import TextEncoder
            self.text_encoder = TextEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                output_dim=embed_dim,
                max_seq_len=max_seq_len,
            )

        # ---- Cross-Modal Fusion Layers ----
        self.fusion_layers = nn.ModuleList([
            CrossModalAttention(
                dim=embed_dim, num_heads=num_heads, dropout=dropout,
            )
            for _ in range(num_fusion_layers)
        ])

        # ---- Fusion Projection ----
        # Combine both modalities into a single cognitive vector
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1),
        )
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
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode image + text into a unified cognitive vector.

        Parameters
        ----------
        images : torch.Tensor
            Image batch of shape ``(B, C, H, W)``.
        token_ids : torch.Tensor
            Token IDs of shape ``(B, L)``.
        attention_mask : torch.Tensor, optional
            Boolean padding mask for text, shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Unified cognitive vector of shape ``(B, output_dim)``.
        """
        # Unimodal encoding
        vis_feat = self.image_encoder(images)       # (B, embed_dim)
        txt_feat = self.text_encoder(token_ids)     # (B, embed_dim)

        # Cross-modal fusion (iterate through fusion layers)
        for fusion_layer in self.fusion_layers:
            vis_feat, txt_feat = fusion_layer(vis_feat, txt_feat)

        # Gated combination
        combined = torch.cat([vis_feat, txt_feat], dim=-1)  # (B, 2*embed_dim)
        gate_weights = self.fusion_gate(combined)           # (B, 2)

        fused = (
            gate_weights[:, 0:1] * vis_feat +
            gate_weights[:, 1:2] * txt_feat
        )  # (B, embed_dim)

        return self.projection(fused)
