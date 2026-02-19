"""
MemoryNode — Retrieves and integrates information from persistent memory.

Uses an attention mechanism to query the external memory bank and fuses
the retrieved context with other upstream signals.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .cognitive_node import CognitiveNode


class MemoryNode(CognitiveNode):
    """
    Memory-retrieval node for the UCGA cognitive graph.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the cognitive state (also the memory slot width).
    num_heads : int
        Number of attention heads for memory retrieval.
    """

    def __init__(self, state_dim: int, num_heads: int = 4):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Memory")
        self.num_heads = num_heads

        # Attention-based memory retrieval
        self.query_proj = nn.Linear(state_dim, state_dim)
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.attn_scale = state_dim ** 0.5

        # Fusion of retrieved context and upstream signals
        self.fusion = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        memory_bank: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retrieve from memory and fuse with upstream inputs.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Upstream signals, each ``(B, state_dim)``.
        memory_bank : torch.Tensor, optional
            External persistent memory of shape ``(B, M, state_dim)`` where
            *M* is the number of memory slots.  If ``None``, skip retrieval.
        """
        # Standard aggregation
        aggregated = torch.stack(inputs, dim=0).sum(dim=0)  # (B, D)

        if memory_bank is not None and memory_bank.size(1) > 0:
            # Attention retrieval
            Q = self.query_proj(aggregated).unsqueeze(1)       # (B, 1, D)
            K = self.key_proj(memory_bank)                      # (B, M, D)
            V = self.value_proj(memory_bank)                    # (B, M, D)

            scores = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale  # (B, 1, M)
            weights = torch.softmax(scores, dim=-1)
            retrieved = torch.bmm(weights, V).squeeze(1)        # (B, D)

            # Fuse retrieved context with aggregated input
            fused = self.fusion(torch.cat([aggregated, retrieved], dim=-1))
        else:
            fused = aggregated

        # Cognitive state update
        self.state = torch.tanh(self.W(fused) + self.b)
        return self.state
