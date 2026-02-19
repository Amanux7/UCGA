"""
BalancerNode — Learned gating to balance contributions across cognitive streams.

Acts as a soft router that dynamically weights the outputs from different
cognitive nodes before they reach the output stage.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class BalancerNode(CognitiveNode):
    """
    Balancer / soft-router for the UCGA cognitive graph.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    num_streams : int
        Number of upstream cognitive streams to balance (default 3).
    """

    def __init__(self, state_dim: int, num_streams: int = 3):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Balancer")
        self.num_streams = num_streams

        # Produces per-stream importance weights
        self.stream_gate = nn.Sequential(
            nn.Linear(state_dim * num_streams, num_streams),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Dynamically weight and combine upstream streams.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Exactly ``num_streams`` tensors of shape ``(B, state_dim)``.
            If fewer are provided, the last input is repeated to fill.
        """
        # Pad inputs to match num_streams
        while len(inputs) < self.num_streams:
            inputs.append(inputs[-1])
        inputs = inputs[: self.num_streams]

        stacked = torch.stack(inputs, dim=1)  # (B, S, D)
        B, S, D = stacked.shape

        concat = stacked.reshape(B, S * D)
        weights = self.stream_gate(concat)  # (B, S)

        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        out = self.output_proj(weighted)

        self.state = torch.tanh(out)
        return self.state
