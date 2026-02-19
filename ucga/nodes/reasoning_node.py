"""
ReasoningNode — Multi-step iterative reasoning within the cognitive graph.

Performs *K* internal refinement steps before producing an output,
allowing the node to emulate chain-of-thought style reasoning.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class ReasoningNode(CognitiveNode):
    """
    Reasoning node with iterative refinement.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    reasoning_steps : int
        Number of internal refinement iterations (default 3).
    """

    def __init__(self, state_dim: int, reasoning_steps: int = 3):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Reasoning")
        self.reasoning_steps = reasoning_steps

        # Each refinement step has its own transform
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.LayerNorm(state_dim),
                nn.GELU(),
            )
            for _ in range(reasoning_steps)
        ])

        # Residual gate for each step
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim * 2, state_dim), nn.Sigmoid())
            for _ in range(reasoning_steps)
        ])

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate inputs, then iteratively refine the internal state.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Upstream signals, each ``(B, state_dim)``.
        """
        # Initial state from base update
        h = super().forward(inputs)

        # Iterative refinement
        for layer, gate in zip(self.refinement_layers, self.gates):
            h_new = layer(h)
            g = gate(torch.cat([h, h_new], dim=-1))
            h = g * h_new + (1 - g) * h  # gated residual

        self.state = h
        return self.state
