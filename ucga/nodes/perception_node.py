"""
PerceptionNode — Processes raw sensory input into internal representations.

Adds a learned projection layer on top of the base cognitive-node update
to handle variable-dimensionality raw inputs (text embeddings, image
features, raw vectors, etc.).

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class PerceptionNode(CognitiveNode):
    """
    Perception layer of the UCGA cognitive graph.

    Parameters
    ----------
    raw_input_dim : int
        Dimensionality of the raw sensory signal *before* projection.
    state_dim : int
        Dimensionality of the internal cognitive state.
    """

    def __init__(self, raw_input_dim: int, state_dim: int):
        # The internal projection maps raw_input_dim → state_dim,
        # so the base node's W operates on state_dim inputs.
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Perception")

        # Projection from raw sensor space to cognitive space
        self.input_projection = nn.Sequential(
            nn.Linear(raw_input_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Project raw inputs into cognitive space and apply state update.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Each tensor has shape ``(B, raw_input_dim)``.
        """
        projected = [self.input_projection(x) for x in inputs]
        return super().forward(projected)
