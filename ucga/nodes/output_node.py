"""
OutputNode — Final projection from cognitive state to task output space.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class OutputNode(CognitiveNode):
    """
    Terminal output node for the UCGA cognitive graph.

    Parameters
    ----------
    state_dim : int
        Internal cognitive state dimensionality.
    output_dim : int
        Dimensionality of the final task output.
    """

    def __init__(self, state_dim: int, output_dim: int):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Output")
        self.output_dim = output_dim

        self.output_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, output_dim),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Produce the final task output.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Upstream signals, each ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Task output of shape ``(B, output_dim)``.
        """
        # Use raw aggregated input (skip tanh) + tanh state for richer signal
        aggregated = torch.stack(inputs, dim=0).sum(dim=0)
        h = super().forward(inputs)
        combined = h + aggregated  # residual bypass
        output = self.output_head(combined)
        return output

