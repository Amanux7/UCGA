"""
CorrectionNode — Applies corrective adjustments based on evaluation feedback.

Receives the evaluation feedback and the original plan, then produces
a corrected output that can be fed back into the reasoning loop.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class CorrectionNode(CognitiveNode):
    """
    Error-correction node for the UCGA cognitive graph.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    """

    def __init__(self, state_dim: int):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Correction")

        # Correction transform
        self.correction_net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

        # Residual gate
        self.gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid(),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply correction.  Expects at least two inputs: the plan state
        and the evaluation feedback.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            ``[plan_state, eval_feedback, ...]`` each ``(B, state_dim)``.
        """
        # Base aggregation
        aggregated = torch.stack(inputs, dim=0).sum(dim=0)

        # If we have two explicit signals, use them for the correction
        if len(inputs) >= 2:
            plan_state = inputs[0]
            eval_feedback = inputs[1]
            combined = torch.cat([plan_state, eval_feedback], dim=-1)
        else:
            combined = torch.cat([aggregated, aggregated], dim=-1)

        correction = self.correction_net(combined)
        g = self.gate(combined)

        self.state = torch.tanh(g * correction + (1 - g) * aggregated)
        return self.state
