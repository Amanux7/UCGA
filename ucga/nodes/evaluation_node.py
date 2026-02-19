"""
EvaluationNode — Evaluates the quality of a proposed plan or action.

Produces a scalar confidence score and a refined signal that indicates
whether the current plan should be accepted or sent back for correction.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .cognitive_node import CognitiveNode


class EvaluationNode(CognitiveNode):
    """
    Evaluation / critic node for the UCGA cognitive loop.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    """

    def __init__(self, state_dim: int):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Evaluation")

        # Confidence estimator (scalar output)
        self.confidence_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Evaluation feedback signal
        self.feedback_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate inputs and update state.  Confidence is stored in
        ``self.last_confidence`` for downstream inspection.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Upstream signals (plan + context).
        """
        h = super().forward(inputs)

        self.last_confidence = self.confidence_head(h)  # (B, 1)
        feedback = self.feedback_head(h)

        self.state = torch.tanh(feedback)
        return self.state

    def get_confidence(self) -> torch.Tensor:
        """Return the last computed confidence score ``(B, 1)``."""
        return self.last_confidence
