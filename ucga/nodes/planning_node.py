"""
PlanningNode — Generates structured action plans from reasoning output.

Takes the refined reasoning state and produces a sequence of planned
sub-goals or action embeddings that downstream nodes can execute.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List

from .cognitive_node import CognitiveNode


class PlanningNode(CognitiveNode):
    """
    Planning node for the UCGA cognitive graph.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    num_plan_steps : int
        Number of planned sub-goal slots (default 4).
    """

    def __init__(self, state_dim: int, num_plan_steps: int = 4):
        super().__init__(input_dim=state_dim, state_dim=state_dim, name="Planning")
        self.num_plan_steps = num_plan_steps

        # Generate a plan as a sequence of sub-goal embeddings
        self.plan_generator = nn.Sequential(
            nn.Linear(state_dim, state_dim * num_plan_steps),
            nn.GELU(),
        )

        # Compress the plan back into a single state vector
        self.plan_compressor = nn.Sequential(
            nn.Linear(state_dim * num_plan_steps, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Produce an action plan and compress it into the cognitive state.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            Upstream signals (typically from ReasoningNode).
        """
        h = super().forward(inputs)

        # Expand into sub-goals
        plan_flat = self.plan_generator(h)  # (B, state_dim * num_plan_steps)

        # Compress back
        plan_state = self.plan_compressor(plan_flat)

        self.state = torch.tanh(plan_state)
        return self.state
