"""
CognitiveNode — Base class for all UCGA cognitive processing units.

Every node in the Unified Cognitive Graph inherits from this class.
Each node maintains an internal state vector and updates it via:

    v_i(t+1) = tanh(W_i · Σ(inputs) + b_i)

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CognitiveNode(nn.Module):
    """
    Base cognitive node for the Unified Cognitive Graph Architecture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each incoming signal.
    state_dim : int
        Dimensionality of the internal cognitive state.
    name : str
        Human-readable identifier for logging / visualization.
    """

    def __init__(self, input_dim: int, state_dim: int, name: str = "CognitiveNode"):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Learnable transformation weights
        self.W = nn.Linear(input_dim, state_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(state_dim))

        # Non-saturating activation + normalisation (replaces hard tanh)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(state_dim)

        # Internal cognitive state (not a parameter — updated during forward pass)
        self.register_buffer("state", torch.zeros(1, state_dim))

    # ------------------------------------------------------------------
    # Core update rule:  v_i(t+1) = LayerNorm(GELU(W_i · Σ(inputs) + b_i))
    # ------------------------------------------------------------------
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate incoming signals and update internal state.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            List of tensors from upstream nodes, each of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Updated state of shape ``(B, state_dim)``.
        """
        # Sum all incoming signals
        aggregated = torch.stack(inputs, dim=0).sum(dim=0)  # (B, input_dim)

        # State update — non-saturating activation
        self.state = self.norm(self.activation(self.W(aggregated) + self.b))
        return self.state


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset the internal state to zeros (detached from any graph)."""
        with torch.no_grad():
            self.state = torch.zeros(batch_size, self.state_dim, device=self.b.device)

    def get_state(self) -> torch.Tensor:
        """Return the current internal cognitive state."""
        return self.state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"input_dim={self.input_dim}, state_dim={self.state_dim})"
        )
