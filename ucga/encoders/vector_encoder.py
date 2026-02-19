"""
VectorEncoder — MLP projection for raw vector inputs.

Author: Aman Singh
"""

import torch
import torch.nn as nn


class VectorEncoder(nn.Module):
    """
    Simple MLP encoder that projects raw numeric vectors into cognitive space.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vector.
    output_dim : int
        Dimensionality of the cognitive vector.
    hidden_dim : int, optional
        Hidden layer width (default: ``output_dim``).
    """

    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw vector input.

        Parameters
        ----------
        x : torch.Tensor
            Raw input of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Cognitive vector of shape ``(B, output_dim)``.
        """
        return self.net(x)
