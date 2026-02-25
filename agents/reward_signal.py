"""
RewardSignal — Reward processing and advantage estimation for RL training.

Provides utilities for computing returns, advantages (GAE), and normalizing
reward signals for use with policy-gradient methods like PPO.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import List


class RewardSignal(nn.Module):
    """
    Reward processing module for reinforcement learning integration.

    Includes:
    - A learned **value function** (critic) for baseline estimation
    - **GAE** (Generalized Advantage Estimation) for variance reduction
    - **Return computation** (discounted cumulative rewards)

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        GAE lambda for bias-variance trade-off.
    """

    def __init__(
        self,
        state_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__()
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Learned value function (critic)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, state_dim // 4),
            nn.GELU(),
            nn.Linear(state_dim // 4, 1),
        )

    def estimate_value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Estimate state values V(s).

        Parameters
        ----------
        states : torch.Tensor
            Batch of state vectors, shape ``(B, D)`` or ``(T, D)``.

        Returns
        -------
        torch.Tensor
            Value estimates, shape ``(B,)`` or ``(T,)``.
        """
        return self.value_head(states).squeeze(-1)

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted cumulative returns (G_t).

        Parameters
        ----------
        rewards : list of float
            Sequence of rewards [r_0, r_1, ..., r_T].

        Returns
        -------
        torch.Tensor
            Returns of shape ``(T,)``.
        """
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE-λ).

        Parameters
        ----------
        rewards : list of float
            Rewards [r_0, ..., r_{T-1}].
        values : torch.Tensor
            Value estimates V(s_0), ..., V(s_{T-1}), shape ``(T,)``.
        next_value : float
            Value estimate for the terminal state.

        Returns
        -------
        torch.Tensor
            Advantages of shape ``(T,)``.
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0

        values_list = values.detach().tolist()
        values_list.append(next_value)

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_list[t + 1] - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        return advantages

    @staticmethod
    def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize a tensor to zero mean and unit variance."""
        return (x - x.mean()) / (x.std() + eps)
