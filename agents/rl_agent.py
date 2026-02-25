"""
RLAgent — Reinforcement-learning-capable UCGA cognitive agent.

Extends the base CognitiveAgent with:
    - PPO-style policy gradient training
    - Episodic memory for experience replay
    - Tool-use via ToolRegistry
    - Value-function baseline (critic) for advantage estimation

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from ucga.memory.episodic_memory import EpisodicMemory
from agents.tool_registry import ToolRegistry
from agents.reward_signal import RewardSignal


class RLAgent(nn.Module):
    """
    PPO-style RL agent built on the UCGA cognitive loop.

    The agent perceives observations, reasons through the cognitive loop,
    and outputs actions.  It learns from reward signals using PPO with
    GAE advantage estimation.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action space dimensionality (discrete = num classes).
    state_dim : int
        Cognitive state dimensionality.
    discrete : bool
        If True, actions are categorical.  If False, continuous (Gaussian).
    gamma : float
        Discount factor.
    clip_eps : float
        PPO clipping epsilon.
    memory_capacity : int
        Episodic memory capacity.
    """

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        state_dim: int = 64,
        discrete: bool = True,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        memory_capacity: int = 500,
        cognitive_steps: int = 1,
        **model_kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.discrete = discrete
        self.clip_eps = clip_eps

        # ---- Encoder ----
        self.encoder = VectorEncoder(input_dim=obs_dim, output_dim=state_dim)

        # ---- UCGA Cognitive Loop ----
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=state_dim,  # outputs cognitive state, not final action
            cognitive_steps=cognitive_steps,
            **model_kwargs,
        )

        # ---- Policy Head (Actor) ----
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, action_dim),
        )

        # ---- Reward Signal (Critic) ----
        self.reward_signal = RewardSignal(state_dim=state_dim, gamma=gamma)

        # ---- Episodic Memory ----
        self.episodic_memory = EpisodicMemory(
            state_dim=state_dim, capacity=memory_capacity,
        )

        # ---- Tool Registry ----
        self.tool_registry = ToolRegistry(state_dim=state_dim)

        # ---- Trajectory Buffer ----
        self._trajectory: List[Dict[str, Any]] = []

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action given an observation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation of shape ``(B, obs_dim)``.
        deterministic : bool
            If True, take the argmax action.

        Returns
        -------
        (action, log_prob, value)
            Action tensor, log probability, and value estimate.
        """
        encoded = self.encoder(obs)
        cognitive_out = self.model(encoded)

        # Policy
        if self.discrete:
            logits = self.policy_head(cognitive_out)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            mean = self.policy_head(cognitive_out)
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Value
        value = self.reward_signal.estimate_value(cognitive_out)

        return action, log_prob, value

    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store a transition in the trajectory buffer."""
        self._trajectory.append({
            "obs": obs.detach(),
            "action": action.detach(),
            "reward": reward,
            "log_prob": log_prob.detach(),
            "value": value.detach(),
        })

        # Also store in episodic memory
        encoded = self.encoder(obs)
        cognitive_out = self.model(encoded)
        self.episodic_memory.store(
            state=cognitive_out.detach().squeeze(0),
            action=action.detach().squeeze(0) if action.dim() > 0 else action.detach().unsqueeze(0),
            reward=reward,
        )

    def compute_ppo_loss(self) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss from the current trajectory buffer.

        Returns
        -------
        dict
            - ``policy_loss``: clipped surrogate loss
            - ``value_loss``: MSE value function loss
            - ``entropy``: policy entropy bonus
            - ``total_loss``: combined loss
        """
        if not self._trajectory:
            return {
                "policy_loss": torch.tensor(0.0),
                "value_loss": torch.tensor(0.0),
                "entropy": torch.tensor(0.0),
                "total_loss": torch.tensor(0.0),
            }

        # Gather trajectory data
        obs_list = [t["obs"] for t in self._trajectory]
        actions = [t["action"] for t in self._trajectory]
        rewards = [t["reward"] for t in self._trajectory]
        old_log_probs = torch.stack([t["log_prob"] for t in self._trajectory]).squeeze()
        old_values = torch.stack([t["value"] for t in self._trajectory]).squeeze()

        # Compute advantages with GAE
        advantages = self.reward_signal.compute_gae(rewards, old_values)
        returns = self.reward_signal.compute_returns(rewards)

        # Normalize advantages
        advantages = RewardSignal.normalize(advantages)

        # Re-evaluate actions under current policy
        all_obs = torch.cat(obs_list, dim=0)
        encoded = self.encoder(all_obs)
        cognitive_out = self.model(encoded)

        if self.discrete:
            logits = self.policy_head(cognitive_out)
            dist = torch.distributions.Categorical(logits=logits)
            actions_t = torch.stack(actions)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
        else:
            mean = self.policy_head(cognitive_out)
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            actions_t = torch.stack(actions)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

        new_values = self.reward_signal.estimate_value(cognitive_out)

        # PPO clipped surrogate loss
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

    def clear_trajectory(self) -> None:
        """Clear the trajectory buffer."""
        self._trajectory.clear()

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
