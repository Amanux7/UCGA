"""
EpisodicMemory — Temporal episodic memory with similarity-based retrieval.

Stores experiences as (state, action, reward, timestamp) tuples and provides
retrieval by recency, similarity, or a hybrid of both.  This enables the
cognitive agent to learn from past episodes and adapt behaviour over time.

Author: Aman Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


class Episode:
    """A single episode record stored in episodic memory."""

    __slots__ = ["state", "action", "reward", "timestamp", "metadata"]

    def __init__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        timestamp: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.state = state.detach()
        self.action = action.detach()
        self.reward = reward
        self.timestamp = timestamp
        self.metadata = metadata or {}


class EpisodicMemory(nn.Module):
    """
    Temporal episodic memory with multiple retrieval strategies.

    Stores episodes and retrieves them by:
    - **Recency**: most recent episodes
    - **Similarity**: most similar states (cosine similarity)
    - **Hybrid**: weighted combination of recency and similarity scores

    Parameters
    ----------
    state_dim : int
        Dimensionality of state vectors.
    capacity : int
        Maximum number of episodes to store (FIFO eviction).
    retrieval_dim : int
        Output dimensionality for the retrieval projection.
    recency_weight : float
        Weight for recency in hybrid retrieval (0=similarity only, 1=recency only).
    """

    def __init__(
        self,
        state_dim: int = 128,
        capacity: int = 1000,
        retrieval_dim: int = 128,
        recency_weight: float = 0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.capacity = capacity
        self.recency_weight = recency_weight
        self._episodes: deque = deque(maxlen=capacity)
        self._global_step: int = 0

        # Learned retrieval projection
        self.query_proj = nn.Linear(state_dim, retrieval_dim)
        self.key_proj = nn.Linear(state_dim, retrieval_dim)
        self.value_proj = nn.Linear(state_dim + 1, retrieval_dim)  # +1 for reward
        self.output_proj = nn.Sequential(
            nn.Linear(retrieval_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    @property
    def size(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a new episode.

        Parameters
        ----------
        state : torch.Tensor
            State vector of shape ``(D,)`` or ``(1, D)``.
        action : torch.Tensor
            Action vector of shape ``(A,)`` or ``(1, A)``.
        reward : float
            Scalar reward received.
        metadata : dict, optional
            Additional metadata to attach.
        """
        if state.dim() > 1:
            state = state.squeeze(0)
        if action.dim() > 1:
            action = action.squeeze(0)

        self._episodes.append(Episode(
            state=state.cpu(),
            action=action.cpu(),
            reward=reward,
            timestamp=self._global_step,
            metadata=metadata,
        ))
        self._global_step += 1

    def retrieve_by_recency(self, k: int = 5) -> List[Episode]:
        """Return the ``k`` most recent episodes."""
        return list(self._episodes)[-k:]

    def retrieve_by_similarity(
        self, query: torch.Tensor, k: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        Return the ``k`` most similar episodes (cosine similarity).

        Parameters
        ----------
        query : torch.Tensor
            Query state vector of shape ``(D,)`` or ``(1, D)``.
        k : int
            Number of episodes to return.

        Returns
        -------
        list of (Episode, float)
            Episodes with their similarity scores, sorted descending.
        """
        if self.size == 0:
            return []
        if query.dim() > 1:
            query = query.squeeze(0)

        query = query.cpu()
        states = torch.stack([ep.state for ep in self._episodes])
        sims = F.cosine_similarity(query.unsqueeze(0), states, dim=-1)

        k = min(k, self.size)
        top_indices = sims.topk(k).indices.tolist()
        episodes = list(self._episodes)
        return [(episodes[i], sims[i].item()) for i in top_indices]

    def retrieve_hybrid(
        self, query: torch.Tensor, k: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        Hybrid retrieval combining recency and similarity scores.

        Score = (1 - recency_weight) * sim + recency_weight * recency_score
        """
        if self.size == 0:
            return []
        if query.dim() > 1:
            query = query.squeeze(0)

        query = query.cpu()
        episodes = list(self._episodes)
        states = torch.stack([ep.state for ep in episodes])

        # Similarity scores
        sims = F.cosine_similarity(query.unsqueeze(0), states, dim=-1)

        # Recency scores (normalize timestamps to [0, 1])
        timestamps = torch.tensor([ep.timestamp for ep in episodes], dtype=torch.float)
        if timestamps.max() > timestamps.min():
            recency = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        else:
            recency = torch.ones_like(timestamps)

        # Hybrid score
        scores = (1 - self.recency_weight) * sims + self.recency_weight * recency

        k = min(k, self.size)
        top_indices = scores.topk(k).indices.tolist()
        return [(episodes[i], scores[i].item()) for i in top_indices]

    def retrieve_as_context(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Retrieve episodes as a differentiable context vector.

        Uses learned projections to attend over stored episodes and
        produce a single context vector that can be fed into the
        cognitive loop.

        Parameters
        ----------
        query : torch.Tensor
            Query state of shape ``(B, D)`` or ``(D,)``.
        k : int
            Number of episodes to attend over.

        Returns
        -------
        torch.Tensor
            Context vector of shape ``(B, D)`` or ``(D,)``.
        """
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        B, D = query.shape
        device = query.device

        if self.size == 0:
            return torch.zeros_like(query)

        # Get top-k by hybrid retrieval (non-differentiable selection)
        retrieved = self.retrieve_hybrid(query[0], k=k)
        ep_states = torch.stack([ep.state for ep, _ in retrieved]).to(device)
        ep_rewards = torch.tensor(
            [ep.reward for ep, _ in retrieved], dtype=torch.float, device=device,
        )

        # Differentiable attention over retrieved memories
        Q = self.query_proj(query)                              # (B, R)
        K = self.key_proj(ep_states)                            # (k, R)

        # Values include reward signal
        val_input = torch.cat([
            ep_states, ep_rewards.unsqueeze(-1),
        ], dim=-1)                                              # (k, D+1)
        V = self.value_proj(val_input)                          # (k, R)

        # Attention scores
        scores = torch.matmul(Q, K.T) / (K.size(-1) ** 0.5)    # (B, k)
        weights = F.softmax(scores, dim=-1)                     # (B, k)
        context = torch.matmul(weights, V)                      # (B, R)

        result = self.output_proj(context)

        if squeeze:
            result = result.squeeze(0)
        return result

    def clear(self) -> None:
        """Clear all stored episodes."""
        self._episodes.clear()
        self._global_step = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity}
        rewards = [ep.reward for ep in self._episodes]
        return {
            "size": self.size,
            "capacity": self.capacity,
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "global_step": self._global_step,
        }
