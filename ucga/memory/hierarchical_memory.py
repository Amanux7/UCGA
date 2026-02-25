"""
hierarchical_memory.py — Multi-tier Hierarchical Memory System

Three-tier memory architecture:
    1. Working Memory  — fast, small FIFO buffer for current episode
    2. Episodic Memory  — mid-term temporal memory (reuses existing module)
    3. Persistent Memory — long-term stable storage (reuses existing module)

Includes consolidation logic to promote important memories up the
hierarchy and demote stale memories down.

Author: Aman Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple


class WorkingMemory(nn.Module):
    """
    Fast, fixed-size FIFO working memory for the current episode.

    Stores the most recent *capacity* states. Provides attention-based
    retrieval and importance scoring for consolidation into higher tiers.

    Parameters
    ----------
    capacity : int
        Maximum number of states to store.
    state_dim : int
        Dimensionality of each state vector.
    """

    def __init__(self, capacity: int = 16, state_dim: int = 128):
        super().__init__()
        self.capacity = capacity
        self.state_dim = state_dim
        self._write_idx = 0
        self._count = 0

        # Buffer
        self.register_buffer("buffer", torch.zeros(capacity, state_dim))

        # Importance scorer
        self.importance_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Read attention
        self.query_proj = nn.Linear(state_dim, state_dim)

    def write(self, state: torch.Tensor) -> None:
        """
        Write a state into working memory (FIFO eviction).

        Parameters
        ----------
        state : torch.Tensor
            State of shape ``(state_dim,)`` or ``(B, state_dim)``.
        """
        if state.dim() == 2:
            state = state[0]  # Take first batch element

        with torch.no_grad():
            self.buffer[self._write_idx] = state.detach()
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Attention-based read from working memory.

        Parameters
        ----------
        query : torch.Tensor
            Query of shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Retrieved content of shape ``(B, state_dim)``.
        """
        if self._count == 0:
            return torch.zeros_like(query)

        active = self.buffer[:self._count]  # (C, D)
        Q = self.query_proj(query)  # (B, D)

        # Attention
        scores = torch.matmul(Q, active.T) / (self.state_dim ** 0.5)  # (B, C)
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, active)  # (B, D)
        return retrieved

    def get_important_states(self, top_k: int = 4) -> List[Tuple[torch.Tensor, float]]:
        """
        Return the top-k most important states for consolidation.

        Returns
        -------
        list of (tensor, importance_score) tuples
        """
        if self._count == 0:
            return []

        active = self.buffer[:self._count]
        with torch.no_grad():
            scores = self.importance_head(active).squeeze(-1)  # (C,)

        k = min(top_k, self._count)
        top_scores, top_indices = scores.topk(k)

        result = []
        for idx, score in zip(top_indices, top_scores):
            result.append((active[idx].clone(), score.item()))

        return result

    def reset(self) -> None:
        """Clear working memory."""
        self.buffer.zero_()
        self._write_idx = 0
        self._count = 0

    @property
    def size(self) -> int:
        return self._count


class HierarchicalMemorySystem(nn.Module):
    """
    Three-tier hierarchical memory orchestrator.

    Manages working, episodic, and persistent memory with automated
    consolidation between tiers.

    Parameters
    ----------
    state_dim : int
        State dimensionality.
    working_capacity : int
        Working memory capacity.
    episodic_capacity : int
        Episodic memory capacity.
    persistent_slots : int
        Persistent memory slots.
    consolidation_threshold : float
        Importance threshold for promoting to episodic memory.
    """

    def __init__(
        self,
        state_dim: int = 128,
        working_capacity: int = 16,
        episodic_capacity: int = 64,
        persistent_slots: int = 128,
        consolidation_threshold: float = 0.6,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.consolidation_threshold = consolidation_threshold

        # Tier 1: Working Memory (fast, current episode)
        self.working = WorkingMemory(
            capacity=working_capacity, state_dim=state_dim,
        )

        # Tier 2: Episodic Memory (mid-term)
        # Use a simple buffer-based approach (compatible with standalone use)
        self.episodic_capacity = episodic_capacity
        self.register_buffer("episodic_buffer", torch.zeros(episodic_capacity, state_dim))
        self.register_buffer("episodic_importance", torch.zeros(episodic_capacity))
        self._episodic_write_idx = 0
        self._episodic_count = 0

        # Tier 3: Persistent Memory (reuse existing PersistentMemory interface)
        from ucga.memory import PersistentMemory
        self.persistent = PersistentMemory(
            num_slots=persistent_slots, slot_dim=state_dim,
        )

        # Tier fusion: combines reads from all tiers
        self.tier_fusion = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

        # Tier gate: learned weighting of tiers
        self.tier_gate = nn.Sequential(
            nn.Linear(state_dim, 3),
            nn.Softmax(dim=-1),
        )

        # Consolidation statistics
        self._consolidation_count = 0

    def write(self, state: torch.Tensor) -> Dict[str, Any]:
        """
        Write a state to working memory and trigger consolidation.

        Parameters
        ----------
        state : torch.Tensor
            State of shape ``(B, state_dim)`` or ``(state_dim,)``.

        Returns
        -------
        dict
            Consolidation info.
        """
        self.working.write(state)

        # Check for consolidation
        info = {"consolidated_to_episodic": 0, "consolidated_to_persistent": 0}

        if self.working.size >= self.working.capacity // 2:
            info = self._consolidate()

        return info

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Multi-tier read with learned tier weighting.

        Parameters
        ----------
        query : torch.Tensor
            Query of shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Fused memory content of shape ``(B, state_dim)``.
        """
        B = query.size(0)

        # Read from each tier
        working_read = self.working.read(query)  # (B, D)
        episodic_read = self._episodic_read(query)  # (B, D)
        persistent_read = self.persistent.read(query)  # (B, D)

        # Compute tier weights
        gate = self.tier_gate(query)  # (B, 3)

        # Weighted combination
        combined = torch.cat([working_read, episodic_read, persistent_read], dim=-1)
        fused = self.tier_fusion(combined)  # (B, D)

        # Apply gate
        stacked = torch.stack([working_read, episodic_read, persistent_read], dim=1)  # (B, 3, D)
        gated = (stacked * gate.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Blend fusion and gated
        return fused + gated

    def _episodic_read(self, query: torch.Tensor) -> torch.Tensor:
        """Attention-based read from episodic buffer."""
        B = query.size(0)
        if self._episodic_count == 0:
            return torch.zeros(B, self.state_dim, device=query.device)

        active = self.episodic_buffer[:self._episodic_count]  # (E, D)
        scores = torch.matmul(query, active.T) / (self.state_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, active)

    def _episodic_write(self, state: torch.Tensor, importance: float) -> None:
        """Write to episodic memory buffer."""
        with torch.no_grad():
            if state.dim() > 1:
                state = state[0]
            self.episodic_buffer[self._episodic_write_idx] = state.detach()
            self.episodic_importance[self._episodic_write_idx] = importance
            self._episodic_write_idx = (self._episodic_write_idx + 1) % self.episodic_capacity
            self._episodic_count = min(self._episodic_count + 1, self.episodic_capacity)

    def _consolidate(self) -> Dict[str, int]:
        """
        Promote important working memories to episodic, and
        important episodic memories to persistent.
        """
        info = {"consolidated_to_episodic": 0, "consolidated_to_persistent": 0}

        # Working → Episodic
        important_states = self.working.get_important_states(top_k=4)
        for state, score in important_states:
            if score >= self.consolidation_threshold:
                self._episodic_write(state, score)
                info["consolidated_to_episodic"] += 1

        # Episodic → Persistent (promote highest importance)
        if self._episodic_count > 0:
            top_idx = self.episodic_importance[:self._episodic_count].argmax()
            top_score = self.episodic_importance[top_idx].item()
            if top_score >= self.consolidation_threshold + 0.1:
                state = self.episodic_buffer[top_idx].unsqueeze(0)
                self.persistent.write(state)
                info["consolidated_to_persistent"] += 1

        self._consolidation_count += 1
        return info

    def get_tier_stats(self) -> Dict[str, Any]:
        """Return statistics about each memory tier."""
        return {
            "working_size": self.working.size,
            "working_capacity": self.working.capacity,
            "episodic_size": self._episodic_count,
            "episodic_capacity": self.episodic_capacity,
            "persistent_usage": self.persistent.usage.sum().item(),
            "consolidation_count": self._consolidation_count,
        }

    def reset(self) -> None:
        """Reset all memory tiers."""
        self.working.reset()
        self.episodic_buffer.zero_()
        self.episodic_importance.zero_()
        self._episodic_write_idx = 0
        self._episodic_count = 0
        self.persistent.reset()
        self._consolidation_count = 0
