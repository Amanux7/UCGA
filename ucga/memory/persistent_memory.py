"""
PersistentMemory — External key-value memory bank with read / write / attention.

This module provides the long-term memory substrate for the UCGA architecture.
Memory slots are stored as a learnable buffer and can be updated during
forward passes or persisted across episodes.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PersistentMemory(nn.Module):
    """
    Differentiable persistent memory bank.

    Parameters
    ----------
    num_slots : int
        Number of memory slots.
    slot_dim : int
        Dimensionality of each memory slot.
    """

    def __init__(self, num_slots: int = 128, slot_dim: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Memory bank — initialised as zeros, progressively filled
        self.register_buffer(
            "memory", torch.zeros(1, num_slots, slot_dim)
        )
        self.register_buffer(
            "usage", torch.zeros(1, num_slots)
        )

        # Write controller
        self.write_gate = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.Sigmoid(),
        )
        self.write_proj = nn.Linear(slot_dim, slot_dim)

        # Read (query) controller
        self.read_query = nn.Linear(slot_dim, slot_dim)
        self.attn_scale = slot_dim ** 0.5

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Attention-based read from memory.

        Parameters
        ----------
        query : torch.Tensor
            Query vector of shape ``(B, slot_dim)``.

        Returns
        -------
        torch.Tensor
            Retrieved memory content of shape ``(B, slot_dim)``.
        """
        B = query.size(0)
        mem = self._expand_memory(B)

        Q = self.read_query(query).unsqueeze(1)             # (B, 1, D)
        scores = torch.bmm(Q, mem.transpose(1, 2)) / self.attn_scale  # (B, 1, M)
        weights = torch.softmax(scores, dim=-1)
        retrieved = torch.bmm(weights, mem).squeeze(1)       # (B, D)
        return retrieved

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(self, content: torch.Tensor) -> None:
        """
        Write *content* into the least-used memory slot.

        Parameters
        ----------
        content : torch.Tensor
            Data to store, shape ``(B, slot_dim)``.
        """
        B = content.size(0)
        mem = self._expand_memory(B)
        usage = self._expand_usage(B)

        # Find least-used slot per batch element
        _, idx = usage.min(dim=1)  # (B,)

        gate = self.write_gate(content)                    # (B, D)
        projected = self.write_proj(content)               # (B, D)
        write_val = gate * projected

        # Scatter write — detach to prevent graph accumulation
        idx_exp = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.slot_dim)
        mem.scatter_(1, idx_exp, write_val.detach().unsqueeze(1))

        # Update usage
        usage.scatter_(1, idx.unsqueeze(1), usage.gather(1, idx.unsqueeze(1)) + 1)

        self.memory = mem[:1].detach()   # keep canonical shape, detach
        self.usage = usage[:1].detach()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_memory_bank(self, batch_size: int = 1) -> torch.Tensor:
        """Return memory expanded to batch size."""
        return self._expand_memory(batch_size)

    def reset(self) -> None:
        """Zero-out the memory bank and usage counters."""
        self.memory.zero_()
        self.usage.zero_()

    def _expand_memory(self, B: int) -> torch.Tensor:
        return self.memory.expand(B, -1, -1).clone()

    def _expand_usage(self, B: int) -> torch.Tensor:
        return self.usage.expand(B, -1).clone()

    def __repr__(self) -> str:
        return f"PersistentMemory(num_slots={self.num_slots}, slot_dim={self.slot_dim})"
