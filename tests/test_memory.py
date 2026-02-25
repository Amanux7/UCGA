"""
test_memory.py — Unit tests for the UCGA persistent memory system.
"""

import pytest
import torch

from ucga.memory import PersistentMemory


BATCH = 4
SLOTS = 16
DIM = 32


class TestPersistentMemory:
    def test_init_shape(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        assert mem.memory.shape == (1, SLOTS, DIM)

    def test_get_memory_bank(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        bank = mem.get_memory_bank(BATCH)
        assert bank.shape == (BATCH, SLOTS, DIM)

    def test_read_shape(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        query = torch.randn(BATCH, DIM)
        retrieved = mem.read(query)
        assert retrieved.shape == (BATCH, DIM)

    def test_write(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        content = torch.randn(BATCH, DIM)
        # Should not raise
        mem.write(content)

    def test_read_after_write(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        # Write a pattern
        pattern = torch.randn(1, DIM) * 5
        mem.write(pattern)
        # Read with the same pattern as query
        retrieved = mem.read(pattern)
        assert retrieved.shape == (1, DIM)


    def test_reset(self):
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        mem.write(torch.randn(BATCH, DIM))
        mem.reset()
        assert mem.memory.abs().sum().item() == 0.0
        assert mem.usage.abs().sum().item() == 0.0

    def test_read_differentiable(self):
        """Read operation should allow gradient flow through the query."""
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        query = torch.randn(BATCH, DIM, requires_grad=True)
        retrieved = mem.read(query)
        loss = retrieved.sum()
        loss.backward()
        assert query.grad is not None

    def test_multiple_writes(self):
        """Multiple writes should not crash or corrupt state."""
        mem = PersistentMemory(num_slots=SLOTS, slot_dim=DIM)
        for _ in range(5):
            mem.write(torch.randn(BATCH, DIM))
        bank = mem.get_memory_bank(1)
        assert bank.shape == (1, SLOTS, DIM)
