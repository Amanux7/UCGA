"""
test_model.py — Integration tests for the full UCGAModel.
"""

import pytest
import torch
import torch.nn as nn

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder


BATCH = 4
INPUT_DIM = 32
STATE_DIM = 64
OUTPUT_DIM = 16


class TestUCGAModel:
    """Core model tests."""

    def test_forward_shape(self):
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
            cognitive_steps=1, reasoning_steps=1,
        )
        x = torch.randn(BATCH, STATE_DIM)
        out = model(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_return_meta(self):
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
            cognitive_steps=2, reasoning_steps=1,
        )
        x = torch.randn(BATCH, STATE_DIM)
        out, meta = model(x, return_meta=True)
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert "confidences" in meta
        assert "corrections" in meta
        assert len(meta["confidences"]) == 2  # cognitive_steps=2

    def test_multi_step(self):
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
            cognitive_steps=3, reasoning_steps=2,
        )
        x = torch.randn(BATCH, STATE_DIM)
        out, meta = model(x, return_meta=True)
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert len(meta["confidences"]) == 3

    def test_count_parameters(self):
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
        )
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_reset_memory(self):
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
        )
        x = torch.randn(BATCH, STATE_DIM)
        model(x)
        model.reset_memory()
        assert model.persistent_memory.memory.abs().sum().item() == 0.0

    def test_repr(self):
        model = UCGAModel()
        s = repr(model)
        assert "UCGAModel" in s
        assert "state_dim" in s


class TestEndToEndTraining:
    """Tests that the full pipeline can train end-to-end."""

    def test_train_step(self):
        enc = VectorEncoder(INPUT_DIM, STATE_DIM)
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
            cognitive_steps=1, reasoning_steps=1,
        )
        opt = torch.optim.Adam(
            list(model.parameters()) + list(enc.parameters()), lr=1e-3
        )
        criterion = nn.MSELoss()

        x = torch.randn(BATCH, INPUT_DIM)
        y = torch.randn(BATCH, OUTPUT_DIM)

        pred = model(enc(x))
        loss = criterion(pred, y)
        loss.backward()
        opt.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_multi_step_train(self):
        """Multiple training steps should work without gradient errors."""
        enc = VectorEncoder(INPUT_DIM, STATE_DIM)
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=OUTPUT_DIM, memory_slots=8,
            cognitive_steps=1, reasoning_steps=1,
        )
        params = list(model.parameters()) + list(enc.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        criterion = nn.MSELoss()

        for _ in range(5):
            x = torch.randn(BATCH, INPUT_DIM)
            y = torch.randn(BATCH, OUTPUT_DIM)
            pred = model(enc(x))
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert not torch.isnan(loss)

    def test_classification(self):
        """Test cross-entropy classification through the model."""
        enc = VectorEncoder(INPUT_DIM, STATE_DIM)
        model = UCGAModel(
            input_dim=STATE_DIM, state_dim=STATE_DIM,
            output_dim=4, memory_slots=8,
            cognitive_steps=1, reasoning_steps=1,
        )
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(BATCH, INPUT_DIM)
        labels = torch.randint(0, 4, (BATCH,))

        logits = model(enc(x))
        loss = criterion(logits, labels)
        loss.backward()

        assert logits.shape == (BATCH, 4)
        assert not torch.isnan(loss)
