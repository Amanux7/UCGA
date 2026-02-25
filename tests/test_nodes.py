"""
test_nodes.py — Unit tests for all UCGA cognitive nodes.
"""

import pytest
import torch

from ucga.nodes import (
    CognitiveNode,
    PerceptionNode,
    MemoryNode,
    ReasoningNode,
    PlanningNode,
    EvaluationNode,
    CorrectionNode,
    BalancerNode,
    OutputNode,
)


BATCH = 4
STATE_DIM = 32
INPUT_DIM = 32
OUTPUT_DIM = 16


# ==================================================================
# Base CognitiveNode
# ==================================================================
class TestCognitiveNode:
    def test_forward_shape(self):
        node = CognitiveNode(input_dim=STATE_DIM, state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)

    def test_multiple_inputs(self):
        node = CognitiveNode(input_dim=STATE_DIM, state_dim=STATE_DIM)
        a = torch.randn(BATCH, STATE_DIM)
        b = torch.randn(BATCH, STATE_DIM)
        out = node([a, b])
        assert out.shape == (BATCH, STATE_DIM)

    def test_reset_state(self):
        node = CognitiveNode(input_dim=STATE_DIM, state_dim=STATE_DIM)
        node.reset_state(BATCH)
        assert node.get_state().shape == (BATCH, STATE_DIM)
        assert node.get_state().abs().sum().item() == 0.0

    def test_gradient_flow(self):
        node = CognitiveNode(input_dim=STATE_DIM, state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM, requires_grad=True)
        out = node([x])
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


# ==================================================================
# PerceptionNode
# ==================================================================
class TestPerceptionNode:
    def test_forward_shape(self):
        node = PerceptionNode(raw_input_dim=INPUT_DIM, state_dim=STATE_DIM)
        x = torch.randn(BATCH, INPUT_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)

    def test_projection(self):
        """Input dim != state dim should still work."""
        node = PerceptionNode(raw_input_dim=64, state_dim=STATE_DIM)
        x = torch.randn(BATCH, 64)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# MemoryNode
# ==================================================================
class TestMemoryNode:
    def test_forward_shape(self):
        node = MemoryNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        bank = torch.randn(BATCH, 16, STATE_DIM)
        out = node([x], memory_bank=bank)
        assert out.shape == (BATCH, STATE_DIM)

    def test_without_memory(self):
        """MemoryNode should work with empty memory bank."""
        node = MemoryNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        bank = torch.zeros(BATCH, 8, STATE_DIM)
        out = node([x], memory_bank=bank)
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# ReasoningNode
# ==================================================================
class TestReasoningNode:
    def test_forward_shape(self):
        node = ReasoningNode(state_dim=STATE_DIM, reasoning_steps=3)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)

    def test_single_step(self):
        node = ReasoningNode(state_dim=STATE_DIM, reasoning_steps=1)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# PlanningNode
# ==================================================================
class TestPlanningNode:
    def test_forward_shape(self):
        node = PlanningNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# EvaluationNode
# ==================================================================
class TestEvaluationNode:
    def test_forward_shape(self):
        node = EvaluationNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, STATE_DIM)

    def test_confidence_range(self):
        node = EvaluationNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        _ = node([x])
        conf = node.get_confidence()
        assert conf.shape == (BATCH, 1)
        assert (conf >= 0).all() and (conf <= 1).all()


# ==================================================================
# CorrectionNode
# ==================================================================
class TestCorrectionNode:
    def test_forward_shape(self):
        node = CorrectionNode(state_dim=STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        y = torch.randn(BATCH, STATE_DIM)
        out = node([x, y])
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# BalancerNode
# ==================================================================
class TestBalancerNode:
    def test_forward_shape(self):
        node = BalancerNode(state_dim=STATE_DIM, num_streams=3)
        a = torch.randn(BATCH, STATE_DIM)
        b = torch.randn(BATCH, STATE_DIM)
        c = torch.randn(BATCH, STATE_DIM)
        out = node([a, b, c])
        assert out.shape == (BATCH, STATE_DIM)


# ==================================================================
# OutputNode
# ==================================================================
class TestOutputNode:
    def test_forward_shape(self):
        node = OutputNode(state_dim=STATE_DIM, output_dim=OUTPUT_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = node([x])
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_gradient_flow(self):
        node = OutputNode(state_dim=STATE_DIM, output_dim=OUTPUT_DIM)
        x = torch.randn(BATCH, STATE_DIM, requires_grad=True)
        out = node([x])
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
