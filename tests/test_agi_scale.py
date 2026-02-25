"""
Tests for Phase 5: AGI-Scale Architecture
    - DistributedCognitiveGraph
    - HierarchicalMemorySystem
    - LifelongLearner (EWC)
    - AdaptiveTopology
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from ucga.distributed_graph import (
    MessageBus, GraphPartition, DistributedCognitiveGraph,
)
from ucga.memory.hierarchical_memory import WorkingMemory, HierarchicalMemorySystem
from ucga.lifelong_learner import EWCRegularizer, LifelongLearner
from ucga.adaptive_topology import NodeImportanceScorer, AdaptiveTopology
from ucga.ucga_model import UCGAModel


# ===========================================================================
# MessageBus
# ===========================================================================
class TestMessageBus:
    def test_send_recv(self):
        bus = MessageBus()
        t = torch.randn(4, 64)
        bus.send(0, 1, t)
        assert bus.recv(0, 1) is not None
        assert bus.recv(1, 0) is None

    def test_clear(self):
        bus = MessageBus()
        bus.send(0, 1, torch.randn(2, 32))
        bus.clear()
        assert bus.recv(0, 1) is None


# ===========================================================================
# DistributedCognitiveGraph
# ===========================================================================
class TestDistributedGraph:
    def test_forward_shape(self):
        g = DistributedCognitiveGraph(state_dim=64, num_partitions=2,
                                       nodes_per_partition=2, communication_rounds=2)
        x = torch.randn(4, 64)
        out = g(x)
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        g = DistributedCognitiveGraph(state_dim=64, num_partitions=2,
                                       nodes_per_partition=2)
        x = torch.randn(4, 64)
        out = g(x)
        out.sum().backward()
        # Check at least some params have gradients
        has_grad = any(p.grad is not None for p in g.parameters())
        assert has_grad

    def test_partition_states(self):
        g = DistributedCognitiveGraph(state_dim=64, num_partitions=3)
        x = torch.randn(4, 64)
        g(x)
        states = g.get_partition_states()
        assert len(states) == 3

    def test_multiple_rounds(self):
        g1 = DistributedCognitiveGraph(state_dim=64, communication_rounds=1)
        g3 = DistributedCognitiveGraph(state_dim=64, communication_rounds=3)
        x = torch.randn(4, 64)
        o1 = g1(x)
        o3 = g3(x)
        assert o1.shape == o3.shape


# ===========================================================================
# WorkingMemory
# ===========================================================================
class TestWorkingMemory:
    def test_write_read(self):
        wm = WorkingMemory(capacity=8, state_dim=32)
        state = torch.randn(32)
        wm.write(state)
        assert wm.size == 1
        query = torch.randn(1, 32)
        out = wm.read(query)
        assert out.shape == (1, 32)

    def test_fifo_eviction(self):
        wm = WorkingMemory(capacity=4, state_dim=16)
        for _ in range(6):
            wm.write(torch.randn(16))
        assert wm.size == 4  # capped at capacity

    def test_importance_scoring(self):
        wm = WorkingMemory(capacity=8, state_dim=32)
        for _ in range(5):
            wm.write(torch.randn(32))
        important = wm.get_important_states(top_k=3)
        assert len(important) == 3
        for state, score in important:
            assert 0.0 <= score <= 1.0

    def test_empty_read(self):
        wm = WorkingMemory(capacity=8, state_dim=32)
        query = torch.randn(2, 32)
        out = wm.read(query)
        assert out.shape == (2, 32)
        assert out.abs().sum().item() == 0.0

    def test_reset(self):
        wm = WorkingMemory(capacity=8, state_dim=32)
        wm.write(torch.randn(32))
        wm.reset()
        assert wm.size == 0


# ===========================================================================
# HierarchicalMemorySystem
# ===========================================================================
class TestHierarchicalMemory:
    def test_write_and_read(self):
        hm = HierarchicalMemorySystem(state_dim=64, working_capacity=8)
        state = torch.randn(1, 64)
        hm.write(state)
        query = torch.randn(1, 64)
        out = hm.read(query)
        assert out.shape == (1, 64)

    def test_tier_stats(self):
        hm = HierarchicalMemorySystem(state_dim=64, working_capacity=8)
        for _ in range(5):
            hm.write(torch.randn(1, 64))
        stats = hm.get_tier_stats()
        assert "working_size" in stats
        assert "episodic_size" in stats
        assert stats["working_size"] <= 8

    def test_consolidation(self):
        hm = HierarchicalMemorySystem(
            state_dim=32, working_capacity=4,
            consolidation_threshold=0.0,  # always consolidate
        )
        for _ in range(10):
            hm.write(torch.randn(1, 32))
        stats = hm.get_tier_stats()
        assert stats["consolidation_count"] > 0

    def test_reset(self):
        hm = HierarchicalMemorySystem(state_dim=32, working_capacity=4)
        hm.write(torch.randn(1, 32))
        hm.reset()
        stats = hm.get_tier_stats()
        assert stats["working_size"] == 0
        assert stats["episodic_size"] == 0


# ===========================================================================
# EWCRegularizer
# ===========================================================================
class TestEWC:
    def _make_simple_model(self):
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def test_penalty_before_consolidation(self):
        model = self._make_simple_model()
        ewc = EWCRegularizer(model, ewc_lambda=1000)
        penalty = ewc.penalty()
        assert penalty.item() == 0.0

    def test_compute_fisher(self):
        model = self._make_simple_model()
        ewc = EWCRegularizer(model)
        # Simple data loader
        data = [(torch.randn(8, 16), torch.randint(0, 4, (8,))) for _ in range(5)]
        ewc.compute_fisher(data, nn.CrossEntropyLoss(), n_samples=40)
        assert ewc.n_tasks == 1

    def test_penalty_after_consolidation(self):
        model = self._make_simple_model()
        ewc = EWCRegularizer(model, ewc_lambda=100)
        data = [(torch.randn(8, 16), torch.randint(0, 4, (8,))) for _ in range(5)]
        ewc.compute_fisher(data, nn.CrossEntropyLoss(), n_samples=20)
        # Perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        penalty = ewc.penalty()
        assert penalty.item() > 0.0


# ===========================================================================
# LifelongLearner
# ===========================================================================
class TestLifelongLearner:
    def _make_setup(self):
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, opt

    def test_train_step(self):
        model, opt = self._make_setup()
        ll = LifelongLearner(model, opt)
        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))
        result = ll.train_step(x, y, nn.CrossEntropyLoss())
        assert "task_loss" in result
        assert "ewc_penalty" in result
        assert "total_loss" in result

    def test_task_lifecycle(self):
        model, opt = self._make_setup()
        ll = LifelongLearner(model, opt)
        ll.begin_task("task_1")
        data = [(torch.randn(8, 16), torch.randint(0, 4, (8,))) for _ in range(3)]
        info = ll.end_task(data, nn.CrossEntropyLoss(), n_samples=20)
        assert info["task_number"] == 1
        assert len(ll.task_history) == 1


# ===========================================================================
# NodeImportanceScorer
# ===========================================================================
class TestNodeImportance:
    def test_score_nodes(self):
        model = UCGAModel(input_dim=64, state_dim=64, output_dim=16)
        x = torch.randn(4, 64)
        model(x)  # populate node states
        scorer = NodeImportanceScorer(state_dim=64)
        scores = scorer.score_nodes(model)
        assert len(scores) > 0
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0


# ===========================================================================
# AdaptiveTopology
# ===========================================================================
class TestAdaptiveTopology:
    def test_forward_shape(self):
        model = UCGAModel(input_dim=64, state_dim=64, output_dim=16)
        at = AdaptiveTopology(model)
        x = torch.randn(4, 64)
        out = at(x)
        assert out.shape == (4, 16)

    def test_return_meta(self):
        model = UCGAModel(input_dim=64, state_dim=64, output_dim=16)
        at = AdaptiveTopology(model)
        x = torch.randn(4, 64)
        out, meta = at(x, return_meta=True)
        assert "topology" in meta
        assert "importance_scores" in meta["topology"]

    def test_topology_stats(self):
        model = UCGAModel(input_dim=64, state_dim=64, output_dim=16)
        at = AdaptiveTopology(model)
        x = torch.randn(4, 64)
        at(x)
        stats = at.get_topology_stats()
        assert "active_nodes" in stats
        assert "n_active" in stats
        assert stats["topology_changes"] == 1

    def test_gradient_flow(self):
        model = UCGAModel(input_dim=64, state_dim=64, output_dim=16)
        at = AdaptiveTopology(model)
        x = torch.randn(4, 64)
        out = at(x)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in at.parameters())
        assert has_grad
