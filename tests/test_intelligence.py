"""
Tests for Intelligence Validation System:
    - ROSValidator
    - LOSValidator
    - GIBValidator
    - Intelligence Dashboard metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from experiments.ros_validation import ROSValidator
from experiments.los_validation import LOSValidator
from experiments.gib_validation import GIBValidator
from utils.intelligence_dashboard import compute_ucga_intelligence_score


# ===========================================================================
# ROSValidator
# ===========================================================================
class TestROSValidator:
    def test_instrumented_forward(self):
        v = ROSValidator(input_dim=32, state_dim=64, output_dim=16,
                         cognitive_steps=3, reasoning_steps=3)
        x = torch.randn(4, 32)
        result = v.run_instrumented_forward(x)
        assert "iteration_data" in result
        assert len(result["iteration_data"]) == 3
        assert "ros_score" in result
        assert "convergence_rate" in result

    def test_iteration_data_fields(self):
        v = ROSValidator(input_dim=32, state_dim=64, output_dim=16,
                         cognitive_steps=2, reasoning_steps=2)
        x = torch.randn(4, 32)
        result = v.run_instrumented_forward(x)
        step = result["iteration_data"][0]
        for key in ["iteration", "reasoning_error", "confidence",
                     "reasoning_output_norm"]:
            assert key in step

    def test_ros_score_bounded(self):
        v = ROSValidator(input_dim=32, state_dim=64, output_dim=16,
                         cognitive_steps=5, reasoning_steps=3)
        x = torch.randn(8, 32)
        result = v.run_instrumented_forward(x)
        # ROS score can be negative (reasoning got worse) or positive
        assert isinstance(result["ros_score"], float)

    def test_batch_validation(self):
        v = ROSValidator(input_dim=32, state_dim=64, output_dim=16,
                         cognitive_steps=2, reasoning_steps=2)
        summary = v.run_batch_validation(n_samples=8, difficulty_levels=3)
        assert "avg_ros_score" in summary
        assert len(summary["per_difficulty"]) == 3


# ===========================================================================
# LOSValidator
# ===========================================================================
class TestLOSValidator:
    def test_memory_contribution(self):
        v = LOSValidator(input_dim=32, state_dim=64, output_dim=16,
                         memory_slots=16)
        x = torch.randn(4, 32)
        result = v.measure_memory_contribution(x)
        assert "memory_contribution" in result
        assert "cosine_similarity" in result
        assert result["memory_contribution"] >= 0

    def test_episode_tracking(self):
        v = LOSValidator(input_dim=32, state_dim=64, output_dim=16,
                         memory_slots=16)
        result = v.track_memory_over_episodes(n_episodes=5, batch_size=4)
        assert len(result["episode_logs"]) == 5
        for log in result["episode_logs"]:
            assert "retrieval_similarity" in log
            assert "memory_slots_used_ratio" in log

    def test_los_score(self):
        v = LOSValidator(input_dim=32, state_dim=64, output_dim=16,
                         memory_slots=16)
        result = v.compute_los_score(n_samples=16)
        assert "los_score" in result
        assert isinstance(result["los_score"], float)


# ===========================================================================
# GIBValidator
# ===========================================================================
class TestGIBValidator:
    def test_capture_balance_factors(self):
        v = GIBValidator(input_dim=32, state_dim=64, output_dim=16)
        x = torch.randn(4, 32)
        result = v.capture_balance_factors(x)
        assert "step_data" in result
        for step in result["step_data"]:
            assert "weights" in step
            assert "reasoning_weight" in step
            assert "planning_weight" in step
            assert "memory_weight" in step
            # Weights should sum to ~1.0 (softmax)
            assert abs(sum(step["weights"]) - 1.0) < 0.01

    def test_adaptivity(self):
        v = GIBValidator(input_dim=32, state_dim=64, output_dim=16)
        result = v.test_adaptivity(n_conditions=6, batch_size=8)
        assert "gib_adaptivity_score" in result
        assert result["gib_adaptivity_score"] >= 0
        assert "stream_analysis" in result

    def test_stream_labels(self):
        v = GIBValidator(input_dim=32, state_dim=64, output_dim=16)
        result = v.test_adaptivity(n_conditions=3, batch_size=4)
        for label in ["reasoning", "planning", "memory"]:
            assert label in result["stream_analysis"]


# ===========================================================================
# Intelligence Metric
# ===========================================================================
class TestIntelligenceMetric:
    def test_compute_score(self):
        score = compute_ucga_intelligence_score(
            ros_score=0.8, los_score=0.5, gib_score=0.1,
        )
        expected = 0.4 * 0.8 + 0.35 * 0.5 + 0.25 * 0.1
        assert abs(score - expected) < 1e-6

    def test_zero_scores(self):
        score = compute_ucga_intelligence_score(
            ros_score=0.0, los_score=0.0, gib_score=0.0,
        )
        assert score == 0.0

    def test_custom_weights(self):
        score = compute_ucga_intelligence_score(
            ros_score=1.0, los_score=0.0, gib_score=0.0,
            ros_weight=1.0, los_weight=0.0, gib_weight=0.0,
        )
        assert abs(score - 1.0) < 1e-6
