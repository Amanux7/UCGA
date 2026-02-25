"""
Tests for Phase 4 cognitive agent modules:
    - EpisodicMemory
    - ToolRegistry
    - RewardSignal
    - RLAgent
    - GridWorldEnv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from ucga.memory.episodic_memory import EpisodicMemory
from agents.tool_registry import ToolRegistry, Tool
from agents.reward_signal import RewardSignal
from agents.rl_agent import RLAgent
from agents.grid_world_env import GridWorldEnv


STATE_DIM = 32
BATCH = 2


# ===========================================================================
# EpisodicMemory
# ===========================================================================
class TestEpisodicMemory:
    def test_store_and_size(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        assert mem.size == 0
        for i in range(5):
            mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=float(i))
        assert mem.size == 5

    def test_capacity_eviction(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=3)
        for i in range(5):
            mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=float(i))
        assert mem.size == 3  # only last 3 kept

    def test_retrieve_by_recency(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        for i in range(5):
            mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=float(i))
        recent = mem.retrieve_by_recency(k=2)
        assert len(recent) == 2
        assert recent[-1].timestamp > recent[-2].timestamp

    def test_retrieve_by_similarity(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        target = torch.ones(STATE_DIM)
        mem.store(target, torch.randn(4), reward=1.0)
        mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=0.0)
        mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=0.0)

        results = mem.retrieve_by_similarity(target, k=1)
        assert len(results) == 1
        assert results[0][1] > 0.9  # highest similarity

    def test_retrieve_as_context_shape(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        for i in range(3):
            mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=0.5)
        query = torch.randn(1, STATE_DIM)
        ctx = mem.retrieve_as_context(query, k=2)
        assert ctx.shape == (1, STATE_DIM)

    def test_retrieve_empty_memory(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        query = torch.randn(1, STATE_DIM)
        ctx = mem.retrieve_as_context(query, k=5)
        assert ctx.shape == (1, STATE_DIM)
        assert (ctx == 0).all()  # zero context when empty

    def test_clear(self):
        mem = EpisodicMemory(state_dim=STATE_DIM, capacity=10)
        mem.store(torch.randn(STATE_DIM), torch.randn(4), reward=1.0)
        mem.clear()
        assert mem.size == 0


# ===========================================================================
# ToolRegistry
# ===========================================================================
class TestToolRegistry:
    def _make_tool(self, name="calc", dim=STATE_DIM):
        return Tool(
            name=name,
            description="Test tool",
            func=lambda x: x * 2,
            input_dim=dim,
            output_dim=dim,
        )

    def test_register_and_list(self):
        reg = ToolRegistry(state_dim=STATE_DIM, max_tools=4)
        reg.register(self._make_tool("add"))
        reg.register(self._make_tool("mul"))
        assert reg.num_tools == 2
        assert "add" in reg.tool_names
        assert "mul" in reg.tool_names

    def test_register_duplicate_raises(self):
        reg = ToolRegistry(state_dim=STATE_DIM, max_tools=4)
        reg.register(self._make_tool("calc"))
        with pytest.raises(ValueError):
            reg.register(self._make_tool("calc"))

    def test_register_over_limit_raises(self):
        reg = ToolRegistry(state_dim=STATE_DIM, max_tools=1)
        reg.register(self._make_tool("t1"))
        with pytest.raises(ValueError):
            reg.register(self._make_tool("t2"))

    def test_unregister(self):
        reg = ToolRegistry(state_dim=STATE_DIM, max_tools=4)
        reg.register(self._make_tool("calc"))
        reg.unregister("calc")
        assert reg.num_tools == 0

    def test_select_and_execute(self):
        reg = ToolRegistry(state_dim=STATE_DIM, max_tools=4)
        reg.register(self._make_tool("t1"))
        state = torch.randn(BATCH, STATE_DIM)
        result = reg.select_and_execute(state)
        assert result["tool_output"].shape == (BATCH, STATE_DIM)
        assert result["tool_probs"].shape[0] == BATCH
        assert len(result["selected_tools"]) == BATCH


# ===========================================================================
# RewardSignal
# ===========================================================================
class TestRewardSignal:
    def test_estimate_value(self):
        rs = RewardSignal(state_dim=STATE_DIM)
        states = torch.randn(BATCH, STATE_DIM)
        values = rs.estimate_value(states)
        assert values.shape == (BATCH,)

    def test_compute_returns(self):
        rs = RewardSignal(gamma=0.99)
        rewards = [1.0, 0.0, 0.0, 1.0]
        returns = rs.compute_returns(rewards)
        assert returns.shape == (4,)
        assert returns[0] > returns[1]  # earlier gets more discounted reward

    def test_compute_gae(self):
        rs = RewardSignal(state_dim=STATE_DIM, gamma=0.99, gae_lambda=0.95)
        rewards = [1.0, 0.0, 0.5]
        values = torch.tensor([0.5, 0.3, 0.2])
        advantages = rs.compute_gae(rewards, values)
        assert advantages.shape == (3,)

    def test_normalize(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = RewardSignal.normalize(x)
        assert abs(normed.mean().item()) < 1e-5
        assert abs(normed.std().item() - 1.0) < 0.1


# ===========================================================================
# RLAgent
# ===========================================================================
class TestRLAgent:
    def test_get_action(self):
        agent = RLAgent(obs_dim=16, action_dim=4, state_dim=STATE_DIM)
        obs = torch.randn(1, 16)
        action, log_prob, value = agent.get_action(obs)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

    def test_deterministic_action(self):
        agent = RLAgent(obs_dim=16, action_dim=4, state_dim=STATE_DIM)
        obs = torch.randn(1, 16)
        a1, _, _ = agent.get_action(obs, deterministic=True)
        a2, _, _ = agent.get_action(obs, deterministic=True)
        assert a1.item() == a2.item()

    def test_ppo_loss_empty_trajectory(self):
        agent = RLAgent(obs_dim=16, action_dim=4, state_dim=STATE_DIM)
        losses = agent.compute_ppo_loss()
        assert losses["total_loss"].item() == 0.0

    def test_store_and_compute_loss(self):
        agent = RLAgent(obs_dim=16, action_dim=4, state_dim=STATE_DIM)
        obs = torch.randn(1, 16)
        action, log_prob, value = agent.get_action(obs)
        agent.store_transition(obs, action, reward=1.0, log_prob=log_prob, value=value)
        agent.store_transition(obs, action, reward=0.5, log_prob=log_prob, value=value)
        losses = agent.compute_ppo_loss()
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "total_loss" in losses

    def test_parameter_count(self):
        agent = RLAgent(obs_dim=16, action_dim=4, state_dim=STATE_DIM)
        assert agent.count_parameters() > 0


# ===========================================================================
# GridWorldEnv
# ===========================================================================
class TestGridWorldEnv:
    def test_reset_shape(self):
        env = GridWorldEnv(grid_size=5, max_steps=20)
        obs = env.reset()
        assert obs.shape == (1, env.obs_dim)

    def test_step_returns(self):
        env = GridWorldEnv(grid_size=5, max_steps=20)
        env.reset()
        obs, reward, done, info = env.step(1)
        assert obs.shape == (1, env.obs_dim)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_goal_reached(self):
        env = GridWorldEnv(grid_size=3, max_steps=100)
        env.reset()
        # Navigate to (2,2) from (0,0): down, down, right, right
        env.step(2); env.step(2)
        _, reward, done, info = env.step(1)
        if not done:
            _, reward, done, info = env.step(1)
        assert done
        assert info["reason"] == "goal_reached"
        assert reward == 1.0

    def test_max_steps_timeout(self):
        env = GridWorldEnv(grid_size=5, max_steps=3)
        env.reset()
        for _ in range(3):
            obs, reward, done, info = env.step(0)  # keep going up (hitting wall)
        assert done
        assert info["reason"] == "timeout"

    def test_obstacles(self):
        env = GridWorldEnv(grid_size=5, max_steps=20, obstacle_ratio=0.3, seed=42)
        env.reset()
        # Start and goal should be clear
        assert env.grid[0, 0] == 0
        assert env.grid[4, 4] == 0

    def test_render(self):
        env = GridWorldEnv(grid_size=4)
        env.reset()
        rendered = env.render_ascii()
        assert "A" in rendered
        assert "G" in rendered
