"""
GridWorldEnv — Simple grid-world environment for UCGA agent benchmarking.

A minimal environment for testing multi-step planning and reinforcement
learning capabilities.  The agent navigates a 2D grid to reach a goal
while optionally avoiding obstacles.

Author: Aman Singh
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


class GridWorldEnv:
    """
    Discrete grid-world environment.

    The agent starts at a fixed position and must navigate to a goal.
    Actions: 0=up, 1=right, 2=down, 3=left.

    Parameters
    ----------
    grid_size : int
        Size of the square grid (grid_size × grid_size).
    max_steps : int
        Maximum steps per episode before truncation.
    obstacle_ratio : float
        Fraction of cells that are obstacles (0.0 = none, 0.3 = 30%).
    seed : int, optional
        Random seed for obstacle placement.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 50,
        obstacle_ratio: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obs_dim = grid_size * grid_size + 4  # flattened grid + one-hot position info

        self.rng = np.random.RandomState(seed)

        # Generate grid: 0=empty, 1=obstacle
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        if obstacle_ratio > 0:
            n_obstacles = int(grid_size * grid_size * obstacle_ratio)
            positions = self.rng.choice(
                grid_size * grid_size, size=n_obstacles, replace=False,
            )
            for pos in positions:
                r, c = divmod(pos, grid_size)
                self.grid[r, c] = 1.0

        # Fixed start (top-left) and goal (bottom-right)
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # Ensure start and goal are clear
        self.grid[self.start_pos] = 0.0
        self.grid[self.goal_pos] = 0.0

        # State
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False

    def _get_obs(self) -> torch.Tensor:
        """Build observation tensor."""
        grid_flat = torch.from_numpy(self.grid.flatten())

        # Agent position as normalized coordinates
        r, c = self.agent_pos
        pos_info = torch.tensor([
            r / self.grid_size,
            c / self.grid_size,
            (self.goal_pos[0] - r) / self.grid_size,
            (self.goal_pos[1] - c) / self.grid_size,
        ])

        obs = torch.cat([grid_flat, pos_info]).float()
        return obs.unsqueeze(0)  # (1, obs_dim)

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False
        return self._get_obs()

    def step(
        self, action: int,
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            Action index (0-3).

        Returns
        -------
        (obs, reward, done, info)
        """
        if self.done:
            return self._get_obs(), 0.0, True, {"reason": "already_done"}

        self.steps += 1
        r, c = self.agent_pos
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc

        # Check boundaries
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            if self.grid[nr, nc] == 0:  # not an obstacle
                self.agent_pos = (nr, nc)

        # Compute reward
        reward = -0.01  # step penalty

        if self.agent_pos == self.goal_pos:
            reward = 1.0
            self.done = True
            return self._get_obs(), reward, True, {"reason": "goal_reached", "steps": self.steps}

        if self.steps >= self.max_steps:
            reward = -0.5  # timeout penalty
            self.done = True
            return self._get_obs(), reward, True, {"reason": "timeout", "steps": self.steps}

        return self._get_obs(), reward, False, {"steps": self.steps}

    @property
    def action_dim(self) -> int:
        return 4

    def render_ascii(self) -> str:
        """Render grid as ASCII art."""
        lines = []
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row += "A "
                elif (r, c) == self.goal_pos:
                    row += "G "
                elif self.grid[r, c] == 1:
                    row += "# "
                else:
                    row += ". "
            lines.append(row.strip())
        return "\n".join(lines)
