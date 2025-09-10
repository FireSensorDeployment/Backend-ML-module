"""
Minimal FireSensorEnv that passes gymnasium.utils.env_checker.check_env.

This is an MVP skeleton: observations are a 50x50 float grid in [0,1],
and the agent controls a cursor with 5 discrete actions (noop, up, down, left, right).
Reward is 0 by default; episodes terminate after a fixed horizon.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

@dataclass
class EnvConfig:
    grid_size : Tuple[int, int] = (50, 50) #对标decision grid
    max_steps: int = 50
    start_pos: Optional[Tuple[int, int]] = None  #没规定位置None => center
    reward_per_step: float = 0.0   # 目前不考虑奖励机制, MVP

class FireSensorEnv(gym.Env):
    """
    A minimal, deterministic-safe Gymnasium env.

    Observation: Box(low=0, high=1, shape=(H, W), dtype=np.float32)
    Action: Discrete(5) -> 0:noop, 1:up, 2:down, 3:left, 4:right
    Episode ends: fixed horizon (max_steps)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config or EnvConfig()
        h, w = self.config.grid_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h, w), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # internal state
        self._grid: Optional[np.ndarray] = None
        self._pos: Tuple[int, int] = (h // 2, w // 2)
        self._steps: int = 0

        # seeding (Gymnasium style)
        self.np_random, _ = seeding.np_random(seed)

    # Optional legacy-style seeding helper to meet "Add seed() support for reproducibility"
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_grid(self) -> np.ndarray:
        """For MVP, deterministic pseudo-random grid using self.np_random."""
        h, w = self.config.grid_size
        grid = self.np_random.random(size=(h, w), dtype=np.float32)
        return grid.astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment. Must return (obs, info)."""
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        h, w = self.config.grid_size
        if self.config.start_pos is None:
            self._pos = (h // 2, w // 2)
        else:
            r = int(np.clip(self.config.start_pos[0], 0, h - 1))
            c = int(np.clip(self.config.start_pos[1], 0, w - 1))
            self._pos = (r, c)

        self._steps = 0
        self._grid = self._sample_grid()
        obs = self._grid.copy()

        info: Dict[str, Any] = {"position": self._pos}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment. Must return (obs, reward, terminated, truncated, info).
        """
        assert self._grid is not None, "Call reset() before step()."
        h, w = self._grid.shape

        # Action effects
        dr = dc = 0
        if action == 1:   # up
            dr = -1
        elif action == 2: # down
            dr = 1
        elif action == 3: # left
            dc = -1
        elif action == 4: # right
            dc = 1
        # 0 is noop

        r, c = self._pos
        new_r, new_c = r + dr, c + dc

        # 🔑 Clamp to legal bounds instead of truncating the episode
        new_r = max(0, min(new_r, h - 1))
        new_c = max(0, min(new_c, w - 1))
        self._pos = (new_r, new_c)

        self._steps += 1
        terminated = False  # no terminal condition in MVP
        truncated = self._steps >= self.config.max_steps

        reward = float(self.config.reward_per_step)

        obs = self._grid.copy()
        info: Dict[str, Any] = {"position": self._pos, "step": self._steps}

        return obs, reward, terminated, truncated, info