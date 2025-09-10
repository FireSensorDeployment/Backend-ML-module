"""
Minimal FireSensorEnv that passes gymnasium.utils.env_checker.check_env.

This is an MVP skeleton: observations are a 50x50 float grid in [0,1],
and the agent controls a cursor with 5 discrete actions (noop, up, down, left, right).
Reward is 0 by default; episodes terminate after a fixed horizon.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Sequence, List

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

    dataset_paths: Optional[Sequence[str]] = None     # list of .npz files
    allow_sample_with_replacement: bool = False       # False => cycle through shuffled queue
    expected_grid_size: Tuple[int, int] = (50, 50)    # strict validation target
    clip_obs_min: float = 0.0
    clip_obs_max: float = 1.0

    place_action_id: int = 5                 # new "Place" Action
    min_sensor_distance: int = 2             # “距离过近”的阈值（格子距离） - 之后会改成半径距离 (sensor size)
    invalid_action_penalty: float = -0.5     # 软惩罚
    invalid_action_threshold: int = 10       # 无效行为累计上限（>= 则 truncate）

class FireSensorEnv(gym.Env):
    """
    A minimal, deterministic-safe Gymnasium env.

    Observation: Box(low=0, high=1, shape=(H, W), dtype=np.float32)
    Action: Discrete(6) -> 0:noop, 1:up, 2:down, 3:left, 4:right, 5: place
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
        self.action_space = spaces.Discrete(6)

        # internal state
        self._grid: Optional[np.ndarray] = None
        self._pos: Tuple[int, int] = (h // 2, w // 2)
        self._steps: int = 0
        self._placed: set[tuple[int, int]] = set()
        self._invalid_actions: int = 0

        # dataset bookkeeping
        self._dataset_paths: Optional[List[str]] = list(self.config.dataset_paths) if self.config.dataset_paths else None
        self._scenario_queue: List[int] = []   # permutation of indices for no-replacement sampling
        self._last_scenario_path: Optional[str] = None

        # RNG
        self.np_random, _ = seeding.np_random(seed)

        if self._dataset_paths and not self.config.allow_sample_with_replacement:
            self._reshuffle_queue()

    def _reshuffle_queue(self) -> None:
        assert self._dataset_paths is not None and len(self._dataset_paths) > 0, "Empty dataset path list"
        indices = np.arange(len(self._dataset_paths))
        self.np_random.shuffle(indices) 
        self._scenario_queue = list(indices)

    def _choose_next_scenario_path(self) -> str:
        assert self._dataset_paths is not None and len(self._dataset_paths) > 0, "No dataset paths configured"
        if self.config.allow_sample_with_replacement:
            idx = int(self.np_random.integers(0, len(self._dataset_paths)))
            return self._dataset_paths[idx]
        # without replacement: cycle through a shuffled queue
        if not self._scenario_queue:
            self._reshuffle_queue()
        idx = self._scenario_queue.pop(0)
        return self._dataset_paths[idx]

    def _load_npz_scenario(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        data = np.load(path, allow_pickle=True)
        # required keys
        for k in ("decision_grid", "display_layer", "metadata"):
            if k not in data:
                raise KeyError(f"NPZ missing key '{k}' in {path}")
        grid = data["decision_grid"]

        # shape validation
        if grid.shape != self.config.expected_grid_size:
            raise ValueError(f"decision_grid has shape {grid.shape}, expected {self.config.expected_grid_size} in {path}")

        # clip & cast
        grid = np.clip(grid, self.config.clip_obs_min, self.config.clip_obs_max).astype(np.float32, copy=False)

        # metadata (allow pickle dict or array of object)
        metadata_raw = data["metadata"]
        if isinstance(metadata_raw, np.ndarray) and metadata_raw.dtype == object:
            metadata = metadata_raw.item()
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {"raw": metadata_raw}

        return grid, {"metadata": metadata, "scenario_path": path}

    def _sample_grid_placeholder(self) -> np.ndarray:
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
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            if self._dataset_paths and not self.config.allow_sample_with_replacement:
                self._reshuffle_queue()

        h, w = self.config.grid_size
        if self.config.start_pos is None:
            self._pos = (h // 2, w // 2)
        else:
            r = int(np.clip(self.config.start_pos[0], 0, h - 1))
            c = int(np.clip(self.config.start_pos[1], 0, w - 1))
            self._pos = (r, c)

        self._steps = 0
        self._placed.clear()
        self._invalid_actions = 0

        # dataset-backed observation or placeholder
        if self._dataset_paths:
            scenario_path = self._choose_next_scenario_path()
            grid, extra = self._load_npz_scenario(scenario_path)
            self._grid = grid
            info: Dict[str, Any] = {"position": self._pos, **extra}
            self._last_scenario_path = scenario_path
        else:
            self._grid = self._sample_grid_placeholder()
            info = {"position": self._pos, "scenario_path": None, "metadata": {}}

        obs = self._grid.copy()
        return obs, info
    
    def step(self, action: int):
        assert self._grid is not None, "Call reset() before step()."
        h, w = self._grid.shape
        obs_before = self._grid  # 用于“状态未改变”的断言

        # place 动作优先处理（避免 place 后还移动）
        if action == self.config.place_action_id:
            r, c = self._pos
            # 规则：重复放置 / 距离过近 -> 无效，罚分，状态不变
            if (r, c) in self._placed:
                return self._apply_invalid_penalty(obs_before, reason="duplicate")
            if self._too_close(r, c):
                return self._apply_invalid_penalty(obs_before, reason="too_close")
            # 合法放置：记录即可（MVP 不改变 obs）
            self._placed.add((r, c))
            self._steps += 1
            reward = float(self.config.reward_per_step)
            info = {
                "position": self._pos,
                "step": self._steps,
                "scenario_path": getattr(self, "_last_scenario_path", None),
                "invalid_actions": self._invalid_actions,
            }
            truncated = self._steps >= self.config.max_steps
            return obs_before.copy(), reward, False, truncated, info

        # 移动：保持 clamp（不计无效，不给惩罚）
        dr = dc = 0
        if action == 1:   dr = -1
        elif action == 2: dr = 1
        elif action == 3: dc = -1
        elif action == 4: dc = 1
        # 0: noop

        r, c = self._pos
        new_r, new_c = r + dr, c + dc
        new_r = max(0, min(new_r, h - 1))
        new_c = max(0, min(new_c, w - 1))
        self._pos = (new_r, new_c)

        self._steps += 1
        reward = float(self.config.reward_per_step)
        truncated = self._steps >= self.config.max_steps
        info = {
            "position": self._pos,
            "step": self._steps,
            "scenario_path": getattr(self, "_last_scenario_path", None),
            "invalid_actions": self._invalid_actions,
        }
        return self._grid.copy(), reward, False, truncated, info

    def _too_close(self, r: int, c: int) -> bool:
        if not self._placed:
            return False
        d = self.config.min_sensor_distance
        for pr, pc in self._placed:
            if max(abs(pr - r), abs(pc - c)) < d:
                return True
        return False

    def _apply_invalid_penalty(self, obs: np.ndarray, reason: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._invalid_actions += 1
        truncated = self._invalid_actions >= self.config.invalid_action_threshold
        info = {
            "position": self._pos,
            "step": self._steps,
            "scenario_path": getattr(self, "_last_scenario_path", None),
            "invalid_actions": self._invalid_actions,
            "invalid_reason": reason,
        }
        # 状态不变，obs 不变（返回拷贝以符合 Gym 惯例）
        return obs.copy(), float(self.config.invalid_action_penalty), False, truncated, info