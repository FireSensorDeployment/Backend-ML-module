# examples/verify_env.py
"""
Run Gymnasium's check_env on FireSensorEnv to validate interface compliance,
and smoke-test the boundary clamp behavior.
"""
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl.environment import FireSensorEnv, EnvConfig

def assert_pos(env, expected_rc):
    pos = env.step(0)[-1]["position"]  # noop to fetch info; doesn't change pos
    assert pos == expected_rc, f"Expected position {expected_rc}, got {pos}"

def main():
    env = FireSensorEnv(EnvConfig(grid_size=(5, 5), max_steps=100), seed=123)
    check_env(env)

    # Reset to top-left corner for clamp tests
    obs, info = env.reset(seed=123)
    env._pos = (0, 0)  # for test clarity; production code would avoid touching internals
    # Try to move further up/left repeatedly -> should clamp to (0,0)
    for _ in range(10):
        env.step(1)  # up
        env.step(3)  # left
    assert_pos(env, (0, 0))

    # Move to bottom-right and attempt to go beyond
    env._pos = (4, 4)
    for _ in range(10):
        env.step(2)  # down
        env.step(4)  # right
    assert_pos(env, (4, 4))

    # Quick random rollout to ensure no crashes
    obs, info = env.reset(seed=42)
    for _ in range(20):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            obs, info = env.reset()

    print("check_env passed and clamp behavior verified.")

if __name__ == "__main__":
    main()
