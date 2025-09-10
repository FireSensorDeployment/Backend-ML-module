# tests/test_environment.py
import os
import sys
import numpy as np
import pytest

# --- 让 tests 在任意工作目录下都能 import 到 src/rl/environment.py ---
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rl.environment import FireSensorEnv, EnvConfig


def make_env_for_invalid_tests():
    """
    构造一个最小环境用于软惩罚机制测试：
    - 使用占位随机网格（不依赖外部数据集）
    - place 动作为 5
    - 将 invalid 阈值设得很小，便于单测
    """
    cfg = EnvConfig(
        grid_size=(5, 5),
        max_steps=100,
        # dataset_paths=None  # 默认 None：使用占位随机网格
        place_action_id=5,
        min_sensor_distance=2,          # Chebyshev 距离 < 2 视为过近
        invalid_action_penalty=-0.5,
        invalid_action_threshold=3,     # 连续 3 次无效动作 -> truncated
    )
    return FireSensorEnv(cfg, seed=123)


def _assert_obs_unchanged(obs_before: np.ndarray, obs_after: np.ndarray):
    # 要求 obs 完全相等（无任何数值变化）
    assert isinstance(obs_before, np.ndarray) and isinstance(obs_after, np.ndarray)
    assert obs_before.shape == obs_after.shape
    assert obs_before.dtype == np.float32 and obs_after.dtype == np.float32
    assert np.array_equal(obs_before, obs_after), "Invalid action must not change observation"


def test_duplicate_place_penalty():
    env = make_env_for_invalid_tests()
    obs0, info0 = env.reset(seed=0)

    # 首次 place：合法（不应惩罚）
    obs1, r1, term1, trunc1, info1 = env.step(env.config.place_action_id)
    assert r1 == 0.0
    assert not term1 and not trunc1
    assert info1.get("invalid_actions", 0) == 0

    # 第二次在同一格 place：重复 -> 应判无效并惩罚；obs 不变
    obs2, r2, term2, trunc2, info2 = env.step(env.config.place_action_id)
    _assert_obs_unchanged(obs1, obs2)
    assert r2 == env.config.invalid_action_penalty
    assert not term2  # 本次未终止
    assert info2["invalid_actions"] == 1
    assert info2.get("invalid_reason") == "duplicate"


def test_too_close_place_penalty():
    env = make_env_for_invalid_tests()
    obs0, info0 = env.reset(seed=0)

    # env.reset() 默认中心 (2,2)（grid_size=5x5）
    # 先在 (2,2) 合法放置一次
    env._pos = (2, 2)
    obs1, r1, term1, trunc1, info1 = env.step(env.config.place_action_id)
    assert r1 == 0.0 and not term1 and not trunc1
    assert info1.get("invalid_actions", 0) == 0

    # 移动到 (1,2)（Chebyshev 距离 1 < 2）再尝试 place -> too_close
    env._pos = (1, 2)
    obs_before = obs1.copy()
    obs2, r2, term2, trunc2, info2 = env.step(env.config.place_action_id)

    _assert_obs_unchanged(obs_before, obs2)
    assert r2 == env.config.invalid_action_penalty
    assert not term2
    assert info2["invalid_actions"] == 1
    assert info2.get("invalid_reason") == "too_close"


def test_invalid_count_threshold_truncates():
    env = make_env_for_invalid_tests()
    env.reset(seed=0)

    # 先在 (0,0) 合法放置一次
    env._pos = (0, 0)
    obs1, r1, term1, trunc1, info1 = env.step(env.config.place_action_id)
    assert r1 == 0.0 and not trunc1

    # 连续重复 place 三次，阈值=3 -> 第 3 次应 truncated=True
    obs2, r2, term2, trunc2, info2 = env.step(env.config.place_action_id)  # invalid #1
    obs3, r3, term3, trunc3, info3 = env.step(env.config.place_action_id)  # invalid #2
    obs4, r4, term4, trunc4, info4 = env.step(env.config.place_action_id)  # invalid #3 -> 触发截断

    _assert_obs_unchanged(obs1, obs2)
    _assert_obs_unchanged(obs2, obs3)
    _assert_obs_unchanged(obs3, obs4)

    assert r2 == env.config.invalid_action_penalty
    assert r3 == env.config.invalid_action_penalty
    assert r4 == env.config.invalid_action_penalty

    assert info2["invalid_actions"] == 1 and not trunc2
    assert info3["invalid_actions"] == 2 and not trunc3
    assert info4["invalid_actions"] == 3 and trunc4, "Should truncate when invalid_actions >= threshold"


def test_invalid_actions_do_not_crash_env_and_obs_dtype_shape():
    """
    兜底健壮性：无效动作不应导致崩溃；obs 形状与 dtype 保持稳定。
    """
    env = make_env_for_invalid_tests()
    obs, info = env.reset(seed=42)
    h, w = env.config.grid_size

    # 故意累积几次无效（在同一格重复 place）
    env._pos = (2, 2)
    env.step(env.config.place_action_id)  # valid
    for _ in range(5):  # > threshold 也应该只是在最后一次返回 truncated=True，而非抛异常
        obs2, r2, term2, trunc2, info2 = env.step(env.config.place_action_id)
        assert obs2.shape == (h, w)
        assert obs2.dtype == np.float32
        assert isinstance(trunc2, bool) and isinstance(term2, bool)
