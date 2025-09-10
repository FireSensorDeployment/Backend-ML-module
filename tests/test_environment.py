# tests/test_environment.py
import os
import sys
import numpy as np
import pytest

# Ensure src is importable regardless of cwd
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rl.environment import FireSensorEnv, EnvConfig


def _write_npz(path, decision_grid, display_layer=None, metadata=None):
    if display_layer is None:
        # dummy high-res layer; not used by env, but required by spec
        display_layer = np.zeros((1000, 1000), dtype=np.float32)
    if metadata is None:
        metadata = {"tag": os.path.basename(path)}
    np.savez(path, decision_grid=decision_grid, display_layer=display_layer, metadata=metadata)


def test_shape_validation(tmp_path):
    # wrong shape (48x48) should raise ValueError
    bad_grid = np.zeros((48, 48), dtype=np.float32)
    p = tmp_path / "bad_shape.npz"
    _write_npz(p, bad_grid)

    cfg = EnvConfig(dataset_paths=[str(p)], expected_grid_size=(50, 50))
    env = FireSensorEnv(cfg, seed=123)

    with pytest.raises(ValueError):
        env.reset()


def test_normalization_clip(tmp_path):
    # values outside [0,1] should be clipped
    grid = np.array([[-0.5, 0.2], [0.9, 1.5]], dtype=np.float32)
    # pad to 50x50 by tiling
    grid = np.tile(grid, (25, 25))[:50, :50]
    p = tmp_path / "clip_grid.npz"
    _write_npz(p, grid)

    cfg = EnvConfig(dataset_paths=[str(p)], expected_grid_size=(50, 50))
    env = FireSensorEnv(cfg, seed=123)
    obs, info = env.reset()

    assert obs.shape == (50, 50)
    assert np.min(obs) >= 0.0 and np.max(obs) <= 1.0, "Observation must be clipped to [0,1]"
    assert obs.dtype == np.float32


def test_multiple_scenarios_cycle_without_replacement(tmp_path):
    # two obviously different scenarios
    g1 = np.zeros((50, 50), dtype=np.float32)
    g2 = np.ones((50, 50), dtype=np.float32) * 0.8
    p1 = tmp_path / "s1.npz"
    p2 = tmp_path / "s2.npz"
    _write_npz(p1, g1)
    _write_npz(p2, g2)

    cfg = EnvConfig(
        dataset_paths=[str(p1), str(p2)],
        expected_grid_size=(50, 50),
        allow_sample_with_replacement=False  # cycle through both before repeating
    )
    env = FireSensorEnv(cfg, seed=123)

    obs1, info1 = env.reset()
    obs2, info2 = env.reset()

    # Should be different scenarios in the first two resets (queue without replacement)
    assert not np.allclose(obs1, obs2), "Expected different scenarios across resets"
    assert info1.get("scenario_path") != info2.get("scenario_path")

    # Ensure still clipped and shaped
    for obs in (obs1, obs2):
        assert obs.shape == (50, 50)
        assert np.min(obs) >= 0.0 and np.max(obs) <= 1.0
        assert obs.dtype == np.float32
