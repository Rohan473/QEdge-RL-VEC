"""
tests/test_environment.py

Pytest tests for VECEnvironment.  Run with: pytest tests/test_environment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from vec_env.environment import VECEnvironment, OBS_DIM, EPISODE_LENGTH
from vec_env.utils import channel_rate, compute_latency_ms, compute_energy_j


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_channel_rate_decreases_with_distance(self):
        r_near = channel_rate(10.0)
        r_far = channel_rate(500.0)
        assert r_near > r_far

    def test_channel_rate_positive(self):
        for d in [1.0, 100.0, 999.0]:
            assert channel_rate(d) > 0.0

    def test_channel_rate_clamps_zero_distance(self):
        # Should not raise or return infinity
        r = channel_rate(0.0)
        assert np.isfinite(r)

    def test_latency_drop_is_zero(self):
        lat = compute_latency_ms(3, 500, 500, 1e6, 1e6, 0, 0)
        assert lat == 0.0

    def test_energy_drop_is_zero(self):
        e = compute_energy_j(3, 500, 500, 1e6)
        assert e == 0.0

    def test_local_latency_proportional_to_cpu(self):
        lat_small = compute_latency_ms(0, 500, 100, 1e6, 1e6, 0, 0)
        lat_large = compute_latency_ms(0, 500, 1000, 1e6, 1e6, 0, 0)
        assert lat_large > lat_small

    def test_edge_latency_increases_with_queue(self):
        lat_empty = compute_latency_ms(1, 500, 500, 50e6, 50e6, 0, 0)
        lat_busy = compute_latency_ms(1, 500, 500, 50e6, 50e6, 10, 0)
        assert lat_busy > lat_empty


# ── Environment contract tests ────────────────────────────────────────────────

class TestVECEnvironmentContract:
    def setup_method(self):
        self.env = VECEnvironment(seed=0)

    def test_reset_returns_correct_obs_shape(self):
        obs, info = self.env.reset(seed=0)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_obs_in_unit_range(self):
        obs, _ = self.env.reset(seed=42)
        assert np.all(obs >= 0.0), "Observation below 0"
        assert np.all(obs <= 1.0), "Observation above 1"

    def test_obs_dtype_float32(self):
        obs, _ = self.env.reset(seed=1)
        assert obs.dtype == np.float32

    def test_step_returns_correct_shapes(self):
        obs, _ = self.env.reset(seed=0)
        obs2, reward, term, trunc, info = self.env.step(0)
        assert obs2.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_unit_range(self):
        self.env.reset(seed=0)
        for a in range(4):
            obs, *_ = self.env.step(a)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_episode_terminates_after_T_steps(self):
        self.env.reset(seed=0)
        trunc = False
        for _ in range(EPISODE_LENGTH - 1):
            _, _, _, trunc, _ = self.env.step(0)
            assert not trunc, "Episode truncated too early"
        _, _, _, trunc, _ = self.env.step(0)
        assert trunc, "Episode should have truncated at T steps"

    def test_invalid_action_raises(self):
        self.env.reset(seed=0)
        with pytest.raises(ValueError):
            self.env.step(99)

    def test_gymnasium_observation_space_contains_obs(self):
        obs, _ = self.env.reset(seed=7)
        assert self.env.observation_space.contains(obs)


# ── Reward sign tests ─────────────────────────────────────────────────────────

class TestRewardSign:
    """Verify reward signs on obvious cases."""

    def _run_until_known(self, env: VECEnvironment, action: int, max_tries: int = 200):
        """Step with a fixed action and return (reward, info) for the first step."""
        obs, _ = env.reset(seed=0)
        _, r, _, _, info = env.step(action)
        return r, info

    def test_drop_always_negative(self):
        env = VECEnvironment(seed=0)
        for seed in range(10):
            env.reset(seed=seed)
            _, r, _, _, _ = env.step(env.ACTION_DROP)
            assert r == -1.0, f"DROP should yield -1.0, got {r}"

    def test_feasible_local_task_positive(self):
        """A task with very large deadline should almost always be completed locally."""
        env = VECEnvironment(seed=0)
        env.reset(seed=0)
        env._task_data_kb = 100.0
        env._task_cpu_mc = 100.0
        env._task_deadline_ms = 500.0   # very generous deadline
        # Local latency = 100 Mc / 1 GHz = 0.1 ms << 500 ms deadline
        _, r, _, _, info = env.step(env.ACTION_LOCAL)
        assert r > 0.0, f"Expected positive reward for easy local task, got {r}"
        assert info["deadline_met"]

    def test_tight_deadline_edge_returns_negative(self):
        """A tiny deadline forces a latency violation → negative reward."""
        env = VECEnvironment(seed=0)
        env.reset(seed=0)
        env._task_deadline_ms = 0.01  # 0.01 ms — no action can meet this
        _, r, _, _, info = env.step(env.ACTION_EDGE_1)
        assert r == -1.0, f"Expected -1.0 for impossible deadline, got {r}"
        assert not info["deadline_met"]

    def test_reward_bounded(self):
        """Reward must stay within [-1, 1]."""
        env = VECEnvironment(seed=42)
        env.reset(seed=42)
        for _ in range(100):
            a = env.action_space.sample()
            _, r, _, trunc, _ = env.step(a)
            assert r <= 1.0, f"Reward exceeded 1.0: {r}"
            assert r >= -1.0, f"Reward below -1.0: {r}"
            if trunc:
                env.reset()
