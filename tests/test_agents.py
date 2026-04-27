"""
tests/test_agents.py

Smoke tests for Random, Greedy, and DQN agents on VECEnvironment.
QRL is import-guarded (PennyLane may not be installed in CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from vec_env.environment import VECEnvironment, OBS_DIM


@pytest.fixture
def env():
    e = VECEnvironment(seed=0)
    e.reset(seed=0)
    return e


@pytest.fixture
def obs(env):
    o, _ = env.reset(seed=0)
    return o


class TestRandomAgent:
    def test_act_returns_valid_action(self, obs):
        from agents.random_agent import RandomAgent
        agent = RandomAgent(seed=0)
        for _ in range(20):
            a = agent.act(obs)
            assert a in range(4)

    def test_different_seeds_differ(self, obs):
        from agents.random_agent import RandomAgent
        a1 = [RandomAgent(seed=0).act(obs) for _ in range(10)]
        a2 = [RandomAgent(seed=99).act(obs) for _ in range(10)]
        assert a1 != a2  # extremely unlikely to be identical for different seeds


class TestGreedyAgent:
    def test_act_returns_valid_action(self, obs):
        from agents.greedy_agent import GreedyAgent
        agent = GreedyAgent()
        a = agent.act(obs)
        assert a in range(4)

    def test_low_queue_prefers_edge(self):
        """With no queue congestion and moderate CPU task, edge should beat local."""
        from agents.greedy_agent import GreedyAgent
        agent = GreedyAgent()
        # Build obs with low queue (indices 6,7 = 0) and high channel rate (indices 8,9 = 1)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = 0.1   # small data (100 KB)
        obs[1] = 0.9   # large cpu (900 Mc) → local very slow
        obs[8] = 1.0   # perfect channel to edge 1
        obs[9] = 1.0   # perfect channel to edge 2
        a = agent.act(obs)
        assert a in (1, 2), f"Expected edge action for high-cpu task with good channel, got {a}"

    def test_deterministic(self, obs):
        from agents.greedy_agent import GreedyAgent
        agent = GreedyAgent()
        assert agent.act(obs) == agent.act(obs)


class TestDQNAgent:
    def test_act_before_training(self, obs):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, seed=0)
        a = agent.act(obs)
        assert a in range(4)

    def test_observe_returns_none_before_warmup(self, obs, env):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, warmup_steps=1000, seed=0)
        obs2, r, _, _, _ = env.step(0)
        loss = agent.observe(obs, 0, r, obs2, False)
        assert loss is None

    def test_observe_returns_loss_after_warmup(self, env):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, warmup_steps=5, batch_size=4, seed=0)
        obs, _ = env.reset()
        for _ in range(10):
            a = agent.act(obs)
            obs2, r, _, trunc, _ = env.step(a)
            loss = agent.observe(obs, a, r, obs2, trunc)
            obs = obs2
            if trunc:
                obs, _ = env.reset()
        # After 10 steps with warmup=5, at least one loss should be non-None
        # (we don't assert the exact value — just that it's a finite float)
        agent2 = DQNAgent(obs_dim=OBS_DIM, warmup_steps=5, batch_size=4, seed=0)
        obs, _ = env.reset()
        losses = []
        for _ in range(20):
            a = agent2.act(obs)
            obs2, r, _, trunc, _ = env.step(a)
            l = agent2.observe(obs, a, r, obs2, trunc)
            if l is not None:
                losses.append(l)
            obs = obs2
            if trunc:
                obs, _ = env.reset()
        assert len(losses) > 0, "Expected at least one learning step"
        assert all(np.isfinite(l) for l in losses), "Loss contains non-finite value"

    def test_save_load_roundtrip(self, obs, tmp_path):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, seed=7)
        path = str(tmp_path / "test_dqn.pt")
        agent.save(path)
        agent2 = DQNAgent(obs_dim=OBS_DIM, seed=99)  # different seed
        agent2.load(path)
        # After loading, both networks should produce the same Q-values
        import torch
        s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q1 = agent.online(s)
            q2 = agent2.online(s)
        assert torch.allclose(q1, q2), "Loaded weights differ from saved"

    def test_epsilon_decays(self, env):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, epsilon_start=1.0, epsilon_end=0.05,
                         epsilon_decay_steps=100, seed=0)
        obs, _ = env.reset()
        eps_before = agent.eps
        for _ in range(50):
            a = agent.act(obs)
            obs2, r, _, trunc, _ = env.step(a)
            agent.observe(obs, a, r, obs2, trunc)
            obs = obs2
            if trunc:
                obs, _ = env.reset()
        assert agent.eps < eps_before, "Epsilon did not decay"


class TestQRLAgentSmoke:
    """Import-guarded smoke tests for QRL (requires PennyLane)."""

    def test_act_returns_valid_action(self, obs):
        try:
            from agents.qrl_agent import QRLAgent
        except ImportError:
            pytest.skip("PennyLane not installed")
        agent = QRLAgent(obs_dim=OBS_DIM, seed=0, batch_size=4, warmup_steps=5)
        a = agent.act(obs)
        assert a in range(4)

    def test_forward_pass_finite(self, obs):
        try:
            from agents.qrl_agent import QRLAgent
            import torch
        except ImportError:
            pytest.skip("PennyLane or torch not installed")
        agent = QRLAgent(obs_dim=OBS_DIM, seed=0)
        s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = agent.online(s)
        assert q.shape == (1, 4)
        assert torch.all(torch.isfinite(q)), "VQC produced non-finite Q-values"
