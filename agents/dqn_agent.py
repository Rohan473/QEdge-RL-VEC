"""
agents/dqn_agent.py

Deep Q-Network (DQN) agent for VECEnvironment.

Architecture:
    - 2-layer MLP: Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 4)
    - Replay buffer (capacity 10 000)
    - Target network with soft update τ = 0.005
    - ε-greedy exploration: ε 1.0 → 0.05 over 10 000 env steps
    - Adam, lr = 1e-3, γ = 0.99
    - Huber loss, gradient clipping at 10

Design note: hidden size is 64 (vs 128 in the legacy rl_agent.py) to keep
inference fast enough that 500 episodes run in <5 min on CPU.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Network ───────────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

@dataclass
class _Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class _ReplayBuffer:
    def __init__(self, capacity: int = 10_000, seed: int = 0) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, t: _Transition) -> None:
        self._buf.append(t)

    def sample(self, batch: int) -> List[_Transition]:
        return self._rng.sample(self._buf, batch)

    def __len__(self) -> int:
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgent:
    """Deep Q-Network agent.

    Args:
        obs_dim: observation space dimension.
        n_actions: number of discrete actions.
        hidden: MLP hidden layer width.
        gamma: discount factor.
        lr: Adam learning rate.
        epsilon_start: initial exploration rate.
        epsilon_end: final exploration rate.
        epsilon_decay_steps: env steps over which ε decays linearly to epsilon_end.
        tau: soft-update coefficient for the target network.
        buffer_capacity: replay buffer size.
        batch_size: mini-batch size for learning.
        warmup_steps: steps before learning starts.
        device: torch device string.
        seed: random seed.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        hidden: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        tau: float = 0.005,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.epsilon_end = epsilon_end
        self._eps_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_steps, 1)
        self.eps = epsilon_start
        self._step = 0

        self.device = torch.device(device)
        self.online = _QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = _QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._buf = _ReplayBuffer(capacity=buffer_capacity, seed=seed)

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection.

        Args:
            obs: current observation.
            greedy: if True, always pick the argmax action (no exploration).

        Returns:
            Integer action.
        """
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe(self, obs: np.ndarray, action: int, reward: float,
                next_obs: np.ndarray, done: bool) -> float | None:
        """Store a transition and trigger a learning step if ready.

        Args:
            obs: state before action.
            action: action taken.
            reward: scalar reward.
            next_obs: resulting state.
            done: whether the episode ended.

        Returns:
            Loss value if a learning step was performed, else None.
        """
        self._buf.push(_Transition(obs, action, reward, next_obs, done))
        self._step += 1
        self.eps = max(self.epsilon_end, self.eps - self._eps_decay)

        if len(self._buf) < self.warmup_steps:
            return None
        if len(self._buf) < self.batch_size:
            return None

        batch = self._buf.sample(self.batch_size)
        return self._learn(batch)

    def _learn(self, batch: Sequence[_Transition]) -> float:
        s = torch.as_tensor(
            np.stack([t.s for t in batch]), dtype=torch.float32, device=self.device
        )
        a = torch.as_tensor([t.a for t in batch], dtype=torch.long, device=self.device)
        r = torch.as_tensor([t.r for t in batch], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(
            np.stack([t.s2 for t in batch]), dtype=torch.float32, device=self.device
        )
        d = torch.as_tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(dim=1).values
            y = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Soft update target network
        for p_online, p_target in zip(self.online.parameters(), self.target.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1.0 - self.tau) * p_target.data)

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save online network weights to *path*."""
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online (and sync target) network weights from *path*."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
        self.eps = self.epsilon_end   # evaluation mode: no exploration
