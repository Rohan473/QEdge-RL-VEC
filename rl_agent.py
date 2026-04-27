"""
rl_agent.py

Minimal DQN agent (PyTorch) for the OffloadingEnv.

- MLP Q-network
- Target network synced every `target_sync` learn steps
- Uniform replay buffer
- ε-greedy exploration with linear-ish (geometric) decay
- Smooth L1 (Huber) loss with gradient clipping
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


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
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


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch: int) -> List[Transition]:
        return self._rng.sample(self.buf, batch)

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_sync: int = 200,
        device: str = "cpu",
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device(device)
        self.online = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = epsilon_start
        self.eps_min = epsilon_end
        self.eps_decay = epsilon_decay
        self.target_sync = target_sync
        self.n_actions = n_actions
        self.step_count = 0

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online(s).squeeze(0).cpu().numpy()

    def learn(self, batch: Sequence[Transition]) -> float:
        s = torch.as_tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor([t.a for t in batch], dtype=torch.long, device=self.device)
        r = torch.as_tensor([t.r for t in batch], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([t.s2 for t in batch]), dtype=torch.float32, device=self.device)
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

        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path: str) -> None:
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
