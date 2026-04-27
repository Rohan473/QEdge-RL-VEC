"""
vec_env/environment.py

VECEnvironment: Gymnasium-compatible simulation of vehicular edge computing.

World
-----
- 1-D highway segment of length L = 1000 m.
- N_VEHICLES = 5 vehicles move along the highway at random speeds (20–120 km/h).
- N_EDGE_SERVERS = 2 fixed at positions 250 m and 750 m.
- Discrete time step Δt = 100 ms; episode length T = 200 steps.

Task generation
---------------
At each step the "current vehicle" (round-robined across the fleet) always
produces one task.  Parameters are sampled independently:
    data_size  ~ Uniform[100, 1000] KB
    cpu_cycles ~ Uniform[100, 1000] Megacycles
    deadline   ~ Uniform[100, 500] ms

Action space
------------
Discrete(4):  0 = LOCAL, 1 = EDGE_1, 2 = EDGE_2, 3 = DROP

Observation (12-dim float32)
----------------------------
Index  Feature                         Normalisation
  0    data_size                       / 1000
  1    cpu_cycles                      / 1000
  2    deadline                        / 500
  3    vehicle_speed                   / 120 (km/h)
  4    dist_to_edge1                   / L
  5    dist_to_edge2                   / L
  6    queue_len_edge1                 / 20
  7    queue_len_edge2                 / 20
  8    channel_rate_edge1              (via normalise_rate)
  9    channel_rate_edge2              (via normalise_rate)
 10    local_cpu_load                  [0, 1]
 11    time_left                       / T

Reward
------
    if action != DROP and latency <= deadline:
        r = +1.0 - α * (latency / deadline) - β * energy
    else:
        r = -1.0
    α = 0.3, β = 0.1
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from vec_env.utils import (
    channel_rate,
    compute_latency_ms,
    compute_energy_j,
    normalise_rate,
)

# ── World constants ───────────────────────────────────────────────────────────
N_VEHICLES: int = 5
L: float = 1000.0                    # highway length (m)
EDGE_POSITIONS: Tuple[float, float] = (250.0, 750.0)  # fixed server positions (m)
DT_S: float = 0.1                    # time step (s = 100 ms)
EPISODE_LENGTH: int = 200            # steps per episode

# Task parameter ranges
DATA_SIZE_RANGE: Tuple[float, float] = (100.0, 1000.0)   # KB
CPU_CYCLES_RANGE: Tuple[float, float] = (100.0, 1000.0)  # Megacycles
DEADLINE_RANGE: Tuple[float, float] = (100.0, 500.0)     # ms

# Vehicle speed range
SPEED_RANGE_KMH: Tuple[float, float] = (20.0, 120.0)

# Queue management
MAX_QUEUE: float = 20.0    # tasks (normalisation upper bound)
# Service rates: average task removed per step from each server's queue
QUEUE_SERVICE_RATE: float = 0.8    # tasks drained per step

# Local CPU load dynamics
LOCAL_LOAD_DELTA: float = 0.15     # added when task runs locally
LOCAL_LOAD_DECAY: float = 0.05     # drained per step

# Reward shaping coefficients
ALPHA: float = 0.3    # latency penalty weight
BETA: float = 0.1     # energy penalty weight

# Observation dimension
OBS_DIM: int = 12


class VECEnvironment(gym.Env):
    """Vehicular Edge Computing task-offloading environment.

    Implements the standard Gymnasium interface: reset() / step() / render().
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    ACTION_LOCAL: int = 0
    ACTION_EDGE_1: int = 1
    ACTION_EDGE_2: int = 2
    ACTION_DROP: int = 3
    ACTION_NAMES: Tuple[str, ...] = ("LOCAL", "EDGE_1", "EDGE_2", "DROP")

    def __init__(
        self,
        n_vehicles: int = N_VEHICLES,
        episode_length: int = EPISODE_LENGTH,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.n_vehicles = n_vehicles
        self.episode_length = episode_length

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = np.random.RandomState(seed if seed is not None else 42)
        self._t: int = 0
        self._vehicle_idx: int = 0

        # Vehicle state (will be initialised in reset)
        self._positions = np.zeros(n_vehicles, dtype=np.float64)
        self._speeds_kmh = np.zeros(n_vehicles, dtype=np.float64)

        # Server queues
        self._queue_edge1: float = 0.0
        self._queue_edge2: float = 0.0

        # Local CPU load of the current vehicle
        self._local_loads = np.zeros(n_vehicles, dtype=np.float64)

        # Current task
        self._task_data_kb: float = 0.0
        self._task_cpu_mc: float = 0.0
        self._task_deadline_ms: float = 0.0

        self.reset(seed=seed)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sample_task(self) -> None:
        """Sample task parameters from uniform distributions."""
        self._task_data_kb = self._rng.uniform(*DATA_SIZE_RANGE)
        self._task_cpu_mc = self._rng.uniform(*CPU_CYCLES_RANGE)
        self._task_deadline_ms = self._rng.uniform(*DEADLINE_RANGE)

    def _channel_rates(self) -> Tuple[float, float]:
        """Compute channel rates from the current vehicle to both edge servers."""
        pos = self._positions[self._vehicle_idx]
        r1 = channel_rate(abs(pos - EDGE_POSITIONS[0]))
        r2 = channel_rate(abs(pos - EDGE_POSITIONS[1]))
        return r1, r2

    def _build_obs(self) -> np.ndarray:
        r1, r2 = self._channel_rates()
        v_idx = self._vehicle_idx
        obs = np.array([
            self._task_data_kb / DATA_SIZE_RANGE[1],
            self._task_cpu_mc / CPU_CYCLES_RANGE[1],
            self._task_deadline_ms / DEADLINE_RANGE[1],
            self._speeds_kmh[v_idx] / SPEED_RANGE_KMH[1],
            abs(self._positions[v_idx] - EDGE_POSITIONS[0]) / L,
            abs(self._positions[v_idx] - EDGE_POSITIONS[1]) / L,
            self._queue_edge1 / MAX_QUEUE,
            self._queue_edge2 / MAX_QUEUE,
            normalise_rate(r1),
            normalise_rate(r2),
            float(self._local_loads[v_idx]),
            float(self._t) / self.episode_length,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _advance_world(self) -> None:
        """Move vehicles, drain queues, decay local CPU load."""
        dt_h = DT_S / 3600.0
        for i in range(self.n_vehicles):
            self._positions[i] += self._speeds_kmh[i] * 1000.0 * dt_h
            # Wrap-around highway (periodic boundary)
            self._positions[i] %= L
            # Random speed fluctuation ±2 km/h
            self._speeds_kmh[i] = np.clip(
                self._speeds_kmh[i] + self._rng.uniform(-2.0, 2.0),
                *SPEED_RANGE_KMH,
            )

        # Drain server queues
        self._queue_edge1 = max(0.0, self._queue_edge1 - QUEUE_SERVICE_RATE)
        self._queue_edge2 = max(0.0, self._queue_edge2 - QUEUE_SERVICE_RATE)

        # Decay local CPU loads
        self._local_loads = np.clip(
            self._local_loads - LOCAL_LOAD_DECAY, 0.0, 1.0
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._t = 0
        self._vehicle_idx = 0

        # Uniformly distribute vehicles along the highway
        self._positions = self._rng.uniform(0.0, L, size=self.n_vehicles)
        self._speeds_kmh = self._rng.uniform(*SPEED_RANGE_KMH, size=self.n_vehicles)
        self._local_loads = np.zeros(self.n_vehicles, dtype=np.float64)
        self._queue_edge1 = 0.0
        self._queue_edge2 = 0.0

        self._sample_task()
        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one task-offloading decision.

        Args:
            action: integer in {0, 1, 2, 3}.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if action not in range(4):
            raise ValueError(f"Invalid action {action}; must be in {{0,1,2,3}}")

        r1, r2 = self._channel_rates()
        rate = r1 if action == self.ACTION_EDGE_1 else r2

        latency_ms = compute_latency_ms(
            action,
            self._task_data_kb,
            self._task_cpu_mc,
            r1, r2,
            self._queue_edge1,
            self._queue_edge2,
        )
        energy_j = compute_energy_j(action, self._task_data_kb, self._task_cpu_mc, rate)

        # Reward function from spec
        if action != self.ACTION_DROP and latency_ms <= self._task_deadline_ms:
            reward = 1.0 - ALPHA * (latency_ms / self._task_deadline_ms) - BETA * energy_j
        else:
            reward = -1.0

        # Update server queues and local load
        if action == self.ACTION_EDGE_1:
            self._queue_edge1 = min(self._queue_edge1 + 1.0, MAX_QUEUE)
        elif action == self.ACTION_EDGE_2:
            self._queue_edge2 = min(self._queue_edge2 + 1.0, MAX_QUEUE)
        elif action == self.ACTION_LOCAL:
            v_idx = self._vehicle_idx
            self._local_loads[v_idx] = min(
                self._local_loads[v_idx] + LOCAL_LOAD_DELTA, 1.0
            )

        # Capture info for the task that was just decided (before sampling the next one)
        deadline_met = (action != self.ACTION_DROP and latency_ms <= self._task_deadline_ms)
        info = {
            "latency_ms": latency_ms,
            "energy_j": energy_j,
            "deadline_ms": self._task_deadline_ms,
            "deadline_met": deadline_met,
            "dropped": action == self.ACTION_DROP,
            "queue_edge1": self._queue_edge1,
            "queue_edge2": self._queue_edge2,
            "channel_rate_edge1": r1,
            "channel_rate_edge2": r2,
        }

        # Advance world state, then sample the next task
        self._advance_world()
        self._t += 1
        self._vehicle_idx = (self._vehicle_idx + 1) % self.n_vehicles
        self._sample_task()

        terminated = False
        truncated = self._t >= self.episode_length

        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a one-line ASCII summary of the world state."""
        v = self._vehicle_idx
        r1, r2 = self._channel_rates()
        print(
            f"t={self._t:3d} | vehicle={v} pos={self._positions[v]:.0f}m "
            f"speed={self._speeds_kmh[v]:.1f}km/h | "
            f"q1={self._queue_edge1:.1f} q2={self._queue_edge2:.1f} | "
            f"R1={r1/1e6:.1f}Mbps R2={r2/1e6:.1f}Mbps | "
            f"task=[{self._task_data_kb:.0f}KB "
            f"{self._task_cpu_mc:.0f}Mc "
            f"{self._task_deadline_ms:.0f}ms]"
        )
