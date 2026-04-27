"""
agents/greedy_agent.py

Greedy heuristic baseline: pick the action with the lowest predicted latency
given the current channel state and server queue occupancy.

Observation index mapping (must match vec_env/environment.py):
    0: data_size_norm         (/ 1000 KB)
    1: cpu_cycles_norm        (/ 1000 Mc)
    2: deadline_norm          (/ 500 ms)   — unused here
    3: speed_norm             — unused
    4: dist_edge1_norm        — unused (rate already encoded)
    5: dist_edge2_norm        — unused
    6: queue_edge1_norm       (/ 20)
    7: queue_edge2_norm       (/ 20)
    8: rate_edge1_norm        (normalise_rate)
    9: rate_edge2_norm        (normalise_rate)
   10: local_cpu_load_norm
   11: time_left_norm         — unused

The greedy agent reconstructs approximate physical quantities from the
normalised observation and calls compute_latency_ms directly.
"""

from __future__ import annotations

import numpy as np

from vec_env.utils import compute_latency_ms, _R_REF
from vec_env.environment import DATA_SIZE_RANGE, CPU_CYCLES_RANGE

_DATA_MAX = DATA_SIZE_RANGE[1]   # 1000 KB
_CPU_MAX = CPU_CYCLES_RANGE[1]   # 1000 Mc
_QUEUE_MAX = 20.0

# Penalty applied to DROP so it is only chosen if all compute options
# have latency > DROP_PENALTY_MS. Set very high so greedy almost never drops.
DROP_PENALTY_MS: float = 1e6


class GreedyAgent:
    """Latency-minimising greedy heuristic agent.

    At each step the agent reconstructs approximate latency for LOCAL,
    EDGE_1, and EDGE_2 from the observation vector, then picks the minimum.
    DROP is only selected when explicitly chosen via ``drop_penalty`` override.

    Args:
        drop_penalty_ms: artificial latency assigned to the DROP action.
    """

    def __init__(self, drop_penalty_ms: float = DROP_PENALTY_MS) -> None:
        self.drop_penalty_ms = drop_penalty_ms

    def act(self, obs: np.ndarray) -> int:
        """Select the action with lowest predicted latency.

        Args:
            obs: 12-dim float32 observation from VECEnvironment.

        Returns:
            Integer action in {0, 1, 2, 3}.
        """
        obs = np.asarray(obs, dtype=np.float64)

        # Denormalise the relevant components
        data_kb = float(obs[0]) * _DATA_MAX
        cpu_mc = float(obs[1]) * _CPU_MAX
        q1 = float(obs[6]) * _QUEUE_MAX
        q2 = float(obs[7]) * _QUEUE_MAX
        r1 = float(obs[8]) * _R_REF   # bps
        r2 = float(obs[9]) * _R_REF   # bps

        lat_local = compute_latency_ms(0, data_kb, cpu_mc, r1, r2, q1, q2)
        lat_edge1 = compute_latency_ms(1, data_kb, cpu_mc, r1, r2, q1, q2)
        lat_edge2 = compute_latency_ms(2, data_kb, cpu_mc, r1, r2, q1, q2)

        latencies = [lat_local, lat_edge1, lat_edge2, self.drop_penalty_ms]
        return int(np.argmin(latencies))

    def reset(self) -> None:
        """No-op — greedy agent carries no state between episodes."""
        pass
