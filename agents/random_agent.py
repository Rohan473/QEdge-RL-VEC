"""
agents/random_agent.py

Uniform-random baseline: selects each action with equal probability 1/4.
Serves as a lower-bound reference in evaluation.
"""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Uniform-random action selector.

    Args:
        n_actions: size of the discrete action space (default 4).
        seed: random seed for reproducibility.
    """

    def __init__(self, n_actions: int = 4, seed: int = 42) -> None:
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)

    def act(self, obs: np.ndarray) -> int:
        """Return a uniformly random action, ignoring the observation.

        Args:
            obs: current environment observation (ignored).

        Returns:
            Integer action in [0, n_actions).
        """
        return int(self._rng.randint(self.n_actions))

    def reset(self) -> None:
        """No-op — random agent carries no state between episodes."""
        pass
