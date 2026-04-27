"""
rl_env.py

Vehicular edge offloading as a gym-compatible short-horizon MDP.

Observation (15-dim float32):
    [0:8]   task features (sampled from the synthetic classification pool)
    [8]     predicted class (0 or 1)
    [9]     classifier confidence in [0, 1]
    [10]    edge_1 queue length (tasks)
    [11]    edge_2 queue length (tasks)
    [12]    link latency to edge_1 (ms / 100)
    [13]    link latency to edge_2 (ms / 100)
    [14]    battery SoC in [0, 1]

Actions (Discrete(4)):
    0 = local, 1 = edge_1, 2 = edge_2, 3 = drop

Reward per step:
    r = -(w_lat * latency_ms / 100 + w_energy * energy + w_safety * safety_risk)

The "safety_risk" term fires mainly on DROP when the task is judged critical
(predicted class 1 with confidence above a threshold). This makes DROP nearly
prohibitive for safety-critical inputs, matching the PDF's framing.
"""

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class OffloadingEnv(gym.Env):
    metadata = {"render_modes": []}

    ACTION_LOCAL = 0
    ACTION_EDGE_1 = 1
    ACTION_EDGE_2 = 2
    ACTION_DROP = 3
    ACTION_NAMES = ["LOCAL", "EDGE_1", "EDGE_2", "DROP"]

    def __init__(
        self,
        episode_length: int = 100,
        latency_weight: float = 1.0,
        energy_weight: float = 0.5,
        safety_weight: float = 5.0,
        pool_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.w_lat = latency_weight
        self.w_energy = energy_weight
        self.w_safety = safety_weight
        self._base_seed = seed

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(15,), dtype=np.float32
        )

        X, y = make_classification(
            n_samples=pool_size, n_features=8, n_informative=5, n_redundant=1,
            random_state=seed,
        )
        self._clf = LogisticRegression(max_iter=1000).fit(X, y)
        self._X_pool = X.astype(np.float32)
        self._y_pool = y.astype(np.int64)

        self._rng = np.random.RandomState(seed)
        self.reset(seed=seed)

    def _sample_task(self) -> Tuple[np.ndarray, int]:
        idx = self._rng.randint(0, len(self._X_pool))
        return self._X_pool[idx], int(self._y_pool[idx])

    def _classify(self, feat: np.ndarray) -> Tuple[int, float]:
        probs = self._clf.predict_proba(feat.reshape(1, -1))[0]
        pred = int(np.argmax(probs))
        return pred, float(probs[pred])

    def _obs(self) -> np.ndarray:
        feat = self._feat
        pred, prob = self._classify(feat)
        return np.concatenate([
            feat,
            np.array([
                float(pred),
                prob,
                self._q1,
                self._q2,
                self._lat1 / 100.0,
                self._lat2 / 100.0,
                self._soc,
            ], dtype=np.float32),
        ]).astype(np.float32)

    def reset(self, *, seed: int = None, options: Dict[str, Any] = None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._q1 = 0.0
        self._q2 = 0.0
        self._lat1 = 30.0 + 10.0 * self._rng.rand()
        self._lat2 = 50.0 + 20.0 * self._rng.rand()
        self._soc = 1.0
        self._t = 0
        self._feat, self._label = self._sample_task()
        return self._obs(), {}

    def step(self, action: int):
        feat = self._feat
        norm = float(np.linalg.norm(feat)) / feat.size
        pred, prob = self._classify(feat)
        is_critical = (pred == 1) and (prob > 0.65)

        service1 = 1.5
        service2 = 1.0

        if action == self.ACTION_LOCAL:
            latency = 40.0 + 20.0 * norm
            energy = 3.0 * (1.0 + norm)
            safety = 1.0 if (is_critical and prob < 0.8) else 0.0
        elif action == self.ACTION_EDGE_1:
            queue_wait = self._q1 * (1.0 / service1) * 10.0
            latency = self._lat1 + queue_wait + 8.0
            energy = 0.6
            safety = 0.0
            self._q1 += 1.0
        elif action == self.ACTION_EDGE_2:
            queue_wait = self._q2 * (1.0 / service2) * 10.0
            latency = self._lat2 + queue_wait + 8.0
            energy = 0.9
            safety = 0.0
            self._q2 += 1.0
        elif action == self.ACTION_DROP:
            latency = 0.0
            energy = 0.0
            safety = 10.0 if is_critical else 2.0
        else:
            raise ValueError(f"invalid action {action}")

        self._q1 = max(0.0, self._q1 - service1 * 0.4)
        self._q2 = max(0.0, self._q2 - service2 * 0.4)
        self._soc = max(0.0, self._soc - energy * 0.002)
        self._lat1 = max(5.0, self._lat1 + self._rng.normal(0.0, 2.0))
        self._lat2 = max(5.0, self._lat2 + self._rng.normal(0.0, 2.0))

        reward = -(
            self.w_lat * latency / 100.0
            + self.w_energy * energy
            + self.w_safety * safety
        )

        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_length or self._soc <= 0.0

        self._feat, self._label = self._sample_task()

        info = {
            "latency": latency,
            "energy": energy,
            "safety": safety,
            "q1": self._q1,
            "q2": self._q2,
            "soc": self._soc,
            "is_critical": is_critical,
            "pred": pred,
            "prob": prob,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    env = OffloadingEnv(seed=0)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape, "obs[:5]:", obs[:5])
    total = 0.0
    for t in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        print(f"t={t:02d} a={env.ACTION_NAMES[a]:6s} r={r:7.3f} crit={info['is_critical']!s:5s} q1={info['q1']:.1f} q2={info['q2']:.1f} soc={info['soc']:.3f}")
        if term or trunc:
            break
    print(f"total={total:.2f}")
