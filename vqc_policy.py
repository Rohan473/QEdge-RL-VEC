"""
vqc_policy.py

Variational quantum circuit (VQC) policy for the OffloadingEnv, trained with
REINFORCE and analytical parameter-shift gradients.

Circuit (2 qubits, 2 data-reupload layers):

    Layer k:
        Ry(pi * f_{2k},   q0)       # data encoding
        Ry(pi * f_{2k+1}, q1)
        CX(q0, q1)
        Ry(theta_{2k},   q0)        # variational
        Ry(theta_{2k+1}, q1)

The 4-dim feature vector used for data encoding is derived from the 15-dim
observation:
    f = [feature_norm, confidence, queue_total, soc]
    (each scaled roughly into [0, 1])

Action probabilities come directly from the |psi> basis-state probabilities:
    P(|00>) -> LOCAL, |01> -> EDGE_1, |10> -> EDGE_2, |11> -> DROP
with bit-string ordering matching format(i, "02b").

Parameter-shift rule:
    dP(a|s)/d theta_i = 0.5 * [P(a|s; theta_i + pi/2) - P(a|s; theta_i - pi/2)]

This module depends on Qiskit (uses the noiseless Statevector simulator for
speed). It runs without qiskit-aer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _HAS_QISKIT = True
except Exception:
    _HAS_QISKIT = False


def _compress_obs(obs: np.ndarray) -> np.ndarray:
    """15-dim observation -> 4-dim feature vector used for data encoding.

    Each entry is clipped into [0, 1] so that `pi * f` stays in a sensible
    range for Ry encoding.
    """
    task = obs[:8]
    conf = float(obs[9])
    q1, q2 = float(obs[10]), float(obs[11])
    soc = float(obs[14])
    feat_norm = float(np.linalg.norm(task)) / np.sqrt(task.size)
    queue_total = (q1 + q2) / 10.0
    f = np.array([feat_norm, conf, queue_total, soc], dtype=np.float64)
    return np.clip(f, 0.0, 1.0)


class VQCPolicy:
    N_QUBITS = 2
    N_ACTIONS = 4
    N_LAYERS = 2
    N_PARAMS = N_LAYERS * N_QUBITS

    def __init__(self, seed: int = 0):
        if not _HAS_QISKIT:
            raise ImportError("qiskit required for VQCPolicy")
        rng = np.random.RandomState(seed)
        self.theta = rng.normal(0.0, 0.3, size=self.N_PARAMS)
        self._rng = np.random.RandomState(seed + 1)

    def _build_circuit(self, f: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.N_QUBITS)
        for layer in range(self.N_LAYERS):
            qc.ry(float(np.pi * f[2 * layer]), 0)
            qc.ry(float(np.pi * f[2 * layer + 1]), 1)
            qc.cx(0, 1)
            qc.ry(float(theta[2 * layer]), 0)
            qc.ry(float(theta[2 * layer + 1]), 1)
        return qc

    def _probs_from_theta(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        qc = self._build_circuit(f, theta)
        sv = Statevector.from_instruction(qc)
        probs = np.asarray(sv.probabilities(), dtype=np.float64)
        # For 2 qubits there are exactly 4 basis states => 4 actions.
        # Qiskit's `probabilities()` returns them in little-endian integer
        # order, matching format(i, "02b") with bit_{n-1}...bit_0.
        probs = probs / probs.sum()
        return probs

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return self._probs_from_theta(_compress_obs(obs), self.theta)

    def act(self, obs: np.ndarray, greedy: bool = False) -> Tuple[int, np.ndarray]:
        p = self.probs(obs)
        if greedy:
            return int(np.argmax(p)), p
        return int(self._rng.choice(self.N_ACTIONS, p=p)), p

    def _grad_log_pi(self, f: np.ndarray, action: int) -> np.ndarray:
        """Parameter-shift gradient of log P(action | obs) wrt self.theta."""
        shift = np.pi / 2
        grads = np.zeros_like(self.theta)
        base_p = self._probs_from_theta(f, self.theta)[action] + 1e-10
        for i in range(self.N_PARAMS):
            theta_plus = self.theta.copy()
            theta_plus[i] += shift
            theta_minus = self.theta.copy()
            theta_minus[i] -= shift
            p_plus = self._probs_from_theta(f, theta_plus)[action]
            p_minus = self._probs_from_theta(f, theta_minus)[action]
            grads[i] = 0.5 * (p_plus - p_minus) / base_p
        return grads


@dataclass
class _EpisodeStats:
    ret: float
    action_counts: np.ndarray


def train_reinforce(
    env,
    policy: VQCPolicy,
    episodes: int = 50,
    lr: float = 0.05,
    gamma: float = 0.95,
    log_every: int = 5,
) -> List[float]:
    """REINFORCE training loop for the VQC policy."""
    returns: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        traj_f: List[np.ndarray] = []
        traj_a: List[int] = []
        rewards: List[float] = []
        done = False
        while not done:
            a, _ = policy.act(obs, greedy=False)
            obs2, r, term, trunc, _ = env.step(a)
            traj_f.append(_compress_obs(obs))
            traj_a.append(a)
            rewards.append(r)
            obs = obs2
            done = term or trunc

        # discounted returns
        G = np.zeros(len(rewards), dtype=np.float64)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            G[t] = running
        ret_total = float(np.sum(rewards))
        returns.append(ret_total)

        # baseline / normalization for variance reduction
        if G.std() > 1e-6:
            G_norm = (G - G.mean()) / G.std()
        else:
            G_norm = G - G.mean()

        # REINFORCE gradient
        grad_accum = np.zeros_like(policy.theta)
        for t, (f, a) in enumerate(zip(traj_f, traj_a)):
            grad_accum += G_norm[t] * policy._grad_log_pi(f, a)
        policy.theta += lr * grad_accum / max(1, len(traj_f))

        if (ep + 1) % log_every == 0:
            window = returns[-log_every:]
            print(
                f"ep={ep + 1:3d} return={ret_total:8.2f} "
                f"return_mean{log_every}={np.mean(window):8.2f} "
                f"theta_norm={np.linalg.norm(policy.theta):.3f}"
            )

    return returns


def evaluate(env, policy: VQCPolicy, episodes: int = 20, greedy: bool = True) -> dict:
    returns: List[float] = []
    action_counts = np.zeros(policy.N_ACTIONS, dtype=np.int64)
    drops_critical = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            a, _ = policy.act(obs, greedy=greedy)
            action_counts[a] += 1
            obs, r, term, trunc, info = env.step(a)
            if a == env.ACTION_DROP and info["is_critical"]:
                drops_critical += 1
            total += r
            done = term or trunc
        returns.append(total)
    rr = np.array(returns)
    return {
        "mean": float(rr.mean()),
        "std": float(rr.std()),
        "action_counts": action_counts.tolist(),
        "drops_critical": int(drops_critical),
    }


if __name__ == "__main__":
    import argparse

    from rl_env import OffloadingEnv

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()

    env = OffloadingEnv(seed=args.seed)
    policy = VQCPolicy(seed=args.seed)

    print("Pre-training (stochastic) eval:")
    stats0 = evaluate(env, policy, episodes=args.eval_episodes, greedy=False)
    print(f"  mean return = {stats0['mean']:.2f} +/- {stats0['std']:.2f}")
    print(f"  action_counts = {stats0['action_counts']}  drops_critical = {stats0['drops_critical']}")

    print("\nTraining (REINFORCE + parameter-shift):")
    train_reinforce(env, policy, episodes=args.episodes, lr=args.lr)

    print("\nPost-training (greedy) eval:")
    stats1 = evaluate(env, policy, episodes=args.eval_episodes, greedy=True)
    print(f"  mean return = {stats1['mean']:.2f} +/- {stats1['std']:.2f}")
    print(f"  action_counts = {stats1['action_counts']}  drops_critical = {stats1['drops_critical']}")
    print(f"  final theta = {policy.theta}")
