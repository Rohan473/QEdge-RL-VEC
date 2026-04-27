"""
evaluate.py

Evaluate all four agents (Random, Greedy, DQN, QRL) on 100 evaluation episodes
with fixed seeds and save results/metrics.csv.

Usage:
    python evaluate.py
    python evaluate.py --episodes 50 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from vec_env.environment import VECEnvironment, OBS_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _evaluate_agent(
    agent,
    n_episodes: int,
    seed: int,
    agent_name: str,
) -> Dict[str, float]:
    """Run *n_episodes* evaluation episodes and return aggregated metrics.

    Args:
        agent: any object with an `act(obs) -> int` method.
        n_episodes: number of evaluation episodes.
        seed: environment seed (each episode gets seed+episode_idx).
        agent_name: human-readable label for logging.

    Returns:
        Dict with keys: avg_return, deadline_hit_rate, avg_latency_ms,
        avg_energy_j, drop_rate.
    """
    env = VECEnvironment(seed=seed)
    all_returns: List[float] = []
    all_latencies: List[float] = []
    all_energies: List[float] = []
    deadline_hits = 0
    drops = 0
    total_tasks = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_return += reward
            total_tasks += 1
            if info["deadline_met"]:
                deadline_hits += 1
                all_latencies.append(info["latency_ms"])
                all_energies.append(info["energy_j"])
            if info["dropped"]:
                drops += 1

        all_returns.append(ep_return)

    metrics = {
        "agent": agent_name,
        "avg_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "deadline_hit_rate_pct": 100.0 * deadline_hits / max(total_tasks, 1),
        "avg_latency_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "avg_energy_j": float(np.mean(all_energies)) if all_energies else 0.0,
        "drop_rate_pct": 100.0 * drops / max(total_tasks, 1),
    }

    logger.info(
        "%s | return=%.2f±%.2f  deadline_hit=%.1f%%  latency=%.1fms  drop=%.1f%%",
        agent_name,
        metrics["avg_return"],
        metrics["std_return"],
        metrics["deadline_hit_rate_pct"],
        metrics["avg_latency_ms"],
        metrics["drop_rate_pct"],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all agents on VECEnvironment")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--dqn-ckpt", default="checkpoints/dqn.pt")
    parser.add_argument("--qrl-ckpt", default="checkpoints/qrl.npz")
    args = parser.parse_args()

    _set_seeds(args.seed)
    Path("results").mkdir(exist_ok=True)

    results = []

    # ── Random ────────────────────────────────────────────────────────────────
    from agents.random_agent import RandomAgent
    results.append(_evaluate_agent(RandomAgent(seed=args.seed), args.episodes, args.seed, "Random"))

    # ── Greedy ────────────────────────────────────────────────────────────────
    from agents.greedy_agent import GreedyAgent
    results.append(_evaluate_agent(GreedyAgent(), args.episodes, args.seed, "Greedy"))

    # ── DQN ───────────────────────────────────────────────────────────────────
    if Path(args.dqn_ckpt).exists():
        from agents.dqn_agent import DQNAgent
        dqn = DQNAgent(obs_dim=OBS_DIM, seed=args.seed)
        dqn.load(args.dqn_ckpt)
        results.append(_evaluate_agent(dqn, args.episodes, args.seed, "DQN"))
    else:
        logger.warning("DQN checkpoint not found at %s — skipping", args.dqn_ckpt)

    # ── QRL ───────────────────────────────────────────────────────────────────
    if Path(args.qrl_ckpt).exists():
        try:
            from agents.qrl_agent import QRLAgent
            qrl = QRLAgent(obs_dim=OBS_DIM, seed=args.seed)
            qrl.load(args.qrl_ckpt)
            results.append(_evaluate_agent(qrl, args.episodes, args.seed, "QRL"))
        except ImportError as e:
            logger.warning("QRL skipped: %s", e)
    else:
        logger.warning("QRL checkpoint not found at %s — skipping", args.qrl_ckpt)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = "results/metrics.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Metrics saved → %s", csv_path)


if __name__ == "__main__":
    main()
