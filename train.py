"""
train.py

Unified training entrypoint for DQN and QRL agents on VECEnvironment.

Usage:
    python train.py --agent dqn --episodes 500
    python train.py --agent qrl --episodes 200
    python train.py --agent dqn --episodes 500 --seed 42 --log-every 25
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from vec_env.environment import VECEnvironment, OBS_DIM


def _setup_logging(agent_name: str) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{agent_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rollout_stats(info_list: list) -> dict:
    """Aggregate step-level infos into episode-level stats."""
    latencies = [i["latency_ms"] for i in info_list if not i["dropped"]]
    energies = [i["energy_j"] for i in info_list if not i["dropped"]]
    deadline_hits = sum(1 for i in info_list if i["deadline_met"])
    drops = sum(1 for i in info_list if i["dropped"])
    return {
        "deadline_hit_rate": deadline_hits / max(len(info_list), 1),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "avg_energy_j": float(np.mean(energies)) if energies else 0.0,
        "drop_rate": drops / max(len(info_list), 1),
    }


def train_dqn(
    episodes: int = 500,
    seed: int = 0,
    log_every: int = 50,
    checkpoint_path: str = "checkpoints/dqn.pt",
) -> List[float]:
    """Train a DQN agent for *episodes* episodes and save the checkpoint.

    Returns episode return history for plotting.
    """
    from agents.dqn_agent import DQNAgent

    logger = _setup_logging("dqn")
    _set_seeds(seed)
    Path("checkpoints").mkdir(exist_ok=True)

    env = VECEnvironment(seed=seed)
    agent = DQNAgent(obs_dim=OBS_DIM, seed=seed)

    returns: List[float] = []
    logger.info("Training DQN for %d episodes (seed=%d)", episodes, seed)

    for ep in tqdm(range(1, episodes + 1), desc="DQN", unit="ep"):
        obs, _ = env.reset()
        ep_return = 0.0
        infos: list = []
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            infos.append(info)

        returns.append(ep_return)

        if ep % log_every == 0:
            window = returns[-log_every:]
            stats = _rollout_stats(infos)
            logger.info(
                "ep=%d  eps=%.3f  return_mean=%.2f  "
                "deadline_hit=%.1f%%  drop=%.1f%%  lat=%.1fms",
                ep, agent.eps, np.mean(window),
                stats["deadline_hit_rate"] * 100,
                stats["drop_rate"] * 100,
                stats["avg_latency_ms"],
            )

    agent.save(checkpoint_path)
    logger.info("DQN checkpoint saved -> %s", checkpoint_path)
    np.save("results/dqn_returns.npy", np.array(returns))
    return returns


def train_qrl(
    episodes: int = 200,
    seed: int = 0,
    log_every: int = 20,
    checkpoint_path: str = "checkpoints/qrl.npz",
) -> List[float]:
    """Train a QRL agent for *episodes* episodes and save the checkpoint.

    Returns episode return history for plotting.

    Note: QRL is significantly slower than DQN on CPU because the VQC
    forward pass is simulated classically. 200 episodes is sufficient to
    demonstrate learning; the circuit runs in O(2^n_qubits) time per sample.
    """
    from agents.qrl_agent import QRLAgent

    logger = _setup_logging("qrl")
    _set_seeds(seed)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env = VECEnvironment(seed=seed)
    # batch_size=16 and warmup=100 so first update happens quickly;
    # larger batches slow down the VQC forward pass substantially on CPU.
    agent = QRLAgent(obs_dim=OBS_DIM, seed=seed, batch_size=16, warmup_steps=100)

    returns: List[float] = []
    logger.info("Training QRL for %d episodes (seed=%d)", episodes, seed)

    for ep in tqdm(range(1, episodes + 1), desc="QRL", unit="ep"):
        obs, _ = env.reset()
        ep_return = 0.0
        infos: list = []
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            infos.append(info)

        returns.append(ep_return)

        if ep % log_every == 0:
            window = returns[-log_every:]
            stats = _rollout_stats(infos)
            logger.info(
                "ep=%d  eps=%.3f  return_mean=%.2f  "
                "deadline_hit=%.1f%%  drop=%.1f%%",
                ep, agent.eps, np.mean(window),
                stats["deadline_hit_rate"] * 100,
                stats["drop_rate"] * 100,
            )

    agent.save(checkpoint_path)
    logger.info("QRL checkpoint saved -> %s", checkpoint_path)
    np.save("results/qrl_returns.npy", np.array(returns))
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Train offloading agents on VECEnvironment")
    parser.add_argument("--agent", choices=["dqn", "qrl"], required=True)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (default: 500 for DQN, 200 for QRL)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    if args.agent == "dqn":
        episodes = args.episodes or 500
        train_dqn(episodes=episodes, seed=args.seed, log_every=args.log_every)
    else:
        episodes = args.episodes or 200
        train_qrl(episodes=episodes, seed=args.seed, log_every=args.log_every)


if __name__ == "__main__":
    main()
