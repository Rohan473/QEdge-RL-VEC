"""
train_rl.py

CLI to train / evaluate the DQN agent on OffloadingEnv.

Examples:
    python train_rl.py train --total-steps 20000 --save dqn_offload.pt
    python train_rl.py eval --load dqn_offload.pt --episodes 20
    python train_rl.py baseline --episodes 20          # random policy
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from rl_env import OffloadingEnv
from rl_agent import DQNAgent, ReplayBuffer, Transition


def train(
    total_steps: int = 20_000,
    batch_size: int = 64,
    warmup: int = 500,
    buffer_capacity: int = 20_000,
    seed: int = 0,
    save_path: str = "dqn_offload.pt",
    log_every: int = 10,
) -> List[float]:
    env = OffloadingEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(obs_dim=obs.shape[0], n_actions=env.action_space.n, seed=seed)
    buf = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    episode_returns: List[float] = []
    ep_ret = 0.0

    for step in range(total_steps):
        a = agent.act(obs, greedy=False)
        obs2, r, term, trunc, _ = env.step(a)
        buf.push(Transition(obs, a, r, obs2, term or trunc))
        obs = obs2
        ep_ret += r

        if len(buf) >= warmup:
            agent.learn(buf.sample(batch_size))

        if term or trunc:
            episode_returns.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0.0
            agent.decay_epsilon()
            if len(episode_returns) % log_every == 0:
                window = episode_returns[-log_every:]
                print(
                    f"step={step + 1:6d} ep={len(episode_returns):4d} "
                    f"eps={agent.eps:.3f} return_mean{log_every}={np.mean(window):8.2f}"
                )

    agent.save(save_path)
    print(f"saved -> {save_path}")
    return episode_returns


def _rollout(env: OffloadingEnv, choose_action, n_episodes: int) -> Dict:
    returns: List[float] = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)
    drops_critical = 0
    drops_noncritical = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            a = choose_action(obs)
            action_counts[a] += 1
            obs, r, term, trunc, info = env.step(a)
            if a == env.ACTION_DROP:
                if info["is_critical"]:
                    drops_critical += 1
                else:
                    drops_noncritical += 1
            total += r
            done = term or trunc
        returns.append(total)
    returns_arr = np.array(returns)
    return {
        "mean": float(returns_arr.mean()),
        "std": float(returns_arr.std()),
        "action_counts": action_counts.tolist(),
        "drops_critical": int(drops_critical),
        "drops_noncritical": int(drops_noncritical),
    }


def evaluate(load_path: str, episodes: int = 20, seed: int = 42) -> Dict:
    env = OffloadingEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(obs_dim=obs.shape[0], n_actions=env.action_space.n, seed=seed)
    agent.load(load_path)
    agent.eps = 0.0
    stats = _rollout(env, lambda o: agent.act(o, greedy=True), episodes)
    _print_stats(f"DQN ({load_path})", stats, env)
    return stats


def baseline_random(episodes: int = 20, seed: int = 42) -> Dict:
    env = OffloadingEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.RandomState(seed)
    stats = _rollout(env, lambda o: int(rng.randint(env.action_space.n)), episodes)
    _print_stats("Random baseline", stats, env)
    return stats


def _print_stats(label: str, stats: Dict, env: OffloadingEnv) -> None:
    print(
        f"{label}: mean return = {stats['mean']:.2f} +/- {stats['std']:.2f} "
        f"over {len(stats['action_counts'])} actions"
    )
    action_hist = dict(zip(env.ACTION_NAMES, stats["action_counts"]))
    print(f"  action counts: {action_hist}")
    print(
        f"  drops: critical={stats['drops_critical']}  "
        f"non-critical={stats['drops_noncritical']}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DQN on OffloadingEnv")
    sub = p.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train DQN and save weights")
    tp.add_argument("--total-steps", type=int, default=20_000)
    tp.add_argument("--batch-size", type=int, default=64)
    tp.add_argument("--warmup", type=int, default=500)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--save", default="dqn_offload.pt")

    ep = sub.add_parser("eval", help="evaluate a trained DQN policy")
    ep.add_argument("--load", default="dqn_offload.pt")
    ep.add_argument("--episodes", type=int, default=20)
    ep.add_argument("--seed", type=int, default=42)

    bp = sub.add_parser("baseline", help="evaluate uniform-random policy")
    bp.add_argument("--episodes", type=int, default=20)
    bp.add_argument("--seed", type=int, default=42)

    vp = sub.add_parser("vqc", help="train + eval VQC policy via REINFORCE (requires qiskit)")
    vp.add_argument("--episodes", type=int, default=40)
    vp.add_argument("--lr", type=float, default=0.2)
    vp.add_argument("--eval-episodes", type=int, default=10)
    vp.add_argument("--seed", type=int, default=0)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            warmup=args.warmup,
            seed=args.seed,
            save_path=args.save,
        )
    elif args.cmd == "eval":
        evaluate(load_path=args.load, episodes=args.episodes, seed=args.seed)
    elif args.cmd == "baseline":
        baseline_random(episodes=args.episodes, seed=args.seed)
    elif args.cmd == "vqc":
        from vqc_policy import VQCPolicy, train_reinforce, evaluate as vqc_eval
        env = OffloadingEnv(seed=args.seed)
        policy = VQCPolicy(seed=args.seed)
        pre = vqc_eval(env, policy, episodes=args.eval_episodes, greedy=False)
        print(f"VQC pre-train (stochastic): mean = {pre['mean']:.2f} +/- {pre['std']:.2f}  actions = {pre['action_counts']}")
        train_reinforce(env, policy, episodes=args.episodes, lr=args.lr)
        post = vqc_eval(env, policy, episodes=args.eval_episodes, greedy=True)
        print(f"VQC post-train (greedy):    mean = {post['mean']:.2f} +/- {post['std']:.2f}  actions = {post['action_counts']}  drops_critical = {post['drops_critical']}")


if __name__ == "__main__":
    main()
