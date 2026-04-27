"""
main.py

CLI for:
- Training / using the classical classifier (baseline pipeline)
- Sampling an offloading action via the quantum / classical-fallback selector
- Training / evaluating the DQN offloading policy (rl subcommand)

Usage:
    python main.py --train
    python main.py --predict
    python main.py --predict --force-quantum
    python main.py --predict --features "0.1,0.2,-0.3,0.4,0.5,0.6,0.7,0.8"
    python main.py rl train --total-steps 20000
    python main.py rl eval  --load dqn_offload.pt
    python main.py rl baseline
"""

import argparse
import logging
import sys

import numpy as np

from classical_model import train, load_model, predict, compute_action_costs
from quantum_decision import select_action_from_costs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_features(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts])


def run_predict(args: argparse.Namespace) -> None:
    if args.features:
        features = parse_features(args.features)
    else:
        rng = np.random.RandomState(0)
        features = rng.normal(size=8)

    try:
        model = load_model()
    except FileNotFoundError:
        logger.error("No trained model found. Train first with --train")
        return

    pred, prob = predict(features, model=model)
    logger.info("Classical model predicted: %s with confidence %.4f", pred, prob)

    costs = compute_action_costs(features, model=model)
    print("Costs:")
    print(f"LOCAL = {costs['local']:.3f}")
    print(f"EDGE_1 = {costs['edge_1']:.3f}")
    print(f"EDGE_2 = {costs['edge_2']:.3f}")
    print(f"DROP = {costs['drop']:.3f}\n")

    measured, action, _details = select_action_from_costs(
        costs, shots=1, seed=0, force_quantum=args.force_quantum
    )
    print("Quantum measurement:", measured)
    print("Selected action ->", action)


def run_rl(rl_args: list) -> None:
    """Forward the `rl` subcommand to train_rl's parser."""
    from train_rl import build_parser as rl_parser, train as rl_train, evaluate as rl_eval, baseline_random

    args = rl_parser().parse_args(rl_args)
    if args.cmd == "train":
        rl_train(
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            warmup=args.warmup,
            seed=args.seed,
            save_path=args.save,
        )
    elif args.cmd == "eval":
        rl_eval(load_path=args.load, episodes=args.episodes, seed=args.seed)
    elif args.cmd == "baseline":
        baseline_random(episodes=args.episodes, seed=args.seed)
    elif args.cmd == "vqc":
        from vqc_policy import VQCPolicy, train_reinforce, evaluate as vqc_eval
        from rl_env import OffloadingEnv
        env = OffloadingEnv(seed=args.seed)
        policy = VQCPolicy(seed=args.seed)
        pre = vqc_eval(env, policy, episodes=args.eval_episodes, greedy=False)
        print(f"VQC pre-train (stochastic): mean={pre['mean']:.2f}  actions={pre['action_counts']}")
        train_reinforce(env, policy, episodes=args.episodes, lr=args.lr)
        post = vqc_eval(env, policy, episodes=args.eval_episodes, greedy=True)
        print(f"VQC post-train (greedy):    mean={post['mean']:.2f}  actions={post['action_counts']}  drops_critical={post['drops_critical']}")


def main() -> None:
    # Intercept the `rl` subcommand before the top-level parser so we can
    # forward the remaining args to train_rl's parser unchanged.
    argv = sys.argv[1:]
    if argv and argv[0] == "rl":
        run_rl(argv[1:])
        return

    parser = argparse.ArgumentParser(description="Quantum edge offloading decision CLI")
    parser.add_argument("--train", action="store_true", help="Train and save the classical model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction + offloading decision")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature vector (defaults to a random sample)")
    parser.add_argument("--force-quantum", action="store_true",
                        help="If Qiskit is installed, use the quantum sampler for the decision")
    parser.add_argument("--quantum-threshold", type=float, default=0.6)
    parser.add_argument("--heuristic-threshold", type=float, default=0.7)
    args = parser.parse_args()

    if not (args.train or args.predict):
        parser.print_help()
        print("\nRL subcommand:")
        print("  python main.py rl train|eval|baseline [options]")
        return

    if args.train:
        acc, path = train()
        logger.info("Trained classical model; validation accuracy: %.4f; saved to %s", acc, path)

    if args.predict:
        run_predict(args)


if __name__ == "__main__":
    main()
