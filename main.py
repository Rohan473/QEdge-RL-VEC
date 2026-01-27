"""
main.py

Small CLI to:
- Train the classical model
- Run a prediction and decide whether to offload

Usage examples:
- python main.py --train
- python main.py --predict
- python main.py --predict --force-quantum
- python main.py --predict --features "0.1,0.2,-0.3,0.4,0.5,0.6,0.7,0.8"
"""

import argparse
import logging
import numpy as np
from classical_model import train, load_model, predict, compute_action_costs
from quantum_decision import select_action_from_costs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_features(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts])


def main():
    parser = argparse.ArgumentParser(description="Quantum edge offloading decision CLI")
    parser.add_argument("--train", action="store_true", help="Train and save the classical model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction + offloading decision")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature vector to predict on (defaults to random sample)")
    parser.add_argument("--force-quantum", action="store_true",
                        help="If Qiskit is installed, force using the quantum decision routine")
    parser.add_argument("--quantum-threshold", type=float, default=0.6, help="Threshold for quantum score")
    parser.add_argument("--heuristic-threshold", type=float, default=0.7, help="Threshold for heuristic confidence")
    args = parser.parse_args()

    # If no action provided, show help and exit (avoid silent run)
    if not (args.train or args.predict):
        parser.print_help()
        return

    if args.train:
        acc, path = train()
        logger.info("Trained classical model; validation accuracy: %.4f; saved to %s", acc, path)

    if args.predict:
        # Prepare features
        if args.features:
            features = parse_features(args.features)
        else:
            # random sample features consistent with classical_model training n_features=8
            rng = np.random.RandomState(0)
            features = rng.normal(size=8)

        # Load model and predict
        try:
            model = load_model()
        except FileNotFoundError:
            logger.error("No trained model found. Train first with --train")
            return

        pred, prob = predict(features, model=model)
        logger.info("Classical model predicted: %s with confidence %.4f", pred, prob)

        # Compute costs for all four actions
        costs = compute_action_costs(features, model=model)
        print("Costs:")
        print(f"LOCAL = {costs['local']:.3f}")
        print(f"EDGE_1 = {costs['edge_1']:.3f}")
        print(f"EDGE_2 = {costs['edge_2']:.3f}")
        print(f"DROP = {costs['drop']:.3f}\n")

        # Run the 2-qubit quantum selection (or fallback classical sampling)
        measured, action, details = select_action_from_costs(costs, shots=1, seed=0, force_quantum=args.force_quantum)
        print("Quantum measurement:", measured)
        print("Selected action ->", action)


if __name__ == "__main__":
    main()
