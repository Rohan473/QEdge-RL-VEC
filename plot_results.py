"""
plot_results.py

Generate evaluation plots from training curves and results/metrics.csv.

Outputs (saved to results/plots/):
    1. training_curves.png     — smoothed episode return: DQN vs QRL
    2. deadline_hit_rate.png   — bar chart across all 4 agents
    3. avg_latency.png         — bar chart across all 4 agents
    4. convergence.png         — smoothed reward (window=20) DQN vs QRL

Usage:
    python plot_results.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLOT_DIR = Path("results/plots")
COLORS = {
    "DQN": "#2196F3",
    "QRL": "#E91E63",
    "Random": "#9E9E9E",
    "Greedy": "#FF9800",
}


def _smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Exponential-like moving average using uniform convolution."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    pad = np.full(window - 1, values[0])
    padded = np.concatenate([pad, values])
    return np.convolve(padded, kernel, mode="valid")


def plot_training_curves() -> None:
    """Plot raw + smoothed training returns for DQN and QRL."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves: Episode Return", fontsize=14)

    for ax, (name, path, color) in zip(
        axes,
        [("DQN", "results/dqn_returns.npy", COLORS["DQN"]),
         ("QRL", "results/qrl_returns.npy", COLORS["QRL"])],
    ):
        if not Path(path).exists():
            ax.set_title(f"{name} (no data)")
            ax.text(0.5, 0.5, "Run train.py first", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue

        returns = np.load(path)
        episodes = np.arange(1, len(returns) + 1)
        smoothed = _smooth(returns, window=20)

        ax.plot(episodes, returns, alpha=0.25, color=color, lw=0.8, label="Raw")
        ax.plot(episodes[:len(smoothed)], smoothed, color=color, lw=2.0,
                label="Smoothed (w=20)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"{name} Training")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def _load_metrics() -> list[dict]:
    path = Path("results/metrics.csv")
    if not path.exists():
        logger.warning("metrics.csv not found; run evaluate.py first")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_bar(
    metrics: list[dict],
    key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if not metrics:
        logger.warning("No metrics to plot for %s", filename)
        return

    agents = [m["agent"] for m in metrics]
    values = [float(m[key]) for m in metrics]
    colors = [COLORS.get(a, "#607D8B") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def plot_convergence() -> None:
    """Overlay smoothed returns for DQN and QRL (normalised to same x-axis length)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Convergence Comparison: DQN vs QRL (smoothed return, window=20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Episode Return")

    any_data = False
    for name, path, color in [
        ("DQN", "results/dqn_returns.npy", COLORS["DQN"]),
        ("QRL", "results/qrl_returns.npy", COLORS["QRL"]),
    ]:
        if not Path(path).exists():
            continue
        returns = np.load(path)
        smoothed = _smooth(returns, window=20)
        episodes = np.arange(1, len(smoothed) + 1)
        ax.plot(episodes, smoothed, color=color, lw=2.0, label=name)
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "Run train.py for DQN and QRL first",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = PLOT_DIR / "convergence.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_curves()

    metrics = _load_metrics()
    plot_bar(
        metrics,
        key="deadline_hit_rate_pct",
        ylabel="Deadline Hit Rate (%)",
        title="Deadline Hit Rate by Agent",
        filename="deadline_hit_rate.png",
    )
    plot_bar(
        metrics,
        key="avg_latency_ms",
        ylabel="Average Latency (ms)",
        title="Average Task Latency by Agent (deadline-met tasks only)",
        filename="avg_latency.png",
    )
    plot_convergence()

    logger.info("All plots saved to %s/", PLOT_DIR)


if __name__ == "__main__":
    main()
