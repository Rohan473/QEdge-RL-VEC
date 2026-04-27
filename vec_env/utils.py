"""
vec_env/utils.py

Physical helper functions for the VEC simulation:
channel capacity, latency, and energy models.
"""

from __future__ import annotations

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
BANDWIDTH_HZ: float = 10e6          # B  = 10 MHz
TX_POWER_W: float = 0.5             # P  = 0.5 W
NOISE_POWER_W: float = 1e-9         # N0 = 1 nW  (thermal noise floor)
PATH_LOSS_EXPONENT: float = 3.0     # α  = 3 (urban/highway)

F_EDGE_HZ: float = 5e9              # edge server CPU: 5 GHz
F_LOCAL_HZ: float = 1e9             # vehicle local CPU: 1 GHz
KAPPA: float = 1e-28                # effective switched capacitance (F·s²)

# Normalisation reference: max channel rate (at 1 m distance, theoretical)
_R_REF: float = BANDWIDTH_HZ * np.log2(
    1.0 + TX_POWER_W / (NOISE_POWER_W)
)  # ≈ 290 Mbps — used only for observation normalisation


def channel_rate(distance_m: float) -> float:
    """Shannon capacity between a vehicle and an edge server.

    Args:
        distance_m: Euclidean distance in metres (clamped to ≥ 1 m).

    Returns:
        Transmission rate in bps.
    """
    d = max(1.0, distance_m)
    h = d ** (-PATH_LOSS_EXPONENT)          # path-loss gain
    snr = TX_POWER_W * h / NOISE_POWER_W
    return BANDWIDTH_HZ * np.log2(1.0 + snr)


def transmission_latency_ms(data_size_kb: float, rate_bps: float) -> float:
    """Time to transmit *data_size_kb* kilobytes over a link at *rate_bps* bps.

    Returns:
        Latency in milliseconds.
    """
    bits = data_size_kb * 1024 * 8
    return (bits / max(rate_bps, 1.0)) * 1000.0


def compute_latency_ms(
    action: int,
    data_size_kb: float,
    cpu_cycles_mc: float,
    rate_edge1_bps: float,
    rate_edge2_bps: float,
    queue_edge1: float,
    queue_edge2: float,
) -> float:
    """End-to-end latency in ms for a given offloading action.

    Queue wait is modelled as: queue_len × avg_task_service_time_ms.
    Average task CPU = 550 Mc, so avg service time = 550e6 / 5e9 = 0.11 ms.

    Args:
        action: 0=LOCAL, 1=EDGE_1, 2=EDGE_2, 3=DROP
        data_size_kb: task data size (KB)
        cpu_cycles_mc: task CPU requirement (Megacycles)
        rate_edge1_bps: channel rate to edge server 1 (bps)
        rate_edge2_bps: channel rate to edge server 2 (bps)
        queue_edge1: number of tasks queued at edge 1
        queue_edge2: number of tasks queued at edge 2

    Returns:
        Latency in milliseconds (0 if DROP).
    """
    avg_service_ms = 550e6 / F_EDGE_HZ * 1000.0   # ≈ 0.11 ms

    if action == 0:   # LOCAL
        return (cpu_cycles_mc * 1e6 / F_LOCAL_HZ) * 1000.0
    elif action == 1:  # EDGE_1
        t_tx = transmission_latency_ms(data_size_kb, rate_edge1_bps)
        t_queue = queue_edge1 * avg_service_ms
        t_comp = (cpu_cycles_mc * 1e6 / F_EDGE_HZ) * 1000.0
        return t_tx + t_queue + t_comp
    elif action == 2:  # EDGE_2
        t_tx = transmission_latency_ms(data_size_kb, rate_edge2_bps)
        t_queue = queue_edge2 * avg_service_ms
        t_comp = (cpu_cycles_mc * 1e6 / F_EDGE_HZ) * 1000.0
        return t_tx + t_queue + t_comp
    else:              # DROP
        return 0.0


def compute_energy_j(
    action: int,
    data_size_kb: float,
    cpu_cycles_mc: float,
    rate_bps: float,
) -> float:
    """Energy consumed by the vehicle for a given offloading action.

    Local: cubic power model  E = κ · f² · cycles
    Offload: transmit energy  E = P · t_tx

    Args:
        action: 0=LOCAL, 1=EDGE_1, 2=EDGE_2, 3=DROP
        data_size_kb: task data size (KB)
        cpu_cycles_mc: task CPU (Megacycles)
        rate_bps: channel rate to the selected edge server (bps); ignored for LOCAL/DROP

    Returns:
        Energy in Joules.
    """
    if action == 0:   # LOCAL
        cycles = cpu_cycles_mc * 1e6
        return KAPPA * (F_LOCAL_HZ ** 2) * cycles
    elif action in (1, 2):  # EDGE offload
        t_tx_s = (data_size_kb * 1024 * 8) / max(rate_bps, 1.0)
        return TX_POWER_W * t_tx_s
    else:              # DROP
        return 0.0


def normalise_rate(rate_bps: float) -> float:
    """Map channel rate to [0, 1] using the reference maximum rate."""
    return float(np.clip(rate_bps / _R_REF, 0.0, 1.0))
