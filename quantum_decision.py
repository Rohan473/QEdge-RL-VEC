"""
quantum_decision.py

Decision logic for whether to offload to a quantum resource, plus a generalized
n-action quantum selector.

Behavior:
- If Qiskit is available and --force-quantum is used, run an n-qubit amplitude-
  encoded circuit that samples exactly from the target distribution over actions.
- Otherwise, fall back to classical probabilistic sampling with the same
  distribution.

The quantum path uses amplitude encoding: n_qubits = ceil(log2(N)) where N is
the number of actions; unused computational-basis states (when N < 2^n_qubits)
get zero amplitude.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import logging

# Optional import of qiskit. Supports Qiskit >= 1.0 (qiskit_aer.AerSimulator)
# with a fallback to the legacy `qiskit.Aer` API for older installs.
try:
    from qiskit import QuantumCircuit, transpile
    try:
        from qiskit_aer import AerSimulator
        _AER_BACKEND = AerSimulator()
    except Exception:
        from qiskit import Aer  # legacy <1.0
        _AER_BACKEND = Aer.get_backend("qasm_simulator")
    HAS_QISKIT = True
except Exception:
    _AER_BACKEND = None
    HAS_QISKIT = False

logger = logging.getLogger(__name__)

# Human-readable labels for the default 4-action set. Any key not in this map
# is returned upper-cased.
_DEFAULT_LABELS: Dict[str, str] = {
    "local": "LOCAL",
    "edge_1": "EDGE_SERVER_1",
    "edge_2": "EDGE_SERVER_2",
    "drop": "DROP_TASK",
}


def _label_for(key: str, action_labels: Optional[Dict[str, str]] = None) -> str:
    if action_labels and key in action_labels:
        return action_labels[key]
    return _DEFAULT_LABELS.get(key, key.upper())


def heuristic_should_offload(features: np.ndarray, model_confidence: float, threshold: float = 0.7) -> bool:
    """Low confidence + large feature norm -> offload."""
    features = np.asarray(features).ravel()
    norm = float(np.linalg.norm(features))
    logger.debug("Heuristic decision: norm=%f, model_confidence=%f", norm, model_confidence)
    return (model_confidence < threshold) and (norm > (features.size * 0.5))


def _costs_to_distribution(costs: Dict[str, float]) -> Tuple[list, np.ndarray]:
    """Convert a costs dict to (ordered keys, probability vector).

    Higher preference is assigned to lower cost via inverse weighting, then
    normalized.
    """
    if len(costs) < 2:
        raise ValueError("costs must define at least 2 actions")
    keys = list(costs.keys())
    prefs = np.array([1.0 / max(1e-9, float(costs[k])) for k in keys], dtype=float)
    probs = prefs / prefs.sum()
    return keys, probs


def select_action_from_costs(
    costs: Dict[str, float],
    shots: int = 1,
    seed: int = 42,
    force_quantum: bool = False,
    action_labels: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, dict]:
    """Sample an action from a cost-derived distribution via amplitude encoding.

    Works for any N >= 2 actions. Uses n_qubits = ceil(log2(N)); when N is not
    a power of 2 the remaining basis states are zero-padded so the sampler
    never selects them.

    Returns:
        (measured_bitstring, action_label, details)
    """
    action_keys, p_action_vec = _costs_to_distribution(costs)
    N = len(action_keys)
    n_qubits = max(1, int(np.ceil(np.log2(N)))) if N > 1 else 1
    slots = 2 ** n_qubits

    p_padded = np.zeros(slots, dtype=float)
    p_padded[:N] = p_action_vec
    # Renormalize defensively; p_padded already sums to 1 when N==slots.
    p_padded = p_padded / p_padded.sum()
    amplitudes = np.sqrt(p_padded)

    p_action = {k: float(p) for k, p in zip(action_keys, p_action_vec)}
    # bitstring (MSB-left, width n_qubits) -> action key
    action_bit_map = {format(i, f"0{n_qubits}b"): action_keys[i] for i in range(N)}

    details: dict = {
        "p_action": p_action,
        "n_qubits": n_qubits,
        "n_actions": N,
        "amplitudes": [float(a) for a in amplitudes],
    }

    if HAS_QISKIT and force_quantum:
        try:
            backend = _AER_BACKEND
            qc = QuantumCircuit(n_qubits, n_qubits)
            # Amplitude-encode the target distribution. Qubit 0 is the LSB of
            # the basis index, qubit n_qubits-1 is the MSB.
            qc.initialize(amplitudes.tolist(), list(range(n_qubits)))
            qc.measure(range(n_qubits), range(n_qubits))
            qc_t = transpile(qc, backend=backend)
            result = backend.run(qc_t, shots=shots, seed_simulator=seed).result()
            counts = result.get_counts(qc_t)
            # Qiskit reports bitstrings as "c_{n-1} ... c_1 c_0" (MSB-left),
            # matching our format(i, f"0{n}b") convention.
            measured = max(counts.items(), key=lambda kv: kv[1])[0].zfill(n_qubits)
            key = action_bit_map.get(measured)
            if key is None:
                # Should only happen if amplitudes leaked probability into a
                # padded slot, which shouldn't occur with noiseless sim.
                raise RuntimeError(f"Measured unmapped basis state {measured}")
            details["counts"] = counts
            return measured, _label_for(key, action_labels), details
        except Exception as e:
            logger.warning("Quantum circuit failed (%s), falling back to classical sampling.", e)

    # Classical fallback: sample directly from p_action_vec.
    rng = np.random.RandomState(seed)
    idx = int(rng.choice(N, p=p_action_vec))
    key = action_keys[idx]
    measured = format(idx, f"0{n_qubits}b")
    details["sampled_choice"] = key
    return measured, _label_for(key, action_labels), details


def should_offload(features: np.ndarray, model_confidence: float, force_quantum: bool = False,
                   quantum_threshold: float = 0.6, heuristic_threshold: float = 0.7) -> Tuple[bool, dict]:
    """Decide whether to offload. Currently always uses the heuristic path."""
    features = np.asarray(features).ravel()
    decision = heuristic_should_offload(features, model_confidence, threshold=heuristic_threshold)
    return decision, {"method": "heuristic", "model_confidence": float(model_confidence)}


if __name__ == "__main__":
    sample = [0.1, 0.2, -0.5, 1.2, 0.0, 0.3, 0.6, -0.1]
    print("Has Qiskit:", HAS_QISKIT)
    print("Heuristic offload:", should_offload(sample, model_confidence=0.65))
    from classical_model import compute_action_costs
    costs = compute_action_costs(np.array(sample))
    measured, action, details = select_action_from_costs(costs, shots=1, seed=42, force_quantum=False)
    print("Costs:", costs)
    print("n_qubits:", details["n_qubits"], "n_actions:", details["n_actions"])
    print("Measured:", measured)
    print("Action:", action)

    # Smoke test with 6 actions (non-power-of-two) — exercises zero-padding.
    costs6 = {f"a{i}": float(i + 1) for i in range(6)}
    m, a, d = select_action_from_costs(costs6, shots=1, seed=7, force_quantum=False)
    print("\n6-action sample:", m, a, "n_qubits=", d["n_qubits"])
