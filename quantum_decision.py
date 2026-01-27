"""
quantum_decision.py

Decision logic for whether to offload to a quantum resource.

Behavior:
- If Qiskit is available and --force-quantum is used, run a lightweight simulated quantum routine that
  returns a "quantum score" and uses a threshold to decide offloading.
- Otherwise, use a deterministic heuristic based on input feature magnitude and model confidence.

The file is intentionally lightweight: the quantum routine is a minimal example using statevector simulator
(if Qiskit is installed). The heuristics are tunable and intended as placeholders.
"""

from typing import Tuple
import numpy as np
import logging

# Optional import of qiskit — the quantum path is used only if Qiskit is installed.
try:
    from qiskit import QuantumCircuit, Aer, transpile
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False

logger = logging.getLogger(__name__)


def heuristic_should_offload(features: np.ndarray, model_confidence: float, threshold: float = 0.7) -> bool:
    """
    Simple heuristic:
    - If model confidence is below threshold, and the input 'complexity' is high (L2 norm),
      decide to offload.
    """
    features = np.asarray(features).ravel()
    norm = float(np.linalg.norm(features))
    logger.debug("Heuristic decision: norm=%f, model_confidence=%f", norm, model_confidence)
    # Tunable rule: low confidence and large norm -> offload
    return (model_confidence < threshold) and (norm > (features.size * 0.5))


def _sample_by_prob_dict(pdict: dict, rng: np.random.RandomState = None) -> str:
    """Sample a single key from a probability dict (values sum to 1)."""
    keys = list(pdict.keys())
    probs = np.array([pdict[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    if rng is None:
        rng = np.random.RandomState()
    choice = rng.choice(len(keys), p=probs)
    return keys[choice]


def select_action_from_costs(costs: dict, shots: int = 1, seed: int = 42, force_quantum: bool = False) -> Tuple[str, str, dict]:
    """
    Use a 2-qubit quantum circuit to probabilistically se zlect one of four actions based on costs.

    Arguments:                          
        costs: dict with keys 'local','edge_1','edge_2','drop' and positive numeric costs.
        shots: number of measurement shots (default 1). We return the most frequent outcome if >1.
        seed: RNG seed for simulator or fallback.
        force_quantum: if False and Qiskit not available, fallback to classical sampling.

    Returns: (measured_bitstring, decoded_action, details)
    details includes intermediate probabilities and used angles.
    """
    # convert costs to preferences (higher preference for lower cost)
    prefs = {k: 1.0 / max(1e-9, float(v)) for k, v in costs.items()}
    total = sum(prefs.values())
    p_action = {k: float(v / total) for k, v in prefs.items()}

    # Marginals for each qubit (bit order is b1 b0, left to right)
    # mapping: 00->local, 01->edge_1, 10->edge_2, 11->drop
    p_qubit0_1 = p_action["edge_1"] + p_action["drop"]  # LSB
    p_qubit1_1 = p_action["edge_2"] + p_action["drop"]  # MSB

    # convert marginal probabilities into Ry angles for each qubit
    # For a single qubit, Ry(theta) on |0> gives P(1) = sin^2(theta/2)
    def theta_from_p(p):
        p = float(min(max(p, 0.0), 1.0))
        return 2.0 * float(np.arcsin(np.sqrt(p)))

    theta0 = theta_from_p(p_qubit0_1)
    theta1 = theta_from_p(p_qubit1_1)

    details = {
        "p_action": p_action,
        "p_qubit0_1": p_qubit0_1,
        "p_qubit1_1": p_qubit1_1,
        "theta0": float(theta0),
        "theta1": float(theta1),
    }

    # If Qiskit available and requested, run a 2-qubit circuit
    if HAS_QISKIT and force_quantum:
        try:
            backend = Aer.get_backend("qasm_simulator")
            qc = QuantumCircuit(2, 2)
            # Put both qubits into superposition
            qc.h([0, 1])
            # Apply Ry rotations to bias each qubit according to marginals
            qc.ry(theta0, 0)
            qc.ry(theta1, 1)
            # Measure qubits into classical bits; map qubit1 -> bit1, qubit0 -> bit0
            qc.measure([1, 0], [1, 0])
            qc = transpile(qc, backend=backend)
            job = backend.run(qc, shots=shots, seed_simulator=seed)
            result = job.result()
            counts = result.get_counts(qc)
            # choose most frequent outcome
            measured = max(counts.items(), key=lambda kv: kv[1])[0]
            # ensure bitstring length 2
            if len(measured) == 1:
                measured = "0" + measured
            # decode mapping
            mapping = {"00": "LOCAL", "01": "EDGE_SERVER_1", "10": "EDGE_SERVER_2", "11": "DROP_TASK"}
            action = mapping.get(measured, "UNKNOWN")
            details["counts"] = counts
            return measured, action, details
        except Exception as e:
            logger.warning("Quantum circuit failed (%s), falling back to classical sampling.", e)

    # Fallback: classical probabilistic sampling using p_action
    rng = np.random.RandomState(seed)
    choice = _sample_by_prob_dict(p_action, rng=rng)
    # convert sampled action key into bitstring according to mapping
    # We have keys 'local','edge_1','edge_2','drop'
    reverse_map = {"local": "00", "edge_1": "01", "edge_2": "10", "drop": "11"}
    measured = reverse_map[choice]
    mapping = {"00": "LOCAL", "01": "EDGE_SERVER_1", "10": "EDGE_SERVER_2", "11": "DROP_TASK"}
    action = mapping[measured]
    details["sampled_choice"] = choice
    return measured, action, details


def should_offload(features: np.ndarray, model_confidence: float, force_quantum: bool = False,
                   quantum_threshold: float = 0.6, heuristic_threshold: float = 0.7) -> Tuple[bool, dict]:
    """
    Decide whether to offload.

    Returns:
        (decision_bool, details)
    details includes:
       - 'method': 'quantum' or 'heuristic'
       - 'score' or 'model_confidence' etc.
    """
    features = np.asarray(features).ravel()
    # Keep original heuristic behavior available
    decision = heuristic_should_offload(features, model_confidence, threshold=heuristic_threshold)
    return decision, {"method": "heuristic", "model_confidence": float(model_confidence)}


if __name__ == "__main__":
    # Simple smoke test
    sample = [0.1, 0.2, -0.5, 1.2, 0.0, 0.3, 0.6, -0.1]
    print("Has Qiskit:", HAS_QISKIT)
    print("Heuristic offload:", should_offload(sample, model_confidence=0.65))
    # Example of the new 2-qubit selection (classical fallback if qiskit not installed)
    from classical_model import compute_action_costs
    costs = compute_action_costs(np.array(sample))
    measured, action, details = select_action_from_costs(costs, shots=1, seed=42, force_quantum=False)
    print("Costs:", costs)
    print("Measured:", measured)
    print("Action:", action)
