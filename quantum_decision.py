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


def _quantum_score_simulation(features: np.ndarray, seed: int = 42) -> float:
    """
    Minimal quantum-inspired routine using Qiskit statevector simulation.
    Produces a deterministic score in [0, 1] derived from a small circuit.
    Falls back to a simple deterministic mapping if Qiskit not available.
    """
    features = np.asarray(features).ravel()
    # Normalize to [0, pi/2] for rotation angles — keep mapping stable across runs
    if features.size == 0:
        return 0.0
    angles = (features - features.min()) / (np.ptp(features) + 1e-9) * (np.pi / 2)
    if HAS_QISKIT:
        try:
            # small circuit: ry rotations + hadamards, measure expectation-like score from statevector
            n_qubits = min(6, features.size)  # cap number of qubits to keep simulation cheap
            qc = QuantumCircuit(n_qubits)
            # map angles to first n_qubits
            for i in range(n_qubits):
                qc.ry(float(angles[i % angles.size]), i)
            # add a layer of H to create superposition
            qc.h(range(n_qubits))
            # simulate statevector
            backend = Aer.get_backend("statevector_simulator")
            qc = transpile(qc, backend=backend)
            result = backend.run(qc).result()
            state = result.get_statevector(qc)
            # compute a simple score: sum of squared magnitudes for basis states where MSB is 1
            # This gives a reproducible scalar in [0,1]
            probs = np.abs(state) ** 2
            half = len(probs) // 2
            score = float(probs[half:].sum())
            return score
        except Exception as e:
            logger.warning("Qiskit simulation failed: %s. Falling back to deterministic mapping.", e)
    # Fallback deterministic mapping (no qiskit)
    # Map mean and variance into [0,1]
    m = float(np.mean(features))
    s = float(np.std(features))
    score = 1.0 / (1.0 + np.exp(- (m / (s + 1e-6))))
    # clamp
    score = max(0.0, min(1.0, score))
    return score


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
    if force_quantum and HAS_QISKIT:
        score = _quantum_score_simulation(features)
        decision = score > quantum_threshold
        return decision, {"method": "quantum", "score": score, "threshold": quantum_threshold}
    # Otherwise use heuristic. If Qiskit available and not forced we still prefer heuristic in this simple demo.
    decision = heuristic_should_offload(features, model_confidence, threshold=heuristic_threshold)
    return decision, {"method": "heuristic", "model_confidence": float(model_confidence)}


if __name__ == "__main__":
    # Simple smoke test
    sample = [0.1, 0.2, -0.5, 1.2, 0.0, 0.3, 0.6, -0.1]
    print("Has Qiskit:", HAS_QISKIT)
    print("Heuristic offload:", should_offload(sample, model_confidence=0.65))
    if HAS_QISKIT:
        print("Quantum score:", _quantum_score_simulation(sample))
