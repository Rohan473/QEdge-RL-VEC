"""
agents/qrl_agent.py

Hybrid Quantum-Classical DQN agent for VECEnvironment.

The Q-function is approximated by a Variational Quantum Circuit (VQC) that
maps the 12-dim observation to 4 Q-values.  The VQC is differentiated through
PennyLane's auto-diff interface and optimised with Adam — no manual
parameter-shift required (PennyLane handles it via its JAX/PyTorch backend).

Circuit architecture (data re-uploading, 4 qubits, 3 variational layers)
-------------------------------------------------------------------------
Encoding (per layer):
    For qubit q in {0,1,2,3}:
        RY(π × obs[q])   — encode 4 of the 12 obs features
        RZ(π × obs[q+4]) — encode the next 4 features
    (Layer 0 encodes obs[0..3] & obs[4..7], layer 1 encodes obs[4..7] & obs[8..11],
     layer 2 re-encodes obs[8..11] & obs[0..3] — re-uploading strategy.)

Variational block (per layer):
    [RY(θ_{l,q,0}), RZ(θ_{l,q,1})] on each qubit  (learnable)
    CNOT ring: (0→1), (1→2), (2→3), (3→0)

Measurement:
    ⟨Z⟩ on each of the 4 qubits → 4 raw Q-value estimates.
    Values are in [-1, +1]; they are scaled by a learnable scalar w (initialised 1)
    to match the scale of the true Q-values.

References
----------
- Pérez-Salinas et al. (2020) "Data re-uploading for a universal quantum
  classifier." Quantum 4, 226.  arXiv:1907.02085
- Jerbi et al. (2021) "Parametrized quantum policies for reinforcement
  learning." NeurIPS 2021.  arXiv:2103.05577

Training protocol
-----------------
Identical to DQNAgent: replay buffer, ε-greedy, Huber loss, soft target update.
The VQC parameters θ and the output scale w are optimised jointly via Adam.
PennyLane's `qml.QNode` with `diff_method="best"` selects backprop on the
default.qubit simulator, giving exact gradients at the cost of classical memory.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import pennylane as qml
    import torch
    import torch.nn as nn
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False


# ── Replay buffer (identical to DQN) ─────────────────────────────────────────

@dataclass
class _Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class _ReplayBuffer:
    def __init__(self, capacity: int = 10_000, seed: int = 0) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, t: _Transition) -> None:
        self._buf.append(t)

    def sample(self, batch: int) -> List[_Transition]:
        return self._rng.sample(self._buf, batch)

    def __len__(self) -> int:
        return len(self._buf)


# ── VQC Q-network ─────────────────────────────────────────────────────────────

def _build_vqc(n_qubits: int = 4, n_layers: int = 3):
    """Construct the PennyLane QNode for the VQC Q-network.

    Returns a callable ``vqc(obs, weights)`` that maps
        obs:     (batch, 12) float Tensor
        weights: (n_layers, n_qubits, 2) float Tensor
    to a (batch, n_qubits) Tensor of ⟨Z⟩ expectations.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    # Encode feature slices for each layer (data re-uploading)
    # Layer 0: obs[0:4], obs[4:8]
    # Layer 1: obs[4:8], obs[8:12]
    # Layer 2: obs[8:12], obs[0:4]
    _ry_slices = [(0, 4), (4, 8), (8, 12)]
    _rz_slices = [(4, 8), (8, 12), (0, 4)]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(obs: "torch.Tensor", weights: "torch.Tensor") -> list:
        """Forward pass supporting both single (12,) and batched (B, 12) obs.

        PennyLane's default.qubit device broadcasts batched rotation angles
        across the batch dimension, running all B samples in one vectorised
        statevector computation — roughly 10-15x faster than a Python loop.
        """
        for layer in range(n_layers):
            ry_start = _ry_slices[layer][0]
            rz_start = _rz_slices[layer][0]
            # Data encoding — obs[..., idx] works for both (12,) and (B,12)
            for q in range(n_qubits):
                qml.RY(torch.pi * obs[..., ry_start + q], wires=q)
                qml.RZ(torch.pi * obs[..., rz_start + q], wires=q)
            # Variational block
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            # CNOT ring entangler
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class VQCQNetwork(nn.Module):
    """Torch nn.Module wrapping the PennyLane VQC Q-network.

    Accepts a batch of observations and returns Q-values for all 4 actions.

    Args:
        n_qubits: number of qubits (= number of actions = 4).
        n_layers: number of data-reupload + variational layers.
        seed: random seed for weight initialisation.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 0) -> None:
        if not _HAS_PENNYLANE:
            raise ImportError(
                "PennyLane is required for QRLAgent. Install with: pip install pennylane"
            )
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        torch.manual_seed(seed)
        # Variational weights: shape (n_layers, n_qubits, 2)
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float32) * 0.1
        )
        # Output scale to match Q-value magnitude (ℤ outputs are in [-1,1])
        self.output_scale = nn.Parameter(torch.ones(1))

        self._circuit = _build_vqc(n_qubits, n_layers)

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        """Compute Q-values for a batch of observations.

        Args:
            obs: Tensor of shape (batch, 12).

        Returns:
            Tensor of shape (batch, 4) — one Q-value per action.
        """
        ev_list = self._circuit(obs, self.weights)  # list of 4 tensors, each (batch,)
        q_tensor = torch.stack(ev_list, dim=-1)      # (batch, 4)
        return q_tensor * self.output_scale


# ── QRL Agent ─────────────────────────────────────────────────────────────────

class QRLAgent:
    """Hybrid quantum DQN agent.

    The Q-network is a VQC (see VQCQNetwork).  Training follows the standard
    DQN recipe: replay buffer, ε-greedy, Huber loss, soft target update.
    The target network is a second VQCQNetwork whose weights are soft-updated
    from the online network every step.

    Args:
        obs_dim: observation dimension (must be 12 for the 4-qubit circuit).
        n_actions: number of actions (4).
        n_qubits: VQC qubits (must equal n_actions = 4).
        n_layers: variational layers (3 by default — cite Pérez-Salinas 2020).
        gamma: discount factor.
        lr: Adam learning rate for VQC parameters.
        epsilon_start: initial ε for ε-greedy.
        epsilon_end: final ε.
        epsilon_decay_steps: linear ε decay duration.
        tau: soft-update coefficient.
        buffer_capacity: replay buffer size.
        batch_size: learning batch size.
        warmup_steps: steps before first learning update.
        seed: random seed.
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_qubits: int = 4,
        n_layers: int = 3,
        gamma: float = 0.99,
        lr: float = 5e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        tau: float = 0.005,
        buffer_capacity: int = 10_000,
        batch_size: int = 32,
        warmup_steps: int = 200,
        seed: int = 0,
    ) -> None:
        if not _HAS_PENNYLANE:
            raise ImportError(
                "PennyLane is required for QRLAgent. Install with: pip install pennylane"
            )
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.epsilon_end = epsilon_end
        self._eps_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_steps, 1)
        self.eps = epsilon_start
        self._step = 0

        self.online = VQCQNetwork(n_qubits, n_layers, seed)
        self.target = VQCQNetwork(n_qubits, n_layers, seed)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._buf = _ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection via VQC Q-values.

        Args:
            obs: current observation (12-dim float array).
            greedy: if True, always return argmax Q.

        Returns:
            Integer action in {0, 1, 2, 3}.
        """
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and trigger learning step.

        Returns:
            Loss value if learning occurred, else None.
        """
        self._buf.push(_Transition(obs, action, reward, next_obs, done))
        self._step += 1
        self.eps = max(self.epsilon_end, self.eps - self._eps_decay)

        if len(self._buf) < self.warmup_steps or len(self._buf) < self.batch_size:
            return None

        return self._learn(self._buf.sample(self.batch_size))

    def _learn(self, batch: Sequence[_Transition]) -> float:
        import torch.nn.functional as F

        s = torch.as_tensor(np.stack([t.s for t in batch]), dtype=torch.float32)
        a = torch.as_tensor([t.a for t in batch], dtype=torch.long)
        r = torch.as_tensor([t.r for t in batch], dtype=torch.float32)
        s2 = torch.as_tensor(np.stack([t.s2 for t in batch]), dtype=torch.float32)
        d = torch.as_tensor([float(t.done) for t in batch], dtype=torch.float32)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(dim=1).values
            y = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Soft-update target
        for p_on, p_tg in zip(self.online.parameters(), self.target.parameters()):
            p_tg.data.copy_(self.tau * p_on.data + (1.0 - self.tau) * p_tg.data)

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save online VQC weights to *path* (npz format)."""
        weights = self.online.weights.detach().cpu().numpy()
        scale = self.online.output_scale.detach().cpu().numpy()
        np.savez(path, weights=weights, output_scale=scale)

    def load(self, path: str) -> None:
        """Load VQC weights from *path* (npz format)."""
        data = np.load(path)
        with torch.no_grad():
            self.online.weights.copy_(torch.as_tensor(data["weights"]))
            self.online.output_scale.copy_(torch.as_tensor(data["output_scale"]))
            self.target.load_state_dict(self.online.state_dict())
        self.eps = self.epsilon_end
