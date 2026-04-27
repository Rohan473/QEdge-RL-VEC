# Viva Defence Q&A — QEdge-RL-VEC

15 likely examiner questions with crisp, defensible answers.

---

**Q1. Define the MDP you have formulated. What are S, A, P, R, and γ?**

The MDP is a finite-horizon episodic problem:
- **S** — 12-dimensional normalised state vector capturing task parameters (data size, CPU cycles, deadline), vehicle kinematics (speed, distance to each server), network state (channel rates, queue lengths), local CPU load, and episode progress.
- **A** — Discrete(4): LOCAL, EDGE\_1, EDGE\_2, DROP.
- **P** — Stochastic transition determined by the environment's physics: vehicle movement, random task generation, and queue dynamics. We don't model P explicitly; the agent learns from samples.
- **R** — Shaped reward: +1 − 0.3·(latency/deadline) − 0.1·energy if the task meets its deadline; −1 otherwise.
- **γ = 0.99** — near-unity discount, appropriate for a 200-step episode where future task completions matter as much as current ones.

---

**Q2. Why did you use DQN rather than simpler tabular Q-learning?**

The state space is a 12-dimensional continuous vector — $[0,1]^{12}$ — which cannot be discretised tractably (even 10 bins per dimension gives $10^{12}$ states). DQN uses a neural network to generalise Q-value estimates across nearby states, enabling learning from a finite number of samples.

---

**Q3. What is data re-uploading, and why is it necessary for your VQC?**

A single-layer quantum circuit with $n$ qubits can encode $n$ features and implement only a fixed unitary transformation. Without re-uploading, the circuit is not a universal function approximator. Data re-uploading (Pérez-Salinas et al., 2020) repeats the encoding before each variational layer, analogous to how classical deep networks process inputs at multiple levels. With 3 re-upload layers and 4 qubits, the VQC can approximate a much richer class of functions than a single-layer circuit.

---

**Q4. How is the VQC differentiated during training?**

PennyLane's `default.qubit` device supports automatic differentiation through statevector simulation: the full quantum state is tracked as a PyTorch tensor and gradients flow back through gate operations via standard reverse-mode AD (backpropagation). This gives *exact* gradients at $O(2^n)$ classical memory cost. On real hardware — where statevectors are inaccessible — the **parameter-shift rule** would be used instead: $\partial_{\theta_i} \langle Z \rangle = \frac{1}{2}[\langle Z \rangle(\theta_i + \pi/2) - \langle Z \rangle(\theta_i - \pi/2)]$, requiring two circuit evaluations per parameter per update.

---

**Q5. What quantum advantage does your VQC actually provide?**

Two demonstrable advantages:
1. **Parameter efficiency:** 25 parameters vs 5 188 for the comparable DQN, while achieving similar performance. The VQC's inductive bias (Hilbert space geometry) provides implicit regularisation.
2. **Theoretical expressiveness:** A depth-$d$ VQC on $n$ qubits can represent exponentially many Fourier modes in the input (Schuld et al., 2021), giving richer representations per parameter than an MLP.

**Honest caveat:** There is no *runtime* quantum speedup in this project. The statevector simulator is exponentially slower than real hardware evaluation would be. Quantum advantage in this context is architectural, not computational.

---

**Q6. What would happen if you deployed this on real quantum hardware (e.g., IBM Quantum)?**

Several practical challenges arise:
1. **Gate noise:** Real qubits have coherence times of ~100 µs and gate fidelities of 99–99.9%, introducing random errors. The learned policy would degrade gracefully but not catastrophically for short-depth circuits (depth ≈ 15 for 3 layers × 4 qubits × ~5 gates/layer).
2. **Shot noise:** Expectation values must be estimated from $K$ measurements. At $K=1000$ shots, shot noise $\sigma \sim 1/\sqrt{K} \approx 3\%$ adds stochasticity to the Q-value estimates.
3. **Transpilation overhead:** The CNOT ring must be mapped to the device's connectivity graph (not fully connected), adding SWAP gates and increasing depth.
4. **Latency:** A single circuit evaluation takes ~10 ms on IBM Quantum (queue + execution), far too slow for real-time VEC decisions. Inference would need to run on a pre-trained parameter table, not live quantum execution.

---

**Q7. Why is the greedy agent worse than DQN despite using the correct latency formula?**

The greedy agent minimises predicted latency but ignores the deadline explicitly and does not model long-term queue dynamics. Specifically: (a) it may offload to an edge server that is faster right now but will queue up tasks, increasing latency for future tasks; (b) it sometimes selects a low-latency action that still exceeds the deadline (DQN learns to prefer DROP in such cases, avoiding queue pollution for no benefit). DQN maximises cumulative reward over 200 steps, naturally accounting for these temporal effects.

---

**Q8. How did you verify the environment is physically realistic?**

Several sanity checks:
- At 100 m from a server with $P = 0.5$ W, $B = 10$ MHz, the Shannon formula gives ≈ 90 Mbps — consistent with 5G NR short-range rates.
- A 500 Mc task takes exactly 500 ms at 1 GHz local CPU and 100 ms at 5 GHz edge CPU — matches the setup.
- Energy for local execution of 500 Mc: $\kappa f^2 C = 10^{-28} \times (10^9)^2 \times 500 \times 10^6 = 0.05$ J — in the expected range for mobile CPUs (50 mJ per operation).
- Pytest suite: 19 environment tests including reward-sign verification on degenerate cases.

---

**Q9. Why did you choose CNOT ring entanglement rather than all-to-all or random entanglement?**

CNOT ring (nearest-neighbour on a 1-D topology) was chosen for three reasons: (1) it matches the native connectivity of many NISQ devices (IBM Falcon processors have linear connectivity), minimising SWAP overhead on real hardware; (2) it creates entanglement that propagates information around all qubits within one layer, sufficient for 4 qubits; (3) it is a standard ansatz in hardware-efficient VQCs (Kandala et al., 2017), making results comparable to published benchmarks.

---

**Q10. Your episode length is 200 steps. How does a vehicle generate exactly one task per step if real arrival is stochastic?**

The spec describes task generation probability $\lambda$ per vehicle per step, but for clean MDP formulation we generate exactly one task per step (round-robined across the 5 vehicles). This is equivalent to conditioning on a task being present and is standard in VEC offloading literature (Huang et al., 2019). An alternative formulation would include a "no task" action when $\lambda < 1$, adding a 5th action but adding no learning signal (trivially correct: do nothing).

---

**Q11. What is the role of the replay buffer and why is its size 10 000?**

The replay buffer breaks temporal correlation in the training data (consecutive environment steps are highly correlated, violating the i.i.d. assumption of stochastic gradient descent) and enables data reuse. Size 10 000 covers approximately 50 episodes of experience (10 000 / 200 steps), ensuring the buffer holds a diverse mix of early-episode (random) and late-episode (converged) transitions. Larger buffers improve diversity but slow down the recency of new experience.

---

**Q12. How does your reward function encourage energy efficiency?**

The $\beta \cdot E$ term in the positive-reward branch directly penalises energy. A task that completes 10 ms before its deadline with high energy costs less reward than one completing at the same time with low energy. Specifically, $\beta = 0.1$ means an extra 0.1 J of energy costs 0.01 reward units — a small but non-zero incentive. This is intentionally weak: safety (meeting the deadline) is the primary objective, and $\alpha = 0.3 > \beta = 0.1$ reflects that latency matters more than energy.

---

**Q13. What are the limitations of your simulation?**

1. **1-D highway:** Real road networks are 2-D graphs; the channel model extends naturally but vehicle trajectories become more complex.
2. **Two edge servers:** Scaling to 10+ servers requires either a larger action space or hierarchical decision-making.
3. **No handover:** A vehicle moves between server coverage zones; in practice, handover protocols add latency that is not modelled.
4. **Homogeneous vehicles:** All vehicles have the same hardware; heterogeneous fleets (trucks vs cars) would need per-vehicle state encoding.
5. **Perfect queue knowledge:** The agent observes exact queue lengths. In practice, queue state must be estimated from delayed signalling (5G MEC APIs have ~1 ms latency).

---

**Q14. How would you scale this to 100 vehicles and 10 edge servers?**

With 10 servers, the action space grows to 12 (LOCAL + 10 servers + DROP), still tractable for DQN. The state vector grows by 4 features per extra server (queue, rate). The main challenge is the VQC: encoding 20+ features requires more qubits or more data-reuploading layers. Options: (a) use a classical embedding layer to compress the 20-dim state to 12 dims before the VQC; (b) use quantum amplitude encoding (exponential compression, but requires $O(2^n)$ CNOT gates); (c) switch to a multi-agent formulation where each vehicle runs its own local agent.

---

**Q15. Your report mentions that QRL is parameter-efficient. Does fewer parameters always mean better generalisation?**

Not necessarily. Fewer parameters reduces overfitting risk but also limits capacity. The VQC's 25 parameters may be *insufficient* for the full 12-dim state space when the training set is large (500+ episodes), explaining why DQN eventually outperforms QRL with sufficient training. The sweet spot is the low-data regime: when training is limited to 50–100 episodes, the VQC's regularisation bias is a genuine advantage. This trade-off is analogous to comparing a linear model (high bias, low variance) to a deep network (low bias, high variance) in classical ML.
