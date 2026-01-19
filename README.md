# Quantum Edge Offloading

This repository contains a small example project that demonstrates a decision pipeline for "quantum edge offloading" — deciding whether to run inference locally on a classical model or offload to a (simulated) quantum resource. The code is intended as a lightweight, educational prototype you can run locally.

Contents
- `classical_model.py` — simple classical training / prediction utilities (scikit-learn).
- `quantum_decision.py` — offloading decision logic. Uses a simple heuristic and an optional Qiskit-based simulated quantum decision routine (if Qiskit is installed).
- `main.py` — CLI entrypoint to train the classical model and run predictions with offloading decision.
- `requirements.txt` — Python dependencies.
- `report/project_summary.md` — short project summary and notes.

Features
- Train a classical classifier on synthetic data.
- Decide whether to offload to a quantum resource using a heuristic or an optional simulated quantum circuit.
- Easy CLI to run training and inference.

Quick start
1. Create a Python 3.9+ virtual environment and activate it:
   - python -m venv .venv
   - source .venv/bin/activate  (Linux / macOS)
   - .venv\Scripts\activate     (Windows)

2. Install dependencies:
   - pip install -r requirements.txt

3. Train the classical model:
   - python main.py --train

4. Run a prediction (with decision whether to offload):
   - python main.py --predict

5. Force quantum decision routine if Qiskit is installed:
   - python main.py --predict --force-quantum

Notes
- The quantum decision routine is optional and will only be used if Qiskit is available; otherwise a deterministic heuristic is used.
- This repository is intended as a template / demo. Replace the synthetic data and decision logic with real models and decision criteria for production use.

License
- MIT (add your license file as needed)
