"""
classical_model.py

Simple classical model utilities:
- train: creates and saves a small classifier on synthetic data
- load_model: loads a saved model
- predict: returns class and probability for input features
"""

from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_PATH = "classical_model.joblib"


def train(random_state: int = 42, n_samples: int = 1000) -> Tuple[float, str]:
    """
    Train a simple logistic regression classifier on synthetic data and persist it.

    Returns:
        tuple: (validation_accuracy, model_path)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    joblib.dump(clf, MODEL_PATH)
    return acc, os.path.abspath(MODEL_PATH)


def load_model(path: str = None):
    """
    Load a persisted model. If path is None, uses default MODEL_PATH.

    Raises FileNotFoundError if model not found.
    """
    if path is None:
        path = MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Train first with train().")
    return joblib.load(path)


def predict(features: np.ndarray, model=None) -> Tuple[int, float]:
    """
    Predict label and probability using the provided model or loaded model.

    Args:
        features: 1D or 2D numpy array of shape (n_features,) or (1, n_features)
        model: optional scikit-learn model instance

    Returns:
        (predicted_label, probability_of_predicted_label)
    """
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if model is None:
        model = load_model()
    probs = model.predict_proba(features)[0]
    pred = int(model.predict(features)[0])
    prob = float(max(probs))
    return pred, prob


def compute_action_costs(features: np.ndarray, model=None, drop_penalty: float = 1.2) -> dict:
    """
    Compute relative costs for four actions: local, edge_server_1, edge_server_2, drop.

    Returns a dict with keys: 'local', 'edge_1', 'edge_2', 'drop'. Values are floats
    normalized so the minimum cost is 1.0 (relative costs).
    """
    features = np.asarray(features).ravel()
    # get model prediction and confidence (loads model if needed)
    try:
        _, prob = predict(features, model=model)
    except Exception:
        prob = 0.5

    # simple feature-derived factor
    norm = float(np.linalg.norm(features)) / max(1.0, features.size)

    # raw costs: lower model confidence -> higher costs
    raw_local = (1.0 - prob) + 0.5 * norm
    raw_edge_1 = (1.0 - prob) + 0.2 + 0.3 * norm
    raw_edge_2 = (1.0 - prob) + 0.25 + 0.15 * norm
    raw_drop = float(drop_penalty)

    raws = {
        "local": max(1e-6, raw_local),
        "edge_1": max(1e-6, raw_edge_1),
        "edge_2": max(1e-6, raw_edge_2),
        "drop": max(1e-6, raw_drop),
    }

    # normalize so the minimum cost becomes 1.0 for easier interpretation
    min_raw = min(raws.values())
    costs = {k: float(v / min_raw) for k, v in raws.items()}
    return costs


if __name__ == "__main__":
    acc, path = train()
    print(f"Trained classical model; validation accuracy: {acc:.4f}; saved to {path}")
