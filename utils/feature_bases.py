"""
Linear function-approximation feature bases for 1-D state spaces (Chapter 9).
All functions take a state index s ∈ {1, …, n_states} and return a 1-D
numpy array φ(s) suitable for dot-product value approximation: V(s) ≈ w · φ(s).
States are normalised to x = s / n_states ∈ (0, 1] before feature computation.
"""

import numpy as np


def state_aggregation(s: int, n_states: int, n_groups: int) -> np.ndarray:
    """One-hot encoding of the group that state s belongs to."""
    group = min(int((s - 1) * n_groups / n_states), n_groups - 1)
    phi   = np.zeros(n_groups)
    phi[group] = 1.0
    return phi


def polynomial_features(s: int, n_states: int, degree: int) -> np.ndarray:
    """φ_i(s) = x^i for i = 0, 1, …, degree where x = s / n_states."""
    x   = s / n_states
    return np.array([x ** i for i in range(degree + 1)])


def fourier_features(s: int, n_states: int, order: int) -> np.ndarray:
    """Fourier cosine basis: φ_i(s) = cos(i·π·x) for i = 0, …, order."""
    x   = s / n_states
    return np.array([np.cos(i * np.pi * x) for i in range(order + 1)])


def rbf_features(s: int, n_states: int, n_centers: int, sigma: float = 0.1) -> np.ndarray:
    """Gaussian RBF: φ_i(s) = exp(-(x - c_i)² / (2σ²)) with evenly-spaced centres."""
    x       = s / n_states
    centres = np.linspace(0.0, 1.0, n_centers)
    return np.exp(-((x - centres) ** 2) / (2 * sigma ** 2))


# ── Registry used by pages to enumerate bases ────────────────────────────────

BASIS_REGISTRY = {
    "State Aggregation": {
        "fn":     state_aggregation,
        "param":  ("Groups", 10, 2, 50, 1),   # (label, default, min, max, step)
        "n_feat": lambda p: p,
    },
    "Polynomial": {
        "fn":     polynomial_features,
        "param":  ("Degree", 5, 1, 20, 1),
        "n_feat": lambda p: p + 1,
    },
    "Fourier Cosine": {
        "fn":     fourier_features,
        "param":  ("Order", 5, 1, 20, 1),
        "n_feat": lambda p: p + 1,
    },
    "RBF (Gaussian)": {
        "fn":     rbf_features,
        "param":  ("Centres", 10, 2, 50, 1),
        "n_feat": lambda p: p,
    },
}


def make_feature_matrix(basis_fn, n_states: int, param: int) -> np.ndarray:
    """
    Build Φ: shape (n_states, n_features) where row s is φ(s+1).
    """
    rows = [basis_fn(s + 1, n_states, param) for s in range(n_states)]
    return np.array(rows)
