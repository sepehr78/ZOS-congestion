"""Euclidean projection onto the leader's feasible set Theta = {theta >= 0, 1^T theta = n}."""

import numpy as np


def project_simplex(v: np.ndarray, z: float) -> np.ndarray:
    """Project ``v`` onto the scaled simplex {x >= 0, sum(x) = z} by sorting, in O(n log n)."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - z))[0][-1]
    tau = (cssv[k] - z) / (k + 1.0)
    return np.maximum(v - tau, 0.0)


def project_theta(theta: np.ndarray) -> np.ndarray:
    """Pi_Theta: projection onto Theta = {theta >= 0, 1^T theta = n} with n = len(theta)."""
    return project_simplex(theta, float(theta.size))
