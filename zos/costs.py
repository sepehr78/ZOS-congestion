"""Fractional congestion cost model used in all experiments (Section 6, Appendix F.1).

For resource (edge) i with load y_i and leader investment theta_i,

    c_i(y_i; theta_i) = d_i * (1 + C * y_i / (theta_i + 1)),

the leader objective (social cost) is F(theta, y) = sum_i c_i(y_i; theta_i) y_i, and the
followers' Beckmann potential is

    f(theta, y) = sum_i d_i * (y_i + C * y_i^2 / (2 (theta_i + 1))),

so that grad_y f(theta, y) = c(y; theta).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FractionalCost:
    """Cost model with free-flow costs ``d`` (scaled so that max d = 1) and congestion scale ``C``."""

    d: np.ndarray
    C: float

    @classmethod
    def from_raw(cls, d_raw: np.ndarray, C: float) -> "FractionalCost":
        """Rescale raw free-flow costs so that max_i d_i = 1, as in the experiments."""
        d_raw = np.asarray(d_raw, dtype=float)
        return cls(d=d_raw / float(np.max(d_raw)), C=float(C))

    def edge_costs(self, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """c(y; theta), which is also grad_y f(theta, y)."""
        return self.d * (1.0 + self.C * (y / (theta + 1.0)))

    def social_cost(self, theta: np.ndarray, y: np.ndarray) -> float:
        """Leader objective F(theta, y) = <c(y; theta), y>."""
        return float(np.dot(self.edge_costs(theta, y), y))

    def potential(self, theta: np.ndarray, y: np.ndarray) -> float:
        """Follower potential f(theta, y)."""
        return float(np.sum(self.d * (y + 0.5 * self.C * y * y / (theta + 1.0))))
