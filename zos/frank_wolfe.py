"""FW-Equilibrium(theta, T): Frank-Wolfe solver for the Wardrop equilibrium (Appendix B).

The equilibrium y*(theta) minimizes the Beckmann potential f(theta, .) over the strategy polytope
C = conv{1_S : S in S}. Each FW step forms w_t = grad_y f(theta, y_t) = c(y_t; theta), calls the
LMO and moves toward the returned atom.

Step size. Appendix B describes exact line search; the code that produced the paper's results
uses the classical open-loop schedule gamma_t = 2 / (t + 2) for a cold start and the shifted
schedule gamma_t = 2 / (t + 2 + T) for a warm start, which treats the warm start y_0 as the
output of T earlier steps. There is no early stopping. We keep that behavior.
"""

import numpy as np

from .costs import FractionalCost


def frank_wolfe(theta: np.ndarray, cost: FractionalCost, lmo, T: int, y0: np.ndarray,
                warm_start: bool = False, seed: int | None = None) -> np.ndarray:
    """Run T Frank-Wolfe steps from ``y0`` and return y_T(theta).

    ``seed`` is only used by stochastic oracles: step t draws its m strategies with seed + t.
    """
    y = np.array(y0, dtype=float)
    for t in range(T):
        g = cost.edge_costs(theta, y)
        s = lmo(g, None if seed is None else seed + t)
        gamma = 2.0 / (t + 2.0 + T) if warm_start else 2.0 / (t + 2.0)
        y = y + gamma * (s - y)
    return y


def fw_gap(theta: np.ndarray, y: np.ndarray, cost: FractionalCost, exact_lmo) -> float:
    """Exact Frank-Wolfe gap <grad_y f(theta, y), y - LMO(grad_y f(theta, y))>."""
    g = cost.edge_costs(theta, y)
    return float(np.dot(g, y - exact_lmo(g)))
