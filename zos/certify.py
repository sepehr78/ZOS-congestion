"""Certified social cost: F(theta, y*) with the follower re-solved to an exact FW gap <= 1e-8.

This is the post-hoc evaluation behind the "certified social cost" in Section 6 and Appendix F.
It is independent of the follower state the optimizer kept. For the fractional cost model the
potential is the separable convex quadratic

    f(theta, y) = <d, y> + (1/2) y^T diag(h) y,    h = C d / (theta + 1),

which we minimize by fully corrective Frank-Wolfe (column generation): keep the atoms returned so
far by the exact LMO, minimize f exactly over their convex hull (a small QP in the convex
weights), and add the exact-LMO atom at the new point, until the exact FW gap
<grad_y f(theta, y), y - LMO(grad_y f(theta, y))> is at most ``tol``. The certificate is a
numerical near-equilibrium check, not an exact-equilibrium proof.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .costs import FractionalCost


@dataclass
class Certificate:
    social_cost: float  # F(theta, y) at the re-solved follower state
    fw_gap: float       # exact FW gap at that state
    converged: bool     # fw_gap <= tol
    lmo_calls: int
    y: np.ndarray


def certify(theta: np.ndarray, cost: FractionalCost, exact_lmo, tol: float = 1e-8,
            max_lmo_calls: int = 1024) -> Certificate:
    """Re-solve the follower at ``theta`` with an exact LMO and return the certified F."""
    d = cost.d
    h = d * cost.C / (theta + 1.0)
    atoms = [exact_lmo(d)]
    active = {tuple(np.flatnonzero(atoms[0]))}
    weights = np.ones(1)
    y = atoms[0]
    for calls in range(2, max_lmo_calls + 2):
        grad = d + h * y
        atom = exact_lmo(grad)
        gap = max(float(np.dot(grad, y - atom)), 0.0)
        if gap <= tol or calls > max_lmo_calls:
            break
        key = tuple(np.flatnonzero(atom))
        if key in active:
            raise RuntimeError(f"exact LMO returned an active atom at gap {gap:.3g}; "
                               "the restricted QP was not solved accurately")
        active.add(key)
        atoms.append(atom)
        A = np.column_stack(atoms)
        weights = _restricted_qp(A, d, h, np.append(weights, 0.0))
        y = A @ weights
    return Certificate(float(np.dot(d, y) + np.dot(h, y * y)), gap, gap <= tol, calls, y)


def _restricted_qp(A: np.ndarray, d: np.ndarray, h: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Minimize f(A w) over the simplex {w >= 0, sum w = 1}.

    SLSQP gives a warm start; an active-set solve of the KKT system on its support then polishes
    the weights to machine precision.
    """
    lin = A.T @ d
    Q = A.T @ (h[:, None] * A)
    k = lin.size
    w = minimize(lambda w: float(lin @ w + 0.5 * w @ Q @ w), w0, jac=lambda w: lin + Q @ w,
                 method="SLSQP", bounds=[(0.0, 1.0)] * k,
                 constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                              "jac": lambda w: np.ones_like(w)},
                 options={"ftol": 1e-15, "maxiter": 4000}).x

    support = set(np.flatnonzero(w > 1e-10).tolist()) or {int(np.argmax(w))}
    for _ in range(max(100, 20 * k)):
        idx = np.array(sorted(support))
        kkt = np.block([[Q[np.ix_(idx, idx)], -np.ones((idx.size, 1))],
                        [np.ones((1, idx.size)), np.zeros((1, 1))]])
        rhs = np.concatenate((-lin[idx], [1.0]))
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(kkt, rhs, rcond=1e-13)[0]
        w_support, multiplier = sol[:-1], sol[-1]
        if np.min(w_support) < -1e-12:  # drop the most negative weight
            support.remove(int(idx[np.argmin(w_support)]))
            support = support or {int(np.argmin(lin + 0.5 * np.diag(Q)))}
            continue
        refined = np.zeros(k)
        refined[idx] = np.maximum(w_support, 0.0)
        refined /= refined.sum()
        outside = np.array([i for i in range(k) if i not in support], dtype=int)
        reduced = (lin + Q @ refined)[outside] - multiplier
        if outside.size and reduced.min() < -1e-11:  # add the most violated column
            support.add(int(outside[np.argmin(reduced)]))
            continue
        w = refined
        break
    else:
        raise RuntimeError("active-set refinement of the restricted QP did not converge")

    w[np.abs(w) < 1e-15] = 0.0
    w = np.maximum(w, 0.0)
    return w / w.sum()
