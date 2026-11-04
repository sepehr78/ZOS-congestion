"""ZOS: zeroth-order Stackelberg optimization (Algorithm 1).

The leader minimizes Phi_T(theta) = F(theta, y_T(theta)) over Theta = {theta >= 0, 1^T theta = n},
where y_T(theta) is the output of T Frank-Wolfe steps. Each outer iteration t draws B_out
directions u_{t,b}, solves the follower problem at the two projected queries
Pi_Theta(theta_t +- rho u_{t,b}), and takes a projected step along the averaged two-point estimate

    g_t = (1 / B_out) sum_b scale * (Phi_T(Pi_Theta(theta_t + rho u)) - Phi_T(Pi_Theta(theta_t - rho u))) / (2 rho) * u,
    theta_{t+1} = Pi_Theta(theta_t - eta g_t).

Estimator used for the paper's experiments (the default, ``directions="rademacher"``): the u are
unnormalized Rademacher vectors (entries +-1 with probability 1/2, so ||u|| = sqrt(n)) and
scale = 1, i.e. the factor is 1 / (2 rho). Algorithm 1 in the paper instead writes unit-sphere
directions with the factor d_theta / (2 rho); that estimator is available as
``directions="sphere"`` (scale = n) but did not produce the reported numbers. In both cases the
queries are projected onto Theta before the follower solve.

Follower solves. The first solve at theta_0 = 1 starts from the oracle's initial atom with the
cold-start FW schedule; every later solve (the 2 B_out probes and the solve at theta_{t+1})
warm-starts from the current follower iterate y_t with the shifted schedule (see frank_wolfe.py).
The probes of one iteration run in parallel threads.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np

from .costs import FractionalCost
from .frank_wolfe import frank_wolfe
from .projection import project_theta


@dataclass
class ZOSResult:
    """Chronological trajectory: entry k is the state after k outer updates (k = 0..K)."""

    theta: np.ndarray      # (K+1, n) leader iterates theta_k
    y: np.ndarray          # (K+1, n) follower iterates y_T(theta_k)
    objective: np.ndarray  # (K+1,) Phi_T(theta_k) = F(theta_k, y_T(theta_k))
    wall_time: np.ndarray  # (K+1,) cumulative seconds since the start of the run


def zos(cost: FractionalCost, lmo, *, K: int = 500, T: int = 3000, rho: float = 0.05,
        eta: float = 0.05, B_out: int = 4, seed: int = 0, directions: str = "rademacher",
        threads: int | None = None, log_every: int = 0) -> ZOSResult:
    """Run K outer ZOS iterations from theta_0 = 1 and return the trajectory.

    Randomness: one generator seeded with ``seed`` draws the directions; a second generator
    with the same seed draws the per-solve sampling seeds used by a stochastic LMO; the first
    follower atom of a stochastic LMO is drawn with ``seed`` itself.
    ``threads`` (default: $OMP_NUM_THREADS or all cores) is split between the probe threads
    and the Numba kernels inside each probe.
    """
    if directions not in ("rademacher", "sphere"):
        raise ValueError(f"directions must be 'rademacher' or 'sphere', got {directions!r}")
    n = cost.d.size
    scale = n if directions == "sphere" else 1
    direction_rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(seed)
    draw_seed = lambda size=None: seed_rng.integers(0, 2**31 - 1, size=size, dtype=np.int64)  # noqa: E731

    threads = threads or int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
    workers = min(B_out, threads)
    probe_threads = min(max(1, threads // workers), numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))

    def phi(theta, y0, warm_start, solve_seed):
        y = frank_wolfe(theta, cost, lmo, T, y0, warm_start, int(solve_seed))
        return cost.social_cost(theta, y), y

    def probe_pair(u, seed_plus, seed_minus, theta, y):
        numba.set_num_threads(probe_threads)
        f_plus, _ = phi(project_theta(theta + rho * u), y, True, seed_plus)
        f_minus, _ = phi(project_theta(theta - rho * u), y, True, seed_minus)
        return f_plus, f_minus

    start = time.perf_counter()
    theta = np.ones(n)
    f, y = phi(theta, lmo.initial(seed), False, draw_seed())
    thetas, ys, objectives, times = [theta], [y], [f], [time.perf_counter() - start]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k in range(K):
            if directions == "rademacher":
                U = [direction_rng.choice([-1.0, 1.0], size=n) for _ in range(B_out)]
            else:
                U = [u / np.linalg.norm(u) for u in direction_rng.standard_normal((B_out, n))]
            seeds = draw_seed(2 * B_out)
            pairs = pool.map(probe_pair, U, seeds[0::2], seeds[1::2], [theta] * B_out, [y] * B_out)

            g_hat = np.zeros(n)
            for u, (f_plus, f_minus) in zip(U, pairs):
                g_hat += (scale * (f_plus - f_minus) / (2.0 * rho)) * u
            g_hat /= B_out

            theta = project_theta(theta - eta * g_hat)
            f, y = phi(theta, y, True, draw_seed())
            thetas.append(theta)
            ys.append(y)
            objectives.append(f)
            times.append(time.perf_counter() - start)
            if log_every and (k + 1) % log_every == 0:
                print(f"outer {k + 1:4d}/{K}  Phi_T = {f:.6f}  ({times[-1]:.0f} s)", flush=True)

    return ZOSResult(np.array(thetas), np.array(ys), np.array(objectives), np.array(times))
