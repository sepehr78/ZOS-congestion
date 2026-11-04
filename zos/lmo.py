"""Linear minimization oracles LMO(g) in argmin_{S in S} sum_{i in S} g_i (Section 4).

Each oracle returns the incidence vector 1_S in {0, 1}^n of the selected strategy and offers
``initial(seed)``, the feasible atom y_0 = 1_{S_0} that starts the first Frank-Wolfe solve.

* :class:`ShortestPathLMO`: exact, Dijkstra's algorithm on the s-t path family (Scenario 1).
* :class:`ZDDLMO`: exact, dynamic program over the ZDD of the family (Scenarios 2-3).
* :class:`SampledLMO`: LMO_m, the best of m strategies drawn from the ZDD under US, UL or HL.
"""

from typing import Sequence

import networkx as nx
import numpy as np
from numba import njit

from .samplers import ZDDSampler
from .zdd import ZDD


class ShortestPathLMO:
    """Exact LMO for s-t paths via Dijkstra's algorithm (NetworkX)."""

    stochastic = False

    def __init__(self, graph: nx.Graph, edges: Sequence[tuple[int, int]], source: int, target: int):
        self.graph, self.source, self.target, self.n = graph, source, target, len(edges)
        self._index = {}
        for i, (u, v) in enumerate(edges):
            self._index[(u, v)] = self._index[(v, u)] = i

    def __call__(self, g: np.ndarray, seed: int | None = None) -> np.ndarray:
        path = nx.shortest_path(self.graph, self.source, self.target,
                                weight=lambda u, v, _: g[self._index[(u, v)]])
        return self._incidence(path)

    def initial(self, seed: int | None = None) -> np.ndarray:
        """A path with the fewest edges (unweighted shortest path)."""
        return self._incidence(nx.shortest_path(self.graph, self.source, self.target))

    def _incidence(self, path: list) -> np.ndarray:
        s = np.zeros(self.n)
        s[[self._index[e] for e in zip(path[:-1], path[1:])]] = 1.0
        return s


class ZDDLMO:
    """Exact LMO by minimum-weight root-to-TRUE path in the ZDD."""

    stochastic = False

    def __init__(self, zdd: ZDD):
        self.zdd, self.n = zdd, zdd.n

    def __call__(self, g: np.ndarray, seed: int | None = None) -> np.ndarray:
        return self.zdd.argmin(g)

    def initial(self, seed: int | None = None) -> np.ndarray:
        """A strategy with the fewest resources (argmin under unit weights)."""
        return self.zdd.argmin(np.ones(self.n))


class SampledLMO:
    """Subsampled oracle LMO_m: draw m strategies with ``sampler`` and return the cheapest."""

    stochastic = True

    def __init__(self, sampler: ZDDSampler, m: int):
        self.sampler, self.m, self.n = sampler, int(m), sampler.zdd.n

    def __call__(self, g: np.ndarray, seed: int) -> np.ndarray:
        strategies, lengths = self.sampler.sample(self.m, seed)
        best = _cheapest(strategies, lengths, g)
        return self._incidence(strategies[best, : lengths[best]])

    def initial(self, seed: int) -> np.ndarray:
        """The shortest of m sampled strategies, i.e. LMO_m under unit weights."""
        strategies, lengths = self.sampler.sample(self.m, seed)
        best = int(np.argmin(lengths))
        return self._incidence(strategies[best, : lengths[best]])

    def _incidence(self, edges: np.ndarray) -> np.ndarray:
        s = np.zeros(self.n)
        s[edges] = 1.0
        return s


@njit(cache=True, nogil=True)
def _cheapest(strategies, lengths, g):
    best, best_cost = -1, 1e18
    for i in range(strategies.shape[0]):
        cost = 0.0
        for j in range(lengths[i]):
            cost += g[strategies[i, j]]
        if cost < best_cost:
            best, best_cost = i, cost
    return best
