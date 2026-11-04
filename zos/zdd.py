"""Zero-suppressed decision diagrams (ZDDs) of strategy families, and the exact ZDD LMO.

A ZDD represents the strategy family S (e.g. all Steiner cycles through a terminal set) as a
DAG: each internal node v tests one resource ``label[v]``; its 0-child ``lo[v]`` excludes the
resource and its 1-child ``hi[v]`` includes it. Root-to-TRUE paths are in bijection with S.
We build the ZDD with Graphillion and convert it to flat arrays, on which the exact LMO

    LMO(g) in argmin_{S in S} sum_{i in S} g_i

is a single bottom-up dynamic program (Section 4, Appendix I).
"""

import tempfile
import threading
from typing import Sequence

import numpy as np
from graphillion import GraphSet
from numba import njit

FAMILIES = ("st_path", "hamiltonian_path", "steiner_cycle")


class ZDD:
    """Array form of a ZDD over resources 0..n-1.

    Internal nodes are 0..N-1; index N is the TRUE terminal and N+1 the FALSE terminal.
    ``label[v]`` is the resource (edge index) tested at v and ``order`` lists the internal
    nodes children-first.
    """

    def __init__(self, label: np.ndarray, lo: np.ndarray, hi: np.ndarray, root: int,
                 order: np.ndarray, n: int):
        self.label, self.lo, self.hi, self.root, self.order, self.n = label, lo, hi, root, order, n
        self._local = threading.local()  # per-thread DP buffers for concurrent LMO calls

    @property
    def num_nodes(self) -> int:
        return int(self.label.size)

    def completion_counts(self) -> list[int]:
        """Exact number of root-to-TRUE completions below every node (terminals included)."""
        N = self.num_nodes
        counts = [0] * (N + 2)
        counts[N] = 1
        lo, hi = self.lo.tolist(), self.hi.tolist()
        for v in self.order.tolist():
            counts[v] = counts[lo[v]] + counts[hi[v]]
        return counts

    def count(self) -> int:
        """|S|, the number of strategies."""
        return self.completion_counts()[self.root]

    def argmin(self, g: np.ndarray) -> np.ndarray:
        """Incidence vector of a minimum-weight strategy under resource weights ``g``."""
        buffers = getattr(self._local, "buffers", None)
        if buffers is None:
            buffers = self._local.buffers = (
                np.empty(self.num_nodes + 2), np.empty(self.num_nodes, dtype=np.uint8),
                np.empty(self.n, dtype=np.int64))
        k = _argmin(self.label, self.lo, self.hi, self.order, self.root, g, *buffers)
        s = np.zeros(self.n)
        s[buffers[2][:k]] = 1.0
        return s


@njit(cache=True, nogil=True)
def _argmin(label, lo, hi, order, root, g, dp, choice, chosen):
    """dp[v] = min(dp[lo[v]], g[label[v]] + dp[hi[v]]), ties to the 0-child; writes the argmin to ``chosen``."""
    N = label.shape[0]
    dp[N] = 0.0
    dp[N + 1] = np.inf
    for v in order:
        c_lo = dp[lo[v]]
        c_hi = dp[hi[v]] + g[label[v]]
        if c_hi < c_lo:
            dp[v] = c_hi
            choice[v] = 1
        else:
            dp[v] = c_lo
            choice[v] = 0
    k = 0
    v = root
    while v < N:
        if choice[v] == 1:
            chosen[k] = label[v]
            k += 1
            v = hi[v]
        else:
            v = lo[v]
    return k


def build_zdd(edges: Sequence[tuple[int, int]], family: str, source: int | None = None,
              target: int | None = None, terminals: Sequence[int] | None = None) -> ZDD:
    """Build the ZDD of s-t paths, Hamiltonian s-t paths or Steiner cycles with Graphillion.

    ``edges`` fixes the resource order: ``label`` values index into it.
    """
    GraphSet.set_universe(sorted((min(u, v), max(u, v)) for u, v in edges))
    if family == "st_path":
        family_set = GraphSet.paths(source, target)
    elif family == "hamiltonian_path":
        family_set = GraphSet.paths(source, target, is_hamilton=True)
    elif family == "steiner_cycle":
        family_set = GraphSet.steiner_cycles(list(terminals))
    else:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")

    with tempfile.TemporaryFile("w+") as fh:
        family_set.dump(fh)
        fh.seek(0)
        rows = [line.split() for line in fh if line.strip() and line.strip() != "."]

    # Graphillion dump rows are "node_id label lo hi" with 1-based labels into
    # GraphSet.universe(); the root is the last row; terminals are "T"/"B".
    node_ids = sorted(int(r[0]) for r in rows)
    index = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    child = lambda c: N if c == "T" else N + 1 if c == "B" else index[int(c)]  # noqa: E731

    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = edge_index[(v, u)] = i
    universe = [edge_index[tuple(e)] for e in GraphSet.universe()]

    gl_label = np.empty(N, dtype=np.int64)
    label, lo, hi = (np.empty(N, dtype=np.int32) for _ in range(3))
    for nid, lab, l, h in rows:
        v = index[int(nid)]
        gl_label[v] = int(lab)
        label[v] = universe[int(lab) - 1]
        lo[v], hi[v] = child(l), child(h)
    # Children test later universe items than their parents, so decreasing
    # Graphillion label is a children-first order.
    order = np.argsort(-gl_label, kind="stable").astype(np.int32)
    return ZDD(label, lo, hi, index[int(rows[-1][0])], order, len(edges))
