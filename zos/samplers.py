"""ZDD-guided strategy samplers for the subsampled LMO (Sections 4 and 5.1).

Three proposal distributions q over the strategy family S:

* ``US`` (uniform over strategies): q(S) = 1 / |S|.
* ``UL`` (uniform over lengths): q(S) = 1 / (|L| N_{|S|}), with L the set of feasible
  lengths and N_l the number of strategies of length l.
* ``HL`` (harmonic length): q(S) proportional to 1 / (|S| N_{|S|}).

US walks the ZDD with branch probabilities proportional to the number of completions; UL and HL
first draw a length l and then walk with probabilities proportional to the number of
completions of the remaining length. Every draw uses its own counter-based random stream
derived from ``(seed, draw index)``, so samples are reproducible and thread-safe.
"""

import numpy as np
from numba import njit, prange

from .zdd import ZDD

SCHEMES = ("US", "UL", "HL")


class ZDDSampler:
    """Draw strategies from a ZDD under the US, UL or HL proposal."""

    def __init__(self, zdd: ZDD, scheme: str):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        self.zdd, self.scheme = zdd, scheme
        if scheme == "US":
            self._counts = np.array([float(c) for c in zdd.completion_counts()])
        else:
            self._log_counts, self._length_cdf = self._length_tables(scheme)

    def sample(self, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``m`` strategies. Row i of the first array holds the edge indices of draw i,
        of which the first ``lengths[i]`` entries are valid."""
        z = self.zdd
        if self.scheme == "US":
            return _sample_us(int(m), int(seed), z.label, z.lo, z.hi, z.root, self._counts, z.n)
        return _sample_stratified(int(m), int(seed), z.label, z.lo, z.hi, z.root,
                                  self._log_counts, self._length_cdf, z.n)

    def _length_tables(self, scheme: str) -> tuple[np.ndarray, np.ndarray]:
        """log N_v(r): log number of completions of length r below node v, and the CDF of the
        length distribution (uniform for UL, proportional to 1/l for HL)."""
        z = self.zdd
        N = z.num_nodes
        lo, hi = z.lo.tolist(), z.hi.tolist()
        max_len = [-1] * (N + 2)
        max_len[N] = 0
        for v in z.order.tolist():
            max_len[v] = max(max_len[lo[v]], max_len[hi[v]] + 1 if max_len[hi[v]] >= 0 else -1)
        L_max = max_len[z.root]

        log_counts = np.full((N + 2, L_max + 1), -np.inf)
        log_counts[N, 0] = 0.0
        shifted = np.full(L_max + 1, -np.inf)
        for v in z.order.tolist():
            r = max_len[v] + 1
            shifted[1:r] = log_counts[hi[v], : r - 1]
            log_counts[v, :r] = np.logaddexp(log_counts[lo[v], :r], shifted[:r])

        feasible = np.isfinite(log_counts[z.root])
        lengths = np.arange(L_max + 1)
        if scheme == "UL":
            weights = feasible.astype(float)
        else:
            weights = np.where(feasible & (lengths > 0), 1.0 / np.maximum(lengths, 1), 0.0)
        cdf = np.cumsum(weights / np.sum(weights))
        cdf[-1] = 1.0
        return log_counts, cdf


# ------------------------- counter-based random streams -------------------------

@njit(cache=True, nogil=True)
def _splitmix64(x):
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, nogil=True)
def _xorshift64star(x):
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    return x * np.uint64(2685821657736338717)


@njit(cache=True, nogil=True)
def _uniform01(x):
    return float((x >> np.uint64(11)) & np.uint64((1 << 53) - 1)) * (1.0 / float(1 << 53))


@njit(cache=True, nogil=True)
def _logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    c = a if a > b else b
    return c + np.log(np.exp(a - c) + np.exp(b - c))


@njit(cache=True, nogil=True)
def _cdf_search(cdf, u):
    """Smallest index l with u < cdf[l]."""
    lo, hi = 0, cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u < cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ------------------------- sampling kernels -------------------------

@njit(cache=True, nogil=True, parallel=True)
def _sample_us(m, seed, label, lo, hi, root, counts, n):
    N = label.shape[0]
    out = np.zeros((m, n), dtype=np.int32)
    lengths = np.zeros(m, dtype=np.int32)
    base = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    for i in prange(m):
        state = _splitmix64(base + np.uint64(i))
        v, k = root, 0
        while v != N:
            state = _xorshift64star(state)
            if _uniform01(state) < counts[lo[v]] / counts[v]:
                v = lo[v]
            else:
                out[i, k] = label[v]
                k += 1
                v = hi[v]
        lengths[i] = k
    return out, lengths


@njit(cache=True, nogil=True, parallel=True)
def _sample_stratified(m, seed, label, lo, hi, root, log_counts, cdf, n):
    N = label.shape[0]
    out = np.zeros((m, n), dtype=np.int32)
    lengths = np.zeros(m, dtype=np.int32)
    base = np.uint64(seed) ^ np.uint64(0xD1B54A32D192ED03)
    for i in prange(m):
        state = _xorshift64star(_splitmix64(base + np.uint64(i)))
        remaining = _cdf_search(cdf, _uniform01(state))
        v, k = root, 0
        while v != N:
            log_lo = log_counts[lo[v], remaining]
            log_hi = log_counts[hi[v], remaining - 1] if remaining > 0 else -np.inf
            take = log_lo == -np.inf
            if log_hi != -np.inf and log_lo != -np.inf:
                state = _xorshift64star(state)
                take = _uniform01(state) >= np.exp(log_lo - _logaddexp(log_lo, log_hi))
            if take:
                out[i, k] = label[v]
                k += 1
                remaining -= 1
                v = hi[v]
            else:
                v = lo[v]
        lengths[i] = k
    return out, lengths
