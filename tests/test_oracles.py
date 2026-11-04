"""ZDD construction, exact LMOs and samplers on a 3x4 grid, checked against brute force."""

import numpy as np
import pytest

from zos import SampledLMO, ShortestPathLMO, ZDDLMO, ZDDSampler

from conftest import EDGES, FAMILIES, GRID, SOURCE, TARGET


def strategy(s: np.ndarray) -> frozenset:
    return frozenset(np.flatnonzero(s).tolist())


@pytest.mark.parametrize("family", FAMILIES)
def test_zdd_count_matches_enumeration(tiny, family):
    zdd, strategies = tiny[family]
    assert len(strategies) > 1
    assert zdd.count() == len(strategies)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_lmo_matches_brute_force(tiny, family):
    zdd, strategies = tiny[family]
    rng = np.random.default_rng(0)
    oracles = [ZDDLMO(zdd)]
    if family == "st_path":
        oracles.append(ShortestPathLMO(GRID, EDGES, SOURCE, TARGET))
    for _ in range(50):
        g = rng.uniform(0.1, 1.0, len(EDGES))
        best = min(sum(g[i] for i in S) for S in strategies)
        for lmo in oracles:
            s = lmo(g)
            assert strategy(s) in strategies
            assert np.dot(g, s) == pytest.approx(best, rel=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("scheme", ["US", "UL", "HL"])
def test_samplers_draw_valid_strategies_with_the_right_law(tiny, family, scheme):
    zdd, strategies = tiny[family]
    draws, lengths = ZDDSampler(zdd, scheme).sample(20000, seed=7)
    sampled = [frozenset(row[:k].tolist()) for row, k in zip(draws, lengths)]
    assert all(S in strategies for S in sampled)
    assert np.array_equal(draws, ZDDSampler(zdd, scheme).sample(20000, seed=7)[0])

    # q(S) for US / UL / HL, from the length strata N_l of the enumerated family.
    sizes = np.array([len(S) for S in strategies])
    N = {l: np.sum(sizes == l) for l in np.unique(sizes)}
    w = {"US": {l: N[l] for l in N}, "UL": {l: 1.0 for l in N}, "HL": {l: 1.0 / l for l in N}}[scheme]
    total = sum(w.values())
    for l in N:
        expected = w[l] / total
        observed = np.mean(lengths == l)
        assert abs(observed - expected) < 4 * np.sqrt(expected * (1 - expected) / 20000) + 1e-12


def test_sampled_lmo_is_best_of_its_draws(tiny):
    zdd, strategies = tiny["steiner_cycle"]
    lmo = SampledLMO(ZDDSampler(zdd, "HL"), m=5)
    g = np.random.default_rng(1).uniform(0.1, 1.0, len(EDGES))
    draws, lengths = lmo.sampler.sample(5, seed=3)
    best = min(g[row[:k]].sum() for row, k in zip(draws, lengths))
    s = lmo(g, seed=3)
    assert strategy(s) in strategies
    assert np.dot(g, s) == pytest.approx(best)
