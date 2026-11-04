"""Projection onto Theta, Frank-Wolfe convergence, certification and a short ZOS run."""

import numpy as np
import pytest

from zos import FractionalCost, ZDDLMO, certify, frank_wolfe, fw_gap, load_scenario, project_theta, zos

from conftest import EDGES


def test_projection_onto_theta():
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = rng.normal(1.0, 2.0, 30)
        p = project_theta(v)
        assert p.min() >= 0.0 and p.sum() == pytest.approx(30.0)
        assert np.allclose(project_theta(p), p)
        # Variational inequality <v - p, z - p> <= 0 for every z in Theta.
        for z in 30.0 * rng.dirichlet(np.ones(30), size=20):
            assert np.dot(v - p, z - p) <= 1e-9


def test_frank_wolfe_gap_vanishes(tiny):
    zdd, _ = tiny["steiner_cycle"]
    lmo = ZDDLMO(zdd)
    rng = np.random.default_rng(0)
    cost = FractionalCost.from_raw(rng.uniform(0.5, 1.0, len(EDGES)), C=10.0)
    theta = project_theta(rng.uniform(0.0, 2.0, len(EDGES)))
    horizons = (10, 100, 1000, 10000)
    gaps = [fw_gap(theta, frank_wolfe(theta, cost, lmo, T, lmo.initial()), cost, lmo) for T in horizons]
    assert all(a > b for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 1e-3
    assert max(g * T for g, T in zip(gaps, horizons)) < 20  # O(1/T) rate of the 2/(t+2) schedule

    cert = certify(theta, cost, lmo)
    assert cert.converged and cert.fw_gap <= 1e-8
    y_fw = frank_wolfe(theta, cost, lmo, 10000, lmo.initial())
    assert cost.social_cost(theta, y_fw) == pytest.approx(cert.social_cost, rel=1e-3)
    assert cert.social_cost == pytest.approx(cost.social_cost(theta, cert.y))


def test_short_zos_run_decreases_social_cost():
    scenario = load_scenario(2)
    lmo = scenario.lmo("exact")
    result = zos(scenario.cost, lmo, K=5, T=300, seed=0)
    assert result.theta.shape == (6, scenario.n)
    assert np.allclose(result.theta.sum(axis=1), scenario.n) and result.theta.min() >= 0.0
    assert result.objective[-1] < result.objective[0]
    certified = [certify(result.theta[k], scenario.cost, lmo).social_cost for k in (0, 5)]
    assert certified[1] < certified[0]
