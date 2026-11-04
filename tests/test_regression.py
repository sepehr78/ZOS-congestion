"""The package reproduces the paper's Scenario 1 and 2 trajectories (one outer update, T = 3000).

Scenario 3 (HL, m = 1000) needs a ~10^6-node ZDD (about 16 GB to build); run ``scripts/regression_check.py s3_hl`` on a
machine with several cores instead.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "regression_check", Path(__file__).resolve().parents[1] / "scripts" / "regression_check.py")
regression_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression_check)


@pytest.mark.parametrize("key", ["s2", "s1"])
def test_matches_reference_trajectory(key):
    diff = regression_check.compare(key, K=1)
    assert diff["y"] <= 1e-12
    assert diff["theta"] <= 1e-10
    assert diff["objective"] <= 1e-9
