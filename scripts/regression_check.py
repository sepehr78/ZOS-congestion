"""Compare a short ZOS run with the trajectories recorded by the code that produced the paper.

``tests/data/reference_trajectories.npz`` holds the first three outer updates (theta_k, y_k and
Phi_T(theta_k), k = 0..3) of three runs from the rebuttal / camera-ready experiments:

* s1:    Scenario 1, exact Dijkstra LMO, seed 0       (ZOS-S1-rho0.05-T3000, matched_zos)
* s2:    Scenario 2, exact ZDD LMO, seed 0            (ZOS-S2-rho0.05-T3000, matched_zos)
* s3_hl: Scenario 3, HL sampled LMO, m = 1000, seed 6000000 (ZOS-HL-m1000, runtime_s3_matched_v1)

all with T = 3000, rho = eta = 0.05 and B_out = 4. Scenario 3 builds a ZDD with about one
million nodes (about 16 GB peak memory) and takes about 3 minutes on 32 cores.

    python scripts/regression_check.py s1 s2 [s3_hl] [--K 3]
"""

import argparse
from pathlib import Path

import numpy as np

from zos import load_scenario, zos

REFERENCE = Path(__file__).resolve().parents[1] / "tests" / "data" / "reference_trajectories.npz"
RUNS = {"s1": (1, "exact"), "s2": (2, "exact"), "s3_hl": (3, "HL")}


def compare(key: str, K: int = 3) -> dict[str, float]:
    """Run ``key`` for K outer updates and return the largest deviations from the reference."""
    ref = np.load(REFERENCE)
    number, oracle = RUNS[key]
    scenario = load_scenario(number)
    result = zos(scenario.cost, scenario.lmo(oracle, m=1000), K=K, T=3000, rho=0.05, eta=0.05,
                 B_out=4, seed=int(ref[f"{key}_seed"]))
    return {
        "theta": float(np.abs(result.theta - ref[f"{key}_theta"][: K + 1]).max()),
        "y": float(np.abs(result.y - ref[f"{key}_y"][: K + 1]).max()),
        "objective": float(np.abs(result.objective - ref[f"{key}_objective"][: K + 1]).max()),
        "objective_final": float(result.objective[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", choices=sorted(RUNS))
    parser.add_argument("--K", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    for key in args.runs:
        d = compare(key, args.K)
        print(f"{key}: K = {args.K}, Phi_T(theta_K) = {d['objective_final']:.10f}; max |diff| vs reference: "
              f"theta {d['theta']:.1e}, y {d['y']:.1e}, Phi_T {d['objective']:.1e}", flush=True)


if __name__ == "__main__":
    main()
