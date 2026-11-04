"""Certified social cost of saved ZOS iterates (post-hoc exact follower re-solve, FW gap <= 1e-8).

    python scripts/certify.py runs/s1_seed0.npz                 # final iterate theta_K
    python scripts/certify.py runs/s1_seed0.npz --checkpoints 0 100 500
"""

import argparse
from pathlib import Path

import numpy as np

from zos import certify, load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help=".npz file written by run_zos.py")
    parser.add_argument("--checkpoints", type=int, nargs="+", help="outer iterations k (default: K)")
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()

    run = np.load(args.run)
    scenario = load_scenario(int(run["scenario"]))
    lmo = scenario.exact_lmo()
    K = len(run["theta"]) - 1
    for k in args.checkpoints or [K]:
        cert = certify(run["theta"][k], scenario.cost, lmo, tol=args.tol)
        print(f"k = {k:4d}  certified F = {cert.social_cost:.6f}  exact FW gap = {cert.fw_gap:.1e}  "
              f"converged = {cert.converged}")


if __name__ == "__main__":
    main()
