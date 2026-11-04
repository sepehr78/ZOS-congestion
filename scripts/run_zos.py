"""Run ZOS on one TNTP scenario with the paper's settings and save the trajectory.

Defaults: T = 3000, K = 500, rho = 0.05, eta = 0.05, B_out = 4, and the paper's oracle per
scenario (Dijkstra in Scenario 1, exact ZDD argmin in Scenario 2, HL with m = 1000 in Scenario 3).

    python scripts/run_zos.py --scenario 1 --seed 0 --out runs/s1_seed0.npz --certify

The .npz file holds the chronological trajectory (theta, y, objective, wall_time for k = 0..K);
``--certify`` also prints the certified social cost of the final iterate (see certify.py).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from zos import PAPER_ORACLE, certify, load_scenario, zos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--oracle", choices=("exact", "US", "UL", "HL"),
                        help="inner LMO (default: the paper's choice for the scenario)")
    parser.add_argument("--m", type=int, default=1000, help="candidate budget of the sampled LMO")
    parser.add_argument("--K", type=int, default=500, help="outer iterations")
    parser.add_argument("--T", type=int, default=3000, help="Frank-Wolfe steps per follower solve")
    parser.add_argument("--rho", type=float, default=0.05, help="smoothing radius")
    parser.add_argument("--eta", type=float, default=0.05, help="outer step size")
    parser.add_argument("--B-out", type=int, default=4, help="directions per outer iteration")
    parser.add_argument("--directions", choices=("rademacher", "sphere"), default="rademacher",
                        help="rademacher: the paper's experiments; sphere: Algorithm 1 as written")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, help="CPU threads (default: $OMP_NUM_THREADS or all)")
    parser.add_argument("--out", type=Path, help="output .npz file")
    parser.add_argument("--certify", action="store_true", help="certify the final iterate")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    oracle = args.oracle or PAPER_ORACLE[args.scenario]
    start = time.perf_counter()
    zdd = scenario.build_zdd() if oracle != "exact" or scenario.family != "st_path" else None
    if zdd is not None:
        print(f"ZDD: {zdd.num_nodes} nodes, built in {time.perf_counter() - start:.1f} s", flush=True)
    lmo = scenario.lmo(oracle, args.m, zdd)

    result = zos(scenario.cost, lmo, K=args.K, T=args.T, rho=args.rho, eta=args.eta,
                 B_out=args.B_out, seed=args.seed, directions=args.directions,
                 threads=args.threads, log_every=args.log_every)
    print(f"Scenario {args.scenario}, oracle {oracle}: Phi_T(theta_K) = {result.objective[-1]:.6f} "
          f"after {result.wall_time[-1]:.0f} s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, scenario=args.scenario, oracle=oracle, m=args.m, seed=args.seed,
                            theta=result.theta, y=result.y, objective=result.objective,
                            wall_time=result.wall_time)
        print(f"saved {args.out}")

    if args.certify:
        cert = certify(result.theta[-1], scenario.cost, scenario.exact_lmo(zdd))
        print(f"certified social cost F(theta_K, y*) = {cert.social_cost:.6f} "
              f"(exact FW gap {cert.fw_gap:.1e}, {cert.lmo_calls} LMO calls)")


if __name__ == "__main__":
    main()
