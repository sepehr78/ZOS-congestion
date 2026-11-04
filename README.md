# Zeroth-Order Stackelberg Control in Combinatorial Congestion Games

Code for the NeurIPS 2026 paper *Zeroth-Order Stackelberg Control in Combinatorial Congestion
Games* by Saeed Masiha, Sepehr Elahi, Negar Kiyavash and Patrick Thiran (EPFL).

```bibtex
@inproceedings{masiha2026zeroth,
  title     = {Zeroth-Order Stackelberg Control in Combinatorial Congestion Games},
  author    = {Masiha, Saeed and Elahi, Sepehr and Kiyavash, Negar and Thiran, Patrick},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## What's in this repository

```
zos/
  zos.py           ZOS outer loop (Algorithm 1): two-point estimates over B_out directions, projection onto Theta
  frank_wolfe.py   FW-Equilibrium inner solver (Appendix B) and the exact FW gap
  lmo.py           exact LMOs (Dijkstra, ZDD argmin) and the subsampled LMO_m (best of m sampled strategies)
  zdd.py           ZDD construction with Graphillion (s-t paths, Hamiltonian s-t paths, Steiner cycles), counting, argmin
  samplers.py      US / UL / HL strategy samplers on the ZDD
  costs.py         cost model c_i = d_i (1 + C y_i / (theta_i + 1)), social cost F, Beckmann potential f
  projection.py    Euclidean projection onto Theta = {theta >= 0, 1^T theta = n}
  certify.py       certified social cost: exact-LMO re-solve of the follower to FW gap <= 1e-8
  scenarios.py     the three TNTP scenarios; instances in zos/data/scenario{1,2,3}.json
scripts/
  run_zos.py           run ZOS on a scenario with the paper's settings
  certify.py           certified social cost of saved iterates
  regression_check.py  compare a short run with trajectories recorded by the paper's code
tests/                 unit tests and the Scenario 1-2 regression test
```

The scenarios are fixed subgraphs of the Winnipeg, Chicago-Sketch and Philadelphia networks of the
[TNTP benchmark](https://github.com/bstabler/TransportationNetworks) (Appendix F.1):

| Scenario | Network | Strategies | \|V\| | \|E\| | \|S\| | C | Oracle in the paper |
|---|---|---|---|---|---|---|---|
| 1 | Winnipeg | s-t paths | 49 | 82 | 3.12e7 | 500 | Dijkstra |
| 2 | Chicago-Sketch | Hamiltonian s-t paths | 63 | 118 | 2.53e6 | 20 | exact ZDD argmin |
| 3 | Philadelphia | Steiner cycles (6 terminals) | 110 | 176 | 3.81e14 | 10 | HL sampling, m = 1000 |

Defaults follow the paper: T = 3000, K = 500, rho = 0.05, eta = 0.05, B_out = 4.

Note on the outer estimator: the experiments draw the directions as unnormalized Rademacher vectors
(entries ±1) with the factor 1/(2 rho) and project the queries onto Theta. This is the default. The
unit-sphere estimator with the factor d_theta/(2 rho), as written in Algorithm 1, is available with
`--directions sphere`. The inner FW solver uses the step sizes 2/(t+2) (cold start) and
2/(t+2+T) (warm start from the previous follower iterate). This is the schedule that produced the
reported results.

## Getting started

Requires Python 3.11–3.13. Dependencies are pinned to the versions used for the paper.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                       # about a minute; includes the Scenario 1-2 regression test
```

Run ZOS on a scenario with the paper's settings (here Scenario 1, seed 0) and certify the final iterate:

```bash
python scripts/run_zos.py --scenario 1 --seed 0 --out runs/s1_seed0.npz --certify
```

Use `--oracle {exact,US,UL,HL}` and `--m` to change the inner oracle, and `--K`, `--T`, `--rho`,
`--eta` and `--B-out` to change the settings. Thread count follows `--threads` or `OMP_NUM_THREADS`.
A full Scenario 1 run takes about an hour on 4 cores; Scenario 3 builds a ZDD with about 10^6
nodes (several minutes) and is best run on 16 or more cores.

The certified social cost is F(theta, y*) with the follower re-solved by fully corrective
Frank-Wolfe with an exact LMO until the exact FW gap is at most 1e-8. `--certify` reports it for
theta_K; for other saved iterates use

```bash
python scripts/certify.py runs/s1_seed0.npz --checkpoints 0 100 500
```

To compare a short run with the trajectories recorded by the code that produced the paper:

```bash
python scripts/regression_check.py s1 s2 s3_hl
```
