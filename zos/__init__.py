"""Zeroth-order Stackelberg control in combinatorial congestion games (NeurIPS 2026).

Public API:

* :func:`zos` -- the outer zeroth-order loop (Algorithm 1),
* :func:`frank_wolfe` -- the inner Wardrop-equilibrium solver (Appendix B),
* the oracles :class:`ShortestPathLMO`, :class:`ZDDLMO` and :class:`SampledLMO`,
* :func:`build_zdd` and :class:`ZDDSampler` (US / UL / HL),
* :class:`FractionalCost`, :func:`project_theta`, :func:`certify`, :func:`load_scenario`.
"""

from .certify import Certificate, certify
from .costs import FractionalCost
from .frank_wolfe import frank_wolfe, fw_gap
from .lmo import SampledLMO, ShortestPathLMO, ZDDLMO
from .projection import project_simplex, project_theta
from .samplers import ZDDSampler
from .scenarios import PAPER_ORACLE, Scenario, load_scenario
from .zdd import ZDD, build_zdd
from .zos import ZOSResult, zos

__all__ = [
    "Certificate", "certify", "FractionalCost", "frank_wolfe", "fw_gap", "SampledLMO",
    "ShortestPathLMO", "ZDDLMO", "project_simplex", "project_theta", "ZDDSampler",
    "PAPER_ORACLE", "Scenario", "load_scenario", "ZDD", "build_zdd", "ZOSResult", "zos",
]
