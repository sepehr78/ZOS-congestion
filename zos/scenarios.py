"""The three TNTP scenarios of Section 6 (Appendix F.1, Table "TNTP scenarios").

=====  ==============  =========================  ====  ====  ==========  ===
 No.   network         strategy family            |V|   |E|   |S|          C
=====  ==============  =========================  ====  ====  ==========  ===
 1     Winnipeg        s-t paths                   49    82   3.12e7      500
 2     Chicago-Sketch  Hamiltonian s-t paths       63   118   2.53e6       20
 3     Philadelphia    Steiner cycles, 6 terminals 110   176   3.81e14      10
=====  ==============  =========================  ====  ====  ==========  ===

Each instance is a fixed connected subgraph of the undirected TNTP network (antiparallel arcs
merged) found by an offline randomized search. ``zos/data/scenario{1,2,3}.json`` stores the
node and edge lists in the order used for the paper's runs (the edge order fixes the coordinates
of theta), the endpoints or terminals, and the raw free-flow costs ``d_raw``; see the
``d_raw_note`` field of each file for how they were derived from the TNTP network files.
"""

import json
from dataclasses import dataclass
from importlib.resources import files

import networkx as nx
import numpy as np

from .costs import FractionalCost
from .lmo import SampledLMO, ShortestPathLMO, ZDDLMO
from .samplers import ZDDSampler
from .zdd import ZDD, build_zdd

# Oracle used for each scenario in the paper's main results.
PAPER_ORACLE = {1: "exact", 2: "exact", 3: "HL"}


@dataclass
class Scenario:
    number: int
    network: str
    family: str  # "st_path", "hamiltonian_path" or "steiner_cycle"
    graph: nx.Graph
    edges: list[tuple[int, int]]
    cost: FractionalCost
    source: int | None = None
    target: int | None = None
    terminals: list[int] | None = None
    num_strategies: int | None = None

    @property
    def n(self) -> int:
        """Number of resources (edges), the dimension of theta."""
        return len(self.edges)

    def build_zdd(self) -> ZDD:
        return build_zdd(self.edges, self.family, self.source, self.target, self.terminals)

    def exact_lmo(self, zdd: ZDD | None = None):
        """Dijkstra for s-t paths (Scenario 1), exact ZDD argmin otherwise."""
        if self.family == "st_path":
            return ShortestPathLMO(self.graph, self.edges, self.source, self.target)
        return ZDDLMO(zdd or self.build_zdd())

    def lmo(self, oracle: str = "exact", m: int = 1000, zdd: ZDD | None = None):
        """``oracle`` is "exact" or one of the sampling schemes "US", "UL", "HL" (LMO_m)."""
        if oracle == "exact":
            return self.exact_lmo(zdd)
        return SampledLMO(ZDDSampler(zdd or self.build_zdd(), oracle), m)


def load_scenario(number: int) -> Scenario:
    data = json.loads(files("zos").joinpath(f"data/scenario{number}.json").read_text())
    graph = nx.Graph()
    graph.add_nodes_from(data["nodes"])
    edges = [tuple(e) for e in data["edges"]]
    graph.add_edges_from(edges)
    return Scenario(
        number=number, network=data["network"], family=data["family"], graph=graph, edges=edges,
        cost=FractionalCost.from_raw(np.array(data["d_raw"]), data["C"]),
        source=data.get("source"), target=data.get("target"), terminals=data.get("terminals"),
        num_strategies=data["num_strategies"],
    )
