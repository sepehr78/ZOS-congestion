"""Small instances whose strategy families can be enumerated by brute force."""

import networkx as nx
import pytest

GRID = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4), first_label=1)
EDGES = list(GRID.edges())
EDGE_INDEX = {frozenset(e): i for i, e in enumerate(EDGES)}
SOURCE, TARGET, TERMINALS = 1, 12, [1, 6, 12]


def _edge_set(nodes, closed=False):
    pairs = zip(nodes, nodes[1:] + nodes[:1] if closed else nodes[1:])
    return frozenset(EDGE_INDEX[frozenset(p)] for p in pairs)


def enumerate_family(family):
    """All strategies of ``family`` on GRID, as frozensets of edge indices."""
    if family == "st_path":
        return {_edge_set(p) for p in nx.all_simple_paths(GRID, SOURCE, TARGET)}
    if family == "hamiltonian_path":
        return {_edge_set(p) for p in nx.all_simple_paths(GRID, SOURCE, TARGET)
                if len(p) == GRID.number_of_nodes()}
    if family == "steiner_cycle":
        return {_edge_set(c, closed=True) for c in nx.simple_cycles(GRID)
                if set(TERMINALS) <= set(c)}
    raise ValueError(family)


FAMILIES = ["st_path", "hamiltonian_path", "steiner_cycle"]


@pytest.fixture(scope="session")
def tiny():
    """family -> (ZDD of the family on GRID, enumerated strategies)."""
    from zos import build_zdd

    out = {}
    for family in FAMILIES:
        zdd = build_zdd(EDGES, family, SOURCE, TARGET, TERMINALS)
        out[family] = (zdd, enumerate_family(family))
    return out

