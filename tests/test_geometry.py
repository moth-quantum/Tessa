"""Geometry and capacity: the checkerboard lattice and the bipartite device graph."""

from __future__ import annotations

import numpy as np
import pytest

from tessa import (
    DataGroup,
    Lattice,
    TessaGraph,
    graph_capacity,
    is_data_site,
    lattice_capacity,
    square_lattice_edges,
)
from tessa.core.lattice import encoding_coupling_map, encoding_edges


def test_checkerboard_sites():
    assert is_data_site(0, 0)
    assert not is_data_site(0, 1)
    lat = Lattice(3, 4)
    assert lat.ascii().splitlines() == ["D A D A", "A D A D", "D A D A"]
    assert lat.n_qubits == 12
    assert len(lat.groups) == 6


def test_capacity_formula_3x4():
    """Capacity == sum of 2**degree over data qubits, split by type."""
    lat = Lattice(3, 4)
    bd = lat.capacity_breakdown()
    assert bd["corner"] == 2 and bd["values_from_corners"] == 2 * 2**2
    assert bd["edge"] == 3 and bd["values_from_edges"] == 3 * 2**3
    assert bd["interior"] == 1 and bd["values_from_interior"] == 1 * 2**4
    assert lat.capacity == 2 * 4 + 3 * 8 + 1 * 16 == 48
    assert lattice_capacity(3, 4) == 48
    assert bd["n_data_qubits"] + bd["n_address_qubits"] == bd["n_qubits"] == 12


@pytest.mark.parametrize(
    ("rows", "cols", "capacity"),
    [(1, 1, 1), (2, 2, 8), (3, 3, 32), (3, 4, 48), (4, 4, 72), (12, 10, 792)],
)
def test_lattice_capacities(rows, cols, capacity):
    assert lattice_capacity(rows, cols) == capacity


def test_lattice_rejects_degenerate_shape():
    with pytest.raises(ValueError):
        Lattice(0, 4)


def test_pack_unpack_roundtrip():
    lat = Lattice(3, 4)
    v = np.arange(lat.capacity, dtype=float)
    assert np.array_equal(lat.unpack(lat.pack(v)), v)
    with pytest.raises(ValueError):
        lat.pack(v[:-1])


def test_address_qubits_are_neighbours():
    lat = Lattice(4, 4)
    for g in lat.groups:
        r, c = g.site
        for ar, ac in g.address_sites:
            assert abs(ar - r) + abs(ac - c) == 1
            assert not is_data_site(ar, ac)
        assert g.n_values == 1 << g.degree


# ----------------------------------------------------------------------
# TessaGraph
# ----------------------------------------------------------------------
def test_graph_matches_lattice_capacity():
    """A square-lattice coupling map bipartitions to the same capacity."""
    edges, n = square_lattice_edges(4, 4)
    g = TessaGraph(n, edges)
    assert g.capacity == Lattice(4, 4).capacity == 72
    assert g.n_qubits == n == 16


def test_graph_rejects_odd_cycle():
    """A triangle is not bipartite, so there is no data/address 2-colouring."""
    with pytest.raises(ValueError, match="not bipartite"):
        TessaGraph(3, [(0, 1), (1, 2), (2, 0)])


def test_graph_rejects_invalid_forced_bipartition():
    with pytest.raises(ValueError, match="same-role"):
        TessaGraph(3, [(0, 1), (1, 2)], data_side={0, 1})


def test_graph_picks_max_capacity_side():
    """A star: the single centre (degree 3 -> 8 values) beats 3 leaves (3 x 2)."""
    g = TessaGraph(4, [(0, 1), (0, 2), (0, 3)])
    assert g.data_qubits == [0]
    assert g.capacity == 8


def test_from_groups_and_edges():
    g = TessaGraph.from_groups(4, [(0, [1, 2, 3])])
    assert g.data_qubits == [0]
    assert g.groups[0].address_qubits == (1, 2, 3)
    assert encoding_edges(g) == {(0, 1), (0, 2), (0, 3)}
    # The coupling map lists both orientations of every edge.
    assert len(encoding_coupling_map(g)) == 6


def test_heavyhex_capacity_448(fez_graph):
    assert fez_graph.n_qubits == 156
    assert (len(fez_graph.data_qubits), len(fez_graph.address_qubits)) == (64, 92)
    assert fez_graph.capacity == 448
    bd = fez_graph.capacity_breakdown()
    assert bd["capacity"] == 448
    assert sum(v["values"] for v in bd["by_degree"].values()) == 448


def test_miami_capacity_792(fake_miami):
    assert graph_capacity(fake_miami) == 792 == Lattice(12, 10).capacity


def test_connected_subgraph_is_compact(fez_graph):
    sub, phys = fez_graph.connected_subgraph(12)
    assert sub.n_qubits == 12 == len(phys)
    assert len(set(phys)) == 12
    assert sorted(sub.data_qubits + sub.address_qubits) == list(range(12))
    assert sub.capacity > 0


def test_datagroup_kinds():
    assert DataGroup((0, 0), 0, (1, 2), (1, 2)).kind == "corner"
    assert DataGroup((0, 0), 0, (1, 2, 3), (1, 2, 3)).kind == "edge"
    assert DataGroup((0, 0), 0, (1, 2, 3, 4), (1, 2, 3, 4)).kind == "interior"
