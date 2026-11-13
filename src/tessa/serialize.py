"""Plain-JSON form of the qubit graphs an :class:`~tessa.codec.Encoding` refers to.

A :class:`~tessa.graph.TessaGraph` is fully determined by its qubit count, its
undirected edges and which qubits are data qubits; a
:class:`~tessa.core.lattice.Lattice` by its rows and columns. The groups (data
qubit, ordered address qubits) are stored as well and checked on loading, so a
file can never silently decode with a different address-bit order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core.lattice import Lattice
from .graph import TessaGraph

__all__ = ["graph_from_dict", "graph_to_dict"]


def _groups(graph) -> list[dict[str, Any]]:
    return [
        {"data": int(g.data_qubit), "addresses": [int(a) for a in g.address_qubits]}
        for g in graph.groups
    ]


def graph_to_dict(graph: TessaGraph | Lattice) -> dict[str, Any]:
    """``graph`` as a JSON-serialisable dict; inverse of :func:`graph_from_dict`."""
    if isinstance(graph, Lattice):
        return {
            "kind": "lattice",
            "rows": graph.rows,
            "cols": graph.cols,
            "groups": _groups(graph),
        }
    if isinstance(graph, TessaGraph):
        return {
            "kind": "graph",
            "n_qubits": int(graph.n_qubits),
            "edges": [[int(a), int(b)] for a, b in sorted(graph.edges)],
            "data_qubits": [int(q) for q in graph.data_qubits],
            "groups": _groups(graph),
        }
    raise TypeError(f"cannot serialise a {type(graph).__name__}")


def graph_from_dict(data: Mapping[str, Any]) -> TessaGraph | Lattice:
    """Rebuild a graph written by :func:`graph_to_dict`."""
    kind = data.get("kind")
    if kind == "lattice":
        graph: TessaGraph | Lattice = Lattice(int(data["rows"]), int(data["cols"]))
    elif kind == "graph":
        graph = TessaGraph(
            int(data["n_qubits"]),
            [tuple(e) for e in data["edges"]],
            data_side={int(q) for q in data["data_qubits"]},
        )
    else:
        raise ValueError(f"unknown graph kind {kind!r}")
    if "groups" in data and _groups(graph) != [
        {"data": int(g["data"]), "addresses": [int(a) for a in g["addresses"]]}
        for g in data["groups"]
    ]:
        raise ValueError("the stored groups do not match the rebuilt graph")
    return graph
