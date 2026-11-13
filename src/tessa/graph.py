"""The Tessa encoding on a general bipartite device graph.

The square lattice of :mod:`tessa.core` is a special case: the construction
works on **any bipartite coupling graph**. 2-colour the qubits into a *data* set
and an *address* set so every hardware edge joins one of each, and each data
qubit carries a uniformly controlled ``RY`` on its neighbouring address qubits
(``2**degree`` values). Readout is the same simultaneous partial tomography --
the decoding argument only ever needed bipartiteness.

IBM heavy-hex processors (``ibm_fez``, ``ibm_marrakesh``, ``ibm_kingston``, ...)
are bipartite, so this maps onto them natively. Their qubits have degree <= 3, so
per-qubit capacity (``2``, ``4`` or ``8`` values) is lower than on a square
lattice, but the encoding uses only native edges -- **no SWAPs**.

:class:`TessaGraph` duck-types :class:`tessa.core.Lattice` (it exposes
``groups``, ``n_qubits``, ``pack``/``unpack``, ``capacity``), so the encoder and
decoders work on it unchanged.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .core.lattice import DataGroup, _pack_values, _unpack_values

__all__ = [
    "TessaGraph",
    "graph_capacity",
    "graph_from_backend",
    "graph_from_coupling",
    "square_lattice_edges",
]


def _undirected_edges(edges) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for a, b in edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        out.add((a, b) if a < b else (b, a))
    return out


class TessaGraph:
    """The Tessa encoding on an arbitrary bipartite coupling graph.

    Parameters
    ----------
    n_qubits:
        Number of qubits (graph vertices ``0 .. n_qubits-1``).
    edges:
        Iterable of undirected ``(a, b)`` couplings.
    data_side:
        ``"max_capacity"`` (default) chooses, per connected component, the colour
        class that maximises encoded capacity.  Pass a set/list of qubit indices to
        force exactly which qubits are data qubits.
    """

    def __init__(
        self, n_qubits: int, edges, data_side: str | set[int] = "max_capacity"
    ) -> None:
        self.n_qubits = int(n_qubits)
        self._edges = _undirected_edges(edges)
        self._adj: dict[int, set[int]] = defaultdict(set)
        for a, b in self._edges:
            self._adj[a].add(b)
            self._adj[b].add(a)

        if isinstance(data_side, str):
            data_qubits = self._choose_data_side(mode=data_side)
        else:
            data_qubits = {int(q) for q in data_side}
            self._validate_bipartite(data_qubits)

        self.data_qubits = sorted(data_qubits)
        self.address_qubits = sorted(set(range(self.n_qubits)) - data_qubits)
        self.groups = self._build_groups()

    # -- bipartition -----------------------------------------------------
    def _components_coloring(self) -> dict[int, int]:
        color: dict[int, int] = {}
        for s in range(self.n_qubits):
            if s in color:
                continue
            color[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in self._adj[u]:
                    if v not in color:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        raise ValueError(
                            "coupling graph is not bipartite; the encoding needs a "
                            "data/address 2-colouring"
                        )
        return color

    def _components(self) -> tuple[dict[int, int], int]:
        comp_of: dict[int, int] = {}
        comp_id = 0
        seen: set[int] = set()
        for s in range(self.n_qubits):
            if s in seen:
                continue
            stack = [s]
            seen.add(s)
            comp_of[s] = comp_id
            while stack:
                u = stack.pop()
                for v in self._adj[u]:
                    if v not in seen:
                        seen.add(v)
                        comp_of[v] = comp_id
                        stack.append(v)
            comp_id += 1
        return comp_of, comp_id

    def _choose_data_side(self, mode: str = "max_capacity") -> set[int]:
        color = self._components_coloring()
        comp_of, n_comps = self._components()

        members_by_comp: dict[int, list[int]] = defaultdict(list)
        for q in range(self.n_qubits):
            members_by_comp[comp_of[q]].append(q)

        data: set[int] = set()
        for cid in range(n_comps):
            members = members_by_comp[cid]
            side0 = [q for q in members if color[q] == 0]
            side1 = [q for q in members if color[q] == 1]
            if mode == "max_capacity":
                cap0 = sum(1 << len(self._adj[q]) for q in side0)
                cap1 = sum(1 << len(self._adj[q]) for q in side1)
                data.update(side0 if cap0 >= cap1 else side1)
            else:  # "color0"
                data.update(side0)
        return data

    def _validate_bipartite(self, data_qubits: set[int]) -> None:
        for a, b in self._edges:
            if (a in data_qubits) == (b in data_qubits):
                raise ValueError(
                    f"edge ({a}, {b}) joins two same-role qubits; the chosen "
                    "data_side is not a valid bipartition"
                )

    def _build_groups(self) -> list[DataGroup]:
        groups = []
        for d in self.data_qubits:
            addr = tuple(sorted(self._adj[d]))
            groups.append(
                DataGroup(
                    site=d,  # use the data qubit index as the group key
                    data_qubit=d,
                    address_qubits=addr,
                    address_sites=addr,
                )
            )
        return groups

    # -- capacity / packing ----------------------------------------------
    @property
    def edges(self) -> set[tuple[int, int]]:
        """The undirected coupling edges of the device graph."""
        return set(self._edges)

    def neighbours(self, q: int) -> set[int]:
        """Neighbours of qubit ``q`` in the coupling graph."""
        return set(self._adj[int(q)])

    @property
    def capacity(self) -> int:
        return sum(g.n_values for g in self.groups)

    def capacity_breakdown(self) -> dict:
        by_deg: dict[int, int] = defaultdict(int)
        vals: dict[int, int] = defaultdict(int)
        for g in self.groups:
            by_deg[g.degree] += 1
            vals[g.degree] += g.n_values
        return {
            "n_qubits": self.n_qubits,
            "n_data_qubits": len(self.data_qubits),
            "n_address_qubits": len(self.address_qubits),
            "capacity": self.capacity,
            "by_degree": {
                d: {"data_qubits": by_deg[d], "values": vals[d]} for d in sorted(by_deg)
            },
        }

    def pack(self, values: np.ndarray) -> dict:
        """Split a value array into per-data-qubit chunks; see :meth:`Lattice.pack`."""
        return _pack_values(self.groups, self.capacity, values)

    def unpack(self, packed: dict) -> np.ndarray:
        """Inverse of :meth:`pack`."""
        return _unpack_values(self.groups, packed)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_coupling_map(cls, coupling_map, n_qubits: int, **kw) -> TessaGraph:
        edges = (
            list(coupling_map.get_edges())
            if hasattr(coupling_map, "get_edges")
            else list(coupling_map)
        )
        return cls(n_qubits, edges, **kw)

    @classmethod
    def from_backend(cls, backend, **kw) -> TessaGraph:
        """Bipartition a backend's coupling map using **physical** qubit indices.

        Keeping physical indexing is what makes the identity layout valid, so the
        encoding CX gates all sit on native edges and the transpiler inserts no
        SWAPs.
        """
        return cls.from_coupling_map(backend.coupling_map, backend.num_qubits, **kw)

    @classmethod
    def from_groups(cls, n_qubits: int, specs) -> TessaGraph:
        """Build directly from explicit ``(data_qubit, [address_qubits])`` specs."""
        edges = []
        data = set()
        for d, addr in specs:
            data.add(d)
            for a in addr:
                edges.append((d, a))
        return cls(n_qubits, edges, data_side=data)

    def connected_subgraph(self, size: int, start: int | None = None):
        """Return a compact ``TessaGraph`` on a connected induced subgraph.

        BFS-collects ``size`` qubits (starting from ``start`` or the highest-degree
        qubit), relabels them ``0 .. size-1`` and returns
        ``(subgraph, physical_qubits)`` where ``physical_qubits[i]`` is the original
        index of compact qubit ``i``.  Handy for running exact / noisy simulations
        on a small piece of a large device.
        """
        if start is None:
            start = max(range(self.n_qubits), key=lambda q: len(self._adj[q]))
        chosen: list[int] = []
        seen = {start}
        queue = deque([start])
        while queue and len(chosen) < size:
            u = queue.popleft()
            chosen.append(u)
            for v in sorted(self._adj[u]):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        chosen = chosen[:size]
        chosen_set = set(chosen)
        remap = {phys: i for i, phys in enumerate(chosen)}
        sub_edges = [
            (remap[a], remap[b])
            for a, b in self._edges
            if a in chosen_set and b in chosen_set
        ]
        return TessaGraph(len(chosen), sub_edges), chosen

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TessaGraph(n_qubits={self.n_qubits}, data={len(self.data_qubits)}, "
            f"address={len(self.address_qubits)}, capacity={self.capacity})"
        )


def graph_capacity(backend=None, coupling_map=None, n_qubits=None, **kw) -> int:
    """Maximum number of values encodable on a backend / coupling map."""
    if backend is not None:
        return TessaGraph.from_backend(backend, **kw).capacity
    return TessaGraph.from_coupling_map(coupling_map, n_qubits, **kw).capacity


def graph_from_coupling(edges, n_qubits: int | None = None, **kw) -> TessaGraph:
    """Tessa graph from a generic coupling map.

    ``edges`` is a list of qubit pairs a two-qubit gate can act on, e.g.
    ``[[0, 1], [1, 3], ...]`` (undirected). ``n_qubits`` defaults to the highest
    qubit index plus one.
    """
    edges = [tuple(int(x) for x in e) for e in edges]
    if n_qubits is None:
        n_qubits = (max((max(e) for e in edges), default=-1) + 1) if edges else 0
    return TessaGraph(n_qubits, edges, **kw)


def graph_from_backend(backend, **kw) -> TessaGraph:
    """Tessa graph from any backend exposing ``coupling_map`` + ``num_qubits``."""
    return TessaGraph.from_backend(backend, **kw)


def square_lattice_edges(rows: int, cols: int) -> tuple[list[tuple[int, int]], int]:
    """A generic square-lattice coupling map (row-major); returns ``(edges, n_qubits)``."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges, rows * cols
