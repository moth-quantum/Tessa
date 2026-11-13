"""Square-lattice Tessa geometry.

We place qubits on a ``rows x cols`` square lattice in a checkerboard pattern::

    D A D A ...
    A D A D ...
    D A D A ...

where ``D`` are *data* qubits (they carry the encoded angles) and ``A`` are
*address* qubits (they form the shared control/superposition register).  The
top-left site ``(0, 0)`` is always a data qubit, i.e. a site ``(r, c)`` is a data
site iff ``(r + c)`` is even.

Because the lattice is a checkerboard, every neighbour of a data qubit is an
address qubit and vice-versa.  A data qubit therefore "owns" a small local
address register consisting of its von-Neumann neighbours (up / down / left /
right).  Interior data qubits have 4 neighbours, edge data qubits have 3 and
corner data qubits have 2.  A data qubit with ``m`` neighbouring address qubits
encodes ``2**m`` real numbers -- one angle for every computational basis state of
its local address register.

Address qubits are *shared* between the (up to four) data qubits that surround
them, which is exactly what lets a small number of qubits store a lot of data.

This module only deals with the classical geometry / bookkeeping.  The circuit
construction lives in :mod:`tessa.core.encode` and the readout in
:mod:`tessa.core.decode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical ordering of the four possible neighbour directions.  The position of
# a neighbour in this list fixes which *bit* of the local address integer it
# represents.
_DIRECTIONS = (
    (-1, 0),  # up
    (0, -1),  # left
    (0, 1),  # right
    (1, 0),  # down
)

__all__ = [
    "DataGroup",
    "Lattice",
    "encoding_coupling_map",
    "encoding_edges",
    "is_data_site",
    "lattice_capacity",
    "rectangular_layout",
]


def _pack_values(groups, capacity: int, values: np.ndarray) -> dict:
    """Split a flat or paired value array into per-group chunks.

    ``values`` has shape ``(capacity,)`` for the single-value encoding or
    ``(capacity, 2)`` for the two-value (Bloch-slot) encoding -- either way,
    values are handed to ``groups`` in order, ``2**degree`` per group. Shared by
    :class:`Lattice` and :class:`~tessa.graph.TessaGraph` so both topologies
    accept the same payload shapes.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(f"values must be 1-D or (n, 2)-shaped, got shape {values.shape}")
    if values.shape[0] != capacity:
        raise ValueError(f"expected {capacity} values, got {values.shape[0]}")
    packed: dict = {}
    idx = 0
    for g in groups:
        packed[g.site] = values[idx : idx + g.n_values].copy()
        idx += g.n_values
    return packed


def _unpack_values(groups, packed: dict) -> np.ndarray:
    """Inverse of :func:`_pack_values` -- concatenate per-group chunks in order."""
    chunks = []
    for g in groups:
        chunk = np.asarray(packed[g.site], dtype=float)
        if chunk.shape[0] != g.n_values:
            raise ValueError(
                f"site {g.site} expects {g.n_values} values, got {chunk.shape[0]}"
            )
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def is_data_site(row: int, col: int) -> bool:
    """Return ``True`` if site ``(row, col)`` holds a *data* qubit."""
    return (row + col) % 2 == 0


@dataclass(frozen=True)
class DataGroup:
    """A single data qubit together with its local address register.

    Attributes
    ----------
    site:
        Key identifying the group -- ``(row, col)`` on a :class:`Lattice`, or the
        data qubit index on a :class:`~tessa.graph.TessaGraph`.
    data_qubit:
        Global qubit index of the data qubit.
    address_qubits:
        Global qubit indices of the neighbouring address qubits, ordered so that
        ``address_qubits[j]`` carries bit ``j`` of the local address integer.
    address_sites:
        Positions matching ``address_qubits`` (same order).
    """

    site: tuple[int, int] | int
    data_qubit: int
    address_qubits: tuple[int, ...]
    address_sites: tuple[tuple[int, int], ...] | tuple[int, ...]

    @property
    def degree(self) -> int:
        """Number of neighbouring address qubits."""
        return len(self.address_qubits)

    @property
    def n_values(self) -> int:
        """Number of real values this data qubit encodes (``2**degree``)."""
        return 1 << self.degree

    @property
    def kind(self) -> str:
        return {2: "corner", 3: "edge", 4: "interior"}.get(self.degree, "?")


@dataclass
class Lattice:
    """Checkerboard D/A lattice with global qubit indexing.

    Qubits are numbered row-major: ``qubit(r, c) = r * cols + c``.  Every lattice
    site is one physical qubit, so ``n_qubits = rows * cols``.
    """

    rows: int
    cols: int
    groups: list[DataGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.groups = self._build_groups()

    def qubit_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def site_of_qubit(self, q: int) -> tuple[int, int]:
        return divmod(q, self.cols)

    @property
    def n_qubits(self) -> int:
        return self.rows * self.cols

    def _neighbours(self, row: int, col: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in _DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                out.append((r, c))
        return out

    def _build_groups(self) -> list[DataGroup]:
        groups: list[DataGroup] = []
        for row in range(self.rows):
            for col in range(self.cols):
                if not is_data_site(row, col):
                    continue
                # Every neighbour of a data site is an address site.
                addr_sites = tuple(self._neighbours(row, col))
                addr_qubits = tuple(self.qubit_index(r, c) for r, c in addr_sites)
                groups.append(
                    DataGroup(
                        site=(row, col),
                        data_qubit=self.qubit_index(row, col),
                        address_qubits=addr_qubits,
                        address_sites=addr_sites,
                    )
                )
        return groups

    @property
    def capacity(self) -> int:
        """Total number of real values the lattice can encode."""
        return sum(g.n_values for g in self.groups)

    def capacity_breakdown(self) -> dict[str, int]:
        """Return counts and value-capacity split by data-qubit type."""
        plural = {"corner": "corners", "edge": "edges", "interior": "interior"}
        out: dict[str, int] = {
            "n_qubits": self.n_qubits,
            "n_data_qubits": len(self.groups),
            "n_address_qubits": self.n_qubits - len(self.groups),
            "corner": 0,
            "edge": 0,
            "interior": 0,
            "values_from_corners": 0,
            "values_from_edges": 0,
            "values_from_interior": 0,
            "capacity": self.capacity,
        }
        for g in self.groups:
            out[g.kind] += 1
            out[f"values_from_{plural[g.kind]}"] += g.n_values
        return out

    def pack(self, values: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """Split a value array into per-data-qubit chunks.

        ``values`` has shape ``(capacity,)`` (single value per slot) or
        ``(capacity, 2)`` (a value pair per slot, for the two-value Bloch-slot
        encoding).  Values are assigned to data qubits in row-major site order;
        each data qubit receives ``2**degree`` consecutive rows.
        """
        return _pack_values(self.groups, self.capacity, values)

    def unpack(self, packed: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        """Inverse of :meth:`pack` -- concatenate per-site chunks in site order."""
        return _unpack_values(self.groups, packed)

    def ascii(self) -> str:
        """ASCII picture of the lattice (D / A)."""
        return "\n".join(
            " ".join("D" if is_data_site(r, c) else "A" for c in range(self.cols))
            for r in range(self.rows)
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Lattice(rows={self.rows}, cols={self.cols}, "
            f"n_qubits={self.n_qubits}, n_data={len(self.groups)}, "
            f"capacity={self.capacity})"
        )


def lattice_capacity(rows: int, cols: int) -> int:
    """Maximum number of real values a ``rows x cols`` D/A lattice can hold.

    Use this to size an input before encoding.
    """
    return Lattice(rows, cols).capacity


def rectangular_layout(
    lattice: Lattice,
    device_rows: int,
    device_cols: int,
    origin: tuple[int, int] = (0, 0),
) -> list[int]:
    """``initial_layout`` placing ``lattice`` on a sub-block of a square-lattice device.

    Virtual qubit ``i`` (lattice site ``(r, c)``) maps to physical qubit
    ``(r + r0) * device_cols + (c + c0)`` for ``origin=(r0, c0)``.  Both grids are
    row-major square lattices, so every nearest-neighbour edge survives the move
    and the transpiler needs no routing -- placing the full lattice at the origin
    gives the identity layout and zero SWAPs.

    This is device-agnostic arithmetic: it applies to any chip whose coupling map
    is a row-major rectangular grid.  ``ibm_miami``, for instance, is 12 x 10, so
    ``rectangular_layout(Lattice(12, 10), 12, 10) == list(range(120))``.

    ``layout[i]`` is the physical qubit for virtual qubit ``i`` -- the format
    Qiskit's ``transpile(..., initial_layout=...)`` expects.
    """
    r0, c0 = origin
    if r0 + lattice.rows > device_rows or c0 + lattice.cols > device_cols:
        raise ValueError(
            f"lattice {lattice.rows}x{lattice.cols} at origin {origin} does not "
            f"fit on a {device_rows}x{device_cols} device"
        )
    layout = [0] * lattice.n_qubits
    for i in range(lattice.n_qubits):
        r, c = lattice.site_of_qubit(i)
        layout[i] = (r + r0) * device_cols + (c + c0)
    return layout


def encoding_edges(topology) -> set[tuple[int, int]]:
    """Undirected data<->address qubit pairs that carry a ``CX`` in the encoding.

    Works for anything exposing ``groups`` -- a :class:`Lattice` or a
    :class:`~tessa.graph.TessaGraph`.
    """
    edges: set[tuple[int, int]] = set()
    for g in topology.groups:
        for a in g.address_qubits:
            lo, hi = sorted((g.data_qubit, a))
            edges.add((lo, hi))
    return edges


def encoding_coupling_map(topology) -> list[list[int]]:
    """Bidirectional coupling map (list of ``[a, b]``) for :func:`encoding_edges`."""
    cm: list[list[int]] = []
    for a, b in sorted(encoding_edges(topology)):
        cm.append([a, b])
        cm.append([b, a])
    return cm
