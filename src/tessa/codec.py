"""The encode / decode protocol: values in, circuit out, values back.

::

    circuit, enc = encode(graph, values)          # classical -> quantum
    ...                                           # your own gates here
    values_out   = decode(circuit, enc, sampler)  # quantum -> classical

:func:`encode` returns a bare, measurement-free circuit plus an
:class:`Encoding` -- the auxiliary data (which graph, which mode, what the
values were scaled by) that :func:`decode` needs to read it back. Nothing is
executed until :func:`decode`, and anything may happen to the circuit in
between.

:func:`decode` appends the measurement settings the mode calls for, runs them on
a **sampler object** (see :mod:`tessa.backends`) and inverts the encoding.
``per_group=True`` (default) cuts the circuit into one sub-circuit per data
qubit and simulates those; ``per_group=False`` submits it whole, which is how
hardware runs it. The cells commute, so the two agree up to shot noise and
crosstalk. Each cell also feels its *halo* (see :func:`cell_halo`): the noise of
the gates neighbouring cells apply to the shared address qubits. By default that
noise is traced onto the address qubits, so the halo qubits themselves are never
simulated (see :func:`decode`). The paper's simulations used bare cells
(``halo=None``).

:func:`save_encoding` / :func:`load_encoding` store an :class:`Encoding` as plain
JSON, so measured counts can be decoded later with :func:`decode_from_counts`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from .backends.base import counts_of, device_of
from .core.decode import _bit_table, _sampled_configurations, decode_cells
from .core.encode import build_tessa_circuit
from .core.lattice import DataGroup
from .core.mode import DEFAULT_CLAMP, get_mode
from .noise import halo_label
from .serialize import graph_from_dict, graph_to_dict

__all__ = [
    "DEFAULT_CLAMP",
    "Encoding",
    "cell_circuit",
    "cell_halo",
    "decode",
    "decode_from_cell_counts",
    "decode_from_counts",
    "encode",
    "load_encoding",
    "measurement_circuits",
    "save_encoding",
    "transpile_to_backend",
]


@dataclass(frozen=True)
class Encoding:
    """What :func:`decode` needs to invert an :func:`encode` -- the auxiliary data."""

    graph: object
    mode: str
    #: One clamp per field: how far ``theta`` and ``phi`` stay off the ends of
    #: their ranges. See :mod:`tessa.core.mode`.
    clamps: tuple[float, ...]
    #: ``(lo, hi)`` per field: what ``[0, 1]`` on the circuit means in real units.
    ranges: tuple[tuple[float, float], ...]
    #: Shape of the input field(s), and how many of the capacity slots they filled.
    shape: tuple[int, ...]
    n: int

    @property
    def n_fields(self) -> int:
        return get_mode(self.mode).n_fields

    @property
    def data_qubits(self) -> list[int]:
        return [g.data_qubit for g in self.graph.groups]

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form: the graph (edges and data/address assignment) and every
        setting :func:`decode_from_counts` needs. Inverse of :meth:`from_dict`."""
        return {
            "graph": graph_to_dict(self.graph),
            "mode": self.mode,
            "clamps": [float(c) for c in self.clamps],
            "ranges": [[float(lo), float(hi)] for lo, hi in self.ranges],
            "shape": [int(k) for k in self.shape],
            "n": int(self.n),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Encoding:
        """Rebuild an :class:`Encoding` written by :meth:`to_dict`."""
        return cls(
            graph=graph_from_dict(data["graph"]),
            mode=str(data["mode"]),
            clamps=tuple(float(c) for c in data["clamps"]),
            ranges=tuple((float(lo), float(hi)) for lo, hi in data["ranges"]),
            shape=tuple(int(k) for k in data["shape"]),
            n=int(data["n"]),
        )


def _normalise(values, given_range) -> tuple[np.ndarray, tuple[float, float]]:
    flat = np.asarray(values, dtype=float).ravel()
    lo, hi = given_range if given_range is not None else (flat.min(), flat.max())
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((flat - lo) / (hi - lo), 0.0, 1.0), (lo, hi)


def encode(
    graph,
    values,
    values_b=None,
    *,
    mode: str | None = None,
    clamp: float | None = None,
    clamp_b: float | None = None,
    value_range=None,
    value_range_b=None,
    compression: float = 0.0,
    strategy: str = "parallel",
) -> tuple[QuantumCircuit, Encoding]:
    """Encode one or two fields as a circuit plus the :class:`Encoding` to read it.

    Parameters
    ----------
    graph:
        An :class:`~tessa.TessaGraph` (or :class:`~tessa.Lattice`) whose
        qubit indices are the target device's own -- see
        :func:`~tessa.graph_from_backend`.
    values, values_b:
        Arrays of any shape with at most ``graph.capacity`` entries; they are
        flattened and zero-padded to capacity. A second field selects a pair
        mode, storing it in each cell's phase at no extra two-qubit gate cost.
    mode:
        ``"single"``, ``"pair"`` or ``"pair_full"`` (see
        :mod:`tessa.core.mode`); inferred from whether ``values_b`` is given,
        so it is only worth passing to ask for ``"pair_full"``.
    clamp, clamp_b:
        Per-field clamps, holding ``theta`` and ``phi`` off the ends of their
        angle ranges. ``clamp`` defaults to ``0`` for one field and
        :data:`DEFAULT_CLAMP` for two; ``clamp_b`` defaults to ``0`` on the
        half-circle and to ``clamp`` on the full one. See
        :mod:`tessa.core.mode`.
    value_range, value_range_b:
        ``(lo, hi)`` mapped onto the full angle range. Defaults to each field's
        own min/max, so a field round-trips in its original units; pass a fixed
        range when several encodings must share a scale.
    compression, strategy:
        See :func:`~tessa.core.encode.build_tessa_circuit`.
    """
    if mode is None:
        mode = "single" if values_b is None else "pair"
    mode = get_mode(mode)
    clamps = mode.resolve_clamps(clamp, clamp_b)
    if (values_b is None) == mode.is_pair:
        raise ValueError(f"mode {mode.name!r} needs exactly {mode.n_fields} field(s)")

    shape = np.asarray(values).shape
    fields = [(values, value_range)]
    if mode.is_pair:
        if np.asarray(values_b).shape != shape:
            raise ValueError("the two fields must have the same shape")
        fields.append((values_b, value_range_b))

    payload = np.zeros((graph.capacity, len(fields)))
    ranges = []
    for j, (field, rng) in enumerate(fields):
        flat01, rng = _normalise(field, rng)
        if flat01.size > graph.capacity:
            raise ValueError(f"{flat01.size} values exceed capacity {graph.capacity}")
        payload[: flat01.size, j] = flat01
        ranges.append(rng)

    circuit = build_tessa_circuit(
        graph,
        payload if mode.is_pair else payload[:, 0],
        mode=mode,
        clamps=clamps,
        compression=compression,
        strategy=strategy,
    )
    n = int(np.prod(shape))
    return circuit, Encoding(graph, mode.name, clamps, tuple(ranges), shape, n)


# ----------------------------------------------------------------------
# Cutting the circuit into independent cells
# ----------------------------------------------------------------------
def cell_halo(graph, group: DataGroup, shape: str = "diamond") -> tuple[int, ...]:
    """The neighbouring data qubits a cell is simulated with.

    ``shape="cross"`` keeps the data qubits that share exactly one
    address qubit with ``group``. On a square lattice that is the data qubit
    straight across each address, which gives a 9-qubit cross::

            D
            A
        D A D A D
            A
            D

    ``shape="diamond"`` (default) keeps every data qubit that shares an address qubit with
    ``group``. On a square lattice that adds the four diagonal data qubits,
    which share two address qubits, for 13 qubits::

            D
          D A D
        D A D A D
          D A D
            D

    On a heavy-hex graph no two data qubits share two addresses, so the two
    shapes are the same.
    """
    if shape not in ("cross", "diamond"):
        raise ValueError(f"unknown halo shape {shape!r}; use 'cross' or 'diamond'")
    own = set(group.address_qubits)
    halo: list[int] = []
    for a in group.address_qubits:
        for other in graph.groups:
            d = other.data_qubit
            if d == group.data_qubit or d in halo or a not in other.address_qubits:
                continue
            if shape == "diamond" or len(own.intersection(other.address_qubits)) == 1:
                halo.append(d)
    return tuple(halo)


def cell_circuit(
    circuit: QuantumCircuit, group: DataGroup, halo: tuple[int, ...] = ()
) -> QuantumCircuit:
    """The part of ``circuit`` acting on one data qubit and its address qubits.

    Every gate whose qubits all lie inside the cell is kept, in order, remapped
    onto a compact register with the data qubit at index 0, its address qubits
    next and the ``halo`` data qubits (see :func:`cell_halo`) last. Gates
    reaching outside are dropped -- they belong to neighbouring cells, which
    touch the shared address qubits only as controls. A halo keeps the
    neighbours' gates on the shared address qubits, so their noise reaches the
    cell, and drops only the neighbours' gates on address qubits further out.
    """
    qubits = [group.data_qubit, *group.address_qubits, *halo]
    position = {q: i for i, q in enumerate(qubits)}
    cell = QuantumCircuit(len(qubits))
    for instruction in circuit.data:
        idx = [circuit.find_bit(q).index for q in instruction.qubits]
        if all(q in position for q in idx):
            cell.append(instruction.operation, [position[q] for q in idx])
    return cell


def _cell_group(group: DataGroup) -> DataGroup:
    """``group`` re-indexed onto the compact register of :func:`cell_circuit`."""
    return DataGroup(
        site=group.site,
        data_qubit=0,
        address_qubits=tuple(range(1, group.degree + 1)),
        address_sites=group.address_sites,
    )


def _measured(circuit: QuantumCircuit, data_qubits, basis: str) -> QuantumCircuit:
    """``circuit`` with a basis change on the data qubits, then ``measure_all``.

    Address qubits are never rotated: they carry no payload, and turning them
    would break the per-cell independence.
    """
    out = circuit.copy()
    if basis == "x":
        out.h(data_qubits)
    elif basis == "y":
        out.sdg(data_qubits)
        out.h(data_qubits)
    out.measure_all()
    return out


def measurement_circuits(
    circuit: QuantumCircuit, encoding: Encoding
) -> dict[str, QuantumCircuit]:
    """The whole-chip circuits ``decode(..., per_group=False)`` runs, keyed by basis.

    For submitting a hardware job yourself and decoding it later: run each circuit
    with the identity layout (``initial_layout=range(graph.n_qubits)``), collect
    the counts per basis, and pass them to :func:`decode_from_counts`. A job that
    waits in a queue then need not hold a Python session open.
    """
    if circuit.num_clbits:
        raise ValueError("pass the unmeasured circuit returned by encode")
    return {
        basis: _measured(circuit, encoding.data_qubits, basis)
        for basis in get_mode(encoding.mode).bases
    }


def _trace_halo(circuit: QuantumCircuit, halo) -> QuantumCircuit:
    """``circuit`` with its ``halo`` qubits traced out into noise on the addresses.

    A halo data qubit meets the cell only through ``cz`` gates on shared address
    qubits. Those gates are diagonal on the address, and an address qubit gets no
    other gate after its first ``H``, so its populations -- all that is measured
    -- never depend on the halo qubit. Only the noise of those gates reaches the
    cell. Each such ``cz`` therefore becomes an identity on the address that
    carries that noise (labelled :func:`~tessa.noise.halo_label`), and every
    other operation on a halo qubit is dropped.
    """
    halo = set(halo)
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        idx = [circuit.find_bit(q).index for q in instruction.qubits]
        touched = [q for q in idx if q in halo]
        if not touched:
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
        elif instruction.operation.name == "cz" and len(touched) == 1:
            (address,) = (q for q in idx if q not in halo)
            out.append(UnitaryGate(np.eye(2), label=halo_label(touched[0])), [address])
    return out


def _check_traceable(sampler) -> None:
    """Fail unless ``sampler``'s noise model can stand in for a traced halo."""
    if not hasattr(sampler, "noise_model"):
        raise ValueError(
            "per_group=True with halo='traced' simulates each cell, so it needs a "
            "simulator sampler from tessa.backends (aer_sampler, noisy_sampler, "
            "calibrated_sampler, fake_sampler); on hardware, run the whole circuit "
            "with per_group=False"
        )
    nm = sampler.noise_model
    prefix = halo_label(0).removesuffix("0")
    if nm is not None and not any(
        name.startswith(prefix) for name in nm.noise_instructions
    ):
        raise ValueError(
            "halo='traced' needs the noise model to carry halo_label errors; "
            "build it with build_average_noise_model or build_local_noise_model"
        )


# ----------------------------------------------------------------------
# Running and decoding
# ----------------------------------------------------------------------
def _run(sampler, jobs, shots, optimization_level) -> list[dict]:
    """Transpile every ``(circuit, layout, traced)`` and run them in one submission.

    ``traced`` lists the qubits :func:`_trace_halo` removes after transpiling.
    One call rather than one per circuit: a single job on hardware, and on a
    simulator the noise model is set up once instead of once per cell.
    """
    device = device_of(sampler)
    transpiled = []
    for circuit, layout, traced in jobs:
        tqc = transpile(
            circuit,
            device,
            initial_layout=layout,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )
        transpiled.append(_trace_halo(tqc, traced) if traced else tqc)
    result = sampler.run(transpiled, shots=shots).result()
    return [counts_of(result, i) for i in range(len(transpiled))]


def decode(
    circuit: QuantumCircuit,
    encoding: Encoding,
    sampler,
    *,
    per_group: bool = True,
    halo: str | None = "traced",
    shots: int = 4096,
    optimization_level: int = 1,
    return_counts: bool = False,
):
    """Measure ``circuit`` on ``sampler`` and recover the encoded field(s).

    ``circuit`` must be measurement-free; the settings the mode needs are
    appended here. Returns the reconstruction in the input's own shape and units
    -- a single array, or one per field for a pair mode. With ``return_counts``
    the raw counts come along: ``{basis: counts}`` for a joint run,
    ``{site: {basis: counts}}`` per group.

    ``per_group=True`` is for simulation; on hardware, pass ``per_group=False``.
    ``halo`` (per group only) sets how each cell feels its neighbours.
    ``"traced"`` (default) gives the ``"diamond"`` result at the bare cell's
    cost: the halo qubits are traced out, and only the noise their gates leave
    on the shared address qubits is kept (see :func:`_trace_halo`). It needs a
    simulator sampler from :mod:`tessa.backends`, and it is exact for the
    circuit :func:`encode` builds and those samplers' noise models. A gate you
    add to an address qubit that is not diagonal makes it approximate.
    ``"diamond"`` and ``"cross"`` simulate the halo qubits in full (see
    :func:`cell_halo`), and ``None`` runs the bare cell.

    The simulations reported in the paper (the local-noise and noiseless
    reconstructions of the QCAT hardware experiment) were made with bare cells,
    i.e. ``halo=None``; pass it explicitly to reproduce them. The default
    ``"traced"`` was added afterwards and gives slightly different noisy results.
    """
    if circuit.num_clbits:
        raise ValueError(
            "decode measures the circuit itself; pass the unmeasured circuit"
        )
    graph = encoding.graph
    mode = get_mode(encoding.mode)
    device = device_of(sampler)
    n_device = getattr(device, "num_qubits", None)
    if n_device is not None and graph.n_qubits > n_device:
        raise ValueError(
            f"graph uses {graph.n_qubits} qubits but the device has {n_device}; "
            "build the graph from this device with graph_from_backend"
        )

    if per_group:
        if halo == "traced":
            _check_traceable(sampler)
        keys, jobs = [], []
        n_cell = {}
        for g in graph.groups:
            shape = "diamond" if halo == "traced" else halo
            extra = cell_halo(graph, g, shape) if shape else ()
            traced = extra if halo == "traced" else ()
            cell = cell_circuit(circuit, g, extra)
            n_cell[g.site] = cell.num_qubits
            layout = [g.data_qubit, *g.address_qubits, *extra]
            for basis in mode.bases:
                keys.append((g.site, basis))
                jobs.append((_measured(cell, [0], basis), layout, traced))
        every = _run(sampler, jobs, shots, optimization_level)

        counts = {g.site: {} for g in graph.groups}
        for (site, basis), c in zip(keys, every, strict=True):
            counts[site][basis] = c
        packed = {}
        for g in graph.groups:
            packed.update(
                decode_cells(
                    counts[g.site],
                    [_cell_group(g)],
                    n_cell[g.site],
                    mode,
                    encoding.clamps,
                )
            )
        values = _restore(graph.unpack(packed), encoding)
    else:
        layout = list(range(graph.n_qubits))
        jobs = [
            (qc, layout, ()) for qc in measurement_circuits(circuit, encoding).values()
        ]
        counts = dict(
            zip(mode.bases, _run(sampler, jobs, shots, optimization_level), strict=True)
        )
        values = decode_from_counts(counts, encoding)
    return (values, counts) if return_counts else values


def decode_from_counts(
    counts_by_basis: dict[str, dict], encoding: Encoding, *, unsampled: str = "zero"
):
    """Invert an encoding from counts you measured yourself.

    ``counts_by_basis`` are the counts of the whole encoded circuit, one entry
    per basis the mode asks for.

    An address configuration of a cell that received no shot in a basis has no
    estimate. With ``unsampled="zero"`` (the default, and what every result of the
    paper used) its Pauli expectations are taken as 0, so in the single encoding
    the value decodes to the middle of its range (0.5 for ``value_range=(0, 1)``)
    without a warning. With ``unsampled="nan"`` every value of such a configuration
    (both fields in the pair encodings) is returned as NaN instead.
    """
    if unsampled not in ("zero", "nan"):
        raise ValueError(f"unsampled must be 'zero' or 'nan', not {unsampled!r}")
    graph = encoding.graph
    packed = decode_cells(
        counts_by_basis,
        graph.groups,
        graph.n_qubits,
        encoding.mode,
        encoding.clamps,
    )
    if unsampled == "nan":
        for basis in get_mode(encoding.mode).bases:
            table = _bit_table(counts_by_basis[basis], graph.n_qubits)
            for g in graph.groups:
                seen = _sampled_configurations(*table, g)
                values = np.asarray(packed[g.site], dtype=float)
                mask = seen if values.ndim == 1 else seen[:, None]
                packed[g.site] = np.where(mask, values, np.nan)
    return _restore(graph.unpack(packed), encoding)


def decode_from_cell_counts(
    counts_by_site: Mapping, encoding: Encoding, n_cell_qubits: Mapping | None = None
):
    """Invert an encoding from per-cell counts, as ``decode(..., per_group=True,
    return_counts=True)`` returns them: ``{site: {basis: counts}}``.

    Each cell's counts are over its own compact register (data qubit, then its
    address qubits, then any halo qubits). ``n_cell_qubits`` gives the register
    size per site; by default the bare cell, ``degree + 1`` (``halo=None``, as in
    the paper's simulations). Site keys may also be given as strings (as after a
    JSON round trip).
    """
    mode = get_mode(encoding.mode)
    packed = {}
    for g in encoding.graph.groups:
        counts = counts_by_site.get(g.site, counts_by_site.get(str(g.site)))
        if counts is None:
            raise KeyError(f"no counts for cell {g.site}")
        n = g.degree + 1 if n_cell_qubits is None else n_cell_qubits[g.site]
        packed.update(decode_cells(counts, [_cell_group(g)], n, mode, encoding.clamps))
    return _restore(encoding.graph.unpack(packed), encoding)


def _restore(payload01: np.ndarray, encoding: Encoding):
    """``[0, 1]`` capacity vector(s) -> the input's shape and units."""
    payload01 = payload01.reshape(len(payload01), -1)  # single mode -> one column
    out = []
    for j, (lo, hi) in enumerate(encoding.ranges):
        field = lo + (hi - lo) * payload01[: encoding.n, j]
        out.append(field.reshape(encoding.shape))
    return out[0] if len(out) == 1 else tuple(out)


def transpile_to_backend(
    circuit: QuantumCircuit, sampler, optimization_level: int = 1
) -> tuple[QuantumCircuit, dict]:
    """Transpile a full-graph circuit onto a sampler's device with the identity layout.

    Returns ``(transpiled, report)``; ``report`` holds the depths, the SWAP and
    two-qubit gate counts and the device size.
    """
    device = device_of(sampler)
    tqc = transpile(
        circuit,
        device,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=optimization_level,
        seed_transpiler=0,
    )
    ops = dict(tqc.count_ops())
    return tqc, {
        "logical_depth": circuit.depth(),
        "transpiled_depth": tqc.depth(),
        "swaps": ops.get("swap", 0),
        "two_qubit_gates": sum(ops.get(g, 0) for g in ("cx", "cz", "ecr")),
        "n_physical_qubits": device.num_qubits,
    }


# ----------------------------------------------------------------------
# Saving and loading encodings (plain JSON)
# ----------------------------------------------------------------------
_FORMAT = "tessa-encoding"
_VERSION = 1


def save_encoding(path: str | Path, encoding: Encoding | Mapping[str, Encoding]) -> Path:
    """Write one :class:`Encoding`, or a ``{name: Encoding}`` mapping, as JSON.

    The file holds everything :func:`decode_from_counts` needs, so stored counts
    can be decoded later without the circuit. Floats are written with full
    precision, so :func:`load_encoding` returns an identical encoding.
    """
    path = Path(path)
    if isinstance(encoding, Encoding):
        doc: dict[str, Any] = {"format": _FORMAT, "version": _VERSION}
        doc["encoding"] = encoding.to_dict()
    else:
        doc = {
            "format": _FORMAT,
            "version": _VERSION,
            "encodings": {str(k): enc.to_dict() for k, enc in encoding.items()},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_encoding(path: str | Path) -> Encoding | dict[str, Encoding]:
    """Read a file written by :func:`save_encoding`.

    Returns an :class:`Encoding`, or a ``{name: Encoding}`` dict if a mapping was
    saved.
    """
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != _FORMAT:
        raise ValueError(f"{path} is not a Tessa encoding file")
    if doc.get("version") != _VERSION:
        raise ValueError(
            f"{path}: unsupported encoding file version {doc.get('version')}"
        )
    if "encoding" in doc:
        return Encoding.from_dict(doc["encoding"])
    return {name: Encoding.from_dict(d) for name, d in doc["encodings"].items()}
