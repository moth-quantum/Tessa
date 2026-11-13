"""Symbols -- DNA bases, amino acids, palette colours -- through the encoding.

A *symbol* is one of ``n_symbols`` discrete values, given as an integer
``0 .. n_symbols - 1``. Only which symbol was encoded has to come back, so the
decoder can snap to the nearest allowed state instead of returning a real number.
Two schemes place the symbols on the cells:

``"constellation"`` (default)
    Each slot holds ``per_slot`` symbols as **one point** out of
    ``n_symbols ** per_slot``, spread evenly over the whole Bloch sphere (see
    :func:`constellation`). The cell is read in ``Z``, ``X`` and ``Y``, the
    measured Bloch vector is normalised and the nearest point wins. Normalising
    cancels any uniform shrinking from depolarising noise. 400 points (two
    amino acids per slot) sit about 0.16 rad apart -- 1.6x the spacing of 20
    levels per field on the pair-mode half-circle, which is what the shot count
    depends on. The point set comes from a short repulsion iteration whose last
    digits depend on the floating-point platform; to decode stored results
    bit-for-bit, pass the stored points with ``encode_symbols(points=...)``.

``"levels"``
    Each symbol is a level, evenly spaced on ``[0, 1]``, in one field: the
    amplitude only (``per_slot=1``, ``mode="single"``), or the amplitude and the
    phase (``per_slot=2``, ``mode="pair"``). With ``snap_theta`` (default) the
    pair decode snaps the amplitude to its level first and reads the phase with
    that level's exact ``sin(theta)``, so the phase no longer carries the
    amplitude's shot noise.

**Placement.** Symbols fill the slots in graph order unless ``slot_order`` says
otherwise. A sequence shorter than the capacity can go on the best data qubits:
:func:`probe_qubits` scores every data qubit by how much of a random probe
constellation it reads back under a sampler (for example the chip's calibrated
noise model, which needs no hardware shots), and :func:`slot_order` lists the
slots best qubit first.

Both schemes accept the dynamic-range methods of
:func:`tessa.apply_dynamic_range` (``drr=``): per field for ``"levels"``, per
Bloch axis before normalising for ``"constellation"``. The reference statistics
are those of the encoded (ideal) values.

::

    circuit, enc = encode_symbols(graph, symbols, n_symbols=20)
    out = decode_symbols(circuit, enc, sampler, shots=32768)   # same shape, ints

    out, counts = decode_symbols(circuit, enc, sampler, return_counts=True)
    again = symbols_from_counts(counts, enc, drr="mean_std")   # no re-run

    order = slot_order(graph, probe_qubits(graph, calibrated_sampler(backend)))
    circuit, enc = encode_symbols(graph, short_sequence, 20, slot_order=order)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import Encoding, _cell_group, decode, encode
from .core.decode import _bit_table, _cell_expectation, _table_expectation
from .core.mode import DEFAULT_CLAMP, get_mode
from .transforms import apply_dynamic_range

__all__ = [
    "CONSTELLATION_CLAMP",
    "SYMBOL_SCHEMES",
    "SymbolEncoding",
    "constellation",
    "decode_symbols",
    "encode_symbols",
    "probe_qubits",
    "slot_order",
    "symbols_from_counts",
]

SYMBOL_SCHEMES = ("constellation", "levels")

#: ``theta`` clamp of a constellation: as small as the pair modes allow, so the
#: points can use the whole sphere.
CONSTELLATION_CLAMP = 1e-3

#: Above this many points, :func:`constellation` skips the O(n^2) relaxation.
_RELAX_LIMIT = 1024


@dataclass(frozen=True)
class SymbolEncoding:
    """What :func:`symbols_from_counts` needs to read the symbols back."""

    #: The value-level encoding underneath, as :func:`tessa.encode` made it.
    encoding: Encoding
    scheme: str
    n_symbols: int
    per_slot: int
    #: Shape of the input symbols, and how many there are.
    shape: tuple[int, ...]
    n: int
    #: Number of slots the symbols fill.
    slots: int
    #: ``"constellation"``: the ``(n_symbols ** per_slot, 3)`` unit Bloch vectors.
    points: np.ndarray | None
    #: The ideal encoded values per used slot -- ``(slots, fields)`` in ``[0, 1]``
    #: for ``"levels"``, ``(slots, 3)`` Bloch vectors for ``"constellation"``.
    #: Dynamic-range recovery reads only summary statistics of it.
    reference: np.ndarray
    #: The capacity slot of each used slot, in the order the symbols fill them.
    slot_index: np.ndarray

    @property
    def graph(self):
        return self.encoding.graph

    @property
    def mode(self) -> str:
        return self.encoding.mode

    @property
    def bases(self) -> tuple[str, ...]:
        """The measurement settings to run, in order."""
        return get_mode(self.encoding.mode).bases

    @property
    def data_qubits(self) -> list[int]:
        return self.encoding.data_qubits


def constellation(n_points: int, *, relax: bool = True, iters: int = 500) -> np.ndarray:
    """``n_points`` unit vectors spread evenly over the sphere, shape ``(n, 3)``.

    A Fibonacci lattice, then (``relax``, up to 1024 points) a short mutual
    repulsion that pushes the closest pairs apart. Deterministic on one machine:
    the same ``n_points`` always gives the same points, in the same order. The
    repulsion is sensitive to rounding (a relative change of 1e-10 in the start
    points moves the minimum spacing of 400 points by about 0.004 rad), so a
    different CPU, BLAS or numpy build can give slightly different points. Store
    the points with the encoded data when a decode must be reproducible across
    platforms, and pass them to :func:`encode_symbols` (``points=``).
    """
    n = int(n_points)
    if n < 1:
        raise ValueError("a constellation needs at least one point")
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.mod(np.pi * (1.0 + 5.0**0.5) * i, 2.0 * np.pi)
    r = np.sqrt(1.0 - z**2)
    pts = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    if not relax or n < 2 or n > _RELAX_LIMIT:
        return pts
    for _ in range(iters):
        d = pts[:, None, :] - pts[None, :, :]
        r2 = (d**2).sum(-1) + np.eye(n)
        force = (d / r2[..., None] ** 1.5).sum(1)
        force -= (force * pts).sum(1, keepdims=True) * pts  # tangential part only
        pts = pts + 0.02 * force / np.abs(force).max()
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _split(symbols: np.ndarray, per_slot: int, slots: int) -> list[np.ndarray]:
    """Flat symbols -> one array per position in the slot, zero-padded to ``slots``.

    Position ``j`` holds the ``j``-th consecutive block of ``slots`` symbols, so a
    sequence's first half goes in the first position and its second half in the
    second.
    """
    padded = np.zeros(per_slot * slots, dtype=int)
    padded[: symbols.size] = symbols
    return [padded[j * slots : (j + 1) * slots] for j in range(per_slot)]


def _join(parts: list[np.ndarray], enc: SymbolEncoding) -> np.ndarray:
    return np.concatenate(parts)[: enc.n].reshape(enc.shape)


def _slot_ranges(graph) -> list[tuple[int, int]]:
    """``(start, stop)`` of each data qubit's slots, in ``graph.groups`` order."""
    stops = np.cumsum([g.n_values for g in graph.groups])
    return [
        (int(b - g.n_values), int(b)) for g, b in zip(graph.groups, stops, strict=True)
    ]


def slot_order(graph, quality) -> np.ndarray:
    """Every slot, best data qubit first: the order for ``encode_symbols(slot_order=)``.

    ``quality`` holds one score per data qubit, in ``graph.groups`` order, higher
    is better (for example from :func:`probe_qubits`). Ties keep graph order, and
    the slots of one data qubit stay together, in their own order.
    """
    q = np.asarray(quality, dtype=float)
    ranges = _slot_ranges(graph)
    if q.shape != (len(ranges),):
        raise ValueError(f"need one quality score per data qubit ({len(ranges)})")
    best = np.argsort(-q, kind="stable")
    return np.concatenate([np.arange(*ranges[i]) for i in best])


def encode_symbols(
    graph,
    symbols,
    n_symbols: int | None = None,
    *,
    scheme: str = "constellation",
    per_slot: int | None = None,
    clamp: float | None = None,
    clamp_b: float | None = None,
    relax: bool = True,
    slot_order=None,
    points=None,
):
    """Encode integer symbols as a circuit plus the :class:`SymbolEncoding`.

    Parameters
    ----------
    graph:
        An :class:`~tessa.TessaGraph` or :class:`~tessa.Lattice`.
    symbols:
        Integers in ``[0, n_symbols)``, any shape.
    n_symbols:
        Alphabet size; defaults to ``max(symbols) + 1``. Pass it whenever a symbol
        might be missing from the data, so the levels / points stay fixed.
    scheme:
        ``"constellation"`` or ``"levels"`` (see the module docstring).
    per_slot:
        Symbols per slot, 1 or 2. Defaults to 1 when they fit in
        ``graph.capacity`` and 2 otherwise.
    clamp, clamp_b:
        ``"levels"`` only. ``clamp`` defaults to 0 for one symbol per slot and to
        :data:`~tessa.DEFAULT_CLAMP` for two; ``clamp_b`` defaults to ``clamp``,
        which keeps the first and last phase level away from ``0`` and ``pi``,
        where the phase read-out is least accurate.
    relax:
        ``"constellation"`` only: passed to :func:`constellation`.
    slot_order:
        The slots to fill, in order (see :func:`slot_order`); the first ones
        needed are used and the others hold symbol 0. Defaults to graph order.
    points:
        ``"constellation"`` only: the ``(n_symbols ** per_slot, 3)`` unit vectors
        to use instead of ``constellation(n_symbols ** per_slot, relax=relax)``,
        for example the point set stored with a hardware run.
    """
    if scheme not in SYMBOL_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SYMBOL_SCHEMES}")
    arr = np.asarray(symbols)
    flat = arr.ravel().astype(int)
    if flat.size == 0:
        raise ValueError("no symbols to encode")
    if not np.array_equal(flat, arr.ravel()):
        raise ValueError("symbols must be integers")
    k = int(flat.max()) + 1 if n_symbols is None else int(n_symbols)
    if flat.min() < 0 or flat.max() >= k:
        raise ValueError(f"symbols must lie in [0, {k})")
    if per_slot is None:
        per_slot = 1 if flat.size <= graph.capacity else 2
    if per_slot not in (1, 2):
        raise ValueError("per_slot must be 1 or 2")
    slots = -(-flat.size // per_slot)
    if slots > graph.capacity:
        raise ValueError(
            f"{flat.size} symbols need {slots} slots at {per_slot} per slot, "
            f"but the graph has {graph.capacity}"
        )
    if slot_order is None:
        where = np.arange(slots)
    else:
        order = np.asarray(slot_order, dtype=int).ravel()
        if order.size < slots or len(set(order.tolist())) != order.size:
            raise ValueError(f"slot_order must list at least {slots} distinct slots")
        if order.min() < 0 or order.max() >= graph.capacity:
            raise ValueError(f"slot_order must lie in [0, {graph.capacity})")
        where = order[:slots]
    parts = _split(flat, per_slot, slots)
    unit = (0.0, 1.0)

    def full(values):
        # Values of the used slots, placed at their slots of the whole capacity.
        out = np.zeros(graph.capacity)
        out[where] = values
        return out

    if scheme == "levels":
        if points is not None:
            raise ValueError("points apply to the constellation scheme only")
        if clamp_b is not None and per_slot == 1:
            raise ValueError("clamp_b needs two symbols per slot")
        top = max(k - 1, 1)
        fields = [p / top for p in parts]
        if per_slot == 1:
            c = 0.0 if clamp is None else float(clamp)
            circuit, enc = encode(graph, full(fields[0]), clamp=c, value_range=unit)
        else:
            c = DEFAULT_CLAMP if clamp is None else float(clamp)
            circuit, enc = encode(
                graph,
                full(fields[0]),
                full(fields[1]),
                mode="pair",
                clamp=c,
                clamp_b=c if clamp_b is None else float(clamp_b),
                value_range=unit,
                value_range_b=unit,
            )
        reference, points = np.stack(fields, axis=1), None
    else:
        if clamp is not None or clamp_b is not None:
            raise ValueError("a constellation sets its own clamps")
        if points is None:
            points = constellation(k**per_slot, relax=relax)
        else:
            points = np.array(points, dtype=float)
            if points.shape != (k**per_slot, 3):
                raise ValueError(
                    f"points must have shape ({k**per_slot}, 3), not {points.shape}"
                )
            if not np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-9):
                raise ValueError("points must be unit vectors")
        index = parts[0] * k + parts[1] if per_slot == 2 else parts[0]
        reference = points[index]
        theta = np.arccos(np.clip(reference[:, 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(reference[:, 1], reference[:, 0]), 2.0 * np.pi)
        c = CONSTELLATION_CLAMP
        circuit, enc = encode(
            graph,
            full((theta - c) / (np.pi - 2.0 * c)),
            full(phi / (2.0 * np.pi)),
            mode="pair_full",
            clamp=c,
            clamp_b=0.0,
            value_range=unit,
            value_range_b=unit,
        )
    return circuit, SymbolEncoding(
        enc, scheme, k, per_slot, arr.shape, flat.size, slots, points, reference, where
    )


def _slot_expectations(counts: dict, enc: SymbolEncoding) -> dict[str, np.ndarray]:
    """``{basis: <P>}`` per used slot, from joint or per-group counts.

    Joint counts (``{basis: counts}``) come from the whole circuit; per-group
    counts (``{site: {basis: counts}}``) from :func:`tessa.decode` with
    ``per_group=True``, where each cell's register holds the data qubit at 0 and
    its address qubits next -- so its last ``1 + degree`` bits are enough, with
    or without a halo.
    """
    graph = enc.graph
    joint = set(counts) <= {"z", "x", "y"}
    if joint:  # parse each setting's bitstrings once, for every cell
        tables = {b: _bit_table(counts[b], graph.n_qubits) for b in enc.bases}
    per_basis: dict[str, dict] = {b: {} for b in enc.bases}
    for g in graph.groups:
        for b in enc.bases:
            if joint:
                value = _table_expectation(*tables[b], g)
            else:
                value = _cell_expectation(counts[g.site][b], _cell_group(g), 1 + g.degree)
            per_basis[b][g.site] = value
    return {b: graph.unpack(v)[enc.slot_index] for b, v in per_basis.items()}


def symbols_from_counts(
    counts: dict, enc: SymbolEncoding, *, drr: str = "none", snap_theta: bool = True
) -> np.ndarray:
    """Read the symbols back from counts, in the input's shape.

    ``counts`` is either ``{basis: counts}`` from the whole encoded circuit (one
    entry per basis in ``enc.bases``, as a hardware job returns them) or the
    per-group counts :func:`decode_symbols` returns with ``return_counts=True``.
    Circuit-free and cheap, so decode once and call this per ``drr`` method.
    ``snap_theta`` applies to the ``"levels"`` scheme with two symbols per slot.
    """
    e = _slot_expectations(counts, enc)
    k = enc.n_symbols
    if enc.scheme == "constellation":
        v = np.stack([e["x"], e["y"], e["z"]], axis=1)
        if drr != "none":
            v = np.stack(
                [
                    apply_dynamic_range(v[:, j], enc.reference[:, j], drr)
                    for j in range(3)
                ],
                axis=1,
            )
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        index = (v @ enc.points.T).argmax(axis=1)
        parts = [index // k, index % k] if enc.per_slot == 2 else [index]
        return _join(parts, enc)

    top = max(k - 1, 1)

    def snap(x01, ref):
        x01 = np.clip(apply_dynamic_range(x01, ref, drr), 0.0, 1.0)
        return np.rint(x01 * top).astype(int)

    clamps = enc.encoding.clamps
    ca = clamps[0]
    theta = np.arccos(np.clip(e["z"], -1.0, 1.0))
    first = snap((theta - ca) / (np.pi - 2.0 * ca), enc.reference[:, 0])
    if enc.per_slot == 1:
        return _join([first], enc)
    if snap_theta:
        theta = ca + (np.pi - 2.0 * ca) * first / top
    sin_theta = np.where(np.sin(theta) == 0, 1.0, np.sin(theta))
    phi = np.arccos(np.clip(e["x"] / sin_theta, -1.0, 1.0))
    cb = clamps[1]
    second = snap((phi - cb) / (np.pi - 2.0 * cb), enc.reference[:, 1])
    return _join([first, second], enc)


def decode_symbols(
    circuit,
    enc: SymbolEncoding,
    sampler,
    *,
    drr: str = "none",
    snap_theta: bool = True,
    return_counts: bool = False,
    **decode_options,
):
    """Measure ``circuit`` on ``sampler`` and read the symbols back.

    ``decode_options`` go to :func:`tessa.decode` (``shots``, ``per_group``,
    ``halo``, ...). With ``return_counts`` the counts come along, ready for
    :func:`symbols_from_counts` with another ``drr``.
    """
    _, counts = decode(
        circuit, enc.encoding, sampler, return_counts=True, **decode_options
    )
    out = symbols_from_counts(counts, enc, drr=drr, snap_theta=snap_theta)
    return (out, counts) if return_counts else out


def probe_qubits(
    graph, sampler, *, n_symbols: int = 20, shots: int = 32768, seed: int = 0, **options
) -> np.ndarray:
    """How well each data qubit reads symbols, in ``graph.groups`` order.

    Encodes ``2 * graph.capacity`` random symbols as a two-per-slot constellation
    of ``n_symbols ** 2`` points, decodes them on ``sampler`` and returns, per data
    qubit, the fraction that came back right. With a simulator carrying a chip's
    calibrated noise (:func:`tessa.calibrated_sampler`) this ranks the qubits
    without hardware shots. ``options`` go to :func:`tessa.decode`.
    """
    probe = np.random.default_rng(seed).integers(0, n_symbols, 2 * graph.capacity)
    circuit, enc = encode_symbols(graph, probe, n_symbols, per_slot=2)
    got = decode_symbols(circuit, enc, sampler, shots=shots, **options)
    right = (got == probe).reshape(2, graph.capacity).mean(axis=0)  # per slot
    return np.array([right[a:b].mean() for a, b in _slot_ranges(graph)])
