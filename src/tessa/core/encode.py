"""Build the Tessa circuit for a normalised value array.

Each data qubit carries a uniformly controlled ``RY`` ("multiplexer") whose
controls are its neighbouring address qubits: for local address state ``i`` the
data qubit ends up in ``cos(theta_i / 2)|0> + sin(theta_i / 2)|1>``. The
multiplexer compiles into plain ``RY``/``CX`` via the scaled fast
Walsh-Hadamard transform plus a Gray-code walk (the QCrank / compressed-FRQI
construction, restricted to each data qubit's *local* address register).

Address qubits act only as controls, so different data qubits' multiplexers
commute and can be appended -- or interleaved -- in any order. The pair modes
add a second multiplexer, ``RZ(phi)`` on the same qubit and controls.

Values arrive here already normalised to ``[0, 1]``; :mod:`tessa.codec` owns
the mapping from real units and back.
"""

from __future__ import annotations

from itertools import zip_longest

import numpy as np
from qiskit import QuantumCircuit

from .lattice import Lattice
from .mode import Mode, get_mode
from .walsh import count_trailing_zeros, gray_code, gray_permutation, sfwht

__all__ = ["build_tessa_circuit", "color_data_groups"]


def _multiplexer_coeffs(net_angles: np.ndarray, compression: float) -> np.ndarray:
    """Net angles -> the RY/RZ coefficient sequence of the Gray-code emission loop.

    The smallest ``compression`` percent of the sfwht + Gray-permuted
    coefficients are zeroed, which removes gates at the cost of accuracy.
    """
    a = gray_permutation(sfwht(np.asarray(net_angles, dtype=float)))
    if compression > 0:
        cutoff = int((compression / 100.0) * a.size)
        if cutoff > 0:
            a = a.copy()
            a[np.argsort(np.abs(a))[:cutoff]] = 0.0
    return a


def color_data_groups(lattice: Lattice) -> list[list[int]]:
    """Partition data qubits into classes whose multiplexers share no qubits.

    Two data qubits conflict if they share an address qubit; a greedy colouring
    of that conflict graph yields classes that can be scheduled in parallel.
    Returns a list of colour classes, each a list of indices into
    ``lattice.groups``.
    """
    groups = lattice.groups
    qubit_to_groups: dict[int, list[int]] = {}
    for gi, g in enumerate(groups):
        for q in (g.data_qubit, *g.address_qubits):
            qubit_to_groups.setdefault(q, []).append(gi)

    adj: list[set[int]] = [set() for _ in groups]
    for members in qubit_to_groups.values():
        for a in members:
            adj[a].update(b for b in members if b != a)

    color: dict[int, int] = {}
    for i in sorted(range(len(groups)), key=lambda i: len(adj[i]), reverse=True):
        used = {color[j] for j in adj[i] if j in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c

    classes: list[list[int]] = [[] for _ in range(max(color.values(), default=-1) + 1)]
    for i in sorted(color):  # keep row-major order within a class
        classes[color[i]].append(i)
    return classes


def _gray_control_bit(i: int, n: int, k: int) -> int:
    """Bit position toggled between Gray words ``i`` and ``i + 1`` (0 at the end)."""
    if i == n - 1:
        return 0
    ctrl = gray_code(i) ^ gray_code(i + 1)
    return k - count_trailing_zeros(ctrl, n_bits=k + 1) - 1


def _multiplexer_ops(
    coeffs: np.ndarray,
    address_qubits: tuple[int, ...],
    data_qubit: int,
    gate: str = "ry",
) -> list[tuple]:
    """Ordered gate list for one uniformly controlled rotation.

    Each element is ``(gate, angle, data_qubit)`` or ``("cx", control,
    data_qubit)``. Bit ``j`` of the Gray-code loop is the *most* significant end
    of the local address integer, hence ``address_qubits[k - 1 - j]``. This
    explicit indexing replaces the original QPIXL implementation's global
    ``reverse_bits()``, and the decoder indexes qubits the same way -- do not
    reintroduce a global bit reversal.
    """
    k = len(address_qubits)
    n = 1 << k
    ops: list[tuple] = []
    i = 0
    while i < n:
        pending = 0
        if coeffs[i] != 0:
            ops.append((gate, float(coeffs[i]), data_qubit))
        pending ^= 1 << _gray_control_bit(i, n, k)
        i += 1
        # Coefficients that were zeroed (by compression, or structurally) need no
        # rotation, but their control toggles still have to be accumulated.
        while i < n and coeffs[i] == 0:
            pending ^= 1 << _gray_control_bit(i, n, k)
            i += 1
        for j in range(k):
            if (pending >> j) & 1:
                ops.append(("cx", address_qubits[k - 1 - j], data_qubit))
    return ops


def _append_ops(circuit: QuantumCircuit, ops) -> None:
    for kind, a, b in ops:
        if kind == "cx":
            circuit.cx(a, b)
        elif kind == "ry":
            circuit.ry(a, b)
        else:  # "rz"
            circuit.rz(a, b)


def build_tessa_circuit(
    lattice,
    values01: np.ndarray,
    mode: str | Mode = "single",
    clamps=0.0,
    compression: float = 0.0,
    strategy: str = "parallel",
) -> QuantumCircuit:
    """Circuit encoding ``values01`` on ``lattice``; no measurements.

    Parameters
    ----------
    lattice:
        Anything exposing ``groups``, ``n_qubits`` and ``pack`` -- a
        :class:`~tessa.core.lattice.Lattice` or an
        :class:`~tessa.TessaGraph`.
    values01:
        ``(capacity,)`` in ``[0, 1]`` for ``mode="single"``, ``(capacity, 2)``
        for the pair modes.
    mode, clamps:
        The mode, and one clamp per field (a single number applies to both).
        See :mod:`tessa.core.mode`.
    compression:
        Percentage (0-100) of the smallest multiplexer coefficients to drop per
        data qubit and rotation axis -- lossy, but removes gates.
    strategy:
        ``"parallel"`` (default) interleaves the multiplexers gate by gate, so
        the per-data-qubit chains overlap in time and the depth collapses to
        roughly one multiplexer's instead of their sum -- same resulting state.
        ``"sequential"`` emits each multiplexer in full before the next.
    """
    if strategy not in ("parallel", "sequential"):
        raise ValueError(f"unknown strategy {strategy!r}; use 'parallel'/'sequential'")
    mode = get_mode(mode)
    values01 = np.asarray(values01, dtype=float)
    expected = (lattice.capacity, 2) if mode.is_pair else (lattice.capacity,)
    if values01.shape != expected:
        raise ValueError(
            f"mode {mode.name!r} expects values shaped {expected}, got {values01.shape}"
        )

    packed = lattice.pack(values01)
    per_group_ops = []
    for g in lattice.groups:
        chunk = packed[g.site]
        theta = mode.angles(chunk[:, 0] if mode.is_pair else chunk, clamps)
        ops = _multiplexer_ops(
            _multiplexer_coeffs(theta, compression), g.address_qubits, g.data_qubit, "ry"
        )
        if mode.is_pair:
            phi = mode.phases(chunk[:, 1], clamps)
            ops += _multiplexer_ops(
                _multiplexer_coeffs(phi, compression),
                g.address_qubits,
                g.data_qubit,
                "rz",
            )
        per_group_ops.append(ops)

    qc = QuantumCircuit(lattice.n_qubits)
    addr_qubits = sorted({q for g in lattice.groups for q in g.address_qubits})
    if addr_qubits:
        qc.h(addr_qubits)
    if strategy == "parallel":
        for layer in zip_longest(*per_group_ops):
            _append_ops(qc, [op for op in layer if op is not None])
    else:
        for ops in per_group_ops:
            _append_ops(qc, ops)
    return qc
