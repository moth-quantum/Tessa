"""Turn measurement counts back into the values that were encoded.

One measurement setting reads out the whole lattice: each cell is recovered from
the same shots by marginalising them onto its own ``(data, address...)`` bits.
Per address configuration the marginal gives the cell's Pauli expectations, and
the mode says how to read values out of them:

- ``theta = arccos(<Z>)``, or ``atan2(hypot(<X>, <Y>), <Z>)`` on the full circle
- ``phi = arccos(<X> / sin(theta))`` on the half-circle
- ``phi = atan2(<Y>, <X>)`` on the full circle
"""

from __future__ import annotations

import numpy as np

from .lattice import DataGroup
from .mode import Mode, get_mode

__all__ = ["decode_cells"]


def _bit_table(counts: dict, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts -> ``(bits, weights)``: ``bits[i, q]`` is qubit ``q`` in outcome ``i``.

    One pass over the keys, so every cell of a whole-chip run reads the same
    table instead of re-parsing every bitstring. Bit order: Qiskit's, the highest
    qubit index is leftmost in a key. A configuration that received no shot has
    ``c0 + c1 = 0``; :func:`_table_expectation` then returns 0 for it (see
    ``tessa.decode_from_counts(..., unsampled=...)``).
    """
    keys = list(counts)
    weights = np.fromiter((counts[k] for k in keys), dtype=float, count=len(keys))
    clean = [k.replace(" ", "")[-n_qubits:].rjust(n_qubits, "0") for k in keys]
    chars = np.frombuffer("".join(clean).encode("ascii"), dtype=np.uint8)
    bits = (chars.reshape(len(keys), n_qubits) - ord("0"))[:, ::-1]
    return bits, weights


def _table_expectation(
    bits: np.ndarray, weights: np.ndarray, group: DataGroup
) -> np.ndarray:
    """``(c0 - c1) / (c0 + c1)`` per address configuration, from a bit table."""
    addr = np.zeros(len(weights), dtype=np.int64)
    for j, a in enumerate(group.address_qubits):
        addr |= bits[:, a].astype(np.int64) << j
    one = bits[:, group.data_qubit].astype(bool)
    c1 = np.bincount(addr[one], weights=weights[one], minlength=group.n_values)
    c0 = np.bincount(addr[~one], weights=weights[~one], minlength=group.n_values)
    total = np.where(c0 + c1 == 0, 1.0, c0 + c1)
    return (c0 - c1) / total


def _sampled_configurations(
    bits: np.ndarray, weights: np.ndarray, group: DataGroup
) -> np.ndarray:
    """Whether each address configuration of ``group`` received at least one shot."""
    addr = np.zeros(len(weights), dtype=np.int64)
    for j, a in enumerate(group.address_qubits):
        addr |= bits[:, a].astype(np.int64) << j
    return np.bincount(addr, weights=weights, minlength=group.n_values) > 0


def _cell_expectation(counts: dict, group: DataGroup, n_qubits: int) -> np.ndarray:
    """``(c0 - c1) / (c0 + c1)`` per address configuration, from one setting's counts."""
    return _table_expectation(*_bit_table(counts, n_qubits), group)


def decode_cells(
    counts_by_basis: dict[str, dict],
    groups,
    n_qubits: int,
    mode: str | Mode = "single",
    clamps=0.0,
) -> dict:
    """Decode one measured circuit into ``{site: values in [0, 1]}``.

    ``counts_by_basis`` holds one counts dict per basis the mode asks for
    (``"z"``, plus ``"x"`` and ``"y"``); ``groups`` and ``n_qubits`` describe the
    circuit those counts came from, which may be the full lattice or a single
    extracted cell. ``clamps`` must be the ones the payload was encoded with.
    """
    mode = get_mode(mode)
    missing = [b for b in mode.bases if b not in counts_by_basis]
    if missing:
        raise ValueError(f"mode {mode.name!r} needs counts for bases {missing}")

    tables = {b: _bit_table(counts_by_basis[b], n_qubits) for b in mode.bases}
    out: dict = {}
    for g in groups:
        ez = _table_expectation(*tables["z"], g)
        if not mode.is_pair:
            out[g.site] = mode.values_from_angles(np.arccos(np.clip(ez, -1, 1)), clamps)
            continue
        ex = _table_expectation(*tables["x"], g)
        if mode.full_circle:
            ey = _table_expectation(*tables["y"], g)
            theta = np.arctan2(np.hypot(ex, ey), ez)
            phi = np.mod(np.arctan2(ey, ex), 2.0 * np.pi)
        else:
            theta = np.arccos(np.clip(ez, -1, 1))
            sin_theta = np.where(np.sin(theta) == 0, 1.0, np.sin(theta))
            phi = np.arccos(np.clip(ex / sin_theta, -1.0, 1.0))
        out[g.site] = np.stack(
            [
                mode.values_from_angles(theta, clamps),
                mode.values_from_phases(phi, clamps),
            ],
            axis=-1,
        )
    return out
