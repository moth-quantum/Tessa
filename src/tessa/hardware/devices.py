"""The two IBM chips of the paper, rebuilt offline from calibration snapshots.

``snapshots/ibm_<name>.json`` holds, per chip, the coupling map, drawing
coordinates and the per-qubit (``T1``, ``T2``, ``sx`` error and duration, readout
error) and per-coupler (``cz`` error and duration) calibration at the time stamp
``calibrated_at``. From it :func:`snapshot_backend` builds a minimal ``BackendV2``
whose ``Target`` carries exactly what the noise models of :mod:`tessa.noise`
read, so simulations never contact IBM. :func:`snapshot` writes such a file from
a live (or fake) backend.

The stored snapshots are ``ibm_phoenix`` (120 qubits, calibrated
2026-10-09 11:17 +02:00, used for the genetic-sequence appendix) and ``ibm_fez``
(156 qubits).
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, IGate, RZGate, SXGate, XGate
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, QubitProperties, Target

from ..backends.aer import calibrated_sampler
from ..graph import TessaGraph, graph_from_backend

__all__ = [
    "DEVICES",
    "SNAPSHOT_DIR",
    "SnapshotBackend",
    "coordinates",
    "graph",
    "noisy_sampler",
    "snapshot",
    "snapshot_backend",
]

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

#: Short name -> IBM backend name.
DEVICES = {"fez": "ibm_fez", "phoenix": "ibm_phoenix"}


class SnapshotBackend(BackendV2):
    """A device that exists only as a calibration snapshot: transpile, never run."""

    def __init__(self, snap: dict[str, Any]) -> None:
        super().__init__(name=snap["name"])
        self.snapshot = snap
        self._target = _target(snap)

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(self, run_input, **options):
        del run_input, options  # required by the BackendV2 interface, not used
        raise NotImplementedError("a snapshot backend only transpiles; run on Aer")


def _target(snap: dict[str, Any]) -> Target:
    def num(x, default):
        return default if x is None else float(x)

    qubits = snap["qubits"]
    target = Target(
        num_qubits=snap["num_qubits"],
        qubit_properties=[
            QubitProperties(t1=num(q["t1"], 1e-4), t2=num(q["t2"], 1e-4)) for q in qubits
        ],
    )
    one = {
        (i,): InstructionProperties(
            error=num(q["sx_error"], 3e-4), duration=num(q["sx_duration"], 3.2e-8)
        )
        for i, q in enumerate(qubits)
    }
    target.add_instruction(SXGate(), one)
    target.add_instruction(XGate(), dict(one))
    target.add_instruction(
        RZGate(Parameter("theta")),
        {
            (i,): InstructionProperties(error=0.0, duration=0.0)
            for i in range(len(qubits))
        },
    )
    target.add_instruction(
        IGate(), {(i,): InstructionProperties(error=0.0) for i in range(len(qubits))}
    )
    target.add_instruction(
        Measure(),
        {
            (i,): InstructionProperties(error=num(q["readout_error"], 1e-2))
            for i, q in enumerate(qubits)
        },
    )
    cz = {}
    for (a, b), (err, dur) in zip(snap["edges"], snap["cz"], strict=True):
        props = {"error": num(err, 8e-3), "duration": num(dur, 7e-8)}
        cz[(a, b)] = InstructionProperties(**props)
        cz[(b, a)] = InstructionProperties(**props)
    target.add_instruction(CZGate(), cz)
    return target


def _check(device: str) -> str:
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; choose from {sorted(DEVICES)}")
    return DEVICES[device]


@cache
def snapshot_backend(device: str) -> SnapshotBackend:
    """The stored snapshot of ``device`` (``"fez"`` or ``"phoenix"``) as a backend."""
    name = _check(device)
    return SnapshotBackend(json.loads((SNAPSHOT_DIR / f"{name}.json").read_text()))


@cache
def graph(device: str) -> TessaGraph:
    """The Tessa graph of ``device``'s snapshot coupling map."""
    return graph_from_backend(snapshot_backend(device))


@cache
def noisy_sampler(device: str, seed: int | None = None):
    """``device``'s own per-qubit calibration, run on Aer. Built once per device and
    seed; ``seed=None`` (the default) draws fresh shots on every run, an integer
    makes the shots reproducible."""
    return calibrated_sampler(snapshot_backend(device), seed=seed)


# ----------------------------------------------------------------------
# Writing a snapshot
# ----------------------------------------------------------------------
def coordinates(n_qubits: int, edges) -> list[tuple[float, float]]:
    """Chip-drawing ``(x, y)`` per qubit; every coupler has length 1.

    Knows the 120-qubit square lattice (Nighthawk: 12 rows of 10, row-major) and
    the 156-qubit heavy-hex chips (Heron r2: rows of 16 joined by bridge qubits).
    """
    if n_qubits == 120:
        coords = [(q % 10, q // 10) for q in range(n_qubits)]
    elif n_qubits == 156:
        coords = [None] * n_qubits
        for q in range(n_qubits):
            if q % 20 < 16:
                coords[q] = (q % 20, 2 * (q // 20))
        for q in range(n_qubits):
            if coords[q] is None:
                above = min(p for a, b in edges for p in (a, b) if q in (a, b) and p != q)
                coords[q] = (coords[above][0], coords[above][1] + 1)
    else:
        raise ValueError(f"no chip layout for {n_qubits} qubits")
    for a, b in edges:
        (xa, ya), (xb, yb) = coords[a], coords[b]
        if not math.isclose(math.hypot(xa - xb, ya - yb), 1.0):
            raise ValueError(f"coupler {(a, b)} does not fit the chip layout")
    return coords


def _value(props, attr):
    x = getattr(props, attr, None) if props is not None else None
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def snapshot(backend) -> dict[str, Any]:
    """The calibration of ``backend`` in the format :func:`snapshot_backend` reads.

    Needs a backend whose ``Target`` has ``cz``, ``sx`` and ``measure`` (IBM Heron
    and Nighthawk chips, or their fake-provider copies).
    """
    tgt = backend.target
    edges = sorted({tuple(sorted(e)) for e in tgt["cz"]})
    qubits = []
    for q in range(backend.num_qubits):
        qp = tgt.qubit_properties[q]
        qubits.append(
            {
                "t1": _value(qp, "t1"),
                "t2": _value(qp, "t2"),
                "sx_error": _value(tgt["sx"].get((q,)), "error"),
                "sx_duration": _value(tgt["sx"].get((q,)), "duration"),
                "readout_error": _value(tgt["measure"].get((q,)), "error"),
            }
        )
    cz = []
    for a, b in edges:
        props = tgt["cz"].get((a, b)) or tgt["cz"].get((b, a))
        cz.append([_value(props, "error"), _value(props, "duration")])
    props = backend.properties() if hasattr(backend, "properties") else None
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "calibrated_at": str(props.last_update_date) if props is not None else None,
        "edges": [list(e) for e in edges],
        "coords": [list(c) for c in coordinates(backend.num_qubits, edges)],
        "qubits": qubits,
        "cz": cz,
    }
