"""Run a :class:`~tessa.hardware.pipeline.Plan`: whole-chip circuits for hardware,
their counts back into images, or a local simulation on a calibration snapshot.

Hardware runs every readout of a plan as whole-chip circuits, one per measurement
basis (:func:`hardware_circuits`); :func:`decode_hardware` turns their counts, in
the same order, into the result dict of :func:`~tessa.hardware.pipeline.finish`.
:func:`simulate` decodes cell by cell on Aer instead, with the noise of a stored
calibration snapshot (``halo="traced"``) or without noise.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from qiskit import QuantumCircuit

from ..backends.aer import aer_sampler
from ..codec import decode, decode_from_counts, measurement_circuits
from ..core.mode import get_mode
from . import devices
from .pipeline import Plan, build_plan, finish, shots_of

__all__ = ["decode_hardware", "hardware_circuits", "simulate"]


@cache
def _noiseless_sampler(device: str):
    return aer_sampler(devices.graph(device))


def simulate(request: dict[str, Any]) -> dict[str, Any]:
    """Run ``request`` on Aer, on the snapshot graph of ``request["device"]``.

    ``request["noise"]`` is ``"calibrated"`` (default: the snapshot's per-qubit
    calibration, each cell simulated with its traced halo, see
    :func:`tessa.codec.decode`) or ``"none"``.
    """
    device = request.get("device", "phoenix")
    graph = devices.graph(device)
    plan = build_plan(request, graph)
    noise = request.get("noise", "calibrated") != "none"
    sampler = devices.noisy_sampler(device) if noise else _noiseless_sampler(device)
    shots = shots_of(request)
    symbolic = "symbols" in plan.extra  # decoded from counts, see pipeline.finish
    decoded = []
    for circuit, enc in plan.readouts:
        out = decode(
            circuit, enc, sampler, shots=shots, halo="traced", return_counts=symbolic
        )
        decoded.append(out[1] if symbolic else out)
    out = finish(plan, decoded)
    out["run"] = {
        "kind": "simulator",
        "device": devices.DEVICES[device],
        "noise": "calibrated" if noise else "none",
        "shots": shots,
        "calibrated_at": devices.snapshot_backend(device).snapshot["calibrated_at"],
    }
    return out


def hardware_circuits(plan: Plan) -> list[QuantumCircuit]:
    """Every readout of ``plan``, one whole-chip circuit per measurement basis."""
    return [
        qc
        for circuit, enc in plan.readouts
        for qc in measurement_circuits(circuit, enc).values()
    ]


def decode_hardware(plan: Plan, counts: list[dict[str, int]]) -> dict[str, Any]:
    """Counts of :func:`hardware_circuits`, in the same order -> :func:`finish`."""
    symbolic = "symbols" in plan.extra  # decoded from counts, see pipeline.finish
    decoded, i = [], 0
    for _, enc in plan.readouts:
        by_basis = {}
        for basis in get_mode(enc.mode).bases:
            by_basis[basis] = counts[i]
            i += 1
        decoded.append(by_basis if symbolic else decode_from_counts(by_basis, enc))
    if i != len(counts):
        raise ValueError(f"expected {i} circuits' counts, got {len(counts)}")
    return finish(plan, decoded)
