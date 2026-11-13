"""Device-derived noise models sized for a small (simulable) lattice or graph.

A full 120- or 156-qubit noisy simulation is infeasible, so calibration numbers
are read off a backend ``Target`` -- single-qubit and ``CZ`` gate errors, readout
error, ``T1``/``T2``, gate durations -- and used to build a noise model on a
*small* topology's own qubits and edges.

* :func:`build_average_noise_model` -- one uniform machine-average model. The
  default everywhere: fast and easy to reason about.
* :func:`build_local_noise_model` -- per-physical-qubit calibration, so each
  block is simulated with the noise of the qubits it actually occupies. More
  faithful on a non-uniform chip (it captures the worst qubits, which the
  average hides), and correspondingly slower. Reached via
  :func:`tessa.backends.calibrated_sampler`.

Both read from a Qiskit ``Target``, so neither is IBM-specific. Both also put,
on each end of every ``cz`` edge, the noise that gate leaves on that qubit alone,
under the label :func:`halo_label`. ``decode(..., halo="traced")`` uses it to
trace a cell's halo qubits out of the simulation.
"""

from __future__ import annotations

import numpy as np

from .core.lattice import encoding_edges

#: Heron / heavy-hex / ibm_miami native basis (identical across these devices).
NATIVE_BASIS = ["id", "rz", "sx", "x", "cz"]

__all__ = [
    "NATIVE_BASIS",
    "backend_rates",
    "build_average_noise_model",
    "build_local_noise_model",
    "halo_label",
]


def halo_label(partner: int) -> str:
    """Label for the noise a ``cz`` with ``partner`` leaves on the other qubit.

    ``partner`` is the qubit traced out. The noise models attach, to this label on
    each qubit ``q`` next to ``partner``, the error of the ``cz`` on
    ``(q, partner)`` with ``partner`` traced out: a 1-qubit depolarizing error of
    the same strength, then ``q``'s own relaxation for the gate's duration.
    """
    return f"cz_halo_{partner}"


def _add_cz_noise(nm, a: int, b: int, cz_error: float, relax_a, relax_b) -> None:
    """The ``cz`` error on edge ``(a, b)``, and its marginal on each end."""
    from qiskit_aer.noise import depolarizing_error

    cz_error = float(np.clip(cz_error, 0, 0.99))
    e2 = depolarizing_error(cz_error, 2).compose(relax_a.tensor(relax_b))
    nm.add_quantum_error(e2, "cz", [a, b])
    nm.add_quantum_error(e2, "cz", [b, a])
    # Tracing b out of depolarizing(p, 2) leaves depolarizing(p, 1) on a.
    nm.add_quantum_error(
        depolarizing_error(cz_error, 1).compose(relax_a), halo_label(b), [a]
    )
    nm.add_quantum_error(
        depolarizing_error(cz_error, 1).compose(relax_b), halo_label(a), [b]
    )


def backend_rates(backend=None) -> dict:
    """Average calibration numbers from any backend ``Target``.

    Defaults to ``FakeMiami`` when ``backend`` is ``None``.  Missing entries fall
    back to representative values so the result is always a complete dict of
    ``{sx_error, cz_error, readout_error, sx_duration, cz_duration, t1, t2}``.
    """
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeMiami

        backend = FakeMiami()
    tgt = backend.target

    def _avg(name, attr):
        vals = []
        if name in tgt:
            for props in tgt[name].values():
                if props is None:
                    continue
                x = getattr(props, attr, None)
                if x is not None and not np.isnan(x):
                    vals.append(x)
        return float(np.mean(vals)) if vals else None

    t1 = [q.t1 for q in tgt.qubit_properties if q.t1 and not np.isnan(q.t1)]
    t2 = [q.t2 for q in tgt.qubit_properties if q.t2 and not np.isnan(q.t2)]
    return {
        "sx_error": _avg("sx", "error") or 3e-4,
        "cz_error": _avg("cz", "error") or 8e-3,
        "readout_error": _avg("measure", "error") or 1e-2,
        "sx_duration": _avg("sx", "duration") or 3.2e-8,
        "cz_duration": _avg("cz", "duration") or 1.6e-7,
        "t1": float(np.mean(t1)) if t1 else 3e-4,
        "t2": float(np.mean(t2)) if t2 else 2.5e-4,
    }


def _clamp_t2(t1: float, t2: float) -> float:
    """Clamp ``T2 <= 2*T1``.

    Real calibration snapshots occasionally report ``T2`` slightly above the
    physical ``2*T1`` bound; Aer's ``thermal_relaxation_error`` rejects that, so
    clamp before building the channel.
    """
    if t1 and t1 > 0:
        return min(float(t2), 2.0 * float(t1))
    return float(t2)


def build_average_noise_model(topology, rates: dict | None = None):
    """Noise model over ``topology``'s qubits/edges using machine-average rates.

    Combines depolarizing + thermal-relaxation errors on the ``sx``/``x`` gates and
    each ``cz`` edge, plus bit-flip readout error on every qubit.  ``topology`` is a
    :class:`~tessa.Lattice` or :class:`~tessa.TessaGraph`.
    """
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    if rates is None:
        rates = backend_rates()

    nm = NoiseModel(basis_gates=NATIVE_BASIS)
    t1 = rates["t1"]
    t2 = _clamp_t2(t1, rates["t2"])
    sx_d, cz_d = rates["sx_duration"], rates["cz_duration"]
    ro = float(np.clip(rates["readout_error"], 0.0, 0.5))

    for q in range(topology.n_qubits):
        e1 = depolarizing_error(float(np.clip(rates["sx_error"], 0, 0.75)), 1).compose(
            thermal_relaxation_error(t1, t2, sx_d)
        )
        nm.add_quantum_error(e1, ["sx", "x"], [q])
        nm.add_readout_error([[1 - ro, ro], [ro, 1 - ro]], [q])

    relax = thermal_relaxation_error(t1, t2, cz_d)
    for a, b in encoding_edges(topology):
        _add_cz_noise(nm, a, b, rates["cz_error"], relax, relax)

    return nm


# ----------------------------------------------------------------------
# Position-aware (per-physical-qubit) noise
# ----------------------------------------------------------------------
def _prop(target, name, key, attr, default):
    """A single InstructionProperties attribute, or ``default`` if missing/NaN."""
    try:
        props = target[name].get(key)
        val = getattr(props, attr, None)
        return float(val) if val is not None and not np.isnan(val) else default
    except Exception:
        return default


def _qubit_t(target, q, attr, default):
    try:
        val = getattr(target.qubit_properties[q], attr, None)
        return float(val) if val is not None and not np.isnan(val) else default
    except Exception:
        return default


def _edge_prop(target, name, a, b, attr, default):
    """CZ/ECR property for physical edge ``(a, b)`` (tries both orientations)."""
    for key in ((a, b), (b, a)):
        try:
            props = target[name].get(key)
            if props is not None:
                val = getattr(props, attr, None)
                if val is not None and not np.isnan(val):
                    return float(val)
        except Exception:
            pass
    return default


def build_local_noise_model(graph, backend, phys_map, fallback: dict | None = None):
    """Noise model for a small, compact ``graph`` using *per-physical-qubit* calibration.

    ``phys_map[i]`` is the physical qubit that compact qubit ``i`` sits on.  Each
    compact qubit inherits the ``sx``/readout/``T1``/``T2`` of its physical qubit,
    and each compact edge inherits the ``cz`` error/duration of the corresponding
    physical edge -- so a super-pixel is simulated with the noise of where it
    actually runs on the chip.  Missing calibration falls back to the machine
    averages (:func:`backend_rates`).
    """
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    tgt = backend.target
    if fallback is None:
        fallback = backend_rates(backend)

    nm = NoiseModel(basis_gates=NATIVE_BASIS)
    t1_by_i, t2_by_i = {}, {}
    for i in range(graph.n_qubits):
        q = int(phys_map[i])
        sx_e = _prop(tgt, "sx", (q,), "error", fallback["sx_error"])
        sx_d = _prop(tgt, "sx", (q,), "duration", fallback["sx_duration"])
        ro = _prop(tgt, "measure", (q,), "error", fallback["readout_error"])
        t1 = _qubit_t(tgt, q, "t1", fallback["t1"])
        t2 = _clamp_t2(t1, _qubit_t(tgt, q, "t2", fallback["t2"]))
        t1_by_i[i], t2_by_i[i] = t1, t2
        e1 = depolarizing_error(float(np.clip(sx_e, 0, 0.75)), 1).compose(
            thermal_relaxation_error(t1, t2, sx_d)
        )
        nm.add_quantum_error(e1, ["sx", "x"], [i])
        ro = float(np.clip(ro, 0, 0.5))
        nm.add_readout_error([[1 - ro, ro], [ro, 1 - ro]], [i])

    for a, b in encoding_edges(graph):
        pa, pb = int(phys_map[a]), int(phys_map[b])
        cz_e = _edge_prop(tgt, "cz", pa, pb, "error", fallback["cz_error"])
        cz_d = _edge_prop(tgt, "cz", pa, pb, "duration", fallback["cz_duration"])
        _add_cz_noise(
            nm,
            a,
            b,
            cz_e,
            thermal_relaxation_error(t1_by_i[a], t2_by_i[a], cz_d),
            thermal_relaxation_error(t1_by_i[b], t2_by_i[b], cz_d),
        )

    return nm
