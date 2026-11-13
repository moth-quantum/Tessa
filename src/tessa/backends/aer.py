"""Local (``qiskit-aer``) samplers: noiseless, hand-rated noise, or chip-calibrated.

Three factories, all returning a :class:`~tessa.backends.base.DeviceSampler`:

* :func:`aer_sampler`        -- noiseless, on a device carrying the *topology's own*
  coupling map, so the encoding never needs a SWAP and the transpiled circuit
  matches what real hardware would see.
* :func:`noisy_sampler`      -- same device, plus a uniform noise model built from a
  dict of ``rates`` (see :func:`tessa.backend_rates`).
* :func:`calibrated_sampler` -- runs *as* a real/fake chip: the device is that
  backend (physical qubit indices), and the noise model comes from its
  calibration, either per-physical-qubit or machine-average.

None of them is special-cased downstream -- :func:`tessa.decode` treats all three
exactly like a hardware sampler.
"""

from __future__ import annotations

from ..core.lattice import encoding_coupling_map
from ..noise import NATIVE_BASIS, backend_rates
from .base import DeviceSampler

_TWO_QUBIT_GATES = frozenset({"cx", "cz", "ecr", "swap", "iswap", "rzz"})

__all__ = [
    "aer_sampler",
    "calibrated_sampler",
    "noisy_sampler",
    "topology_device",
]


def topology_device(topology, basis_gates: list[str] | None = None):
    """A ``GenericBackendV2`` whose coupling map is exactly ``topology``'s edges.

    ``topology`` is a :class:`~tessa.Lattice` or :class:`~tessa.TessaGraph`.
    Carrying the encoding's own edges (rather than some default connectivity)
    means transpilation behaves identically to the real-hardware path and inserts
    no SWAPs, and it keeps compact qubit ``i`` at device index ``i``.
    """
    from qiskit.providers.fake_provider import GenericBackendV2

    edges = encoding_coupling_map(topology)
    basis = list(basis_gates or NATIVE_BASIS)
    if topology.n_qubits < 2:
        # A single-qubit device can't declare a two-qubit basis gate.
        basis = [g for g in basis if g not in _TWO_QUBIT_GATES]
    return GenericBackendV2(
        num_qubits=topology.n_qubits,
        basis_gates=basis,
        coupling_map=edges or None,
        noise_info=False,
    )


def _aer_sampler(device, noise_model=None, seed: int | None = None, **sim_options):
    """Wrap an ``AerSimulator`` executing for ``device`` as a ``DeviceSampler``."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    sim = AerSimulator(noise_model=noise_model, **sim_options)
    return DeviceSampler(
        AerSamplerV2.from_backend(sim, seed=seed), device, noise_model=noise_model
    )


def aer_sampler(topology, *, seed: int | None = None, **sim_options):
    """Noiseless local sampler on ``topology``'s own coupling map.

    The baseline: any departure from the input values is shot noise alone.  Extra
    ``sim_options`` go to ``AerSimulator`` (e.g. ``method="statevector"``).
    """
    return _aer_sampler(topology_device(topology), seed=seed, **sim_options)


def noisy_sampler(topology, *, rates: dict | None = None, seed=None, **sim_options):
    """Local sampler on ``topology`` with a uniform noise model from ``rates``.

    ``rates`` is a dict of machine-average calibration numbers -- ``None`` uses
    :func:`tessa.backend_rates` defaults, or pass ``backend_rates(FakeFez())``
    for a specific chip's averages.  Use :func:`calibrated_sampler` instead when
    you want each qubit's *own* calibration rather than one average.
    """
    from ..noise import build_average_noise_model

    return _aer_sampler(
        topology_device(topology),
        noise_model=build_average_noise_model(topology, rates),
        seed=seed,
        **sim_options,
    )


def calibrated_sampler(
    backend, *, local: bool = True, rates: dict | None = None, seed=None, **sim_options
):
    """Local stand-in for a real chip: its topology, its calibration, Aer execution.

    The device is ``backend`` itself, so qubit indices are *physical* -- build the
    graph with :func:`tessa.graph_from_backend` and every group runs on the qubits
    it would really occupy.

    ``local=True`` (default) gives each physical qubit and edge its own measured
    error rates, making the run position-aware: a block on a bad corner of the chip
    degrades more than one in a good region. ``local=False`` uses a single
    machine-average model instead -- faster to build, but it hides the worst qubits.
    ``rates`` overrides the measured averages, and ``seed``/``sim_options`` go to
    Aer.

    One noise model is built over the whole chip and reused; Aer only simulates the
    qubits a circuit touches, so this stays cheap on a 156-qubit device.
    """
    from ..graph import TessaGraph
    from ..noise import build_average_noise_model, build_local_noise_model

    graph = TessaGraph.from_backend(backend)
    if local:
        nm = build_local_noise_model(
            graph, backend, phys_map=list(range(graph.n_qubits)), fallback=rates
        )
    else:
        nm = build_average_noise_model(graph, rates or backend_rates(backend))
    return _aer_sampler(backend, noise_model=nm, seed=seed, **sim_options)
