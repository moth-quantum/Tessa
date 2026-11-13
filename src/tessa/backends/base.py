"""What the execution layer requires of a "sampler", and how to satisfy it.

Everything that runs a circuit here takes a **sampler object**, never a backend
name. A sampler is anything exposing the two calls Qiskit's ``SamplerV2``
primitives already expose:

``sampler.run(pubs, shots=...) -> result``
    Execute circuits; ``result[0].data.<creg>.get_counts()`` yields the counts.
``sampler.backend() -> BackendV2-like``
    The device targeted, carrying ``num_qubits``, ``coupling_map`` and usually a
    ``target``. This is what the graph is bipartitioned from and what circuits
    are transpiled against.

``qiskit_ibm_runtime.SamplerV2`` satisfies this as-is. ``qiskit_aer``'s does not
expose ``backend()``, and its internal ``AerSimulator`` is not the device we want
to transpile against anyway, so :class:`DeviceSampler` pairs the two explicitly.

Adding a provider therefore means writing one factory that returns a
sampler-shaped object; nothing downstream branches on who made it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DeviceSampler",
    "Sampler",
    "counts_of",
    "device_of",
]


@runtime_checkable
class Sampler(Protocol):
    """Structural type of an object the execution layer can run circuits on."""

    def backend(self) -> Any:
        """The BackendV2-like device this sampler targets."""

    def run(self, pubs, *, shots: int | None = None) -> Any:
        """Execute ``pubs`` and return a ``SamplerV2``-style result."""


class DeviceSampler:
    """Bind a ``SamplerV2``-like primitive to the device it stands for.

    ``primitive`` executes; ``device`` is what we transpile against and read the
    topology from.  The two differ whenever execution is simulated: an
    ``AerSimulator`` runs the circuit while a real (or fake) chip supplies the
    coupling map, calibration and basis.

    ``name`` is cosmetic -- it only shows up in ``repr`` and error messages.
    ``noise_model`` is the Aer noise model ``primitive`` simulates with, or
    ``None`` for a noiseless one; ``decode(..., halo="traced")`` reads it.
    """

    def __init__(
        self,
        primitive: Any,
        device: Any,
        name: str | None = None,
        noise_model: Any = None,
    ) -> None:
        self.primitive = primitive
        self.device = device
        self.noise_model = noise_model
        self.name = name or getattr(device, "name", None) or type(device).__name__

    def backend(self) -> Any:
        """The device (not the executing simulator)."""
        return self.device

    def run(self, pubs, **kwargs) -> Any:
        return self.primitive.run(pubs, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = getattr(self.device, "num_qubits", "?")
        return f"DeviceSampler(name={self.name!r}, qubits={n})"


def device_of(sampler: Any) -> Any:
    """The device behind ``sampler``.

    Accepts a sampler (``.backend()`` called), a sampler exposing ``backend`` as a
    plain attribute, or a bare backend (returned unchanged) -- so helpers like
    :func:`tessa.transpile_to_backend` work with either.
    """
    backend = getattr(sampler, "backend", None)
    if backend is None:
        return sampler  # already a backend
    return backend() if callable(backend) else backend


def counts_of(result: Any, index: int = 0) -> dict:
    """Counts from a ``SamplerV2`` result, whatever the classical register is called.

    ``build_tessa_circuit(..., add_measurements=True)`` uses ``measure_all``,
    so the register is ``meas`` -- but reading the single field out of the
    ``DataBin`` keeps this working for hand-built circuits too.
    """
    data = result[index].data
    fields = list(data.keys()) if hasattr(data, "keys") else []
    if not fields:
        raise ValueError("sampler result carries no classical register to decode")
    if "meas" in fields:
        return data.meas.get_counts()
    if len(fields) > 1:
        raise ValueError(
            f"result has several classical registers {fields}; expected one "
            "(circuits built here use measure_all -> 'meas')"
        )
    return getattr(data, fields[0]).get_counts()
