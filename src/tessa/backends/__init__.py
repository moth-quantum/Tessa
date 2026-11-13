"""Sampler objects -- the one thing the execution layer accepts.

:func:`tessa.decode` never takes a backend *name*, a service, or a mode string.
It takes a **sampler**: an object that runs circuits and knows its own device
(see :mod:`tessa.backends.base` for the protocol).
Which chip, which noise, which credentials -- all of that is decided once, here,
when the sampler is built.

The factories are conveniences, not requirements; any ``SamplerV2``-like object
works, including one you write yourself for a provider this package has never
heard of.

::

    from tessa import decode, encode, graph_from_backend
    from tessa.backends import aer_sampler, fake_sampler, ibm_sampler

    sampler = fake_sampler("fez")                    # offline, chip-calibrated
    graph   = graph_from_backend(sampler.backend())
    circuit, enc = encode(graph, values)
    out = decode(circuit, enc, sampler, shots=4096)

Swapping in ``ibm_sampler(name="ibm_fez")`` changes nothing else about the call.
Whether the payload is read back cell-by-cell or in one joint measurement is a
separate ``per_group`` flag on :func:`~tessa.decode`, orthogonal to the sampler.
"""

from __future__ import annotations

from .aer import aer_sampler, calibrated_sampler, noisy_sampler, topology_device
from .base import DeviceSampler, Sampler, counts_of, device_of
from .ibm import fake_backend, fake_sampler, ibm_sampler

__all__ = [
    "DeviceSampler",
    "Sampler",
    "aer_sampler",
    "calibrated_sampler",
    "counts_of",
    "device_of",
    "fake_backend",
    "fake_sampler",
    "ibm_sampler",
    "noisy_sampler",
    "topology_device",
]
