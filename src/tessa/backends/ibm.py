"""IBM Quantum samplers -- real hardware, or a fake-provider calibration snapshot.

Thin on purpose: ``qiskit_ibm_runtime.SamplerV2`` already satisfies the sampler
protocol (``.backend()`` + ``.run()``), so this module only resolves *which*
device -- and, for real hardware, the credentials that reach it -- and hands back
the primitive.  Requires the ``ibm`` extra.

No credentials are stored in this package. Both the instance and the API token
come from the arguments or from the environment:

========================  ==============================================
``QISKIT_IBM_INSTANCE``   IBM Cloud instance CRN (no default)
``QISKIT_IBM_TOKEN``      IBM Quantum API token (no default)
========================  ==============================================

If neither the argument nor the variable is set, ``QiskitRuntimeService`` falls back
to an account saved with ``QiskitRuntimeService.save_account``. **Never commit an
API token.**

Hardware submission is opt-in throughout: nothing contacts IBM until you build a
real sampler and run it.
"""

from __future__ import annotations

import os

__all__ = [
    "fake_backend",
    "fake_sampler",
    "ibm_instance",
    "ibm_sampler",
    "ibm_token",
    "make_service",
]


def ibm_instance(instance: str | None = None) -> str | None:
    """Resolve the instance CRN: argument -> ``$QISKIT_IBM_INSTANCE`` -> ``None``."""
    return instance or os.environ.get("QISKIT_IBM_INSTANCE") or None


def ibm_token(token: str | None = None) -> str | None:
    """Resolve the API token: argument -> ``$QISKIT_IBM_TOKEN`` -> ``None``."""
    return token or os.environ.get("QISKIT_IBM_TOKEN") or None


def make_service(token: str | None = None, instance: str | None = None):
    """Create a ``QiskitRuntimeService`` for the IBM Quantum Platform (IBM Cloud).

    ``token`` and ``instance`` fall back to the environment (see the module
    docstring). If no token is found either way, the account saved on this machine
    with ``QiskitRuntimeService.save_account`` is used.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    tok, inst = ibm_token(token), ibm_instance(instance)
    if tok is None:
        return QiskitRuntimeService(instance=inst)
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=inst)


def ibm_sampler(
    backend=None,
    *,
    service=None,
    name: str | None = None,
    token: str | None = None,
    instance: str | None = None,
    **options,
):
    """A Runtime ``SamplerV2`` bound to a real IBM device.

    Give it a ``backend`` object directly, or a ``name`` to resolve one -- via
    ``service`` if you already have one, else via :func:`make_service` with
    ``token``/``instance`` (both falling back to the environment).  ``options``
    are passed to ``SamplerV2``.

    Submitting a job is the caller's decision: nothing here contacts IBM until
    the returned sampler is actually run -- except resolving ``name``, which
    needs the service.
    """
    from qiskit_ibm_runtime import SamplerV2

    if backend is None:
        if name is None:
            raise ValueError("pass a backend object, or a name to resolve")
        if service is None:
            service = make_service(token=token, instance=instance)
        backend = service.backend(name)
    return SamplerV2(mode=backend, **options)


def fake_backend(name: str):
    """A ``qiskit_ibm_runtime.fake_provider`` backend by name.

    Accepts either the class name or its snake-case form: ``"FakeFez"``,
    ``"fake_fez"`` and ``"fez"`` all resolve to ``FakeFez``.
    """
    from qiskit_ibm_runtime import fake_provider

    candidates = [name, f"Fake{name.title().replace('_', '')}"]
    if name.lower().startswith("fake_"):
        candidates.append("Fake" + name[5:].title().replace("_", ""))
    for candidate in candidates:
        cls = getattr(fake_provider, candidate, None)
        if isinstance(cls, type):  # the module also exposes same-named submodules
            return cls()
    raise ValueError(f"no fake backend matching {name!r} in qiskit_ibm_runtime")


def fake_sampler(name: str, *, local: bool = True, seed=None, **kwargs):
    """Noisy local sampler standing in for IBM chip ``name``.

    Shorthand for :func:`tessa.backends.calibrated_sampler` on
    :func:`fake_backend` -- ``fake_sampler("fez")`` runs on ``FakeFez``'s
    topology with ``FakeFez``'s calibration, entirely offline.
    """
    from .aer import calibrated_sampler

    return calibrated_sampler(fake_backend(name), local=local, seed=seed, **kwargs)
