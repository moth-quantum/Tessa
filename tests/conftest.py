"""Shared fixtures.  Everything here is offline: fake backends, small graphs, low shots."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


def _fake(name: str):
    """Instantiate a ``qiskit_ibm_runtime.fake_provider`` backend, or skip."""
    pytest.importorskip("qiskit_ibm_runtime", reason="needs the [ibm] extra")
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, name)()


@pytest.fixture(scope="session")
def data_file():
    """``data_file(name)`` -> path of an input under ``data/``, or skip the test.

    The inputs live in the repository checkout, not in the package, so a test that
    needs one is skipped when it is absent.
    """
    from tessa import data_path

    def find(name: str):
        try:
            return data_path(name)
        except FileNotFoundError as exc:
            pytest.skip(f"input not available: {exc}")

    return find


@pytest.fixture(scope="session")
def results_file():
    """``results_file(pattern)`` -> sorted paths under ``results/`` matching a glob,
    or skip the test if there are none."""
    from tessa import results_dir

    def find(pattern: str):
        try:
            hits = sorted(results_dir().glob(pattern))
        except FileNotFoundError as exc:
            pytest.skip(f"results not available: {exc}")
        if not hits:
            pytest.skip(f"no stored results match {pattern}")
        return hits

    return find


@pytest.fixture(scope="session")
def fake_fez():
    """156-qubit heavy-hex calibration snapshot."""
    return _fake("FakeFez")


@pytest.fixture(scope="session")
def fake_miami():
    """120-qubit square-lattice calibration snapshot."""
    return _fake("FakeMiami")


@pytest.fixture(scope="session")
def fez_graph(fake_fez):
    from tessa import TessaGraph

    return TessaGraph.from_backend(fake_fez)


@pytest.fixture(scope="session")
def fez_rates(fake_fez):
    from tessa import backend_rates

    return backend_rates(fake_fez)
