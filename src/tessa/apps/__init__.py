"""Example applications built on the Tessa encoder.

* :mod:`~tessa.apps.midi` -- two-pass MIDI codec (pitch pass + duration pass).
* :mod:`~tessa.apps.audio` -- wavetable synthesiser fed a quantum-reconstructed table.

The submodules are imported lazily so that ``import tessa.apps`` does not require the
optional ``mido``/``scipy`` dependencies (the ``audio`` extra).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import audio, midi

__all__ = ["audio", "midi"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
