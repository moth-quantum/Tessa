"""Hardware packages: pixel images encoded, run on IBM chips and decoded.

This subpackage holds the tool that produced the pixel-art hardware results of the
paper, and the calibration snapshots of the two chips it targets.

* :mod:`~tessa.hardware.pipeline` -- a request (images + settings) -> a
  :class:`~tessa.hardware.pipeline.Plan` (circuits and bookkeeping), and decoded
  readouts -> images and metrics (:func:`~tessa.hardware.pipeline.finish`).
* :mod:`~tessa.hardware.mapping` -- which pixels each data qubit holds.
* :mod:`~tessa.hardware.runners` -- whole-chip circuits for hardware and their
  decoding, or a local Aer simulation.
* :mod:`~tessa.hardware.devices` -- ``ibm_fez`` / ``ibm_phoenix`` rebuilt offline
  from ``snapshots/*.json``.
* :mod:`~tessa.hardware.package` -- the command line,
  ``python -m tessa.hardware {check,run,resume,decode,snapshot}``.

Decoding a stored package needs only the ``ibm`` extra (to read the raw result)::

    python -m tessa.hardware decode results/hardware/pixel_art/PixelArt --out decoded
"""

from __future__ import annotations

from .devices import DEVICES, snapshot_backend
from .package import decode_folder
from .pipeline import Plan, build_plan, finish, ideal_frames
from .runners import decode_hardware, hardware_circuits, simulate

__all__ = [
    "DEVICES",
    "Plan",
    "build_plan",
    "decode_folder",
    "decode_hardware",
    "finish",
    "hardware_circuits",
    "ideal_frames",
    "simulate",
    "snapshot_backend",
]
