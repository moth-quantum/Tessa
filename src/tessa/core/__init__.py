"""Core Tessa: D/A lattice geometry, the encoding modes, circuit and decoder.

A checkerboard lattice of *data* (D) and *address* (A) qubits::

    D A D A
    A D A D
    D A D A

Each data qubit stores ``2**m`` real numbers, where ``m`` is its number of
neighbouring address qubits (4 interior, 3 edge, 2 corner). Address qubits are
shared between adjacent data qubits, which is what lets capacity exceed the
qubit count. Readout is partial tomography that recovers every cell at once.

This package is the low-level half: :mod:`~tessa.core.mode` fixes what a cell
holds, :mod:`~tessa.core.encode` builds the circuit and
:mod:`~tessa.core.decode` turns counts back into values. The protocol users
actually call lives in :mod:`tessa.codec`.
"""

from .decode import decode_cells
from .encode import build_tessa_circuit, color_data_groups
from .imaging import flat_to_image, reconstruction_error_image, rectangle_shape
from .lattice import (
    DataGroup,
    Lattice,
    encoding_coupling_map,
    encoding_edges,
    is_data_site,
    lattice_capacity,
    rectangular_layout,
)
from .mode import DEFAULT_CLAMP, MODES, Mode, as_clamps, get_mode

__all__ = [
    "DEFAULT_CLAMP",
    "MODES",
    "DataGroup",
    "Lattice",
    "Mode",
    "as_clamps",
    "build_tessa_circuit",
    "color_data_groups",
    "decode_cells",
    "encoding_coupling_map",
    "encoding_edges",
    "flat_to_image",
    "get_mode",
    "is_data_site",
    "lattice_capacity",
    "reconstruction_error_image",
    "rectangle_shape",
    "rectangular_layout",
]
