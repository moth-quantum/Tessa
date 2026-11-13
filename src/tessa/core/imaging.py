"""Helpers to view a reconstructed flat value vector as an image.

The capacity of a lattice or device graph is rarely a perfect square, so the flat
vector is laid out into the smallest near-square rectangle that holds it and any
leftover cells are filled with ``NaN`` (matplotlib renders ``NaN`` as blank).
"""

from __future__ import annotations

import numpy as np

__all__ = ["flat_to_image", "reconstruction_error_image", "rectangle_shape"]


def rectangle_shape(n: int) -> tuple[int, int]:
    """Smallest near-square ``(nrows, ncols)`` rectangle with ``nrows*ncols >= n``."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def flat_to_image(
    values: np.ndarray, shape: tuple[int, int] | None = None, fill: float = np.nan
) -> np.ndarray:
    """Reshape a flat value vector into a 2-D image, padding with ``fill``.

    Parameters
    ----------
    values:
        Flat array (length = lattice/graph capacity).
    shape:
        Optional ``(nrows, ncols)``.  Defaults to a near-square rectangle.
    fill:
        Value used for the leftover cells (``NaN`` -> blank pixels).
    """
    values = np.asarray(values, dtype=float).ravel()
    if shape is None:
        shape = rectangle_shape(values.size)
    nrows, ncols = shape
    if nrows * ncols < values.size:
        raise ValueError(f"shape {shape} too small for {values.size} values")
    img = np.full(nrows * ncols, fill, dtype=float)
    img[: values.size] = values
    return img.reshape(nrows, ncols)


def reconstruction_error_image(
    original: np.ndarray,
    recovered: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Absolute-error image ``|recovered - original|`` in the same layout."""
    err = np.abs(np.asarray(recovered, dtype=float) - np.asarray(original, dtype=float))
    return flat_to_image(err, shape)
