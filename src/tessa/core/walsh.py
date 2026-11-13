"""Gray-code / Walsh-Hadamard primitives for the uniformly-controlled-RY compiler.

These four helpers are all that the Tessa encoder needs from the original
upstream QPIXL ``helper`` module, re-implemented here so the package has no
dependency on that code drop.

The construction they serve: a uniformly controlled ``RY`` ("multiplexer") over
``k`` control qubits is compiled into ``2**k`` single ``RY`` gates interleaved
with ``CX`` gates.  The gate angles are the *scaled* Walsh-Hadamard transform of
the desired net angles (:func:`sfwht`), reordered into Gray-code order
(:func:`gray_permutation`) so that consecutive terms differ in exactly one
control bit and therefore need exactly one ``CX``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["count_trailing_zeros", "gray_code", "gray_permutation", "sfwht"]


def sfwht(values: np.ndarray) -> np.ndarray:
    """Scaled (unitary-normalised by ``1/2`` per stage) fast Walsh-Hadamard transform.

    Operates on a copy of ``values`` whose length must be a power of two.  The
    ``1/2`` per butterfly stage is what makes the resulting coefficients the
    actual ``RY`` gate angles: summed back up along any Gray-code path they
    reproduce the requested net angle for that control configuration.
    """
    a = np.array(values, dtype=float).ravel().copy()
    n = a.size
    if n == 0:
        return a
    if n & (n - 1):
        raise ValueError(f"sfwht needs a power-of-two length, got {n}")
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            for j in range(start, start + h):
                x, y = a[j], a[j + h]
                a[j] = (x + y) / 2.0
                a[j + h] = (x - y) / 2.0
        h *= 2
    return a


def gray_code(x: int) -> int:
    """Binary-reflected Gray code of ``x``."""
    return int(x) ^ (int(x) >> 1)


def gray_permutation(values: np.ndarray) -> np.ndarray:
    """Reorder ``values`` into Gray-code order: ``out[i] = values[gray_code(i)]``."""
    a = np.asarray(values, dtype=float).ravel()
    idx = np.array([gray_code(i) for i in range(a.size)], dtype=int)
    return a[idx]


def count_trailing_zeros(x: int, n_bits: int = 32) -> int:
    """Number of trailing zero bits of ``x``; ``n_bits`` is returned for ``x == 0``."""
    x = int(x)
    if x == 0:
        return int(n_bits)
    count = 0
    while not (x >> count) & 1:
        count += 1
    return count
