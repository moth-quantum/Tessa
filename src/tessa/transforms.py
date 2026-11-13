"""Classical data conditioning: warps, quantisation and dynamic-range recovery.

Three orthogonal knobs for getting real-world data through the *linear* angle
encoding. All are general purpose -- the image, MIDI and audio applications use
the same functions.

**Warps** (:class:`Warp`, :func:`get_warp`) -- a monotonic invertible transform
applied before normalisation and inverted after decoding. The encoder spreads
its resolution uniformly over ``[0, 1]``, so a field with two very different
scales (many short notes, a few very long ones) wastes most of that resolution
on the sparse tail. A ``log`` warp compresses the tail before encoding and
expands it again afterwards. Being exactly invertible, it adds no information.

**Quantisation** (:func:`quantize`, :func:`build_codebook`) -- either uniform
bit-quantisation to ``2**bits`` levels, or a *codebook* mode that maps the
input's distinct values to equal spacing in ``[0, 1]`` by rank (``{1, 5, 100} ->
{0, 0.5, 1}``) and snaps back on the way out. In codebook mode only the ordering
reaches the encoder.

**Dynamic-range recovery** (:func:`apply_dynamic_range`) -- rescales the noisy
reconstruction back toward the input's spread using only a few summary scalars
of the input, so the quantum output itself is preserved. Wraps the primitives in
:mod:`tessa.postprocess`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .postprocess import (
    match_mean_std,
    match_min_avg_max,
    match_min_max,
    match_percentile_range,
    rescale_to_abs_max,
)

__all__ = [
    "DYNAMIC_RANGE_METHODS",
    "WARP_NAMES",
    "FieldPrep",
    "Warp",
    "apply_dynamic_range",
    "build_codebook",
    "get_warp",
    "nearest_index",
    "prepare_field",
    "quantize",
    "restore_field",
]


# ----------------------------------------------------------------------
# Warps
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Warp:
    """A monotonic invertible transform with ``forward``/``inverse`` maps."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]

    def __call__(self, x):
        return self.forward(x)


_WARPS: dict[str, Warp] = {
    "linear": Warp(
        "linear",
        lambda x: np.asarray(x, float),
        lambda y: np.asarray(y, float),
    ),
    "log": Warp(
        "log",
        lambda x: np.log1p(np.asarray(x, float)),
        lambda y: np.expm1(np.asarray(y, float)),
    ),
    "sqrt": Warp(
        "sqrt",
        lambda x: np.sqrt(np.clip(np.asarray(x, float), 0, None)),
        lambda y: np.square(np.asarray(y, float)),
    ),
}

WARP_NAMES = tuple(_WARPS)


def get_warp(warp: str | Warp | None) -> Warp:
    """Resolve a warp name (or pass through a :class:`Warp` instance)."""
    if warp is None:
        return _WARPS["linear"]
    if isinstance(warp, Warp):
        return warp
    if warp not in _WARPS:
        raise ValueError(f"unknown warp {warp!r}; choose from {sorted(_WARPS)}")
    return _WARPS[warp]


# ----------------------------------------------------------------------
# Quantisation
# ----------------------------------------------------------------------
def quantize(values: np.ndarray, bits: int | None) -> np.ndarray:
    """Uniformly quantise ``values`` to ``2**bits`` levels over their own range.

    ``bits=None`` leaves the data untouched (full float precision).
    """
    v = np.asarray(values, dtype=float)
    if bits is None:
        return v
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return v.copy()
    levels = (1 << bits) - 1
    q = np.round((v - lo) / (hi - lo) * levels) / levels
    return lo + q * (hi - lo)


def build_codebook(values: np.ndarray, quant_levels) -> np.ndarray:
    """Sorted set of representative levels derived from ``values``.

    * ``"auto"`` / ``None`` -> every distinct value in the data.
    * an ``int L`` -> ``L`` levels drawn from the data distribution (its equally
      spaced quantiles), or all distinct values when there are fewer than ``L``.
    * a list/array -> used verbatim as the codebook.
    """
    v = np.asarray(values, dtype=float)
    if isinstance(quant_levels, (list, tuple, np.ndarray)):
        return np.unique(np.asarray(quant_levels, dtype=float))
    if quant_levels in ("auto", None):
        return np.unique(v)
    n_levels = int(quant_levels)
    uniq = np.unique(v)
    if uniq.size <= n_levels:
        return uniq
    return np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_levels)))


def nearest_index(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Index of the nearest codebook entry for each value."""
    v = np.asarray(values, dtype=float).ravel()
    cb = np.asarray(codebook, dtype=float).ravel()
    return np.abs(v[:, None] - cb[None, :]).argmin(axis=1)


# ----------------------------------------------------------------------
# Dynamic-range recovery
# ----------------------------------------------------------------------
DYNAMIC_RANGE_METHODS = (
    "none",
    "min_max",
    "percentile",
    "mean_std",
    "abs_max",
    "min_avg_max",
)


def apply_dynamic_range(
    recon: np.ndarray,
    reference: np.ndarray,
    method: str = "none",
    pct: float = 2.0,
    clip: bool = False,
) -> np.ndarray:
    """Rescale ``recon`` toward the spread of ``reference`` using ``method``.

    ``reference`` is the *input* field, in the same (possibly warped) space; only a
    few summary scalars of it are used:

    * ``"none"``        return ``recon`` unchanged.
    * ``"min_max"``     match min and max (2 scalars).
    * ``"percentile"``  match the ``pct``/``100-pct`` percentiles (2 scalars).
    * ``"mean_std"``    match mean and standard deviation (2 scalars).
    * ``"abs_max"``     match the absolute maximum (1 scalar).
    * ``"min_avg_max"`` match min, mean and max (3 scalars).

    ``clip=True`` then clips the result to ``[min, max]`` of ``reference`` (2 more
    scalars). A stretch fitted to the bulk, such as ``mean_std``, overshoots at the
    extremes; since every true value lies inside that range, clipping moves no
    value further from its target and usually brings the overshoot back. It is not
    applied to ``"none"``, which always returns ``recon`` untouched.

    The QCAT hardware analysis of the paper (Tables hw and drr) used
    ``clip=False`` on the value field only (``method="mean_std"`` in Table hw); the
    coherence run's classical route (``z_image_mae_drr``) used ``clip=True``.
    """
    recon = np.asarray(recon, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if method == "none":
        return recon
    out = _stretch(recon, ref, method, pct)
    return np.clip(out, ref.min(), ref.max()) if clip else out


def _stretch(recon: np.ndarray, ref: np.ndarray, method: str, pct: float) -> np.ndarray:
    """The rescaling of :func:`apply_dynamic_range`, before any clipping."""
    if method == "min_max":
        return match_min_max(recon, float(ref.min()), float(ref.max()))
    if method == "percentile":
        lo = float(np.percentile(ref, pct))
        hi = float(np.percentile(ref, 100.0 - pct))
        return match_percentile_range(recon, lo, hi, pct=pct)
    if method == "mean_std":
        return match_mean_std(recon, float(ref.mean()), float(ref.std()))
    if method == "abs_max":
        return rescale_to_abs_max(recon, float(np.max(np.abs(ref))))
    if method == "min_avg_max":
        return match_min_avg_max(
            recon, float(ref.min()), float(ref.mean()), float(ref.max())
        )
    raise ValueError(
        f"unknown dynamic_range method {method!r}; choose from {DYNAMIC_RANGE_METHODS}"
    )


# ----------------------------------------------------------------------
# The three knobs as one round trip
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FieldPrep:
    """What :func:`restore_field` needs to invert a :func:`prepare_field`."""

    warp_name: str
    #: The prepared input, kept as the dynamic-range reference.
    reference: np.ndarray
    codebook: np.ndarray | None = None


def prepare_field(
    data, warp: str = "linear", quant_bits: int | None = None, quant_levels=None
) -> tuple[np.ndarray, FieldPrep]:
    """Condition a field for encoding; returns ``(values, prep)``.

    Two quantisation modes:

    * **continuous** (default) -- optionally uniform-quantise to ``quant_bits``,
      then warp.
    * **codebook** (``quant_levels`` set) -- replace each value by its *rank*
      among the levels derived from the data, evenly spaced in ``[0, 1]``
      (``{1, 5, 100} -> {0, 0.5, 1}``). Only the ordering reaches the encoder;
      the values come back by snapping in :func:`restore_field`.

    The result keeps ``data``'s shape and is not normalised -- :func:`tessa.encode`
    maps whatever range it gets onto the angle range.
    """
    arr = np.asarray(data, dtype=float)
    if quant_levels is not None:
        codebook = build_codebook(arr.ravel(), quant_levels)
        span = max(codebook.size - 1, 1)
        pos = (nearest_index(arr.ravel(), codebook) / span).reshape(arr.shape)
        return pos, FieldPrep("linear", pos, codebook)
    warped = get_warp(warp).forward(quantize(arr, quant_bits))
    return warped, FieldPrep(warp, warped)


def restore_field(
    recon,
    prep: FieldPrep,
    dynamic_range: str = "none",
    pct: float = 2.0,
    clip: bool = False,
) -> np.ndarray:
    """Invert :func:`prepare_field` on a decoded field.

    Cheap and circuit-free, so decode once and call this per ``dynamic_range``
    method you want to compare. ``clip`` is passed to :func:`apply_dynamic_range`.
    """
    out = apply_dynamic_range(
        np.asarray(recon, dtype=float),
        prep.reference,
        method=dynamic_range,
        pct=pct,
        clip=clip,
    )
    if prep.codebook is None:
        return get_warp(prep.warp_name).inverse(out)
    span = max(prep.codebook.size - 1, 1)
    idx = np.clip(np.round(np.clip(out, 0.0, 1.0) * span), 0, prep.codebook.size - 1)
    return prep.codebook[idx.astype(int)]
