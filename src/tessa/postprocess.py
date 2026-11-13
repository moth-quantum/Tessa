"""Amplitude post-processing (dynamic-range recovery) for noisy reconstructions.

Hardware noise tends to *compress* the dynamic range of a decoded image -- values
are pulled toward the middle, so the reconstruction looks washed out even where
the relative structure is intact. These helpers rescale it so its amplitude
statistics match the original signal again.

Every function takes only summary statistics of the input (one to three scalars)
plus the signal to correct, never the input signal itself: that models knowing a
few numbers about the intended signal, and keeps the quantum output from being
overwritten. ``NaN`` cells (blank or missing pixels) are preserved throughout.

* :func:`rescale_to_abs_max`     -- absolute maximum (1 scalar).
* :func:`match_min_max`          -- min and max (2 scalars).
* :func:`match_percentile_range` -- two percentile values (2 scalars).
* :func:`match_mean_std`         -- mean and std (2 scalars).
* :func:`match_min_avg_max`      -- min, mean and max (3 scalars).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "abs_max",
    "match_mean_std",
    "match_min_avg_max",
    "match_min_max",
    "match_percentile_range",
    "min_avg_max",
    "rescale_to_abs_max",
]


def rescale_to_abs_max(image: np.ndarray, input_abs_max: float) -> np.ndarray:
    """Scale ``image`` so its absolute maximum equals ``input_abs_max``.

    One multiplicative factor ``input_abs_max / max(|image|)``: it undoes the overall
    amplitude damping from noise without touching relative structure.
    ``input_abs_max`` is ``max(|input|)`` of the original -- the only input needed.
    ``NaN`` cells are preserved.
    """
    img = np.asarray(image, dtype=float)
    out_abs_max = np.nanmax(np.abs(img))
    if out_abs_max == 0 or not np.isfinite(out_abs_max):
        return img.copy()
    return img * (float(input_abs_max) / out_abs_max)


def match_min_avg_max(
    image: np.ndarray,
    input_min: float,
    input_avg: float,
    input_max: float,
) -> np.ndarray:
    """Rescale ``image`` to share the input's minimum, average and maximum.

    An affine stretch fixes min and max, then one monotonic gamma exponent (solved
    numerically) pins the mean, so the result has exactly the given ``input_min``,
    ``input_avg`` and ``input_max``. The map is monotonic, so pixel ordering
    survives. ``NaN`` cells are preserved.
    """
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    if not finite.any():
        return img.copy()

    o_min = float(np.nanmin(img))
    o_max = float(np.nanmax(img))
    out = np.full_like(img, np.nan)
    if o_max == o_min:
        out[finite] = float(input_avg)
        return out

    norm = (img - o_min) / (o_max - o_min)
    vals = np.clip(norm[finite], 0.0, 1.0)

    span = float(input_max) - float(input_min)
    if span == 0:
        out[finite] = float(input_min)
        return out

    t = (float(input_avg) - float(input_min)) / span
    t = min(max(t, 1e-6), 1.0 - 1e-6)

    safe = np.clip(vals, 1e-12, 1.0)

    def mean_pow(g: float) -> float:
        return float(np.mean(safe**g))

    # mean(x**gamma) is monotonically decreasing in gamma on [0, 1]; bisect in
    # log-space for the exponent that hits the requested normalised mean.
    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if mean_pow(mid) > t:
            lo = mid
        else:
            hi = mid
    gamma = np.sqrt(lo * hi)

    out[finite] = float(input_min) + span * (vals**gamma)
    return out


def match_min_max(image: np.ndarray, input_min: float, input_max: float) -> np.ndarray:
    """Affine dynamic-range recovery: rescale so ``min``/``max`` match the input.

    ``out = input_min + (input_max - input_min) * (image - image_min) /
    (image_max - image_min)`` -- stretches a noise-compressed reconstruction back
    using only the input's ``(min, max)``. ``NaN`` cells are preserved.
    """
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    out = np.full_like(img, np.nan)
    if not finite.any():
        return out
    o_min = float(np.nanmin(img))
    o_max = float(np.nanmax(img))
    if o_max == o_min:
        out[finite] = float(input_min)
        return out
    norm = (img[finite] - o_min) / (o_max - o_min)
    out[finite] = float(input_min) + (float(input_max) - float(input_min)) * norm
    return out


def match_mean_std(image: np.ndarray, input_mean: float, input_std: float) -> np.ndarray:
    """Rescale ``image`` to the input's mean and standard deviation.

    ``out = (img - img_mean) / img_std * input_std + input_mean``. Driven by the bulk
    spread rather than the two extreme samples, so it is much less outlier-sensitive
    than min/max matching. ``NaN`` cells are preserved.
    """
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    out = np.full_like(img, np.nan)
    if not finite.any():
        return out
    o_mean = float(np.nanmean(img))
    o_std = float(np.nanstd(img))
    if o_std == 0:
        out[finite] = float(input_mean)
        return out
    out[finite] = (img[finite] - o_mean) / o_std * float(input_std) + float(input_mean)
    return out


def match_percentile_range(
    image: np.ndarray, input_lo: float, input_hi: float, pct: float = 2.0
) -> np.ndarray:
    """Robust min/max matching, using percentiles instead of the extremes.

    Maps the reconstruction's ``pct`` and ``100-pct`` percentiles onto the input's
    ``input_lo``/``input_hi`` (the input's percentile values). Ignoring the tails
    makes it robust to the odd noisy outlier a strict min/max stretch would latch
    onto. Values outside the band are extrapolated, not clipped. ``NaN`` cells are
    preserved.
    """
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    out = np.full_like(img, np.nan)
    if not finite.any():
        return out
    lo = float(np.nanpercentile(img, pct))
    hi = float(np.nanpercentile(img, 100.0 - pct))
    if hi == lo:
        out[finite] = float(input_lo)
        return out
    norm = (img[finite] - lo) / (hi - lo)
    out[finite] = float(input_lo) + (float(input_hi) - float(input_lo)) * norm
    return out


def abs_max(image: np.ndarray) -> float:
    """Convenience: ``max(|image|)`` ignoring ``NaN`` (to feed the input stat)."""
    return float(np.nanmax(np.abs(np.asarray(image, dtype=float))))


def min_avg_max(image: np.ndarray) -> tuple[float, float, float]:
    """Convenience: ``(min, mean, max)`` ignoring ``NaN`` (to feed the stats)."""
    img = np.asarray(image, dtype=float)
    return float(np.nanmin(img)), float(np.nanmean(img)), float(np.nanmax(img))
