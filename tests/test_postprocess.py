"""Dynamic-range recovery primitives.

Every method must hit the requested statistics *exactly*, preserve ``NaN`` cells, stay
monotonic (structure-preserving) and consume only the 1-3 summary scalars it is given.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tessa import (
    DYNAMIC_RANGE_METHODS,
    abs_max,
    apply_dynamic_range,
    match_mean_std,
    match_min_avg_max,
    match_min_max,
    match_percentile_range,
    min_avg_max,
    rescale_to_abs_max,
)


@pytest.fixture
def compressed():
    """A noise-compressed reconstruction of a 0..1 signal, with a blank pixel."""
    rng = np.random.default_rng(0)
    img = 0.5 + 0.15 * rng.random(60)  # squashed toward the middle
    img[7] = np.nan
    return img


def test_match_min_max_hits_extremes(compressed):
    out = match_min_max(compressed, 0.0, 1.0)
    assert np.nanmin(out) == pytest.approx(0.0)
    assert np.nanmax(out) == pytest.approx(1.0)
    assert np.isnan(out[7])


def test_match_mean_std_hits_moments(compressed):
    out = match_mean_std(compressed, 0.3, 0.25)
    assert np.nanmean(out) == pytest.approx(0.3)
    assert np.nanstd(out) == pytest.approx(0.25)
    assert np.isnan(out[7])


def test_match_min_avg_max_hits_all_three(compressed):
    out = match_min_avg_max(compressed, 0.0, 0.42, 1.0)
    assert np.nanmin(out) == pytest.approx(0.0)
    assert np.nanmax(out) == pytest.approx(1.0)
    assert np.nanmean(out) == pytest.approx(0.42, abs=1e-6)
    assert np.isnan(out[7])


def test_match_min_avg_max_is_monotonic(compressed):
    """The gamma stretch must preserve pixel ordering (structure)."""
    finite = np.isfinite(compressed)
    out = match_min_avg_max(compressed, 0.0, 0.42, 1.0)
    order_in = np.argsort(compressed[finite])
    assert np.all(np.diff(out[finite][order_in]) >= -1e-12)


def test_rescale_to_abs_max(compressed):
    out = rescale_to_abs_max(compressed, 3.0)
    assert np.nanmax(np.abs(out)) == pytest.approx(3.0)
    assert np.isnan(out[7])


def test_match_percentile_range_ignores_outliers():
    img = np.concatenate([np.linspace(0.4, 0.6, 100), [50.0]])  # one wild outlier
    strict = match_min_max(img, 0.0, 1.0)
    robust = match_percentile_range(img, 0.0, 1.0, pct=2.0)
    # A strict min/max stretch collapses the body of the signal onto ~0.
    assert np.ptp(strict[:100]) < 0.05
    assert np.ptp(robust[:100]) > 0.5


def test_convenience_stats():
    img = np.array([-2.0, np.nan, 1.0, 5.0])
    assert abs_max(img) == 5.0
    lo, avg, hi = min_avg_max(img)
    assert (lo, hi) == (-2.0, 5.0)
    assert avg == pytest.approx(4 / 3)


@pytest.mark.parametrize("fn", [match_min_max, match_mean_std, match_percentile_range])
def test_all_nan_input_is_survivable(fn):
    out = fn(np.full(5, np.nan), 0.0, 1.0)
    assert np.all(np.isnan(out))


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: match_min_max(x, 0.0, 1.0),
        lambda x: match_mean_std(x, 0.0, 1.0),
        lambda x: match_min_avg_max(x, 0.0, 0.5, 1.0),
        lambda x: match_percentile_range(x, 0.0, 1.0),
        lambda x: rescale_to_abs_max(x, 1.0),
    ],
)
def test_constant_input_does_not_blow_up(fn):
    out = fn(np.full(6, 0.3))
    assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------
def test_dispatcher_covers_every_advertised_method(compressed):
    ref = np.linspace(0.0, 1.0, compressed.size)
    for method in DYNAMIC_RANGE_METHODS:
        out = apply_dynamic_range(np.nan_to_num(compressed), ref, method=method)
        assert out.shape == compressed.shape


def test_dispatcher_none_is_identity():
    x = np.array([0.1, 0.2])
    assert apply_dynamic_range(x, np.array([0.0, 1.0]), method="none") is not None
    assert apply_dynamic_range(x, np.array([0.0, 1.0]), method="none") == pytest.approx(x)


def test_dispatcher_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown dynamic_range method"):
        apply_dynamic_range(np.zeros(3), np.zeros(3), method="magic")


def test_recovery_uses_only_summary_scalars():
    """Invariant: the primitives take scalars, never the input signal itself.

    Each ``match_*`` / ``rescale_*`` primitive must accept the image plus at most
    three floats -- that is what keeps the quantum output intact rather than
    substituting the known answer.
    """
    for fn in (
        rescale_to_abs_max,
        match_min_max,
        match_mean_std,
        match_percentile_range,
        match_min_avg_max,
    ):
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "image"
        scalars = [p for p in params[1:] if p.name != "pct"]
        assert 1 <= len(scalars) <= 3, fn.__name__


@pytest.mark.parametrize("method", [m for m in DYNAMIC_RANGE_METHODS if m != "none"])
def test_clip_never_moves_a_value_further_from_its_target(method):
    """Clipping to the reference's range is a projection onto a set holding the truth."""
    rng = np.random.default_rng(7)
    truth = rng.random(400) * 0.6 + 0.2  # range [0.2, 0.8]
    recon = (
        0.5 + 0.7 * (truth - 0.5) + rng.normal(0, 0.08, truth.size)
    )  # compressed, noisy
    plain = apply_dynamic_range(recon, truth, method=method)
    clipped = apply_dynamic_range(recon, truth, method=method, clip=True)
    assert clipped.min() >= truth.min() and clipped.max() <= truth.max()
    assert np.all(np.abs(clipped - truth) <= np.abs(plain - truth) + 1e-12)


def test_clip_leaves_none_untouched():
    x = np.array([-0.5, 0.5, 1.5])
    out = apply_dynamic_range(x, np.array([0.0, 1.0]), method="none", clip=True)
    assert out == pytest.approx(x)


def test_clip_helps_mean_std_on_a_saturated_signal():
    """Many values at the extremes: the bulk-fitted stretch overshoots, the clip recovers."""
    truth = np.r_[np.zeros(40), np.ones(40), np.linspace(0, 1, 20)]
    recon = 0.5 + 0.6 * (truth - 0.5)
    plain = np.abs(apply_dynamic_range(recon, truth, method="mean_std") - truth).mean()
    clipped = np.abs(
        apply_dynamic_range(recon, truth, method="mean_std", clip=True) - truth
    ).mean()
    assert clipped <= plain
