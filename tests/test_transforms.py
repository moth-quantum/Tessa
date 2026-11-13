"""Warps and quantisation (bit + codebook)."""

from __future__ import annotations

import numpy as np
import pytest

from tessa import WARP_NAMES, Warp, build_codebook, get_warp, quantize
from tessa.transforms import nearest_index


@pytest.mark.parametrize("name", WARP_NAMES)
def test_warp_inverse_roundtrips(name):
    x = np.linspace(0.0, 25.0, 41)
    w = get_warp(name)
    assert w.inverse(w.forward(x)) == pytest.approx(x, abs=1e-9)


@pytest.mark.parametrize("name", WARP_NAMES)
def test_warps_are_monotonic(name):
    x = np.linspace(0.0, 25.0, 41)
    assert np.all(np.diff(get_warp(name).forward(x)) > 0)


def test_concave_warps_spread_a_heavy_tail():
    """log/sqrt give the dense small values more of the [0,1] budget."""
    x = np.array([0.1, 0.2, 0.3, 100.0])  # three small values + one huge one

    def spread(values):
        v = np.asarray(values, float)
        n = (v - v.min()) / (v.max() - v.min())
        return n[2] - n[0]  # room given to the small values

    assert spread(get_warp("log").forward(x)) > spread(x)
    assert spread(get_warp("sqrt").forward(x)) > spread(x)


def test_get_warp_passthrough_and_errors():
    w = get_warp("linear")
    assert get_warp(w) is w
    assert get_warp(None).name == "linear"
    assert isinstance(w, Warp)
    assert w(np.array([2.0])) == pytest.approx([2.0])  # __call__ == forward
    with pytest.raises(ValueError, match="unknown warp"):
        get_warp("cbrt")


# ----------------------------------------------------------------------
# Quantisation
# ----------------------------------------------------------------------
def test_quantize_none_is_identity():
    x = np.array([0.123, 0.456])
    assert quantize(x, None) is not None
    assert quantize(x, None) == pytest.approx(x)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_quantize_produces_at_most_2_pow_bits_levels(bits):
    x = np.linspace(0.0, 1.0, 500)
    q = quantize(x, bits)
    assert np.unique(q).size == 2**bits
    assert np.abs(q - x).max() <= 1.0 / (2 * ((1 << bits) - 1)) + 1e-12


def test_quantize_constant_input():
    assert quantize(np.full(5, 2.0), 3) == pytest.approx(np.full(5, 2.0))


def test_codebook_auto_is_distinct_values():
    x = np.array([1.0, 5.0, 100.0, 5.0, 1.0])
    assert build_codebook(x, "auto") == pytest.approx([1.0, 5.0, 100.0])
    assert build_codebook(x, None) == pytest.approx([1.0, 5.0, 100.0])


def test_codebook_from_int_and_explicit_list():
    x = np.linspace(0.0, 1.0, 100)
    assert build_codebook(x, 5).size == 5
    assert build_codebook(x, [0.25, 0.75, 0.25]) == pytest.approx([0.25, 0.75])
    # Fewer distinct values than requested levels -> just the distinct values.
    assert build_codebook(np.array([1.0, 2.0]), 8) == pytest.approx([1.0, 2.0])


def test_nearest_index_snaps_to_closest_level():
    cb = np.array([0.0, 1.0, 10.0])
    assert list(nearest_index(np.array([0.4, 0.6, 9.0]), cb)) == [0, 1, 2]
