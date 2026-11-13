"""Symbols through the encoding: the constellation and the snapped-level schemes."""

from __future__ import annotations

import numpy as np
import pytest

from tessa import (
    DYNAMIC_RANGE_METHODS,
    Lattice,
    aer_sampler,
    constellation,
    decode,
    decode_symbols,
    encode_symbols,
    probe_qubits,
    slot_order,
    symbols_from_counts,
)


def _min_separation(pts):
    cos = pts @ pts.T - 2 * np.eye(len(pts))
    return float(np.arccos(cos.max()))


@pytest.fixture(scope="module")
def lattice():
    return Lattice(3, 4)  # 12 qubits, capacity 48


def test_constellation_is_unit_and_deterministic():
    pts = constellation(36)
    assert pts.shape == (36, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert np.array_equal(pts, constellation(36))
    assert constellation(1).shape == (1, 3)


def test_stored_constellation_points_are_used():
    lattice = Lattice(2, 2)
    symbols = np.array([0, 1, 2, 3])
    rotated = constellation(4) @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    _, default = encode_symbols(lattice, symbols, 4)
    _, given = encode_symbols(lattice, symbols, 4, points=rotated)
    assert np.array_equal(given.points, rotated)
    assert not np.allclose(given.reference, default.reference)
    with pytest.raises(ValueError, match="shape"):
        encode_symbols(lattice, symbols, 4, points=constellation(5))
    with pytest.raises(ValueError, match="unit"):
        encode_symbols(lattice, symbols, 4, points=2 * constellation(4))
    with pytest.raises(ValueError, match="constellation scheme"):
        encode_symbols(lattice, symbols, 4, scheme="levels", points=rotated)


def test_relaxing_spreads_the_points():
    assert _min_separation(constellation(16)) > _min_separation(
        constellation(16, relax=False)
    )


@pytest.mark.parametrize(
    ("scheme", "n_symbols", "per_slot"),
    [
        ("levels", 4, 1),
        ("levels", 4, 2),
        ("constellation", 6, 1),
        ("constellation", 6, 2),
    ],
)
def test_noiseless_round_trip_is_exact(lattice, scheme, n_symbols, per_slot):
    n = lattice.capacity * per_slot - 1  # leaves one slot position as padding
    symbols = np.random.default_rng(0).integers(0, n_symbols, n)
    circuit, enc = encode_symbols(
        lattice, symbols, n_symbols, scheme=scheme, per_slot=per_slot
    )
    out = decode_symbols(circuit, enc, aer_sampler(lattice, seed=1), shots=40_000)
    assert out.dtype.kind == "i"
    assert np.array_equal(out, symbols)


def test_constellation_reads_all_three_axes(lattice):
    _, enc = encode_symbols(lattice, np.arange(20) % 5, 5)
    assert enc.bases == ("z", "x", "y")
    assert enc.points.shape == (5, 3)


def test_shape_is_kept_and_per_slot_defaults_to_what_fits(lattice):
    symbols = np.random.default_rng(1).integers(0, 3, (6, 8))
    _, enc = encode_symbols(lattice, symbols, 3)
    assert enc.per_slot == 1  # 48 symbols fit one per slot
    _, enc2 = encode_symbols(lattice, np.zeros(60, int), 3)
    assert enc2.per_slot == 2 and enc2.slots == 30

    circuit, enc = encode_symbols(lattice, symbols, 3, scheme="levels")
    out = decode_symbols(circuit, enc, aer_sampler(lattice, seed=2), shots=20_000)
    assert out.shape == symbols.shape


def test_joint_and_per_group_counts_decode_alike(lattice):
    symbols = np.random.default_rng(3).integers(0, 9, 90)
    circuit, enc = encode_symbols(lattice, symbols, 9)
    sampler = aer_sampler(lattice, seed=4)
    joint = decode_symbols(circuit, enc, sampler, per_group=False, shots=40_000)
    split = decode_symbols(circuit, enc, sampler, per_group=True, shots=40_000)
    assert np.array_equal(joint, symbols)
    assert np.array_equal(split, symbols)


def test_without_snapping_the_levels_match_the_library_decode(lattice):
    """snap_theta=False must be exactly the rounding of tessa.decode's values."""
    symbols = np.random.default_rng(5).integers(0, 12, 96)
    circuit, enc = encode_symbols(lattice, symbols, 12, scheme="levels")
    values, counts = decode(
        circuit, enc.encoding, aer_sampler(lattice, seed=6), shots=500, return_counts=True
    )
    expected = np.rint(np.clip(np.concatenate(values), 0, 1) * 11).astype(int)
    got = symbols_from_counts(counts, enc, snap_theta=False)
    assert np.array_equal(got, expected[: symbols.size])


@pytest.mark.parametrize("scheme", ["levels", "constellation"])
@pytest.mark.parametrize("drr", DYNAMIC_RANGE_METHODS)
def test_every_dynamic_range_method_runs(lattice, scheme, drr):
    symbols = np.random.default_rng(7).integers(0, 5, 80)
    circuit, enc = encode_symbols(lattice, symbols, 5, scheme=scheme)
    _, counts = decode_symbols(
        circuit, enc, aer_sampler(lattice, seed=8), shots=2000, return_counts=True
    )
    out = symbols_from_counts(counts, enc, drr=drr)
    assert out.shape == symbols.shape
    assert out.min() >= 0 and out.max() < 5


def test_bad_input_is_refused(lattice):
    with pytest.raises(ValueError, match="slots"):
        encode_symbols(lattice, np.zeros(97, int), 2)
    with pytest.raises(ValueError, match="integers"):
        encode_symbols(lattice, [0.5, 1.0], 2)
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        encode_symbols(lattice, [0, 3], 3)
    with pytest.raises(ValueError, match="scheme"):
        encode_symbols(lattice, [0, 1], scheme="grid")
    with pytest.raises(ValueError, match="clamps"):
        encode_symbols(lattice, [0, 1], clamp=0.3)


def test_slot_order_puts_the_best_data_qubit_first(lattice):
    quality = np.arange(len(lattice.groups), dtype=float)  # the last qubit is best
    order = slot_order(lattice, quality)
    assert sorted(order.tolist()) == list(range(lattice.capacity))
    last = lattice.groups[-1]
    assert order[: last.n_values].tolist() == list(
        range(lattice.capacity - last.n_values, lattice.capacity)
    )
    with pytest.raises(ValueError, match="one quality score"):
        slot_order(lattice, [1.0, 2.0])


def test_a_short_sequence_round_trips_on_chosen_slots(lattice):
    symbols = np.random.default_rng(9).integers(0, 7, 20)
    order = np.arange(lattice.capacity)[::-1]  # fill from the last slot down
    circuit, enc = encode_symbols(lattice, symbols, 7, slot_order=order)
    assert enc.slot_index.tolist() == order[:20].tolist()
    out = decode_symbols(circuit, enc, aer_sampler(lattice, seed=10), shots=40_000)
    assert np.array_equal(out, symbols)
    with pytest.raises(ValueError, match="distinct"):
        encode_symbols(lattice, symbols, 7, slot_order=[0, 0] + list(range(1, 30)))


def test_a_noiseless_probe_scores_every_qubit_perfect(lattice):
    quality = probe_qubits(
        lattice, aer_sampler(lattice, seed=11), n_symbols=5, shots=40_000
    )
    assert quality.shape == (len(lattice.groups),)
    assert np.all(quality == 1.0)
