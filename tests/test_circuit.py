"""Encoder invariants: the angle convention, the multiplexer bit order, depth.

These are load-bearing properties of the method -- if any of them regresses the
science is wrong, not just the code.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from tessa import DEFAULT_CLAMP, Lattice, TessaGraph, build_tessa_circuit
from tessa.core import MODES, decode_cells, get_mode
from tessa.core.walsh import count_trailing_zeros, gray_code, gray_permutation, sfwht


# ----------------------------------------------------------------------
# Angle convention
# ----------------------------------------------------------------------
def test_angle_convention_is_linear_and_invertible():
    mode = get_mode("single")
    v = np.linspace(0.0, 1.0, 11)
    ang = mode.angles(v, 0.0)
    assert ang[0] == 0.0
    assert ang[-1] == pytest.approx(np.pi)
    assert np.all(np.diff(ang) > 0)  # monotone => invertible from statistics
    assert mode.values_from_angles(ang, 0.0) == pytest.approx(v)


def test_clamp_keeps_theta_off_the_poles():
    mode = get_mode("pair")
    ang = mode.angles(np.array([0.0, 1.0]), DEFAULT_CLAMP)
    assert ang == pytest.approx([DEFAULT_CLAMP, np.pi - DEFAULT_CLAMP])


def test_each_field_reads_its_own_clamp():
    mode = get_mode("pair")
    x = np.array([0.0, 1.0])
    assert mode.angles(x, (0.1, 0.7)) == pytest.approx([0.1, np.pi - 0.1])
    assert mode.phases(x, (0.1, 0.7)) == pytest.approx([0.7, np.pi - 0.7])
    assert mode.values_from_angles(
        mode.angles(x, (0.1, 0.7)), (0.1, 0.7)
    ) == pytest.approx(x)
    assert mode.values_from_phases(
        mode.phases(x, (0.1, 0.7)), (0.1, 0.7)
    ) == pytest.approx(x)


def test_a_zero_clamp_b_identifies_the_full_circles_ends():
    """x01=0 and x01=1 decode to the same phase -- correct for periodic data,
    but only distinguishable up to the exact wrap point."""
    mode = get_mode("pair_full")
    phi = mode.phases(np.array([0.0, 1.0]), (0.0, 0.0))
    assert phi == pytest.approx([0.0, 2 * np.pi])
    assert mode.values_from_phases(phi, (0.0, 0.0)) == pytest.approx([0.0, 0.0])


def test_clamp_b_buys_margin_around_the_branch_cut():
    """A small negative perturbation on a true phase near 0 wraps mod 2*pi to
    near 2*pi and decodes near x01=1 -- the failure clamp_b > 0 guards against.
    With headroom the same perturbation stays close to the true value."""
    mode = get_mode("pair_full")
    true_x01 = 0.001
    noise = -0.01  # a small negative phase error, e.g. from shot noise

    tight_phi = mode.phases(np.array([true_x01]), (0.0, 0.0))[0] + noise
    tight_out = mode.values_from_phases(np.array([tight_phi]), (0.0, 0.0))[0]
    assert tight_out == pytest.approx(1.0, abs=0.01)  # flipped to the other end

    padded_phi = mode.phases(np.array([true_x01]), (0.0, DEFAULT_CLAMP))[0] + noise
    padded_out = mode.values_from_phases(np.array([padded_phi]), (0.0, DEFAULT_CLAMP))[0]
    assert padded_out == pytest.approx(true_x01, abs=0.01)  # unaffected


def test_only_the_full_circle_mode_passes_pi():
    x = np.array([0.0, 1.0])
    assert MODES["pair"].phases(x, 0.0).max() == pytest.approx(np.pi)
    assert MODES["pair_full"].phases(x, 0.0).max() == pytest.approx(2 * np.pi)


def test_value_maps_to_p1_as_sin_squared():
    """value -> P(|1>) = sin^2(pi*value/2)."""
    graph = TessaGraph.from_groups(2, [(0, [1])])
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        qc = build_tessa_circuit(graph, np.array([value, value]))
        probs = Statevector(qc).probabilities([0])  # data qubit
        assert probs[1] == pytest.approx(np.sin(np.pi * value / 2) ** 2, abs=1e-12)


# ----------------------------------------------------------------------
# Walsh / Gray primitives
# ----------------------------------------------------------------------
def test_sfwht_involution_and_scaling():
    a = np.random.default_rng(0).random(8)
    # The scaled transform is its own inverse up to the 1/2-per-stage factor.
    assert sfwht(sfwht(a)) * 8 == pytest.approx(a)
    assert sfwht(np.ones(4)) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_gray_helpers():
    assert [gray_code(i) for i in range(8)] == [0, 1, 3, 2, 6, 7, 5, 4]
    for i in range(15):  # consecutive Gray words differ in exactly one bit
        assert bin(gray_code(i) ^ gray_code(i + 1)).count("1") == 1
    assert gray_permutation(np.arange(4.0)) == pytest.approx([0, 1, 3, 2])
    assert count_trailing_zeros(0b1000) == 3
    assert count_trailing_zeros(0, n_bits=5) == 5


# ----------------------------------------------------------------------
# Bit order: a hand-checked 1-data + 2-address case
# ----------------------------------------------------------------------
def test_hand_checked_two_address_bit_order():
    """Address integer i must select value i: ``address_qubits[j]`` carries bit j.

    With one data qubit (0) and address qubits (1, 2), the local address integer
    is ``bits[1] + 2*bits[2]``. Encoding four distinct values and reading the
    exact conditional probabilities must return them in that order -- this pins
    the Gray-code mapping ``j -> address_qubits[k-1-j]`` in the multiplexer, and
    the absence of any global ``reverse_bits``.
    """
    graph = TessaGraph.from_groups(3, [(0, [1, 2])])
    values = np.array([0.1, 0.4, 0.7, 0.95])
    probs = Statevector(build_tessa_circuit(graph, values)).probabilities_dict()

    recovered = []
    for addr in range(4):
        b1, b2 = addr & 1, (addr >> 1) & 1
        # Qiskit keys print the highest qubit index leftmost: q2 q1 q0.
        p0 = probs.get(f"{b2}{b1}0", 0.0)
        p1 = probs.get(f"{b2}{b1}1", 0.0)
        recovered.append(2 * np.arctan2(np.sqrt(p1), np.sqrt(p0)) / np.pi)
    assert np.array(recovered) == pytest.approx(values, abs=1e-12)

    # And the library decoder agrees, reading the same probabilities as counts.
    assert decode_cells({"z": probs}, graph.groups, graph.n_qubits)[0] == pytest.approx(
        values, abs=1e-12
    )


def test_counts_are_read_by_qubit_index_not_register_order():
    graph = TessaGraph.from_groups(3, [(0, [1, 2])])
    # Counts by hand: address 0 -> data always 1, the others -> data always 0.
    counts = {"000": 0, "001": 100, "010": 100, "100": 100, "110": 100}
    out = decode_cells({"z": counts}, graph.groups, graph.n_qubits)[0]
    assert out[0] == pytest.approx(1.0)  # addr 0 saw only data=1
    assert out[1] == pytest.approx(0.0)  # addr 1 (q1=1) saw only data=0
    assert out[2] == pytest.approx(0.0)  # addr 2 (q2=1) saw only data=0


# ----------------------------------------------------------------------
# Interleaving
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("rows", "cols"), [(3, 4), (4, 4)])
def test_parallel_gives_the_same_state_at_lower_depth(rows, cols):
    lat = Lattice(rows, cols)
    v = np.random.default_rng(5).random(lat.capacity)
    seq = build_tessa_circuit(lat, v, strategy="sequential")
    par = build_tessa_circuit(lat, v, strategy="parallel")
    assert abs(Statevector(seq).inner(Statevector(par))) ** 2 == pytest.approx(
        1.0, abs=1e-9
    )
    assert par.count_ops() == seq.count_ops()
    assert par.depth() < seq.depth()


def test_reference_depths_3x4():
    """The documented 68 -> 34 depth collapse on the 3x4 grid."""
    lat = Lattice(3, 4)
    v = np.random.default_rng(42).random(lat.capacity)
    assert build_tessa_circuit(lat, v, strategy="sequential").depth() == 68
    assert build_tessa_circuit(lat, v, strategy="parallel").depth() == 34


def test_unknown_strategy_raises():
    lat = Lattice(2, 2)
    with pytest.raises(ValueError, match="unknown strategy"):
        build_tessa_circuit(lat, np.zeros(lat.capacity), strategy="diagonal")


# ----------------------------------------------------------------------
# Compression, shape, measurements
# ----------------------------------------------------------------------
def test_compression_reduces_gates():
    lat = Lattice(3, 4)
    v = np.random.default_rng(4).random(lat.capacity)
    plain = build_tessa_circuit(lat, v).count_ops()["cx"]
    squeezed = build_tessa_circuit(lat, v, compression=50).count_ops()["cx"]
    assert squeezed < plain


def test_the_circuit_is_never_measured():
    lat = Lattice(2, 2)
    qc = build_tessa_circuit(lat, np.zeros(lat.capacity))
    assert qc.num_clbits == 0
    assert "measure" not in qc.count_ops()


@pytest.mark.parametrize("mode", ["single", "pair"])
def test_the_payload_shape_must_match_the_mode(mode):
    lat = Lattice(2, 2)
    wrong = np.zeros((lat.capacity, 2)) if mode == "single" else np.zeros(lat.capacity)
    with pytest.raises(ValueError, match="expects values shaped"):
        build_tessa_circuit(lat, wrong, mode=mode, clamps=DEFAULT_CLAMP)
