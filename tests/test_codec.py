"""The encode / decode protocol: round trips, modes, readout paths, failure modes.

Most round trips here are checked *exactly*: ``exact_counts`` builds the same
basis-rotated circuits :func:`tessa.decode` would submit and reads their
probabilities off a statevector, which is a shot-free stand-in for counts. The
sampled paths are then only checked to agree with it to within shot noise.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from tessa import (
    DEFAULT_CLAMP,
    Lattice,
    TessaGraph,
    aer_sampler,
    cell_circuit,
    cell_halo,
    decode,
    decode_from_counts,
    encode,
    graph_from_backend,
    graph_from_coupling,
    measurement_circuits,
    square_lattice_edges,
    transpile_to_backend,
)
from tessa.codec import _cell_group, _measured, _trace_halo
from tessa.core import get_mode
from tessa.core.decode import decode_cells

MODES = ["single", "pair", "pair_full"]


@pytest.fixture(scope="module")
def lattice():
    return Lattice(3, 4)  # 12 qubits, capacity 48


@pytest.fixture(scope="module")
def sampler(lattice):
    return aer_sampler(lattice, seed=5)


def exact_counts(circuit, encoding) -> dict:
    """The counts :func:`decode` would collect, computed exactly from the statevector."""
    out = {}
    for basis in get_mode(encoding.mode).bases:
        qc = circuit.copy()
        if basis == "x":
            qc.h(encoding.data_qubits)
        elif basis == "y":
            qc.sdg(encoding.data_qubits)
            qc.h(encoding.data_qubits)
        out[basis] = Statevector(qc).probabilities_dict()
    return out


def fields(mode, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random(n) for _ in range(get_mode(mode).n_fields)]


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_exact_roundtrip(lattice, mode):
    values = fields(mode, lattice.capacity)
    circuit, enc = encode(lattice, *values, mode=mode)
    out = decode_from_counts(exact_counts(circuit, enc), enc)
    for got, want in zip(np.atleast_2d(out), values, strict=True):
        # Loose for the pair modes: phi comes back through arccos, which loses
        # half its digits at the ends of an unclamped range.
        assert np.abs(got - want).max() < 1e-7


@pytest.mark.parametrize(("rows", "cols"), [(1, 2), (2, 2), (3, 4), (4, 4)])
def test_exact_roundtrip_lattice_shapes(rows, cols):
    lat = Lattice(rows, cols)
    values = np.random.default_rng(1).random(lat.capacity)
    circuit, enc = encode(lat, values)
    assert (
        np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - values).max() < 1e-9
    )


def test_exact_roundtrip_on_a_heavyhex_subgraph(fez_graph):
    sub, _phys = fez_graph.connected_subgraph(10)
    values = np.random.default_rng(2).random(sub.capacity)
    circuit, enc = encode(sub, values)
    assert (
        np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - values).max() < 1e-9
    )


def test_values_come_back_in_their_own_units(lattice):
    values = np.random.default_rng(3).uniform(-40.0, 12.0, lattice.capacity)
    circuit, enc = encode(lattice, values)
    assert enc.ranges[0] == (values.min(), values.max())
    assert (
        np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - values).max() < 1e-8
    )


def test_a_fixed_value_range_is_honoured(lattice):
    values = np.full(lattice.capacity, 0.25)
    circuit, enc = encode(lattice, values, value_range=(0.0, 1.0))
    assert enc.ranges[0] == (0.0, 1.0)
    out = decode_from_counts(exact_counts(circuit, enc), enc)
    assert np.abs(out - 0.25).max() < 1e-9


def test_a_constant_field_gets_a_unit_span(lattice):
    circuit, enc = encode(lattice, np.full(lattice.capacity, 7.0))
    assert enc.ranges[0] == (7.0, 8.0)
    assert np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - 7.0).max() < 1e-9


def test_shape_is_preserved_and_the_padding_dropped(lattice):
    image = np.random.default_rng(4).random((5, 6))  # 30 of 48 slots
    circuit, enc = encode(lattice, image)
    out = decode_from_counts(exact_counts(circuit, enc), enc)
    assert out.shape == image.shape
    assert np.abs(out - image).max() < 1e-9


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def test_a_second_field_selects_a_pair_mode(lattice):
    _, enc = encode(lattice, np.zeros(4), np.ones(4))
    assert enc.mode == "pair" and enc.n_fields == 2
    assert enc.clamps == pytest.approx((DEFAULT_CLAMP, 0.0))


def test_single_mode_needs_no_clamp(lattice):
    _, enc = encode(lattice, np.zeros(4))
    assert enc.clamps == (0.0,)


def test_the_two_fields_get_their_own_clamp(lattice):
    a, b = fields("pair", lattice.capacity, seed=13)
    circuit, enc = encode(lattice, a, b, clamp=0.4, clamp_b=0.9)
    assert enc.clamps == pytest.approx((0.4, 0.9))
    out_a, out_b = decode_from_counts(exact_counts(circuit, enc), enc)
    assert np.abs(out_a - a).max() < 1e-9
    assert np.abs(out_b - b).max() < 1e-9


def test_clamp_b_defaults_by_mode(lattice):
    """The half-circle needs no phi clamp; the full circle must not close up."""
    _, half = encode(lattice, np.zeros(4), np.ones(4), clamp=0.3)
    assert half.clamps == pytest.approx((0.3, 0.0))
    _, full = encode(lattice, np.zeros(4), np.ones(4), mode="pair_full", clamp=0.3)
    assert full.clamps == pytest.approx((0.3, 0.3))


def test_a_wider_theta_clamp_makes_the_phase_easier_to_read(lattice, sampler):
    """<X> and <Y> both scale with sin(theta), so field b gains what a gives up."""
    a, b = fields("pair", lattice.capacity, seed=14)
    errors = {}
    for clamp in (0.05, DEFAULT_CLAMP):
        circuit, enc = encode(lattice, a, b, clamp=clamp)
        out_a, out_b = decode(circuit, enc, sampler, shots=100_000)
        errors[clamp] = (np.abs(out_a - a).mean(), np.abs(out_b - b).mean())
    assert errors[DEFAULT_CLAMP][1] < errors[0.05][1]


def test_clamp_b_is_rejected_for_one_field(lattice):
    with pytest.raises(ValueError, match="clamp_b is unused"):
        encode(lattice, np.zeros(4), clamp_b=0.2)


def test_the_full_circle_allows_a_zero_clamp_b(lattice):
    """A zero clamp_b identifies phi=0 and phi=2*pi -- x01=0 and x01=1 decode
    to the same phase, which is correct for a genuinely periodic field."""
    a = np.full(lattice.capacity, 0.5)
    b = np.linspace(0.0, 1.0, lattice.capacity)
    circuit, enc = encode(lattice, a, b, mode="pair_full", clamp_b=0.0)
    assert enc.clamps[1] == 0.0
    _, out_b = decode_from_counts(exact_counts(circuit, enc), enc)
    # b[-1] == 1.0 decodes as phi=0, indistinguishable from b[0] == 0.0.
    assert np.abs(out_b[:-1] - b[:-1]).max() < 1e-7
    assert out_b[-1] == pytest.approx(0.0, abs=1e-7)


def test_the_full_circle_keeps_the_ends_off_the_branch_cut(lattice, sampler):
    """Without headroom, an end value that noise pushes past the cut decodes as
    the other end -- a full-scale error. The default clamp must prevent it."""
    a = np.full(lattice.capacity, 0.5)
    b = np.linspace(0.0, 1.0, lattice.capacity)  # both ends sit at the cut
    circuit, enc = encode(lattice, a, b, mode="pair_full")
    _, out_b = decode(circuit, enc, sampler, shots=20_000)
    assert np.abs(out_b - b).max() < 0.1


def test_the_half_circle_allows_a_zero_clamp_b(lattice):
    a, b = fields("pair", lattice.capacity, seed=15)
    circuit, enc = encode(lattice, a, b, clamp_b=0.0)
    _out_a, out_b = decode_from_counts(exact_counts(circuit, enc), enc)
    assert np.abs(out_b - b).max() < 1e-7


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"clamp": 2.0}, "clamp must lie"),
        ({"clamp_b": 4.0}, "clamp_b must lie"),
    ],
)
def test_out_of_range_clamps_are_rejected(lattice, kwargs, match):
    with pytest.raises(ValueError, match=match):
        encode(lattice, np.zeros(4), np.ones(4), **kwargs)


def test_decoding_with_the_wrong_clamps_is_wrong(lattice):
    """The clamps are part of the map, so both sides must agree on them."""
    values = np.linspace(0.0, 1.0, lattice.capacity)
    circuit, enc = encode(lattice, values, value_range=(0.0, 1.0), clamp=0.4)
    assert (
        np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - values).max() < 1e-9
    )
    wrong = replace(enc, clamps=(0.0,))
    assert (
        np.abs(decode_from_counts(exact_counts(circuit, wrong), wrong) - values).max()
        > 0.1
    )


def test_pair_modes_reject_a_zero_clamp(lattice):
    with pytest.raises(ValueError, match="clamp"):
        encode(lattice, np.zeros(4), np.ones(4), clamp=0.0)


@pytest.mark.parametrize(("mode", "n_given"), [("single", 2), ("pair", 1)])
def test_mode_and_field_count_must_agree(lattice, mode, n_given):
    args = [np.zeros(4)] * n_given
    with pytest.raises(ValueError, match="field"):
        encode(lattice, *args, mode=mode)


def test_the_two_fields_must_have_the_same_shape(lattice):
    with pytest.raises(ValueError, match="same shape"):
        encode(lattice, np.zeros((2, 2)), np.zeros(4))


def test_unknown_mode_is_rejected(lattice):
    with pytest.raises(ValueError, match="unknown mode"):
        encode(lattice, np.zeros(4), mode="bloch")


def test_pair_full_gives_phi_the_full_circle(lattice):
    """The two phi ranges are different maps: both sides must agree on the mode."""
    a = np.full(lattice.capacity, 0.5)
    b = np.linspace(0.0, 1.0, lattice.capacity)
    circuit, enc = encode(lattice, a, b, mode="pair_full")
    _, out_b = decode_from_counts(exact_counts(circuit, enc), enc)
    assert np.abs(out_b - b).max() < 1e-9

    # Reading the same circuit as a half-circle payload folds phi past pi back.
    half = replace(enc, mode="pair")
    _, folded = decode_from_counts(exact_counts(circuit, half), half)
    assert np.abs(folded - b).max() > 0.4


def test_a_pair_costs_no_extra_two_qubit_gates_per_number(lattice):
    values = np.random.default_rng(6).random(lattice.capacity)
    single, _ = encode(lattice, values)
    pair, _ = encode(lattice, values, values)
    assert pair.count_ops()["cx"] == 2 * single.count_ops()["cx"]  # twice the numbers


# ----------------------------------------------------------------------
# Readout paths
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_per_group_and_joint_agree(lattice, sampler, mode):
    values = fields(mode, lattice.capacity, seed=7)
    circuit, enc = encode(lattice, *values, mode=mode)
    per_group = np.atleast_2d(decode(circuit, enc, sampler, shots=200_000))
    joint = np.atleast_2d(decode(circuit, enc, sampler, per_group=False, shots=200_000))
    for a, b, want in zip(per_group, joint, values, strict=True):
        assert np.abs(a - want).mean() < 0.01
        assert np.abs(b - want).mean() < 0.01


def test_a_cell_only_runs_its_own_qubits(lattice):
    values = np.random.default_rng(8).random(lattice.capacity)
    circuit, _ = encode(lattice, values)
    for g in lattice.groups:
        cell = cell_circuit(circuit, g)
        assert cell.num_qubits == g.degree + 1
        halo = cell_halo(lattice, g)
        assert cell_circuit(circuit, g, halo).num_qubits == g.degree + 1 + len(halo)


def test_the_halo_is_the_next_data_qubit_in_each_direction():
    lattice = Lattice(5, 5)
    centre = next(g for g in lattice.groups if g.site == (2, 2))
    straight = {lattice.qubit_index(r, c) for r, c in [(0, 2), (2, 0), (2, 4), (4, 2)]}
    assert set(cell_halo(lattice, centre, "cross")) == straight  # no diagonals
    corner = next(g for g in lattice.groups if g.site == (0, 0))
    assert set(cell_halo(lattice, corner, "cross")) == {
        lattice.qubit_index(0, 2),
        lattice.qubit_index(2, 0),
    }


def test_the_diamond_halo_adds_the_diagonals():
    lattice = Lattice(5, 5)
    centre = next(g for g in lattice.groups if g.site == (2, 2))
    sites = [(0, 2), (2, 0), (2, 4), (4, 2), (1, 1), (1, 3), (3, 1), (3, 3)]
    diamond = cell_halo(lattice, centre, "diamond")
    assert len(diamond) == 8  # each diagonal once, though it shares two addresses
    assert set(diamond) == {lattice.qubit_index(r, c) for r, c in sites}
    corner = next(g for g in lattice.groups if g.site == (0, 0))
    assert set(cell_halo(lattice, corner, "diamond")) == {
        lattice.qubit_index(r, c) for r, c in [(0, 2), (2, 0), (1, 1)]
    }


def test_an_unknown_halo_shape_is_rejected(lattice):
    with pytest.raises(ValueError, match="halo shape"):
        cell_halo(lattice, lattice.groups[0], "ring")


def test_every_heavy_hex_neighbour_is_in_the_halo(fez_graph):
    for g in fez_graph.groups:
        reached = {
            h.data_qubit
            for h in fez_graph.groups
            if h is not g and set(h.address_qubits) & set(g.address_qubits)
        }
        assert set(cell_halo(fez_graph, g)) == reached
        assert cell_halo(fez_graph, g, "cross") == cell_halo(fez_graph, g)


@pytest.mark.parametrize(("shape", "n_qubits"), [("cross", 9), ("diamond", 13)])
@pytest.mark.parametrize("mode", MODES)
def test_a_cell_with_a_halo_decodes_exactly(mode, shape, n_qubits):
    """The halo's gates only use the shared address qubits as controls."""
    lattice = Lattice(5, 5)
    values = fields(mode, lattice.capacity, seed=12)
    unit = {"value_range": (0.0, 1.0)}
    if len(values) > 1:
        unit["value_range_b"] = (0.0, 1.0)
    circuit, enc = encode(lattice, *values, mode=mode, **unit)
    g = next(g for g in lattice.groups if g.site == (2, 2))
    cell = cell_circuit(circuit, g, cell_halo(lattice, g, shape))
    assert cell.num_qubits == n_qubits

    counts = {}
    for basis in get_mode(mode).bases:
        qc = cell.copy()
        if basis == "x":
            qc.h(0)
        elif basis == "y":
            qc.sdg(0)
            qc.h(0)
        counts[basis] = Statevector(qc).probabilities_dict()
    got = decode_cells(counts, [_cell_group(g)], n_qubits, mode, enc.clamps)[g.site]
    want = lattice.pack(np.stack(values, axis=-1) if len(values) > 1 else values[0])
    assert np.abs(got - want[g.site]).max() < 1e-7


@pytest.mark.parametrize("halo", [None, "cross", "diamond", "traced"])
def test_every_halo_decodes_through_a_sampler(halo):
    lattice = Lattice(4, 4)
    values = np.random.default_rng(13).random(lattice.capacity)
    circuit, enc = encode(lattice, values, value_range=(0.0, 1.0))
    out = decode(circuit, enc, aer_sampler(lattice, seed=3), halo=halo, shots=100_000)
    assert np.abs(out - values).mean() < 0.01


# Strong noise, so a missing or misplaced error shows up in the probabilities.
LOUD_RATES = {
    "sx_error": 0.01,
    "cz_error": 0.08,
    "readout_error": 0.0,
    "sx_duration": 4e-8,
    "cz_duration": 7e-8,
    "t1": 2e-5,
    "t2": 1.5e-5,
}


def _cell_probabilities(lattice, circuit, group, basis, halo):
    """Exact noisy probabilities of the cell qubits, the halo kept or traced out."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from tessa import build_average_noise_model
    from tessa.backends.aer import topology_device

    extra = cell_halo(lattice, group, "diamond") if halo else ()
    cell = _measured(cell_circuit(circuit, group, extra), [0], basis)
    cell.remove_final_measurements()
    layout = [group.data_qubit, *group.address_qubits, *extra]
    tqc = transpile(
        cell, topology_device(lattice), initial_layout=layout, seed_transpiler=0
    )
    if halo == "traced":
        tqc = _trace_halo(tqc, extra)
    tqc.save_probabilities_dict(layout[: group.degree + 1])
    sim = AerSimulator(
        method="density_matrix",
        noise_model=build_average_noise_model(lattice, LOUD_RATES),
    )
    probs = sim.run(tqc).result().data()["probabilities"]
    return np.array([probs.get(i, 0.0) for i in range(1 << (group.degree + 1))])


@pytest.mark.parametrize("basis", ["z", "x", "y"])
def test_tracing_the_halo_out_is_exact(basis):
    """The traced cell sees the same noise as the diamond, on 5 qubits instead of 9."""
    lattice = Lattice(3, 3)  # the centre cell's halo is its four diagonals
    a, b = fields("pair_full", lattice.capacity, seed=14)
    circuit, _ = encode(lattice, a, b, mode="pair_full")
    centre = next(g for g in lattice.groups if g.site == (1, 1))
    diamond = _cell_probabilities(lattice, circuit, centre, basis, "diamond")
    traced = _cell_probabilities(lattice, circuit, centre, basis, "traced")
    bare = _cell_probabilities(lattice, circuit, centre, basis, None)
    assert np.abs(traced - diamond).max() < 1e-12
    # The halo's noise does matter: ~1e-3 here (just below 1e-3 for x and y with
    # qiskit-aer 0.18; the original test asked for > 1e-3).
    assert np.abs(bare - diamond).max() > 5e-4


def test_tracing_needs_a_known_noise_model(lattice):
    circuit, enc = encode(lattice, np.zeros(lattice.capacity))

    class Opaque:  # e.g. hardware: no noise model to trace into
        def backend(self):
            return aer_sampler(lattice).backend()

        def run(self, pubs, **kwargs):
            raise AssertionError("must not run")

    with pytest.raises(ValueError, match="simulator sampler"):
        decode(circuit, enc, Opaque(), halo="traced")

    from qiskit_aer.noise import NoiseModel

    unlabelled = aer_sampler(lattice)
    unlabelled.noise_model = NoiseModel()
    with pytest.raises(ValueError, match="halo_label"):
        decode(circuit, enc, unlabelled, halo="traced")


def test_shot_noise_shrinks_with_shots(lattice, sampler):
    values = np.random.default_rng(9).random(lattice.capacity)
    circuit, enc = encode(lattice, values, value_range=(0.0, 1.0))
    few = np.abs(decode(circuit, enc, sampler, shots=2_000) - values).mean()
    many = np.abs(decode(circuit, enc, sampler, shots=128_000) - values).mean()
    assert many < few / 3


def test_counts_can_come_back_with_the_values(lattice, sampler):
    circuit, enc = encode(lattice, np.random.default_rng(10).random(lattice.capacity))
    joint_out, joint_counts = decode(
        circuit, enc, sampler, per_group=False, shots=1024, return_counts=True
    )
    assert set(joint_counts) == {"z"}
    cell_out, cell_counts = decode(circuit, enc, sampler, shots=1024, return_counts=True)
    assert set(cell_counts) == {g.site for g in lattice.groups}
    assert joint_out.shape == cell_out.shape


def test_the_whole_decode_is_one_submission(lattice):
    """Per cell or joint, decode makes a single run() call -- one job on hardware."""

    class CountingSampler:
        def __init__(self, inner):
            self.inner, self.calls, self.pubs = inner, 0, 0
            self.noise_model = inner.noise_model

        def backend(self):
            return self.inner.backend()

        def run(self, pubs, **kwargs):
            self.calls += 1
            self.pubs += len(pubs)
            return self.inner.run(pubs, **kwargs)

    a, b = fields("pair", lattice.capacity, seed=16)
    circuit, enc = encode(lattice, a, b)
    n_cells = len(lattice.groups)

    per_cell = CountingSampler(aer_sampler(lattice, seed=0))
    decode(circuit, enc, per_cell, shots=256)
    assert per_cell.calls == 1
    assert per_cell.pubs == 2 * n_cells  # one circuit per (cell, basis)

    joint = CountingSampler(aer_sampler(lattice, seed=0))
    decode(circuit, enc, joint, per_group=False, shots=256)
    assert joint.calls == 1
    assert joint.pubs == 2  # one circuit per basis


def test_a_measured_circuit_is_rejected(lattice, sampler):
    circuit, enc = encode(lattice, np.zeros(4))
    circuit.measure_all()
    with pytest.raises(ValueError, match="unmeasured"):
        decode(circuit, enc, sampler)


def test_a_graph_larger_than_the_device_is_rejected(fez_graph):
    small = Lattice(2, 2)
    circuit, enc = encode(fez_graph, np.zeros(fez_graph.capacity))
    with pytest.raises(ValueError, match="device"):
        decode(circuit, enc, aer_sampler(small), per_group=False)


def test_too_many_values_are_rejected(lattice):
    with pytest.raises(ValueError, match="capacity"):
        encode(lattice, np.zeros(lattice.capacity + 1))


# ----------------------------------------------------------------------
# The circuit is the interchange format
# ----------------------------------------------------------------------
def test_a_gate_added_in_between_only_touches_its_own_cell(lattice):
    values = np.random.default_rng(11).random(lattice.capacity)
    circuit, enc = encode(lattice, values, value_range=(0.0, 1.0))
    circuit.x(lattice.groups[0].data_qubit)  # theta -> pi - theta, i.e. v -> 1 - v

    out = decode_from_counts(exact_counts(circuit, enc), enc)
    n0 = lattice.groups[0].n_values
    assert np.abs(out[:n0] - (1.0 - values[:n0])).max() < 1e-9
    assert np.abs(out[n0:] - values[n0:]).max() < 1e-9


def test_cell_extraction_keeps_gates_added_in_between(lattice):
    circuit, _ = encode(lattice, np.zeros(lattice.capacity))
    circuit.x(lattice.groups[0].data_qubit)
    assert cell_circuit(circuit, lattice.groups[0]).count_ops()["x"] == 1
    assert "x" not in cell_circuit(circuit, lattice.groups[1]).count_ops()


# ----------------------------------------------------------------------
# Graph constructors and transpilation
# ----------------------------------------------------------------------
def test_graph_from_coupling_infers_n_qubits():
    edges, n = square_lattice_edges(3, 4)
    graph = graph_from_coupling(edges)
    assert graph.n_qubits == n
    assert graph.capacity == Lattice(3, 4).capacity


def test_graph_from_backend_uses_physical_indices(fake_fez, fez_graph):
    assert graph_from_backend(fake_fez).capacity == fez_graph.capacity == 448
    assert fez_graph.n_qubits == fake_fez.num_qubits


def test_the_whole_chip_transpiles_without_swaps(fake_fez, fez_graph):
    circuit, _ = encode(fez_graph, np.random.default_rng(12).random(fez_graph.capacity))
    _, report = transpile_to_backend(circuit, fake_fez)
    assert report["swaps"] == 0
    assert report["n_physical_qubits"] == fez_graph.n_qubits
    assert report["two_qubit_gates"] == circuit.count_ops()["cx"]


def test_a_degree_zero_group_round_trips():
    """An isolated data qubit still holds exactly one value."""
    graph = TessaGraph.from_groups(1, [(0, [])])
    circuit, enc = encode(graph, np.array([0.3]), value_range=(0.0, 1.0))
    assert np.abs(decode_from_counts(exact_counts(circuit, enc), enc) - 0.3).max() < 1e-9


@pytest.mark.parametrize("mode", MODES)
def test_measurement_circuits_submit_then_decode(lattice, mode):
    """Submit-yourself path: run measurement_circuits, then decode_from_counts."""
    values = fields(mode, lattice.capacity, seed=3)
    circuit, enc = encode(lattice, *values, mode=mode)
    circuits = measurement_circuits(circuit, enc)
    assert list(circuits) == list(get_mode(mode).bases)
    counts = {}
    for basis, qc in circuits.items():
        assert qc.num_clbits == lattice.n_qubits  # measure_all on the whole chip
        counts[basis] = Statevector(
            qc.remove_final_measurements(inplace=False)
        ).probabilities_dict()
    out = decode_from_counts(counts, enc)
    for got, want in zip(np.atleast_2d(out), values, strict=True):
        assert np.abs(got - want).max() < 1e-7
    with pytest.raises(ValueError):
        measurement_circuits(circuits[get_mode(mode).bases[0]], enc)


@pytest.mark.parametrize("halo", [None, "diamond"])
def test_per_cell_counts_decode_as_decode_does(halo):
    """decode_from_cell_counts reads the counts decode(per_group=True) returns."""
    from tessa import decode_from_cell_counts

    lattice = Lattice(3, 3)
    a, b = fields("pair", lattice.capacity, seed=5)
    circuit, enc = encode(lattice, a, b)
    values, counts = decode(
        circuit,
        enc,
        aer_sampler(lattice, seed=2),
        halo=halo,
        shots=2000,
        return_counts=True,
    )
    n_cell = None
    if halo is not None:
        n_cell = {
            g.site: g.degree + 1 + len(cell_halo(lattice, g, halo))
            for g in lattice.groups
        }
    again = decode_from_cell_counts(counts, enc, n_cell)
    for x, y in zip(values, again, strict=True):
        assert np.array_equal(x, y)


def test_decode_from_counts_unsampled_configuration():
    """An address configuration without shots decodes to 0.5 by default, NaN on request."""
    from tessa import decode_from_counts, encode, graph_from_coupling

    graph = graph_from_coupling([(0, 1)], 2)  # data qubit 0, address qubit 1
    _, enc = encode(graph, np.array([0.2, 0.9]), value_range=(0, 1))
    counts = {"z": {"00": 90, "01": 10}}  # address qubit 1 is never 1
    default = decode_from_counts(counts, enc)
    assert default[1] == pytest.approx(0.5)
    flagged = decode_from_counts(counts, enc, unsampled="nan")
    assert flagged[0] == pytest.approx(default[0])
    assert np.isnan(flagged[1])
    with pytest.raises(ValueError):
        decode_from_counts(counts, enc, unsampled="drop")
